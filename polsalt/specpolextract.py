
"""
specpolextract

Background subtract and extract spectropolarimetric data.  The end of the image-specific pipeline.

"""

import os, sys, glob, shutil, inspect

import numpy as np
import pyfits
from scipy.interpolate import interp1d
from scipy import linalg as la

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

from oksmooth import blksmooth2d
from specpolwavmap import configmap
from specpollampextract import specpollampextract
from specpolsignalmap import specpolsignalmap
from skysub2d_khn import make_2d_skyspectrum
from scrunch1d import scrunch1d
from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging

# np.seterr(invalid='raise')
np.set_printoptions(threshold=np.nan)
debug = True

# ---------------------------------------------------------------------------------------------
def bkgsub(hdulist,badbinbkg_orc,isbkgcont_orc,skyflat_orc,maprow_ocd,tnum):
    """
    Remove remaining detector bias and sky continuum
    Use Kotulla 2D sky subtract to remove sky lines

    Parameters
    ----------
    hdu: fits.HDUList
       polarimetric image cube for processing
    badbinbkg_orc: np 3D boolean array
       bins to be avoided in background estimation
    isbkgcont_orc: np 3D boolean array
       bins used for sky continuum background estimation (avoids target, sky lines)
    skyflat_orc: np 3D float array
       skyflat estimate, divided out before sky line removal
    maprow_ocd: np 3D float array
       row number of edge of fov and target, top and bottom, vs column for O,E 
    tnum: int
       image number (to label debug fits files)

    Output
    ------
    target_orc: np 3D float array 
        background-subtracted image

    """

    sci_orc = np.copy(hdulist['SCI'].data)
    wav_orc = hdulist['WAV'].data

    rows,cols = hdulist['SCI'].data.shape[1:3]
    cbin,rbin = np.array(hdulist[0].header["CCDSUM"].split(" ")).astype(int)
    slitid = hdulist[0].header["MASKID"]
    if slitid[0] =="P": slitwidth = float(slitid[2:5])/10.
    else: slitwidth = float(slitid)         # in arcsec

  # evaluate detector bias, for the record, but don't use it 
  # use median of rows outside of fov at bottom and top of full image (bottom of O, top of E)
    dtrbkg_dc = np.zeros((2,cols))
    dtrbkg_dc[0] = np.median(sci_orc[0,2:(maprow_ocd[0,:,0].min()-2)],axis=0)
    dtrbkg_dc[1] = np.median(sci_orc[1,(maprow_ocd[1,:,3].max()+2):(rows-2)],axis=0)

    target_orc = sci_orc

  # make background continuum image, smoothed over resolution element
    rblk,cblk = int(1.5*8./rbin), int(slitwidth*8./cbin)
    isedge_orc = (np.arange(rows)[:,None] < maprow_ocd[:,None,:,0]) | \
          (np.arange(rows)[:,None] > maprow_ocd[:,None,:,3])
    isskycont_orc = (((np.arange(rows)[:,None] < maprow_ocd[:,None,:,0]+rows/16) |  \
          (np.arange(rows)[:,None] > maprow_ocd[:,None,:,3]-rows/16)) & ~isedge_orc)  

    for o in (0,1):
        bkgcont_rc = blksmooth2d(target_orc[o],isbkgcont_orc[o],rblk,cblk,0.25,mode="mean")
           
    # remove sky continuum: ends of bkg continuum * skyflat
        skycont_c = (bkgcont_rc.T[isskycont_orc[o].T]/skyflat_orc[o].T[isskycont_orc[o].T])  \
           .reshape((cols,-1)).mean(axis=1)
        skycont_rc = skycont_c*skyflat_orc[o]
        
    # remove sky lines: image - bkgcont run through 2d sky averaging
        objdata_rc = ((target_orc[o] - bkgcont_rc)/skyflat_orc)[o]
        objdata_rc[badbinbkg_orc[o]] = np.nan

#        pyfits.PrimaryHDU(objdata_rc.astype('float32')).writeto('objdata_'+tnum+'_'+str(o)+'.fits',clobber=True)

        skylines_rc = make_2d_skyspectrum(objdata_rc,wav_orc[o],np.array([[0,rows],]))*skyflat_orc[o]
        target_orc[o] -= skycont_rc + skylines_rc

#        pyfits.PrimaryHDU(skylines_rc.astype('float32')).writeto('skylines_rc_'+tnum+'_'+str(o)+'.fits',clobber=True)
#        pyfits.PrimaryHDU(skycont_rc.astype('float32')).writeto('skycont_rc_'+tnum+'_'+str(o)+'.fits',clobber=True)

    return target_orc,dtrbkg_dc

# ---------------------------------------------------------------------------------
def specpolextract(infilelist, logfile='salt.log', debug=False):
    """Produce a 1-D extract spectra for the O and E beams

    This also cleans the 2-D spectra of a number of artifacts, removes the background, accounts for small 
        spatial shifts in the observation, and resamples the data into a wavelength grid

    Parameters
    ----------
    infile_list: list
        List of filenames that include an extracted spectra

    logfile: str
        Name of file for logging


    """

#set up the files
    obsdate=os.path.basename(infilelist[0])[8:16]

    with logging(logfile, debug) as log:
        #create the observation log
        obs_dict=obslog(infilelist)
    # get rid of arcs
        for i in range(len(infilelist))[::-1]:
            if (obs_dict['OBJECT'][i].upper().strip()=='ARC'): del infilelist[i]            
        infiles = len(infilelist)

    # contiguous images of the same object and config are grouped together as an observation
        obs_dict=obslog(infilelist)
        confno_i,confdatlist = configmap(infilelist)
        configs = len(confdatlist)
        objectlist = list(set(obs_dict['OBJECT']))
        objno_i = np.array([objectlist.index(obs_dict['OBJECT'][i]) for i in range(infiles)],dtype=int)
        obs_i = np.zeros((infiles),dtype=int)
        obs_i[1:] = ((objno_i[1:] != objno_i[:-1]) | (confno_i[1:] != confno_i[:-1]) ).cumsum()
        dum,iarg_b =  np.unique(obs_i,return_index=True)    # gives i for beginning of each obs
        obss = iarg_b.shape[0]
        obscount_b = np.zeros((obss),dtype=int)
        oclist_b = np.array([[objno_i[iarg_b[b]], confno_i[iarg_b[b]]] for b in range(obss)])        
        if obss>1:
            for b in range(1,obss): 
                obscount_b[b] = (oclist_b[b]==oclist_b[0:b]).all(axis=1).sum()

        for b in range(obss):
            ilist = np.where(obs_i==b)[0]
            outfiles = len(ilist)
            outfilelist = [infilelist[i] for i in ilist]
            obs_dict=obslog(outfilelist)
            imagenolist = [int(os.path.basename(infilelist[i]).split('.')[0][-4:]) for i in ilist]
            log.message('\nExtract: '+objectlist[objno_i[ilist[0]]]+'  Grating %s  Grang %6.2f  Artic %6.2f' % \
               confdatlist[confno_i[ilist[0]]], with_header=False)
            log.message('  Images: '+outfiles*'%i ' % tuple(imagenolist), with_header=False)
            hdu0 =  pyfits.open(outfilelist[0])       
            rows,cols = hdu0['SCI'].data.shape[1:3]
            cbin,rbin = np.array(obs_dict["CCDSUM"][0].split(" ")).astype(int)

        # special version for lamp data
            lampid = obs_dict["LAMPID"][0].strip().upper()
            if lampid!="NONE":
                specpollampextract(outfilelist, logfile=logfile)           
                continue

        # sum spectra to find target, background artifacts, and estimate sky flat and psf functions
            count = 0
            for i in range(outfiles):
                badbin_orc = pyfits.open(outfilelist[i])['BPM'].data > 0
                if count == 0: 
                    count_orc = (~badbin_orc).astype(int)
                    image_orc = pyfits.open(outfilelist[i])['SCI'].data*count_orc
                    var_orc = pyfits.open(outfilelist[i])['VAR'].data*count_orc
                else:
                    count_orc += (~badbin_orc).astype(int)
                    image_orc += pyfits.open(outfilelist[i])['SCI'].data*(~badbin_orc).astype(int)
                    var_orc += pyfits.open(outfilelist[i])['VAR'].data*(~badbin_orc).astype(int)
                count += 1
            if count ==0:
                print 'No valid images'
                continue
            image_orc[count_orc>0] /= count_orc[count_orc>0]
            badbinall_orc = (count_orc==0) | (image_orc==0)             # bin is bad in all images
            badbinone_orc = (count_orc < count) | (image_orc==0)        # bin is bad in at least one image
            var_orc[count_orc>0] /= (count_orc[count_orc>0])**2

            wav_orc = pyfits.open(outfilelist[0])['WAV'].data
            slitid = obs_dict["MASKID"][0]
            if slitid[0] =="P": slitwidth = float(slitid[2:5])/10.
            else: slitwidth = float(slitid) 

            obsname = objectlist[oclist_b[b][0]]+"_c"+str(oclist_b[b][1])+"_"+str(obscount_b[b])
            hdusum = pyfits.PrimaryHDU(header=hdu0[0].header)   
            hdusum = pyfits.HDUList(hdusum)
            hdusum[0].header.update('OBJECT',obsname)     
            header=hdu0['SCI'].header.copy()       
            hdusum.append(pyfits.ImageHDU(data=image_orc, header=header, name='SCI'))
            hdusum.append(pyfits.ImageHDU(data=var_orc, header=header, name='VAR'))
            hdusum.append(pyfits.ImageHDU(data=badbinall_orc.astype('uint8'), header=header, name='BPM'))
            hdusum.append(pyfits.ImageHDU(data=wav_orc, header=header, name='WAV'))

            psf_orc,skyflat_orc,badbinnew_orc,isbkgcont_orc,maprow_od,drow_oc = \
                specpolsignalmap(hdusum,logfile=logfile,debug=debug)
 
            maprow_ocd = maprow_od[:,None,:] + np.zeros((2,cols,4)) 
            maprow_ocd -= drow_oc[:,:,None]      

            isedge_orc = (np.arange(rows)[:,None] < maprow_ocd[:,None,:,0]) | \
                (np.arange(rows)[:,None] > maprow_ocd[:,None,:,3])
            istarget_orc = (np.arange(rows)[:,None] > maprow_ocd[:,None,:,1]) & \
                (np.arange(rows)[:,None] < maprow_ocd[:,None,:,2])
                                   
            isbkgcont_orc &= (~badbinall_orc & ~isedge_orc & ~istarget_orc)
            badbinall_orc |= badbinnew_orc
            badbinone_orc |= badbinnew_orc
            hdusum['BPM'].data = badbinnew_orc.astype('uint8')

            if debug: 
                hdusum.writeto(obsname+".fits",clobber=True)
#               pyfits.PrimaryHDU(var_orc.astype('float32')).writeto('var_orc.fits',clobber=True) 
#               pyfits.PrimaryHDU(badbinnew_orc.astype('uint8')).writeto('badbinnew_orc.fits',clobber=True)   
#               pyfits.PrimaryHDU(badbinall_orc.astype('uint8')).writeto('badbinall_orc.fits',clobber=True)  
#               pyfits.PrimaryHDU(badbinone_orc.astype('uint8')).writeto('badbinone_orc.fits',clobber=True)  

        # scrunch and normalize psf from summed images (using badbinone) for optimized extraction
        # psf is normalized so its integral over row is 1.
            psfnormmin = 0.70    # wavelengths with less than this flux in good bins are marked bad
            wbin = wav_orc[0,rows/2,cols/2]-wav_orc[0,rows/2,cols/2-1] 
            wbin = float(int(wbin/0.75))
            wmin,wmax = wav_orc.min(axis=2).max(),wav_orc.max(axis=2).min()
            wedgemin = wbin*int(wmin/wbin+0.5) + wbin/2.
            wedgemax = wbin*int(wmax/wbin-0.5) + wbin/2.
            wedge_w = np.arange(wedgemin,wedgemax+wbin,wbin)
            wavs = wedge_w.shape[0] - 1
            binedge_orw = np.zeros((2,rows,wavs+1))
            psf_orw = np.zeros((2,rows,wavs))
            specrow_or = maprow_od[:,1:3].mean(axis=1)[:,None] + np.arange(-rows/4,rows/4)

            for o in (0,1):
                for r in specrow_or[o]:
                    binedge_orw[o,r] = interp1d(wav_orc[o,r],np.arange(cols))(wedge_w)
                    psf_orw[o,r] = scrunch1d(psf_orc[o,r],binedge_orw[o,r])
            psf_orw /= psf_orw.sum(axis=1)[:,None,:]

#            if debug: 
#               pyfits.PrimaryHDU(psf_orw.astype('float32')).writeto(obsname+'_psf_orw.fits',clobber=True)
#               pyfits.PrimaryHDU(var_orw.astype('float32')).writeto(obsname+'_var_orw.fits',clobber=True) 

        # set up optional image-dependent column shift for slitless data
            colshiftfilename = "colshift.txt"
            docolshift = os.path.isfile(colshiftfilename)
            if docolshift:
                img_I,dcol_I = np.loadtxt(colshiftfilename,dtype=float,unpack=True,usecols=(0,1))
                shifts = img_I.shape[0]
                log.message('Column shift: \n Images '+shifts*'%5i ' % tuple(img_I), with_header=False)                 
                log.message(' Bins    '+shifts*'%5.2f ' % tuple(dcol_I), with_header=False)                 
               
        # background-subtract and extract spectra
            dtrbkg_dic = np.zeros((2,outfiles,cols))
            for i in range(outfiles):
                hdulist = pyfits.open(outfilelist[i])
                tnum = os.path.basename(outfilelist[i]).split('.')[0][-3:]
                badbin_orc = (hdulist['BPM'].data > 0)
                badbinbkg_orc = (badbin_orc | badbinnew_orc | isedge_orc | istarget_orc)
                target_orc,dtrbkg_dic[:,i,:] = bkgsub(hdulist,badbinbkg_orc,isbkgcont_orc,skyflat_orc,maprow_ocd,tnum)
#                pyfits.PrimaryHDU(target_orc.astype('float32')).writeto('target_'+tnum+'_orc.fits',clobber=True)
                target_orc *= (~badbin_orc).astype(int)             
#                pyfits.PrimaryHDU(target_orc.astype('float32')).writeto('target_'+tnum+'_orc.fits',clobber=True)
                var_orc = hdulist['var'].data
                badbin_orc = (hdulist['bpm'].data > 0) | badbinnew_orc
            # extract spectrum optimally (Horne, PASP 1986)
                target_orw = np.zeros((2,rows,wavs))   
                var_orw = np.zeros_like(target_orw)
                badbin_orw = np.ones((2,rows,wavs),dtype='bool')   
                wt_orw = np.zeros_like(target_orw)
                dcol = 0.
                if docolshift:
                    if int(tnum) in img_I:
                        dcol = dcol_I[np.where(img_I==int(tnum))]    # table has observed shift
                for o in (0,1):
                    for r in specrow_or[o]:
                        target_orw[o,r] = scrunch1d(target_orc[o,r],binedge_orw[o,r]+dcol)
                        var_orw[o,r] = scrunch1d(var_orc[o,r],binedge_orw[o,r]+dcol)
                        badbin_orw[o,r] = scrunch1d(badbin_orc[o,r].astype(float),binedge_orw[o,r]+dcol) > 0.001 
                badbin_orw |= (var_orw == 0)
                badbin_orw |= ((psf_orw*(~badbin_orw)).sum(axis=1)[:,None,:] < psfnormmin)
#                pyfits.PrimaryHDU(var_orw.astype('float32')).writeto('var_'+tnum+'_orw.fits',clobber=True)
#                pyfits.PrimaryHDU(badbin_orw.astype('uint8')).writeto('badbin_'+tnum+'_orw.fits',clobber=True)
  
            # use master psf shifted in row to allow for guide errors
                pwidth = 2*int(1./psf_orw.max())
                ok_w = ((psf_orw*badbin_orw).sum(axis=1) < 0.03/float(pwidth/2)).all(axis=0)
                crosscor_s = np.zeros(pwidth)
                for s in range(pwidth):
                    crosscor_s[s] = (psf_orw[:,s:s-pwidth]*target_orw[:,pwidth/2:-pwidth/2]*ok_w).sum()
                smax = np.argmax(crosscor_s)
                s_S = np.arange(smax-pwidth/4,smax-pwidth/4+pwidth/2+1)
                polycof = la.lstsq(np.vstack((s_S**2,s_S,np.ones_like(s_S))).T,crosscor_s[s_S])[0]
                pshift = -(-0.5*polycof[1]/polycof[0] - pwidth/2)
                s = int(pshift+pwidth)-pwidth
                sfrac = pshift-s
                psfsh_orw = np.zeros_like(psf_orw)
                outrow = np.arange(max(0,s+1),rows-(1+int(abs(pshift)))+max(0,s+1))
                psfsh_orw[:,outrow] = (1.-sfrac)*psf_orw[:,outrow-s] + sfrac*psf_orw[:,outrow-s-1]
#                pyfits.PrimaryHDU(psfsh_orw.astype('float32')).writeto('psfsh_'+tnum+'_orw.fits',clobber=True)

                wt_orw[~badbin_orw] = psfsh_orw[~badbin_orw]/var_orw[~badbin_orw]
                var_ow = (psfsh_orw*wt_orw*(~badbin_orw)).sum(axis=1)
                badbin_ow = (var_ow == 0)
                var_ow[~badbin_ow] = 1./var_ow[~badbin_ow]
#                pyfits.PrimaryHDU(var_ow.astype('float32')).writeto('var_'+tnum+'_ow.fits',clobber=True)
#                pyfits.PrimaryHDU(target_orw.astype('float32')).writeto('target_'+tnum+'_orw.fits',clobber=True)
#                pyfits.PrimaryHDU(wt_orw.astype('float32')).writeto('wt_'+tnum+'_orw.fits',clobber=True)

                sci_ow = (target_orw*wt_orw).sum(axis=1)*var_ow

                badlim = 0.20
                psfbadfrac_ow = (psfsh_orw*badbin_orw.astype(int)).sum(axis=1)/psfsh_orw.sum(axis=1)
                badbin_ow |= (psfbadfrac_ow > badlim)

#                cdebug = 39
#                if debug: np.savetxt("xtrct"+str(cdebug)+"_"+tnum+".txt",np.vstack((psf_orw[:,:,cdebug],var_orw[:,:,cdebug], \
#                 wt_orw[:,:,cdebug],target_orw[:,:,cdebug])).reshape((4,2,-1)).transpose(1,0,2).reshape((8,-1)).T,fmt="%12.5e")

            # write O,E spectrum, prefix "s". VAR, BPM for each spectrum. y dim is virtual (length 1)
            # for consistency with other modes
                hduout = pyfits.PrimaryHDU(header=hdulist[0].header)    
                hduout = pyfits.HDUList(hduout)
                header=hdulist['SCI'].header.copy()
                header.update('VAREXT',2)
                header.update('BPMEXT',3)
                header.update('CRVAL1',wedge_w[0]+wbin/2.)
                header.update('CRVAL2',0)
                header.update('CDELT1',wbin)
                header.update('CTYPE1','Angstroms')
            
                hduout.append(pyfits.ImageHDU(data=sci_ow.reshape((2,1,wavs)), header=header, name='SCI'))
                header.update('SCIEXT',1,'Extension for Science Frame',before='VAREXT')
                hduout.append(pyfits.ImageHDU(data=var_ow.reshape((2,1,wavs)), header=header, name='VAR'))
                hduout.append(pyfits.ImageHDU(data=badbin_ow.astype("uint8").reshape((2,1,wavs)), header=header, name='BPM'))            
            
                hduout.writeto('e'+outfilelist[i],clobber=True,output_verify='warn')
                log.message('Output file '+'e'+outfilelist[i] , with_header=False)
            if debug: np.savetxt(obsname+"_dtrbkg_dic.txt",dtrbkg_dic.reshape((-1,cols)).T,fmt="%8.2f")
    return

