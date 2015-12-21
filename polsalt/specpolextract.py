
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

# ---------------------------------------------------------------------------------
def specpolextract(infilelist, logfile='salt.log'):

#set up the files
    obsdate=os.path.basename(infilelist[0])[8:16]

    with logging(logfile, debug) as log:
        #create the observation log
        obs_dict=obslog(infilelist)
    # get rid of arcs
        for i in range(len(infilelist))[::-1]:
            if (obs_dict['OBJECT'][i].upper().strip()=='ARC'): del infilelist[i]            
        infiles = len(infilelist)

    # contiguous images of the same object and config are grouped together
        obs_dict=obslog(infilelist)
        confno_i,confdatlist = configmap(infilelist)
        configs = len(confdatlist)
        objectlist = list(set(obs_dict['OBJECT']))
        objno_i = np.array([objectlist.index(obs_dict['OBJECT'][i]) for i in range(infiles)],dtype=int)
        grp_i = np.zeros((infiles),dtype=int)
        grp_i[1:] = ((confno_i[1:] != confno_i[:-1]) | (objno_i[1:] != objno_i[:-1])).cumsum()

        for g in np.unique(grp_i):
            ilist = np.where(grp_i==g)[0]
            outfiles = len(ilist)
            outfilelist = [infilelist[i] for i in ilist]
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
                    image_orc += pyfits.open(infilelist[i])['SCI'].data*(~badbin_orc).astype(int)
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

            hdusum = pyfits.PrimaryHDU(header=hdu0[0].header)    
            hdusum = pyfits.HDUList(hdusum)
            header=hdu0['SCI'].header.copy()           
            hdusum.append(pyfits.ImageHDU(data=image_orc, header=header, name='SCI'))
            hdusum.append(pyfits.ImageHDU(data=var_orc, header=header, name='VAR'))
            hdusum.append(pyfits.ImageHDU(data=badbinall_orc.astype('uint8'), header=header, name='BPM'))
            hdusum.append(pyfits.ImageHDU(data=wav_orc, header=header, name='WAV'))
#            hdusum.writeto("groupsum_"+str(g)+".fits",clobber=True)

            psf_orc,skyflat_orc,badbinnew_orc,isbkgcont_orc,maprow_od,drow_oc = \
                specpolsignalmap(hdusum,logfile=logfile)

            maprow_ocd = maprow_od[:,None,:] + np.zeros((2,cols,4)) 
            maprow_ocd[:,:,[1,2]] -= drow_oc[:,:,None]      # edge is straight, target curved

            isedge_orc = (np.arange(rows)[:,None] < maprow_ocd[:,None,:,0]) | \
                (np.arange(rows)[:,None] > maprow_ocd[:,None,:,3])
            istarget_orc = (np.arange(rows)[:,None] > maprow_ocd[:,None,:,1]) & \
                (np.arange(rows)[:,None] < maprow_ocd[:,None,:,2])
            isskycont_orc = (((np.arange(rows)[:,None] < maprow_ocd[:,None,:,0]+rows/16) |  \
                (np.arange(rows)[:,None] > maprow_ocd[:,None,:,3]-rows/16)) & ~isedge_orc)                                     
            isbkgcont_orc &= (~badbinall_orc & ~isedge_orc & ~istarget_orc)
            badbinall_orc |= badbinnew_orc
            badbinone_orc |= badbinnew_orc

#            pyfits.PrimaryHDU(var_orc.astype('float32')).writeto('var_orc1.fits',clobber=True) 
#            pyfits.PrimaryHDU(badbinnew_orc.astype('uint8')).writeto('badbinnew_orc.fits',clobber=True)   
#            pyfits.PrimaryHDU(badbinall_orc.astype('uint8')).writeto('badbinall_orc.fits',clobber=True)  
#            pyfits.PrimaryHDU(badbinone_orc.astype('uint8')).writeto('badbinone_orc.fits',clobber=True)  

        # scrunch and normalize psf from summed images (using badbinone) for optimized extraction
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
#            pyfits.PrimaryHDU(var_orc.astype('float32')).writeto('var_orc2.fits',clobber=True)  
            for o in (0,1):
                for r in specrow_or[o]:
                    binedge_orw[o,r] = interp1d(wav_orc[o,r],np.arange(cols))(wedge_w)
                    psf_orw[o,r] = scrunch1d(psf_orc[o,r],binedge_orw[o,r])
            psf_orw /= psf_orw.sum(axis=1)[:,None,:]

#            np.savetxt("psfnorm_ow.txt",(psf_orw*okbin_orw).sum(axis=1).T,fmt="%10.4f") 
#            pyfits.PrimaryHDU(psf_orw.astype('float32')).writeto('psf_orw.fits',clobber=True)
#            pyfits.PrimaryHDU(var_orw.astype('float32')).writeto('var_orw.fits',clobber=True) 

        # set up optional image-dependent column shift for slitless data
            colshiftfilename = "colshift.txt"
            docolshift = os.path.isfile(colshiftfilename)
            if docolshift:
                img_I,dcol_I = np.loadtxt(colshiftfilename,dtype=float,unpack=True,usecols=(0,1))
                shifts = img_I.shape[0]
                log.message('Column shift: \n Images '+shifts*'%5i ' % tuple(img_I), with_header=False)                 
                log.message(' Bins    '+shifts*'%5.2f ' % tuple(dcol_I), with_header=False)                 
               
        # background-subtract and extract spectra                
            psfbadfrac_iow = np.zeros((outfiles,2,wavs))

            for i in range(outfiles):
                hdulist = pyfits.open(outfilelist[i])
                sci_orc = hdulist['sci'].data
                var_orc = hdulist['var'].data
                badbin_orc = (hdulist['bpm'].data > 0) | badbinnew_orc
                tnum = os.path.basename(outfilelist[i]).split('.')[0][-3:]

            # make background continuum image, smoothed over resolution element
                rblk,cblk = int(1.5*8./rbin), int(slitwidth*8./cbin)
                target_orc = np.zeros_like(sci_orc)

                for o in (0,1):
                    bkgcont_rc = blksmooth2d(sci_orc[o],isbkgcont_orc[o],rblk,cblk,0.25,mode="mean")
           
            # remove sky continuum: ends of bkg continuum * skyflat
                    skycont_c = (bkgcont_rc.T[isskycont_orc[o].T]/skyflat_orc[o].T[isskycont_orc[o].T])  \
                            .reshape((cols,-1)).mean(axis=1)
                    skycont_rc = skycont_c*skyflat_orc[o]
        
            # remove sky lines: image - bkg cont run through 2d sky averaging
                    obj_data = ((sci_orc[o] - bkgcont_rc)/skyflat_orc)[o]
                    obj_data[(badbin_orc | isedge_orc | istarget_orc)[o]] = np.nan
#                    pyfits.PrimaryHDU(obj_data.astype('float32')).writeto('obj_data.fits',clobber=True)
                    skylines_rc = make_2d_skyspectrum(obj_data,wav_orc[o],np.array([[0,rows],]))*skyflat_orc[o]
                    target_orc[o] = sci_orc[o] - skycont_rc - skylines_rc
#                    pyfits.PrimaryHDU(skylines_rc.astype('float32')).writeto('skylines_rc_'+tnum+'_'+str(o)+'.fits',clobber=True)
#                    pyfits.PrimaryHDU(skycont_rc.astype('float32')).writeto('skycont_rc_'+tnum+'_'+str(o)+'.fits',clobber=True)
                target_orc *= (~badbin_orc).astype(int)             
#                pyfits.PrimaryHDU(target_orc.astype('float32')).writeto('target_'+tnum+'_orc.fits',clobber=True)

            # extract spectrum optimally (Horne, PASP 1986)
                target_orw = np.zeros((2,rows,wavs));   var_orw = np.zeros_like(target_orw)
                badbin_orw = np.ones((2,rows,wavs),dtype='bool');   wt_orw = np.zeros_like(target_orw)
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
                psfbadfrac_iow[i] = (psfsh_orw*badbin_orw.astype(int)).sum(axis=1)/psfsh_orw.sum(axis=1)
                badbin_ow |= (psfbadfrac_iow[i] > badlim)

#                cdebug = 39
#                np.savetxt("xtrct"+str(cdebug)+"_"+tnum+".txt",np.vstack((psf_orw[:,:,cdebug],var_orw[:,:,cdebug], \
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

#            np.savetxt("psfbadfrac_iow.txt",psfbadfrac_iow.reshape((-1,wavs)).T,fmt="%8.5f")
    return

