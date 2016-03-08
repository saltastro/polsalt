
"""
specpolextract

Background subtract and extract spectropolarimetric data.  The end of the image-specific pipeline.

"""

import os, sys, glob, shutil, inspect

import numpy as np
import pyfits
from scipy.interpolate import interp1d
from scipy import linalg as la

from specpolutils import *

from oksmooth import blksmooth2d
from specpolutils import configmap
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
def bkgsub(hdulist,badbinbkg_orc,isbkgcont_orc,skyflat_orc,maprow_ocd,tnum,debug=False):
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

    target_orc = sci_orc

      # make background continuum image, smoothed over resolution element
    rblk,cblk = int(1.5*8./rbin), int(slitwidth*8./cbin)
    isedge_orc = (np.arange(rows)[:,None] < maprow_ocd[:,None,:,0]) | \
          (np.arange(rows)[:,None] > maprow_ocd[:,None,:,3])
    isskycont_orc = (((np.arange(rows)[:,None] < maprow_ocd[:,None,:,0]+rows/16) |  \
          (np.arange(rows)[:,None] > maprow_ocd[:,None,:,3]-rows/16)) & ~isedge_orc)
    if debug: pyfits.PrimaryHDU(isskycont_orc.astype('uint8')).writeto('isskycont_orc_'+tnum+'.fits',clobber=True)  

    for o in (0,1):
        bkgcont_rc = blksmooth2d(target_orc[o],isbkgcont_orc[o],rblk,cblk,0.25,mode="mean")
           
        # remove sky continuum: ends of bkg continuum * skyflat
        skycont_rc = np.zeros((rows,cols))
        okflat_rc = ~np.isnan(skyflat_orc[o])
        skycont_rc[okflat_rc] = (bkgcont_rc[okflat_rc]/skyflat_orc[o,okflat_rc])*isbkgcont_orc[o,okflat_rc] 
        skycont_c = skycont_rc.sum(axis=0)
        skycontrows_c = (skycont_rc>0.).sum(axis=0)
        skycont_rc[:,skycont_c>0.] = skyflat_orc[o][:,skycont_c>0.]*skycont_c[skycont_c>0.]/skycontrows_c[skycont_c>0.]
        
        # remove sky lines: image - bkgcont run through 2d sky averaging
        objdata_rc = ((target_orc[o] - bkgcont_rc)/skyflat_orc)[o]
        if debug: pyfits.PrimaryHDU(badbinbkg_orc[o].astype('uint8')).writeto('badbinbkg_orc_'+tnum+'_'+str(o)+'.fits',clobber=True)
        objdata_rc[badbinbkg_orc[o]] = np.nan

        if debug: pyfits.PrimaryHDU(objdata_rc.astype('float32')).writeto('objdata_'+tnum+'_'+str(o)+'.fits',clobber=True)

        skylines_rc = make_2d_skyspectrum(objdata_rc,wav_orc[o],np.array([[0,rows],]))*skyflat_orc[o]
        target_orc[o] -= skycont_rc + skylines_rc

        if debug: pyfits.PrimaryHDU(skylines_rc.astype('float32')).writeto('skylines_rc_'+tnum+'_'+str(o)+'.fits',clobber=True)
        if debug: pyfits.PrimaryHDU(skycont_rc.astype('float32')).writeto('skycont_rc_'+tnum+'_'+str(o)+'.fits',clobber=True)

    return target_orc


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

    with logging(logfile, debug) as log:
 
        config_dict = list_configurations(infilelist, log)
        config_count = 0

        for config in config_dict:
            outfilelist = config_dict[config]['object']
            outfiles = len(outfilelist)
            obs_dict=obslog(outfilelist)
            hdu0 =  pyfits.open(outfilelist[0])       
            rows,cols = hdu0['SCI'].data.shape[1:3]
            cbin,rbin = np.array(obs_dict["CCDSUM"][0].split(" ")).astype(int)
            object_name = hdu0[0].header['OBJECT']
            log.message('\nExtract: {3}  Grating {0} Grang {1:6.2f}  Artic {2:6.2f}'.format(
                        config[0], config[1], config[2], object_name))
            log.message(' Images: '+ ' '.join([str(image_number(img)) for img in outfilelist]), with_header=False)

            # special version for lamp data
            # this is now removed and will not be part of this code
            #object = obs_dict["OBJECT"][0].strip().upper()
            #ampid = obs_dict["LAMPID"][0].strip().upper()
            #f ((object != "ARC") & (lampid != "NONE")) :
            #   specpollampextract(outfilelist, logfile=logfile)           
            #   continue

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
            okwav_oc = ~((wav_orc == 0).all(axis=1))

            obsname = object_name + "_c" + str(config_count)+"_"+str(outfiles)
            hdusum = pyfits.PrimaryHDU(header=hdu0[0].header)   
            hdusum = pyfits.HDUList(hdusum)
            hdusum[0].header['OBJECT']=obsname
            header=hdu0['SCI'].header.copy()       
            hdusum.append(pyfits.ImageHDU(data=image_orc, header=header, name='SCI'))
            hdusum.append(pyfits.ImageHDU(data=var_orc, header=header, name='VAR'))
            hdusum.append(pyfits.ImageHDU(data=badbinall_orc.astype('uint8'), header=header, name='BPM'))
            hdusum.append(pyfits.ImageHDU(data=wav_orc, header=header, name='WAV'))

            
            if debug: hdusum.writeto(obsname+".fits",clobber=True)

            # run specpolsignalmap on image
            psf_orc,skyflat_orc,badbinnew_orc,isbkgcont_orc,maprow_od,drow_oc = \
                specpolsignalmap(hdusum,logfile=logfile,debug=debug)

            maprow_ocd = maprow_od[:,None,:] + np.zeros((2,cols,4)) 
            maprow_ocd[okwav_oc] += drow_oc[okwav_oc,None]      

            isedge_orc = (np.arange(rows)[:,None] < maprow_ocd[:,None,:,0]) | \
                (np.arange(rows)[:,None] > maprow_ocd[:,None,:,3])
            istarget_orc = okwav_oc[:,None,:] & (np.arange(rows)[:,None] > maprow_ocd[:,None,:,1]) & \
                (np.arange(rows)[:,None] < maprow_ocd[:,None,:,2])
                                   
            isbkgcont_orc &= (~badbinall_orc & ~isedge_orc & ~istarget_orc)
            badbinall_orc |= badbinnew_orc
            badbinone_orc |= badbinnew_orc
            hdusum['BPM'].data = badbinnew_orc.astype('uint8')
            psf_orc *= istarget_orc.astype(int)

            if debug: 
#                hdusum.writeto(obsname+".fits",clobber=True)
               pyfits.PrimaryHDU(psf_orc.astype('float32')).writeto(obsname+'_psf_orc.fits',clobber=True) 
#               pyfits.PrimaryHDU(badbinnew_orc.astype('uint8')).writeto('badbinnew_orc.fits',clobber=True)   
#               pyfits.PrimaryHDU(badbinall_orc.astype('uint8')).writeto('badbinall_orc.fits',clobber=True)  
#               pyfits.PrimaryHDU(badbinone_orc.astype('uint8')).writeto('badbinone_orc.fits',clobber=True)  

            # set up wavelength binning
            wbin = wav_orc[0,rows/2,cols/2]-wav_orc[0,rows/2,cols/2-1] 
            wbin = 2.**(np.rint(np.log2(wbin)))         # bin to nearest power of 2 angstroms
            wmin = (wav_orc.max(axis=1)[okwav_oc].reshape((2,-1))).min(axis=1).max()
            wmax = wav_orc.max()
            for o in (0,1): 
                colmax = np.where((wav_orc[o] > 0.).any(axis=0))[0][-1]
                row_r = np.where(wav_orc[o,:,colmax] > 0.)[0]
                wmax = min(wmax,wav_orc[o,row_r,colmax].min())
            wedgemin = wbin*int(wmin/wbin+0.5) + wbin/2.
            wedgemax = wbin*int(wmax/wbin-0.5) + wbin/2.
            wedge_w = np.arange(wedgemin,wedgemax+wbin,wbin)
            wavs = wedge_w.shape[0] - 1
            binedge_orw = np.zeros((2,rows,wavs+1))
            specrow_or = (maprow_od[:,1:3].mean(axis=1)[:,None] + np.arange(-rows/4,rows/4)).astype(int)

            # scrunch and normalize psf from summed images (using badbinone) for optimized extraction
            # psf is normalized so its integral over row is 1.
            psfnormmin = 0.70    # wavelengths with less than this flux in good bins are marked bad
            psf_orw = np.zeros((2,rows,wavs))

            for o in (0,1):
                for r in specrow_or[o]:
                    binedge_orw[o,r] = \
                        interp1d(wav_orc[o,r,okwav_oc[o]],np.arange(cols)[okwav_oc[o]], \
                                   kind='linear',bounds_error=False)(wedge_w)
                    psf_orw[o,r] = scrunch1d(psf_orc[o,r],binedge_orw[o,r])

            if debug: 
                pyfits.PrimaryHDU(binedge_orw.astype('float32')).writeto(obsname+'_binedge_orw.fits',clobber=True)
                pyfits.PrimaryHDU(psf_orw.astype('float32')).writeto(obsname+'_psf_orw.fits',clobber=True)

            psfnorm_orw = np.repeat(psf_orw.sum(axis=1),rows,axis=1).reshape(2,rows,-1)
            psf_orw[psfnorm_orw>0.] /= psfnorm_orw[psfnorm_orw>0.]
            pmax = np.minimum(1.,np.median(psf_orw[psfnorm_orw>0.].reshape((2,rows,-1)).max(axis=1)))

            log.message('Stellar profile width: %8.2f arcsec' % ((1./pmax)*rbin/8.), with_header=False)     
            pwidth = int(1./pmax)

            if debug: 
                pyfits.PrimaryHDU(psf_orw.astype('float32')).writeto(obsname+'_psfnormed_orw.fits',clobber=True)

            # set up optional image-dependent column shift for slitless data
            colshiftfilename = "colshift.txt"
            docolshift = os.path.isfile(colshiftfilename)
            if docolshift:
                img_I,dcol_I = np.loadtxt(colshiftfilename,dtype=float,unpack=True,usecols=(0,1))
                shifts = img_I.shape[0]
                log.message('Column shift: \n Images '+shifts*'%5i ' % tuple(img_I), with_header=False)                 
                log.message(' Bins    '+shifts*'%5.2f ' % tuple(dcol_I), with_header=False)                 

            log.message('\nArcsec offset     Output File', with_header=False)                  

            # background-subtract and extract spectra
            for i in range(outfiles):
                hdulist = pyfits.open(outfilelist[i])
                tnum = image_number(outfilelist[i])
                badbin_orc = (hdulist['BPM'].data > 0)
                badbinbkg_orc = (badbin_orc | badbinnew_orc | isedge_orc | istarget_orc)
                if debug:
                    pyfits.PrimaryHDU(isedge_orc.astype('uint8')).writeto('isedge_orc_'+tnum+'.fits',clobber=True)
                    pyfits.PrimaryHDU(istarget_orc.astype('uint8')).writeto('istarget_orc_'+tnum+'.fits',clobber=True) 
                    pyfits.PrimaryHDU(badbinbkg_orc.astype('uint8')).writeto('badbinbkg_orc_'+tnum+'.fits',clobber=True)
                target_orc = bkgsub(hdulist,badbinbkg_orc,isbkgcont_orc,skyflat_orc,maprow_ocd,tnum,debug=debug)
                target_orc *= (~badbin_orc).astype(int)             
                if debug:
                    pyfits.PrimaryHDU(target_orc.astype('float32')).writeto('target_'+tnum+'_orc.fits',clobber=True)
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
                if debug:
#                   pyfits.PrimaryHDU(var_orw.astype('float32')).writeto('var_'+tnum+'_orw.fits',clobber=True)
                    pyfits.PrimaryHDU(badbin_orw.astype('uint8')).writeto('badbin_'+tnum+'_orw.fits',clobber=True)
  
                # use master psf shifted in row to allow for guide errors
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

                cdebug = 83
                if debug: np.savetxt("xtrct"+str(cdebug)+"_"+tnum+".txt",np.vstack((psf_orw[:,:,cdebug],var_orw[:,:,cdebug], \
                    wt_orw[:,:,cdebug],target_orw[:,:,cdebug])).reshape((4,2,-1)).transpose(1,0,2).reshape((8,-1)).T,fmt="%12.5e")

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
                log.message('  %8.2f   e%s' % (pshift*rbin/8.,outfilelist[i]), with_header=False)

            #increate the config count
            config_count += 1

    return

