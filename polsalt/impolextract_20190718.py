
"""
impolextract

Optimal extraction for imaging polarimetric data
Write out extracted data fits (etm*) dimensions wavelength,target #

"""

import os, sys, glob, shutil, inspect

import numpy as np
from scipy.interpolate import griddata, interp1d, UnivariateSpline, LSQUnivariateSpline
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.signal import convolve2d
from scipy.linalg import lstsq
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import Latitude,Longitude,Angle

from pyraf import iraf
from iraf import pysalt

from saltobslog_kn import obslog
from saltsafelog import logging

from scrunch1d import scrunch1d, scrunchvar1d
from specpolutils import datedline, rssdtralign, rssmodelwave, configmap
from rssmaptools import ccdcenter,YXcalc,boxsmooth1d,blksmooth2d,RSScolpolcam,RSSpolgeom,impolguide,Tableinterp

datadir = os.path.dirname(__file__) + '/data/'
np.set_printoptions(threshold=np.nan)

import warnings 
# warnings.filterwarnings("error")

def imfilpolextract(fileList,hi_df,name,image_ftpRC,var_ftpRC,okbin_ftpRC,oktgt_ftpRC,bkg_ftpRC,   \
    logfile='salt.log',debug=False):
    """derive extracted target data vs target

    Parameters 
    ----------
    fileList: list of strings
    hi_df: 2d integer array. h = (0,1..) filter-pair counter, i = (0,1) index within pair 
    image_ftpRC,var_ftpRC,bkg_ftpRC: 5d float arrays of input target data
    okbin_ftpRC,oktgt_ftpRC: 5d boolean arrays. bin: badpix,gap; tgt: tgt overlap

    """
    """
    _d dimension index r,c = 0,1
    _f file index
    _t target index
    _p pol beam = 0,1 for O,E
    _R, _C bin coordinate within target

    """
    dfile,dtarget,dbeam = 0,0,0
    hdr0 = pyfits.open(fileList[0])[0].header
    exptime = hdr0['EXPTIME']
    wavl = float(hdr0['FILTER'][3:])
    rcbin_d = np.array([int(x) for x in hdr0['CCDSUM'].split(" ")])[::-1]

  # flag bad crs (1st iteration) by comparison with normalized "meanimage" summed over targets in all files
  # for bad bins, targets weighted by median flux over central 9 bins 
    files = image_ftpRC.shape[0]
    targets = image_ftpRC.shape[1]
    Rows,Cols = image_ftpRC.shape[3],image_ftpRC.shape[4]
    ok_ftpRC = (okbin_ftpRC & oktgt_ftpRC)              
    wt_ftp = np.median((image_ftpRC*ok_ftpRC)[:,:,:,Rows/2-1:Rows/2+2,Cols/2-1:Cols/2+2],axis=(3,4))
    wtcnt_ftp = ok_ftpRC[:,:,:,Rows/2-1:Rows/2+2,Cols/2-1:Cols/2+2].sum(axis=(3,4))
    ok_ft = (wtcnt_ftp > 5).all(axis=2)                                # >= 4 bad pix in center: bad target
    meanimage_RC = (image_ftpRC*ok_ftpRC*ok_ft[:,:,None,None,None]).sum(axis=(0,1,2))
    wt_RC = (ok_ftpRC*wt_ftp[:,:,:,None,None]*ok_ft[:,:,None,None,None]).sum((0,1,2))/wt_ftp[ok_ft,:].sum()
    meanimage_RC[wt_RC>0] /= wt_RC[wt_RC>0]
    meanimage_RC /= meanimage_RC[Rows/2,Cols/2]

  # flag cr's on signal by looking for spikes in residual from meanimage
  #   allowing for position and seeing variation
  # also evaluate target position and profile rms for extraction
    nbr_rc = np.pad(np.zeros((1,1)),1,'constant',constant_values=1./8.)     # val-nbrs/8
    iscross_rc = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype(bool)           # use for meanimage floating filter
    notrim_RC = np.pad(np.ones((Rows-2,Cols-2)),1,'constant',constant_values=0.).astype(bool)
    dRow_R = (np.arange(Rows)-(Rows-1)/2)
    dCol_C = (np.arange(Cols)-(Cols-1)/2)
    iscr_ftpRC = np.zeros((files,targets,2,Rows,Cols),dtype=bool)
    dR_ftp = np.zeros((files,targets,2))
    dC_ftp = np.zeros_like(dR_ftp)
    sigma_ftp = np.zeros_like(dR_ftp)
    norm_ftp = np.zeros_like(dR_ftp)

    print "Evaluating position, profile rms history"

    wt_ftp[~ok_ft] = 1.                                               # avoid divide by zero
    resid_ftpRC = image_ftpRC/wt_ftp[:,:,:,None,None] - meanimage_RC[None,None,None,:,:]
    imagevar_RC = maximum_filter(meanimage_RC,footprint=iscross_rc,mode='constant') -\
        minimum_filter(meanimage_RC,footprint=iscross_rc,mode='constant') + meanimage_RC/3.

    for (f,t,p) in np.ndindex(files,targets,2):
        if (ok_ft[f,t].all() == 0): continue
        spike_RC = ok_ftpRC[f,t,p]*resid_ftpRC[f,t,p] - \
                    convolve2d(ok_ftpRC[f,t,p]*resid_ftpRC[f,t,p],nbr_rc,mode='same') 
        Q1,Q3 = np.percentile(spike_RC[notrim_RC&ok_ftpRC[f,t,p]],(25.,75.))
        flagit_RC = (notrim_RC & (spike_RC > (Q3+10*(Q3-Q1))) & (spike_RC > imagevar_RC))
        iscr_ftpRC[f,t,p] =  (flagit_RC & ok_ftpRC[f,t,p])
    ok_ftpRC &= ~iscr_ftpRC                                           # has target,BPM and CR's

    print "Computing comp profiles for CR removal and extraction"

  # determine image shift (use center 3 rows/ cols) and Moffat profile fit
    goodrow_R = range(Rows/2-1,Rows/2+2)
    goodcol_C = range(Cols/2-1,Cols/2+2)
    imageok_ftpRC = image_ftpRC*ok_ftpRC

    dR_ftp[ok_ft] = (imageok_ftpRC*dRow_R[:,None])[:,:,:,1:-2,goodcol_C].sum(axis=(3,4))[ok_ft]/   \
                    imageok_ftpRC[:,:,:,1:-2,goodcol_C].sum(axis=(3,4))[ok_ft]
    dC_ftp[ok_ft] = (imageok_ftpRC*dCol_C[None,:])[:,:,:,goodrow_R,1:-1].sum(axis=(3,4))[ok_ft]/   \
                    imageok_ftpRC[:,:,:,goodrow_R,1:-2].sum(axis=(3,4))[ok_ft]
    rad_ftpRC = np.sqrt((dRow_R[None,None,None,:,None]-dR_ftp[:,:,:,None,None])**2 + \
                        (dCol_C[None,None,None,None,:]-dC_ftp[:,:,:,None,None])**2)
    sigma_i,fmax_i,fiterr_ib = moffat2dfit(image_ftpRC,rad_ftpRC,ok_ftpRC)
    sigma_ftp,fmax_ftp = sigma_i.reshape((files,targets,2)),fmax_i.reshape((files,targets,2))
    starbins_ftp = (image_ftpRC > fmax_ftp[:,:,:,None,None]/2.).sum(axis=(3,4))

 #  THIS MAY BE WRONG!
    for h in range(hi_df.shape[0]):
        fList = np.where(hi_df[0]==h)[0]
        sigma_ftp[fList,:,:] = sigma_ftp[fList,:,:].max(axis=0)       # use same width within pair

    if debug:
        crs_ftp = iscr_ftpRC.sum(axis=(3,4))
        crs_RC = iscr_ftpRC.sum(axis=(0,1,2))                
        np.savetxt(name+'dR_fpt.txt',dR_ftp.transpose((0,2,1)).reshape((2*files,-1)).T,fmt="%8.2f") 
        np.savetxt(name+'dC_fpt.txt',dC_ftp.transpose((0,2,1)).reshape((2*files,-1)).T,fmt="%8.2f") 
        np.savetxt(name+'sigma_fpt.txt',sigma_ftp.transpose((0,2,1)).reshape((2*files,-1)).T,fmt="%8.2f")
        np.savetxt(name+'fmax_fpt.txt',fmax_ftp.transpose((0,2,1)).reshape((2*files,-1)).T,fmt="%8.2f")
        np.savetxt(name+'starbins_fpt.txt',starbins_ftp.transpose((0,2,1)).reshape((2*files,-1)).T,fmt="%3i")
        np.savetxt(name+'wt_fpt.txt',wt_ftp.transpose((0,2,1)).reshape((2*files,-1)).T,fmt="%8.2f")
        np.savetxt(name+'crs1_fpt.txt',crs_ftp.transpose((0,2,1)).reshape((2*files,-1)).T,fmt="%3i") 
        np.savetxt(name+'crs1_RC.txt',crs_RC,fmt="%3i")
        debuglbl = ((3*"_%i"+".txt") % (dfile,dtarget,dbeam))  
        np.savetxt(name+'debugimg'+debuglbl,image_ftpRC[dfile,dtarget,dbeam],fmt="%9.1f")
        np.savetxt(name+'debugres1'+debuglbl,resid_ftpRC[dfile,dtarget,dbeam],fmt="%9.3f")
        np.savetxt(name+'debugcr1'+debuglbl,iscr_ftpRC[dfile,dtarget,dbeam],fmt="%2i")
                             
  # form mean comp image for each target by stacking shifted images, evaluate
    compimage_tpRC = np.zeros((targets,2,Rows,Cols))
    compcount_tpRC = np.zeros_like(compimage_tpRC).astype(int)
    dR_f = np.median(dR_ftp,axis=(1,2))
    dC_f = np.median(dC_ftp,axis=(1,2))
    dR_tp = np.median(dR_ftp - dR_f[:,None,None],axis=0)
    dC_tp = np.median(dC_ftp - dC_f[:,None,None],axis=0)
    wt_f = (fmax_ftp*sigma_ftp**2).mean(axis=(1,2))
                
    for f,p in np.ndindex(files,2):                 # ok has BPM and CR's only
        dcompimage_tRC = (wt_f.mean()/wt_f[f])*shift(image_ftpRC[f,:,p], (0,-dR_f[f],-dC_f[f]), order=3)
        okdcomp_tRC = (shift(ok_ftpRC[f,:,p].astype(float), (0,-dR_f[f],-dC_f[f]), order=1) ==1.)
        compimage_tpRC[:,p] += okdcomp_tRC*dcompimage_tRC
        compcount_tpRC[:,p] += okdcomp_tRC.astype(int)

    okcomp_tpRC = (compcount_tpRC > 0)
    compimage_tpRC[okcomp_tpRC] /= compcount_tpRC[okcomp_tpRC]
    rad_tpRC = np.sqrt((dRow_R[None,None,:,None]-dR_tp[:,:,None,None])**2 + \
                       (dCol_C[None,None,None,:]-dC_tp[:,:,None,None])**2)
    sigmacomp_i,fmaxcomp_i,fiterrcomp_ib = moffat2dfit(compimage_tpRC,rad_tpRC,okcomp_tpRC)
    sigmacomp_tp,fmaxcomp_tp = sigmacomp_i.reshape((targets,2)),fmaxcomp_i.reshape((targets,2))

  # form extraction diagnostics
    relerrcomp_tpRC = fiterrcomp_ib.reshape((targets,2,Rows,Cols))/fmaxcomp_tp[:,:,None,None]
    grid_dRC = np.indices((Rows,Cols)) - np.array([Rows/2,Cols/2])[:,None,None]
    rad_RC = np.sqrt((grid_dRC**2).sum(axis=0))
    isouter_tpRC = (okcomp_tpRC & (rad_RC > Rows/2 -2))
    outercount_tp = isouter_tpRC.sum(axis=(2,3))
    compouter_tp = (isouter_tpRC*compimage_tpRC/fmaxcomp_tp[:,:,None,None]).sum(axis=(2,3))/ outercount_tp
    rmsouter_tp = np.sqrt(((isouter_tpRC*relerrcomp_tpRC)**2).sum(axis=(2,3))/ outercount_tp)
    iscrowd_tpRC = (isouter_tpRC & (relerrcomp_tpRC > 0.))
    crowderr_tp = (iscrowd_tpRC*relerrcomp_tpRC).sum(axis=(2,3))/ outercount_tp
    relerrcomp_tpRC[~okcomp_tpRC] = np.nan

  # form comp image for each file, correcting for shift 
    compimage_ftpRC = np.zeros((files,targets,2,Rows,Cols)) 
    okcomp_ftpRC = np.zeros_like(compimage_ftpRC).astype(bool)
    for f,p in np.ndindex(files,2):
        compimage_ftpRC[f,:,p] = shift(compimage_tpRC[:,p], (0,dR_f[f],dC_f[f]), order=3)
        okcomp_ftpRC[f,:,p] = (shift(okcomp_tpRC[:,p].astype(float),    \
                (0,dR_f[f],dC_f[f]), order=1) ==1.).astype(int)
    improf_ftpRC = (1. + (rad_ftpRC/sigma_ftp[:,:,:,None,None])**2)**(-3)
    compprof_ftpRC = (1. + (rad_ftpRC/sigmacomp_tp[None,:,:,None,None])**2)**(-3)
    profcor_ftpRC = improf_ftpRC-compprof_ftpRC

    if debug:
        np.savetxt(name+'sigmacomp_tp.txt',sigmacomp_tp,fmt="%8.2f")
        np.savetxt(name+'fmaxcomp_tp.txt',fmaxcomp_tp,fmt="%8.2f")
        np.savetxt(name+'compouter_tp.txt',compouter_tp,fmt="%8.4f")  
        np.savetxt(name+'rmsouter_tp.txt',rmsouter_tp,fmt="%8.4f")  
        np.savetxt(name+'crowderr_tp.txt',crowderr_tp,fmt="%8.4f") 
        np.savetxt(name+'compimg'+debuglbl[3:],compimage_tpRC[dtarget,dbeam],fmt="%9.2f") 
        np.savetxt(name+'fcompimg'+debuglbl,compimage_ftpRC[dfile,dtarget,dbeam],fmt="%9.2f") 
        np.savetxt(name+'profcor'+debuglbl,profcor_ftpRC[dfile,dtarget,dbeam],fmt="%9.3f")
        np.savetxt(name+'rad'+debuglbl,rad_ftpRC[dfile,dtarget,dbeam],fmt="%9.3f")
        debuglbl = ((2*"_%i"+".txt") % (dtarget,dbeam)) 
        np.savetxt(name+'relerrcomp'+debuglbl,relerrcomp_tpRC[dtarget,dbeam],fmt="%8.4f")

    compimage_ftpRC = fmax_ftp[:,:,:,None,None]*    \
        np.maximum(0.,(compimage_ftpRC/fmaxcomp_tp[None,:,:,None,None] + profcor_ftpRC))  

  # do final cr cull by comparing file and bin residuals from comp for each target
  # allow for profile position error of up to allowshift = 1/2 bin
    Qcull = 5.
    slopecull = 4.                      # need to find a formula
    fmaxcull = 0.7                      # need to find a formula       
    resid_ftpRC = (ok_ftpRC*okcomp_ftpRC*(image_ftpRC - compimage_ftpRC))
    Q1_tp = np.zeros((targets,2))
    Q3_tp = np.zeros((targets,2))
    for (t,p) in np.ndindex(targets,2):
        Q1_tp[t,p],Q3_tp[t,p] = np.percentile(resid_ftpRC[:,t,p][(ok_ftpRC&okcomp_ftpRC)[:,t,p]],(25.,75.))
    isQcull_ftpRC = (ok_ftpRC & okcomp_ftpRC &  \
        (resid_ftpRC > (Q3_tp+Qcull*(Q3_tp-Q1_tp))[None,:,:,None,None]))
    imageslope_ftpRC = 6.*fmax_ftp[:,:,:,None,None]*(rad_ftpRC/sigma_ftp[:,:,:,None,None]**2)* \
            (1. + (rad_ftpRC/sigma_ftp[:,:,:,None,None])**2)**(-4.)
    sloperat_ftpRC = np.zeros((files,targets,2,Rows,Cols))
    okslope_ftpRC = (rad_ftpRC/sigma_ftp[:,:,:,None,None] > 0.5)
    sloperat_ftpRC[okslope_ftpRC] = resid_ftpRC[okslope_ftpRC]/imageslope_ftpRC[okslope_ftpRC]
    isslopecull_ftpRC = (sloperat_ftpRC > slopecull)
    isfmaxcull_ftpRC = ((resid_ftpRC/fmax_ftp[:,:,:,None,None]) > fmaxcull)

    print "    crs found, iter 1: ", iscr_ftpRC.sum(),
    iscr_ftpRC |=  (isQcull_ftpRC & isslopecull_ftpRC & isfmaxcull_ftpRC)
    crs = iscr_ftpRC.sum()
    crrat = crs/(exptime*ok_ftpRC.sum()*np.prod(rcbin_d)/1.e6)
    print (", total: %3i,  tot/Mpix/s %6.2f") % (iscr_ftpRC.sum(),crrat)

    ok_ftpRC &= ~iscr_ftpRC
    relrms_tpRC = ((ok_ftpRC*resid_ftpRC/fmax_ftp[:,:,:,None,None])**2).sum(axis=0)
    okrms_tpRC = (ok_ftpRC.sum(axis=0)>0)
    relrms_tpRC[okrms_tpRC] = np.sqrt(relrms_tpRC[okrms_tpRC]/ok_ftpRC.sum(axis=0)[okrms_tpRC])
    relrms_tp = np.sqrt((relrms_tpRC**2).sum(axis=(2,3))/okrms_tpRC.sum(axis=(2,3)))
    relrms_ftp = ((ok_ftpRC*resid_ftpRC/fmax_ftp[:,:,:,None,None])**2).sum(axis=(3,4))
    okrms_ftp = (ok_ftpRC.sum(axis=(3,4))>0)
    relrms_ftp[okrms_ftp] = np.sqrt(relrms_ftp[okrms_ftp]/ok_ftpRC.sum(axis=(3,4))[okrms_ftp])
    worstf_tp = np.argmax(relrms_ftp,axis=0)
    worstrelrms_tp = relrms_ftp.reshape((files,-1))[worstf_tp.flatten(),range(2*targets)].reshape((targets,2))

    if debug:
        Qcullrat_i = ((resid_ftpRC - Q3_tp[None,:,:,None,None])/(Q3_tp-Q1_tp)[None,:,:,None,None]).flatten()
        showcull = 5.
        cullidx_x = np.where(Qcullrat_i >= showcull)[0]
        Qcullrat_x = Qcullrat_i[cullidx_x]
        errsloperat_x = sloperat_ftpRC.flatten()[cullidx_x]
        errfmaxrat_x = ((resid_ftpRC/fmax_ftp[:,:,:,None,None]).flatten())[cullidx_x]
        ftpRC_dx = (np.indices((files,targets,2,Rows,Cols)).reshape((5,-1)))[:,cullidx_x]
        np.savetxt(name+'culldat_x.txt',np.vstack((ftpRC_dx,Qcullrat_x,errsloperat_x,errfmaxrat_x)).T, \
                fmt = (5*"%3i "+3*"%8.2f "))
        crs_ftp = iscr_ftpRC.sum(axis=(3,4))
        crs_fRC = iscr_ftpRC.sum(axis=(1,2))
        crs_RC = iscr_ftpRC.sum(axis=(0,1,2))
        np.savetxt(name+'corfcompimg'+debuglbl,compimage_ftpRC[dfile,dtarget,dbeam],fmt="%9.2f")
        np.savetxt(name+'wt_f.txt',wt_f.T,fmt="%8.2f")
        np.savetxt(name+'Q13_tp.txt',np.hstack((Q1_tp,Q3_tp)),fmt="%9.2f")
        np.savetxt(name+'crs2_fpt.txt',crs_ftp.transpose((0,2,1)).reshape((2*files,-1)).T,fmt="%3i") 
        np.savetxt(name+'crs2_RC.txt',crs_RC,fmt="%3i")
        np.savetxt(name+'crs2_fRC.txt',crs_fRC.reshape((-1,Cols)),fmt="%3i")
        np.savetxt(name+'relrms_tp.txt',np.vstack((np.arange(targets),worstf_tp.T, \
                 relrms_tp.T,worstrelrms_tp.T)).T,fmt=3*"%3i "+4*"%8.4f ")
        debuglbl = ((3*"_%i"+".txt") % (dfile,dtarget,dbeam))
        np.savetxt(name+'relrms_fp'+str(dtarget)+'.txt',relrms_ftp[:,dtarget],fmt="%8.4f %8.4f")
        np.savetxt(name+'debugimg'+debuglbl,image_ftpRC[dfile,dtarget,dbeam],fmt="%9.2f")
        np.savetxt(name+'debugres2'+debuglbl,resid_ftpRC[dfile,dtarget,dbeam],fmt="%9.2f")
        np.savetxt(name+'debugslope'+debuglbl,imageslope_ftpRC[dfile,dtarget,dbeam],fmt="%9.2f")
        np.savetxt(name+'debugcr2'+debuglbl,iscr_ftpRC[dfile,dtarget,dbeam],fmt="%2i")

  # mark target bad if more than 5% of comp fit flux lost to non-target, bad pix or cr
    frac_ftp = 1. - (ok_ftpRC*compprof_ftpRC).sum(axis=(3,4))/compprof_ftpRC.sum(axis=(3,4))
    cullit = 0.05
    ok_ftp = (frac_ftp < cullit)

    for f,file in enumerate(fileList):
        crs = iscr_ftpRC[f].sum()
        culls = (~ok_ftp[f]).any(axis=1).sum()
                    
      # do optimal extraction using compimage as weights
        xtrwt_tpRC = compimage_ftpRC[f]
        xtrwt_tpRC[xtrwt_tpRC<0] = 0.
        xtrwt_tpRC /= xtrwt_tpRC.sum(axis=(2,3))[:,:,None,None]
        xtrwt_tpRC *= ok_ftpRC[f]
        xtrwt_tpRC /= xtrwt_tpRC.sum(axis=(2,3))[:,:,None,None]
        sigcount_tp = ok_ftpRC[f].sum(axis=(2,3)) 
        fstd_tp = (ok_ftpRC*image_ftpRC)[f].sum(axis=(2,3))/xtrwt_tpRC.sum(axis=(2,3))
        vstd_tp = (ok_ftpRC*var_ftpRC)[f].sum(axis=(2,3)) / xtrwt_tpRC.sum(axis=(2,3))**2
        vopt_tpRC = fstd_tp[:,:,None,None]*xtrwt_tpRC +     \
                    bkg_ftpRC[f,:,:,Rows/2,Cols/2][:,:,None,None]
        norm_tpRC = np.zeros((targets,2,Rows,Cols))
        norm_tpRC[vopt_tpRC != 0] =     \
                    (xtrwt_tpRC[vopt_tpRC != 0]**2/vopt_tpRC[vopt_tpRC != 0])
        norm_tp = norm_tpRC.sum(axis=(2,3))
        fopt_tp = (xtrwt_tpRC*image_ftpRC[f]/vopt_tpRC).sum(axis=(2,3))/norm_tp
        vopt_tp = xtrwt_tpRC.sum(axis=(2,3))/norm_tp
                    
        if (debug & (f==dfile)):
            np.savetxt('fstd_tp_'+str(f)+'.txt',fstd_tp,fmt='%9.2f ')
            imgstd_tpRC = (ok_ftpRC*image_ftpRC)[f]
            np.savetxt('optextr_'+str(f)+'.txt',np.vstack((np.indices((targets,2)).reshape((2,-1)),  \
                (iscr_ftpRC[f].sum(axis=(2,3))).flatten(),frac_ftp[f].flatten(),fstd_tp.flatten(),  \
                vstd_tp.flatten(), fopt_tp.flatten(),vopt_tp.flatten(),dR_tp.flatten(),     \
                dC_tp.flatten())).T, fmt=3*"%4i "+"%8.3f "+4*"%10.0f "+2*"%8.3f ")
 
      # for filter imaging polarimetry 
      #   the column dimension is a single bin corresponding to the filter wavelength 
        hdul = pyfits.open(file)

        outfile = 'e'+file
        hdul['SCI'].data = fopt_tp.astype('float32').T.reshape((2,targets,1))
        for ext in [0,'SCI']:                       # specpolrawstokes wants them in both ext's
            hdul[ext].header['CRVAL1'] = wavl
            hdul[ext].header['CDELT1'] = 1.         # avoid divide by zero NaN's
            hdul[ext].header['CTYPE1'] = 'Angstroms'
        hdul['VAR'].data = vopt_tp.astype('float32').T.reshape((2,targets,1))
        covar_pt = np.zeros(2*targets).astype('float32').reshape((2,targets,1))
        hdul.append(pyfits.ImageHDU(data=covar_pt, header=hdr0, name='COV'))
        hdul['BPM'].data = (~ok_ftp[f]).astype('uint8').T.reshape((2,targets,1)) 
        hdul.writeto(outfile,overwrite=True)
        printstdlog ("Output file %s cr's: %3i    targets culled: %2i" %     \
                (outfile,crs,culls), logfile)
                
    return

# ----------------------------------------------------------
def imsppolextract(fileList,name,logfile='salt.log',debug=False):
    """derive extracted target data vs target and wavelength for slitless imaging spectropolarimetry

    Parameters 
    ----------
    fileList: list of strings
    hi_df: 2d integer array. h = (0,1..) filter-pair counter, i = (0,1) index within pair 

    """
    """
    wavmap_ftpR,Cmap_ftpR: 4d float arrays with wavelength and column position vs Row of spectra in target box
    image_ftpRC,var_ftpRC,bkg_ftpRC: 5d float arrays of input target data
    okbin_ftpRC,oktgt_ftpRC: 5d boolean array. bin: badpix,gap; tgt: tgt overlap
    names without "f" refer to calfil reference
    _d dimension index r,c = 0,1
    _f file index
    _t potential target id index
    _b best id index (for guiding and extraction profile)
    _T final target index    
    _p pol beam = 0,1 for O,E
    _R, _C bin coordinate within target box

    """

    pixmm = 0.015
    pm1_p = np.array([1,-1]) 
    files = len(fileList)      
    hdul0 = pyfits.open(fileList[0])
    hdr0 = hdul0[0].header
    exptime = hdr0['EXPTIME']
    wavl = float(hdr0['FILTER'][3:])
    imwav = 5000.                                           # reference wavelength for imaging    
    rcbin_d = np.array(hdr0['CCDSUM'].split(' ')).astype(int)[::-1]
    prows, cols = hdul0[1].data.shape[1:]
    imgno_f = np.array([f.split('.')[0][-4:] for f in fileList]).astype(int)
    camtem = hdr0['CAMTEM']
    coltem = hdr0['COLTEM']
    yxp0_dp = np.array([map(float,hdr0['YXAXISO'].split()),map(float,hdr0['YXAXISE'].split())]).T
    calimgno_F = np.array(hdr0['CALIMG'].split(' ')).astype(int) 
    calfilters = len(calimgno_F)    
    tgt_prc = hdul0['TMAP'].data                            # these should be the same for all files 
    tgtTab = Table.read(hdul0['TGT'])    
    ids = len(tgtTab)
    RAd0 = Longitude(hdr0['RA0']+' hours').degree
    DECd0 = Latitude(hdr0['DEC0']+' degrees').degree
    PAd0 = hdr0['PA0']    
            
    printstdlog(("candidate ids:       %3i" % ids),logfile)
        
  # get id'd target positions in calfilter(s) at reference (imwav) wavelength
    dateobs =  hdr0['DATE-OBS'].replace('-','')
    trkrho = hdr0['TRKRHO']   
    ur0,uc0,saltfps = rssdtralign(dateobs,trkrho)       # ur, uc =unbinned pixels, saltfps =micr/arcsec
    yx0_d = -0.015*np.array([ur0,uc0])                  # center of CCD relative to optical axis in mm
    yxOEoff_d = -np.diff((yxp0_dp - RSSpolgeom(hdul0,imwav)[2]), axis=1)[:,0]            
    yx0_dp,rshift = RSSpolgeom(hdul0,imwav,yxOEoff_d=yxOEoff_d)[:2]     # use pol image shift for 5000 Ang              
    rccenter_d = ccdcenter(hdul0[1].data[0])[0]*np.array([2,1])         # gets center of unsplit data               

    YX_dt = np.array([tgtTab['Y'],tgtTab['X']])        
    yxcat_dpt = RSScolpolcam(YX_dt,imwav,coltem,camtem,yxOEoff_d=yxOEoff_d)
    yxpcat_dpt = yxcat_dpt - yx0_dp[:,:,None]
    yxp_dpt = np.array([[tgtTab['YO'],tgtTab['YE']],[tgtTab['XO'],tgtTab['XE']]])
    dyxp_dpt = yxp_dpt - yxpcat_dpt                     # target offset from model
 
    Rfovlim = 50.0                                      # avoid effects at edge of SALT FOV 
    okfov_t = (np.sqrt((YX_dt**2).sum(axis=0)) < Rfovlim)
    fovcullList = list(np.where(~okfov_t)[0])
    if len(fovcullList): printstdlog((("%3i ids FOV culled: "+len(fovcullList)*"%2i ") %     \
                tuple([len(fovcullList),]+fovcullList)),logfile)
                
  # form reference wav, dwav, and col maps for guiding estimates
    wav_W = np.arange(3200.,10600.,100.)
    Wavs = wav_W.shape[0]
    Wimref = np.where(wav_W == imwav)[0][0]    
    np.savetxt('onaxisspec.txt', \
        RSScolpolcam(np.zeros((2,Wavs)),wav_W,coltem,camtem,yxOEoff_d=yxOEoff_d).reshape((4,-1)).T,fmt="%8.4f") 
    dY = 0.05                                           # for Y derivative
    wav_s = np.tile(wav_W,ids)
    YX_ds = np.repeat(YX_dt,Wavs,axis=1)
    YX1_ds = np.copy(YX_ds)
    YX1_ds[0] += dY*np.ones(ids*Wavs)       
    yx_dptW = RSScolpolcam(YX_ds,wav_s,coltem,camtem,yxOEoff_d=yxOEoff_d).reshape((2,2,ids,Wavs))   \
        + dyxp_dpt[:,:,:,None]
    yx1_dptW = RSScolpolcam(YX1_ds,wav_s,coltem,camtem,yxOEoff_d=yxOEoff_d).reshape((2,2,ids,Wavs))     \
        + dyxp_dpt[:,:,:,None]  
    dydY_ptW = (yx1_dptW[0] - yx_dptW[0])/dY
    dydW_ptW = np.zeros((2,ids,Wavs)) 
    dWdY_ptW = np.zeros_like(dydW_ptW)    
    for p,id in np.ndindex(2,ids):   
        dydW_ptW[p,id] = UnivariateSpline(wav_W,yx_dptW[0,p,id],s=0).derivative()(wav_W)
    dWdY_ptW = dydY_ptW/dydW_ptW

    rc_dptW = (yx_dptW - yx0_d[:,None,None,None])    \
          /(rcbin_d[:,None,None,None]*pixmm) + rccenter_d[:,None,None,None]       
    rcp_dptW = rc_dptW + np.array([[-rshift,-rshift-prows],[0,0]])[:,:,None,None]                                    
    r0_pt = np.clip(np.ceil(rcp_dptW[0].min(axis=2)).astype(int),0,prows)
    r1_pt = np.clip(np.floor(rcp_dptW[0].max(axis=2)).astype(int),0,prows)
    if debug:
        np.savetxt("r01_pt.txt",np.vstack((r0_pt,r1_pt)).T,fmt="%4i ")
        np.savetxt("dyxp_dpt.txt",np.vstack((np.indices((2,ids)).reshape((2,-1)),  \
            dyxp_dpt.reshape((2,-1)))).T,fmt="%3i %3i %8.4f %8.4f")    

    wavmap_tpr = np.zeros((ids,2,prows),dtype='float32')
    dwavmap_tpr = np.zeros_like(wavmap_tpr)    
    cmap_tpr = np.zeros_like(wavmap_tpr)            
    for id,p in np.ndindex(ids,2):
        Rows = (r1_pt - r0_pt)[p,id]+1            
        dr_W = rcp_dptW[0,p,id] - r0_pt[p,id,None]   
        wavmap_tpr[id,p,r0_pt[p,id]:r1_pt[p,id]+1] =   \
            interp1d(dr_W,wav_W,kind='cubic')(np.arange(Rows)) 
        dwavmap_tpr[id,p,r0_pt[p,id]:r1_pt[p,id]+1] =   \
            interp1d(dr_W,dWdY_ptW[p,id],kind='cubic')(np.arange(Rows))                        
        cmap_tpr[id,p,r0_pt[p,id]:r1_pt[p,id]+1] =   \
            interp1d(dr_W,rcp_dptW[1,p,id],kind='cubic')(np.arange(Rows))                      
        
  # get target box info
    rckey_pd = np.array([['R0O','C0O'],['R0E','C0E']])
    Rows,Cols = np.array(hdul0[0].header['BOXRC'].split()).astype(int)
    ri_tpR = np.zeros((ids,2,Rows),dtype=int)
    ci_tpC = np.zeros((ids,2,Cols),dtype=int)
    bkgwidth = 4 
    cbi_tpC = np.zeros((ids,2,Cols+2*bkgwidth),dtype=int)   # columns around box for background
    for p in (0,1):
        ri_tpR[:,p] = np.clip(tgtTab[rckey_pd[p,0]][:,None] + np.arange(Rows)[None,:], 0,prows-1)
        ci_tpC[:,p] = np.clip(tgtTab[rckey_pd[p,1]][:,None] + np.arange(Cols)[None,:], 0,cols-1)
        cbi_tpC[:,p] = np.clip(tgtTab[rckey_pd[p,1]][:,None] - bkgwidth + \
            np.arange(Cols+2*bkgwidth)[None,:], 0,cols-1) 

  # find good Rows with no target overlap
    okovlap_tpR = np.zeros((ids,2,Rows),dtype=bool)
    for t,p in np.ndindex(ids,2):
        okovlap_tpR[t,p] = (tgt_prc[p][ri_tpR[t,p],:][:,ci_tpC[t,p]]==t+1).all(axis=1)
    okovlap_t = okovlap_tpR.any(axis=2).all(axis=1)
    ovlapcullList = list(np.where(~okovlap_t & okfov_t)[0])
    if len(ovlapcullList): printstdlog((("%3i ids overlap culled: "+len(ovlapcullList)*"%2i ") %     \
                tuple([len(ovlapcullList),]+ovlapcullList)),logfile)

  # assemble target and background data
    image_fprc = np.zeros((files,2,prows,cols))
    var_fprc = np.zeros_like(image_fprc) 
    okbin_fprc =  np.zeros_like(image_fprc).astype(bool)    
    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)
        image_fprc[f] = hdul['SCI'].data
        var_fprc[f] = hdul['VAR'].data
        okbin_fprc[f] = (hdul['BPM'].data==0) 

  # ensure same bins are used within hw pair
    hi_df = findpair(fileList,logfile=logfile)   
    pairs = hi_df[0].max() + 1
    okbin_hprc = np.zeros((pairs,2,prows,cols),dtype=bool)       
    for h in range(pairs): okbin_hprc[h] = okbin_fprc[hi_df[0]==h].all(axis=0)

    if (debug):
        df, dt, db, dp = 14,45,13,1
        dlblt = "_"+str(df)+"_"+str(dt)+"_"+str(dp)+".txt"
        dlblb = "_"+str(df)+"_"+str(db)+"_"+str(dp)+".txt"
        open("bkgsum.txt","w")    # new empty file 
        sumfile = open("bkgsum.txt","a")
        
  # find bad rows based on bkg positive outliers (use median across bkgwidth rows)
  # require (med(bkg) - min(med(bkg))/maxsig < bkglim
    bkglim = 1. 
    okbkg_ftpR = np.zeros((files,ids,2,Rows),dtype=bool)
    bkgleft_ftpR = np.zeros((files,ids,2,Rows))
    bkgright_ftpR = np.zeros((files,ids,2,Rows))
              
    for f in range(files): 
        okbin_fprc[f] = okbin_hprc[hi_df[0,f]]
        relxcess_tpR = np.zeros((ids,2,Rows))
        for t,p in np.ndindex(ids,2):
            if ~(okfov_t & okovlap_t)[t]: continue
            bkg_RC = np.zeros((Rows,Cols+2*bkgwidth))
            okbkg_RC = np.zeros_like(bkg_RC,dtype=bool)
            bkg_RC = image_fprc[f,p][ri_tpR[t,p],:][:,cbi_tpC[t,p]]            
            okbkg_RC = (okovlap_tpR[t,p][:,None] &  \
                (okbin_fprc[f,p] & (tgt_prc[p]==255))[ri_tpR[t,p],:][:,cbi_tpC[t,p]]) 
            bkgleft_ftpR[f,t,p] =np.ma.min(np.ma.masked_array(bkg_RC[:,:bkgwidth], \
                mask=~okbkg_RC[:,:bkgwidth]),axis=1).data*okbkg_RC[:,:bkgwidth].any(axis=1) 
            bkgright_ftpR[f,t,p] =np.ma.min(np.ma.masked_array(bkg_RC[:,-bkgwidth:], \
                mask=~okbkg_RC[:,-bkgwidth:]),axis=1).data*okbkg_RC[:,-bkgwidth:].any(axis=1) 
            okbkg_ftpR[f,t,p] = ((bkgleft_ftpR[f,t,p] >0.) & (bkgright_ftpR[f,t,p] >0.))
            
            if okbkg_ftpR[f,t,p].sum():            
                minbkg = np.minimum(bkgleft_ftpR[f,t,p,okbkg_ftpR[f,t,p]].min(), \
                    bkgright_ftpR[f,t,p,okbkg_ftpR[f,t,p]].min())
                bkgxcess_R = np.maximum(bkgleft_ftpR,bkgright_ftpR)[f,t,p] - minbkg
                okbin_RC = okbin_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
                maxsig_R = (okbin_RC*image_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]) \
                    [:,(Cols/2-2):(Cols/2+3)].max(axis=1)
                maxsig_R = np.clip(maxsig_R - (bkg_RC[:,:bkgwidth].min(axis=1) +    \
                    bkg_RC[:,-bkgwidth:].min(axis=1))/2.,1.,1.e9) 
                relxcess_tpR[t,p] =  \
                    boxsmooth1d(bkgxcess_R/maxsig_R,okbkg_ftpR[f,t,p],4,.2)
                okbkg_ftpR[f,t,p] &= (relxcess_tpR[t,p] < bkglim)
                                 
            bkgrows0 = np.ptp(np.where(okovlap_tpR[t,p])[0])+1
            bkgrows1 = okbkg_ftpR[f,t,p].sum()
            if (debug): 
                print >>sumfile, (5*"%4i "+"%10.2f ") % ((f,t,p,bkgrows0,bkgrows1,minbkg))
                if ((f==df)&(t==dt)&(p==dp)):
                    np.savetxt("bkgdat_RC"+dlblt,bkg_RC,fmt="%9.2f") 
                    np.savetxt("okbkg_RC"+dlblt,okbkg_RC.astype(int),fmt="%3i")
                    np.savetxt("bkgexcess_R"+dlblt, np.vstack((bkgleft_ftpR[f,t,p],bkgright_ftpR[f,t,p], \
                        bkgxcess_R,maxsig_R,relxcess_tpR[t,p])).T,fmt="%9.2f") 
        if (debug): 
            if (f==df):
                np.savetxt("relxcess_"+str(f)+"tpR.txt",    \
                    np.clip(relxcess_tpR,.001,1.e3).reshape((-1,Rows)).T,fmt="%9.3f")  
    
  # equalize row culls, evaluate and remove background
  # background is linear interpolation of left and right background
  # left and right background is median for bkgrow spread < fitrows, linear fit otherwise
  # use good background rows in common among files to avoid systematic changes
    fitrows = Rows/2 
    okbkg_tpR = okbkg_ftpR.all(axis=0)
    okbkg_t = (okbkg_tpR.sum(axis=2) > 0).all(axis=1)    
    bkgcullList = list(np.where(~okbkg_t & okfov_t & okovlap_t)[0])
    if len(bkgcullList): printstdlog((("%3i ids background culled: "+len(bkgcullList)*"%2i ") %     \
                tuple([len(bkgcullList),]+bkgcullList)),logfile)    
    if debug:
        np.savetxt("okbkg_tpR.txt",np.vstack((np.indices((ids,2)).reshape((2,-1)),  \
                    okbkg_tpR.astype(int).reshape((-1,Rows)).T)).T,fmt="%3i %3i  "+Rows*"%3i ") 
        np.savetxt("Rows_ftp.txt",okbkg_ftpR.sum(axis=3).reshape((files,-1)).T,fmt="%3i") 
        np.savetxt("Rows_tp.txt",okbkg_tpR.sum(axis=2),fmt="%3i") 

    image_ftpRC = np.zeros((files,ids,2,Rows,Cols))
    var_ftpRC = np.zeros_like(image_ftpRC)
    bkg_ftpRC = np.zeros_like(image_ftpRC)    
    okbin_ftpRC = np.zeros((files,ids,2,Rows,Cols),dtype=bool)
    oktgt_ftpRC = np.zeros_like(okbin_ftpRC) 
    for f,t,p in np.ndindex(files,ids,2): 
        if ~(okfov_t & okovlap_t & okbkg_t)[t]: continue    
        image_ftpRC[f,t,p] = image_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
        var_ftpRC[f,t,p] = var_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]] 
        okbin_ftpRC[f,t,p] = okbin_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
        oktgt_ftpRC[f,t,p] = (tgt_prc==t+1)[p][ri_tpR[t,p],:][:,ci_tpC[t,p]] 
        row_R = np.arange(Rows)
        okbkgtp_R = okbkg_tpR[t,p]
                
        if np.where(okbkgtp_R)[0].ptp() < fitrows:
            leftcof_d = np.array([0.,np.median(bkgleft_ftpR[f,t,p,okbkgtp_R])])
            rightcof_d = np.array([0.,np.median(bkgright_ftpR[f,t,p,okbkgtp_R])]) 
        else:
            leftcof_d = np.polyfit(row_R[okbkgtp_R], bkgleft_ftpR[f,t,p,okbkgtp_R],1)
            rightcof_d = np.polyfit(row_R[okbkgtp_R], bkgright_ftpR[f,t,p,okbkgtp_R],1) 
        bkgleft_R = np.poly1d(leftcof_d)(row_R)
        bkgright_R = np.poly1d(rightcof_d)(row_R)
        bkgwt_C = (np.arange(Cols)+bkgwidth/2.)/(Cols+bkgwidth) 
        bkg_ftpRC[f,t,p] = (1.-bkgwt_C)*bkgleft_R[:,None] + bkgwt_C*bkgright_R[:,None]
        if debug: 
            if ((f==df)&(t==dt)&(p==dp)): 
                np.savetxt("bkgfit_R"+dlblt,np.vstack((okbkgtp_R.astype(int),bkgleft_R,bkgright_R)).T,  \
                        fmt="%3i %10.2f %10.2f")
                np.savetxt("bkgfit_RC"+dlblt,bkg_ftpRC[f,t,p],fmt="%10.2f")     

    signal_ftpRC = image_ftpRC - bkg_ftpRC 

  # fit mean column profiles for each file and target to 1D Moffat, using rows with all columns
    wavmap_tpR = np.zeros((ids,2,Rows))
    dwavmap_tpR = np.zeros_like(wavmap_tpR)    
    Cmap_tpR = np.zeros_like(wavmap_tpR)
    for t,p in np.ndindex(ids,2):
        wavmap_tpR[t,p] = wavmap_tpr[t,p][ri_tpR[t,p]]
        dwavmap_tpR[t,p] = dwavmap_tpr[t,p][ri_tpR[t,p]] 
        Cmap_tpR[t,p] = (cmap_tpr[t,p][ri_tpR[t,p]] - ci_tpC[t,p,0]).clip(min=0.)  
      # extend Cmap one bin out to allow for guiding errors
        RC0,RC1 = np.where(Cmap_tpR[t,p] > 0.)[0][[0,-1]].clip(1,Rows-2)
        Cmap_tpR[t,p,RC0-1] = Cmap_tpR[t,p,RC0] +(Cmap_tpR[t,p,RC0]-Cmap_tpR[t,p,RC0+1])
        Cmap_tpR[t,p,RC1+1] = Cmap_tpR[t,p,RC1] +(Cmap_tpR[t,p,RC1]-Cmap_tpR[t,p,RC1-1])    
    ok1_ftpRC = (okbin_ftpRC & oktgt_ftpRC) & okbkg_ftpR[:,:,:,:,None]
    cols_ftpR = ok1_ftpRC.sum(axis=4)
    prof_ftpRC = ok1_ftpRC*signal_ftpRC
    prof_ftpRC[cols_ftpR>0,:] /= cols_ftpR[cols_ftpR>0,None]
    norm_ftpR = prof_ftpRC.max(axis=4)
    Ridxsmax_ftp = np.argmax(((okbin_ftpRC & oktgt_ftpRC)*image_ftpRC).max(axis=4),axis=3)
    prof_ftpRC[cols_ftpR>0,:] /= norm_ftpR[cols_ftpR>0,None]    
    fitprof_ftpR = ((cols_ftpR == Cols) & (Cmap_tpR>0.)[None,:,:,:]) 
    fitprof_ftpR[:,~(okfov_t & okovlap_t & okbkg_t)] = False
    i_ftpR = np.zeros((files,ids,2,Rows),dtype=int)
    i_ftpR[fitprof_ftpR] = np.arange(fitprof_ftpR.sum()) 
    if debug: 
       # idebug = i_ftpR[df,dt,dp,28] 
        idebug=None 
        
    sigma_i, fCmax_i, C0_i, fiterr_iC, okprof_i =    \
        moffat1dfit(prof_ftpRC[fitprof_ftpR,:].reshape((-1,Cols)),    \
            np.ones((fitprof_ftpR.sum(),Cols),dtype=bool),beta=2.5,idebug=idebug)
                    
    i_ftpR = np.zeros((files,ids,2,Rows),dtype=int)
    i_ftpR[fitprof_ftpR] = np.arange(fitprof_ftpR.sum()) 
    sigma_ftpR = sigma_i[i_ftpR]
    fCmax_ftpR = fCmax_i[i_ftpR]
    fmax_ftpR = fCmax_i[i_ftpR]*norm_ftpR
    C0_ftpR = C0_i[i_ftpR] 
    dC_ftpR = C0_ftpR - Cmap_tpR[None,:,:,:]
    rmserr_ftpR = np.sqrt((fiterr_iC[i_ftpR,:]**2).mean(axis=4))
    okprof_ftpR = np.zeros((files,ids,2,Rows)).astype(bool)
    okprof_ftpR[fitprof_ftpR] = okprof_i

  # identify good profiles, best targets to use for seeing, guiding
    goodprofrmsmax = 0.1

    isgoodprof1_ftpR = (okprof_ftpR & (rmserr_ftpR < goodprofrmsmax))
   
  # goodprof also requires sigma not above outer quartile limits 
    outersigma_fq = np.zeros((files,2)) 
    for f in range(files):  
        q1,q3 = np.percentile(sigma_ftpR[f][isgoodprof1_ftpR[f]],(25,75))
        outersigma_fq[f] = (q1 - 3.*(q3-q1)),(q3 + 3.*(q3-q1))
    isgoodprof2_ftpR = (isgoodprof1_ftpR & (sigma_ftpR < outersigma_fq[:,1,None,None,None]))
    
  # for guiding targets, do a very rough cut on uncorrected column errors
    dC_ftp = np.ma.median(np.ma.masked_array(dC_ftpR,mask=~isgoodprof2_ftpR),axis=3).data 
    ddC_ftpR = dC_ftpR - dC_ftp[:,:,:,None]
    q1,q3 = np.percentile(ddC_ftpR[isgoodprof2_ftpR],(25,75))    
    lower,upper = (q1 - 5.*(q3-q1)),(q3 + 5.*(q3-q1))
    isgoodprof3_ftpR = (isgoodprof2_ftpR & ((ddC_ftpR > lower) & (ddC_ftpR < upper)))
    contRows_ftp = np.zeros((files,ids,2))
    for f,t,p in np.ndindex((files,ids,2)):
        goodR = np.where(isgoodprof3_ftpR[f,t,p])[0]
        contList = np.split(goodR,np.where(np.diff(goodR) != 1)[0]+1)
        contRows_ftp[f,t,p] = max(map(len,contList))
    
    if debug:
        np.savetxt("fitprof_ftpR.txt",np.vstack((np.indices((files,ids,2)).reshape((3,-1)),  \
            fitprof_ftpR.astype(int).reshape((-1,Rows)).T)).T,fmt="%3i %3i %3i  "+Rows*"%3i ") 
        np.savetxt("proffit.txt",np.vstack((np.where(fitprof_ftpR), isgoodprof2_ftpR[fitprof_ftpR], \
            isgoodprof3_ftpR[fitprof_ftpR].astype(int),sigma_i,fCmax_i,dC_ftpR[fitprof_ftpR],   \
            rmserr_ftpR[fitprof_ftpR],fmax_ftpR[fitprof_ftpR])).T, fmt=6*"%4i "+4*"%9.3f "+"%9.1f ") 
        np.savetxt("outersigma_fq.txt",outersigma_fq, fmt="%8.3f ")
        np.savetxt("Cmap_tpR.txt", np.vstack((np.indices((ids,2)).reshape((2,-1)), \
            Cmap_tpR.reshape((-1,Rows)).T)).T, fmt=2*"%4i "+Rows*"%8.3f ")              
        np.savetxt("C0_ftpR.txt", np.vstack((np.where(fitprof_ftpR), isgoodprof3_ftpR[fitprof_ftpR], \
            C0_ftpR[fitprof_ftpR])).T, fmt=5*"%4i "+"%8.3f ")
        np.savetxt("isgoodprof2_tpRf.txt", np.vstack((np.indices((ids,2,Rows)).reshape((3,-1)),  \
            isgoodprof2_ftpR.transpose((1,2,3,0)).reshape((-1,files)).astype(int).T)).T,fmt="%3i")
        np.savetxt("isgoodprof3_tpRf.txt", np.vstack((np.indices((ids,2,Rows)).reshape((3,-1)),  \
            isgoodprof3_ftpR.transpose((1,2,3,0)).reshape((-1,files)).astype(int).T)).T,fmt="%3i")
        np.savetxt("contRows_tpf.txt", np.vstack((np.indices((ids,2)).reshape((2,-1)),  \
            contRows_ftp.transpose((1,2,0)).reshape((-1,files)).T)).T,fmt="%3i") 

  # best targets require good profs in at least Rows/3 continuous rows (file median), both O,E
  #  eliminate targets with >2 files with less than Rows/4 continuous

    isbest_t = (np.median(contRows_ftp, axis=0) > Rows/3).all(axis=1)
    isbest_t &= ((contRows_ftp < Rows/4).sum(axis=0) < 3).all(axis=1)
    tbest_b = np.where(isbest_t)[0]
    if debug: print (("Rows > %2i: "+len(tbest_b)*"%3i ") % ((Rows/3,)+tuple(tbest_b)))

  # best targets also require good prof at max signal, all files, both O,E     
    isgoodmax_ftp = isgoodprof3_ftpR.reshape((-1,Rows))  \
        [range(files*ids*2),Ridxsmax_ftp.flatten()].reshape((files,ids,2)) 
    isbest_t &= isgoodmax_ftp.all(axis=(0,2)) 
    tbest_b = np.where(isbest_t)[0]  
    if debug: print (("good max:  "+len(tbest_b)*"%3i ") % tuple(tbest_b))
   
  # best targets also require good prof around max signal, all files, both O,E
    for t in tbest_b:
        for dR in (-2,-1,1,2):
            isgoodmax_ftp[:,t] &= isgoodprof3_ftpR[:,t].reshape((-1,Rows))  \
                [range(files*2),Ridxsmax_ftp[:,t].flatten()+dR].reshape((files,2)) 
                       
    isbest_t &= isgoodmax_ftp.all(axis=(0,2)) 
    tbest_b = np.where(isbest_t)[0]
    besttargs = isbest_t.sum()

    printstdlog(("best ids:  "+besttargs*"%3i " % tuple(tbest_b)),logfile)

    isgoodprof_fbpR = isgoodprof3_ftpR[:,isbest_t,:,:]       
    bestprof_fbpR = isgoodprof_fbpR*(4./3.)*(fmax_ftpR*sigma_ftpR)[:,tbest_b]
    bestvar_fbpR = bestprof_fbpR + isgoodprof_fbpR*bkg_ftpRC[:,isbest_t].mean(axis=4)
    Cmap_bpR = Cmap_tpR[isbest_t,:,:]    
    wavmap_bpR = wavmap_tpR[isbest_t,:,:]
    dwavmap_bpR = dwavmap_tpR[isbest_t,:,:] 
 
    if debug:        
        bestprof_bpRf = bestprof_fbpR.transpose((1,2,3,0)).reshape((-1,files))          
        rmserr_bpRf = rmserr_ftpR[:,isbest_t].transpose((1,2,3,0)).reshape((-1,files))
        sigma_bpRf = sigma_ftpR[:,isbest_t].transpose((1,2,3,0)).reshape((-1,files))
        dC_bpRf = dC_ftpR[:,isbest_t].transpose((1,2,3,0)).reshape((-1,files)) 
        C0_bpRf = C0_ftpR[:,isbest_t].transpose((1,2,3,0)).reshape((-1,files))
        okbkg_fbpR = okbkg_ftpR[:,isbest_t]       
        fitprof_bpRf = fitprof_ftpR[:,isbest_t].transpose((1,2,3,0)).reshape((-1,files))
        isgoodprof_bpRf = isgoodprof_fbpR.transpose((1,2,3,0)).reshape((-1,files))        
        np.savetxt("contRows_fbp.txt",contRows_ftp[:,isbest_t].reshape((files,-1)),fmt="%4i ") 
        np.savetxt("Cmap_bpR.txt",np.vstack((np.indices((besttargs,2)).reshape((2,-1)),   \
                    Cmap_bpR.reshape((-1,Rows)).T)).T,fmt="%4i %4i "+Rows*"%8.3f ")         
        np.savetxt("wavmap_bpR.txt",np.vstack((np.indices((besttargs,2)).reshape((2,-1)),   \
                    wavmap_bpR.reshape((-1,Rows)).T)).T,fmt="%4i %4i "+Rows*"%8.2f ") 
        np.savetxt("bestprof_bpRf.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    bestprof_bpRf.T)).T,fmt="%4i %4i %4i "+files*"%8.1f ")    
        np.savetxt("rmserr_bpRf.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    rmserr_bpRf.T)).T,fmt="%4i %4i %4i "+files*"%8.3f ")  
        np.savetxt("sigma_bpRf.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    sigma_bpRf.T)).T,fmt="%4i %4i %4i "+files*"%10.3f ")
        np.savetxt("dC_bpRf.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    dC_bpRf.T)).T,fmt="%4i %4i %4i "+files*"%10.3f ") 
        np.savetxt("C0_bpRf.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    C0_bpRf.T)).T,fmt="%4i %4i %4i "+files*"%8.3f ")
        np.savetxt("okbkg_fbpR.txt",np.vstack((np.indices((files,besttargs,2)).reshape((3,-1)),    \
                    okbkg_fbpR.reshape((-1,Rows)).astype(int).T)).T, fmt="%4i %4i %4i "+Rows*"%3i ")
        np.savetxt("fitprof_bpRf.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    fitprof_bpRf.astype(int).T)).T, fmt="%4i %4i %4i "+files*"%3i ")
        np.savetxt("isgoodprof_bpRf.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    isgoodprof_bpRf.astype(int).T)).T, fmt="%4i %4i %4i "+files*"%3i ")

  # for besttargs compute column motion relative to calfil reference
    dC_fbpR = dC_ftpR[:,isbest_t]
    isgoodprof0_fbpR = (isgoodprof_fbpR & isgoodprof_fbpR[0])
    dC_fbp = np.ma.median(np.ma.masked_array(dC_fbpR, mask=~isgoodprof0_fbpR),axis=3).data

  # for besttargs compute relative row motion comparing difference spectrum with impol reference file 
  #   impol reference file is the image number closest to a cal reference file number
    fminblue = 0.50                                # only use blue rows > 0.5*smax
    fminred = 0.20                                 # only use red rows > 0.2*smax or min+0.05    
    fref = np.unravel_index(np.argmin(np.abs(imgno_f[:,None] - calimgno_F[None,:])),(files,calfilters))[0]
    
    print "imgno_f[fref]: ",imgno_f[fref]
    
    okcomp_fbp = np.ones((files,besttargs,2)).astype(bool)   
    Ridxsmax_fbp = np.argmax((isgoodprof_fbpR*bestprof_fbpR),axis=3)
    smax_fbp = np.zeros((files,besttargs,2))
    Rsmax_fbp = np.zeros((files,besttargs,2))
    Redge_dfbp = np.zeros((2,files,besttargs,2)).astype(int)
    badprofcount_fbp = np.zeros((files,besttargs,2)).astype(int)
    diffRows_bp = np.zeros((besttargs,2)).astype(int)
    comps_fbp = np.zeros((files,besttargs,2)).astype(int)
    meandiffampl_bp = np.zeros((besttargs,2)) 
    meandifferr_bp = np.zeros((besttargs,2))         
    rx01_dpb = np.zeros((2,2,besttargs))                            # note transposition 
    x01diff_dpb = np.zeros((2,2,besttargs)).astype(int)             # note transposition
    R01diff_dfbp = np.zeros((2,files,besttargs,2)).astype(int)    
    rr01diff_dfbp = np.zeros((2,files,besttargs,2)).astype(int)
    rzero_fbp = np.zeros((files,besttargs,2))
    diff_rList_fbp = np.empty((files,besttargs,2)).astype(object)   # a fbp array of Lists of diff_r
    meandiff_xList_bp = np.empty((besttargs,2)).astype(object)      # a bp array of Lists of meandiff_x
    dR_cList_fbp = np.empty((files,besttargs,2)).astype(object)     # a fbp array of Lists of dr_c       
    dRcomp_fbp = np.zeros((files,besttargs,2)) 
    dRcomperr_fbp = np.zeros((files,besttargs,2))        
         
    for b,p in np.ndindex((besttargs,2)):
        for f in range(files):                      # first find consistent row sample size
            Redge_d = np.where(isgoodprof_fbpR[f,b,p])[0][[0,-1]]
            RmaxList = range(max(Redge_d[0],Ridxsmax_fbp[f,b,p]-2),min(Redge_d[1],Ridxsmax_fbp[f,b,p]+2)+1)
            cof_x = np.polyfit(RmaxList,bestprof_fbpR[f,b,p,RmaxList],2)
            Rsmax_fbp[f,b,p] = -0.5*cof_x[1]/cof_x[0]
            Ridxsmax_fbp[f,b,p] = np.around(Rsmax_fbp[f,b,p])  # update max by fitting quadratic around max posn 
            smax_fbp[f,b,p] = np.polyval(cof_x,Rsmax_fbp[f,b,p])
            f_R = bestprof_fbpR[f,b,p,:]/smax_fbp[f,b,p]
                                                    # first go out to 50%, or end of goodprofs, whichever 1st
            ok_R = (isgoodprof_fbpR[f,b,p] & (f_R > fminblue))
            dR0 = -(list(np.where(~ok_R[Ridxsmax_fbp[f,b,p]::-1])[0])+[Ridxsmax_fbp[f,b,p]])[0] +1
            dR1 = (list(np.where(~ok_R[Ridxsmax_fbp[f,b,p]:-1])[0])+[Rows-Ridxsmax_fbp[f,b,p]])[0] -1
            Redge_d = np.array([dR0,dR1]) + Ridxsmax_fbp[f,b,p]                                                    # now extend red side
            min_d = np.array([f_R[:Redge_d[0]].min(),f_R[(Redge_d[1]+1):].min()])
            fminredp = max(fminred, min_d[1-p]+.05)
            ok_R = (isgoodprof_fbpR[f,b,p] & (f_R > fminredp))            
            if (p==0):                                  # O beam: red is on right (d=1)
                dR = (list(np.where(~ok_R[Redge_d[1]:-1])[0])+[Rows-Redge_d[1]])[0] -1
            else:
                dR = -(list(np.where(~ok_R[Redge_d[0]::-1])[0])+[Redge_d[0]])[0] +1
            Redge_d[1-p] = Redge_d[1-p] + dR          
            Redge_dfbp[:,f,b,p] = Redge_d
        diffRows_bp[b,p] = (Redge_dfbp[1,:,b,p]-Redge_dfbp[0,:,b,p] +1).min(axis=0).clip(0,Rows/4)
     
        for f in range(files):                      # compute diff, mean diff for target by smoothed spline
            f_R = bestprof_fbpR[f,b,p,:]/smax_fbp[f,b,p]        
            clipRows = Redge_dfbp[1,f,b,p]-Redge_dfbp[0,f,b,p] +1 - diffRows_bp[b,p]
            R0 = Redge_dfbp[0,f,b,p] + (1-p)*clipRows
            R1 = Redge_dfbp[1,f,b,p] - p*clipRows                                    
            badprofcount_fbp[f,b,p] = (~isgoodprof_fbpR[f,b,p,R0:(R1+1)]).sum()  
            diff_r = f_R[R0:(R1-3)] - f_R[(R0+4):(R1+1)] 
            r0idx = np.where(diff_r > 0.)[0][0]
            rzero_fbp[f,b,p] = r0idx-1 - diff_r[r0idx-1]/(diff_r[r0idx] - diff_r[r0idx-1])                
            R01diff_dfbp[:,f,b,p] = R0,R1
            diff_rList_fbp[f,b,p] = list(diff_r)
            
        diff_Xd = np.empty((0,2))                   # all samples of [X=r-drzero,diff] for target 
        f_X = np.empty(0,dtype=int)
        for f in range(files):                      # allow for sampling difference with reference
            diff_r = diff_rList_fbp[f,b,p]             
            for r in range(len(diff_r)):
                diff_Xd = np.vstack((diff_Xd,np.array([r-(rzero_fbp[f,b,p]-rzero_fbp[fref,b,p]), diff_r[r]])))
            f_X = np.append(f_X,np.repeat(f,len(diff_r)))
            
        rx0 = np.floor(diff_Xd.min(axis=0)[0]+.001)+1   # avoid having knot fall right on max or min 
        rx1 = np.floor(diff_Xd.max(axis=0)[0]-.001)                 
        Xsort_X = np.argsort(diff_Xd[:,0])          # spline requires sorted x's 
               
        meandiffSpline = LSQUnivariateSpline(diff_Xd[Xsort_X,0],diff_Xd[Xsort_X,1],np.arange(rx0,rx1+1.))
        r_x = np.arange(rx0,rx1+0.25,0.25)          # oversample spline 4x for reverse interpolation            
        meandiff_x = meandiffSpline(r_x)        
        meandiff_xList_bp[b,p] = list(meandiff_x)
        knots = len(meandiff_x)
        rx01_dpb[:,p,b] = rx0,rx1                   # for debug 
        meandifferr_bp[b,p] =  np.sqrt(meandiffSpline.get_residual()/len(Xsort_X))
        if debug:
            if ((b==db) & (p==dp)): 
                np.savetxt('differr_'+str(b)+'_'+str(p)+'_X.txt',np.vstack((f_X[Xsort_X],   \
                diff_Xd[Xsort_X,0], diff_Xd[Xsort_X,1],meandiffSpline(diff_Xd[Xsort_X,0]))).T,  \
                fmt="%3i "+3*"%9.4f ")
           
      # only use meandiff around zero cross with x slope > xdifflim (stop when not)
        xdifflim = 0.01
        xidxzero = np.where(meandiff_x > 0.)[0][0]
        badslope_x = (np.diff(meandiff_x) < xdifflim)     # note: the diff makes badslope length knots-1
        x0 = xidxzero - ((list(np.where(badslope_x[xidxzero::-1])[0])+[xidxzero+1])[0]-1)
        x1 = xidxzero + (list(np.where(badslope_x[xidxzero:])[0] +1 )+[knots-xidxzero])[0]-1 
        xzero = griddata(meandiff_x[x0:(x1+1)],r_x[x0:(x1+1)],0.,method='cubic')
        meandiffampl_bp[b,p] = meandiff_x[x1] - meandiff_x[x0]

        x01diff_dpb[:,p,b] = x0,x1                  # for debug
        rrcenter = [0.5,-0.5][p]                    # favor red side                    
                
        for f in range(files):                      # do comparison, avoiding points outside mean diff
            diff_r = np.array(diff_rList_fbp[f,b,p])
            ridxzero = int(np.around(rzero_fbp[f,b,p]))
            ratlim_d = [[0.8,0.95],[0.95,0.8]][p]   # favor blue side
            baddiff_r = ((diff_r < ratlim_d[0]*meandiff_x[x0]) | (diff_r > ratlim_d[1]*meandiff_x[x1]))
            rr0 = ridxzero - ((list(np.where(baddiff_r[ridxzero::-1])[0])+[ridxzero+1])[0]-1)
            rr1 = ridxzero + (list(np.where(baddiff_r[ridxzero:])[0])+[len(diff_r)-ridxzero])[0]-1       
            rr01diff_dfbp[:,f,b,p] = rr0,rr1        # for debug            
            comps_fbp[f,b,p] = rr1 - rr0 + 1
            if (comps_fbp[f,b,p] < 3):
                okcomp_fbp[f,b,p] = False
                continue
                                            
            dr_c = -(griddata(meandiff_x[x0:(x1+1)],r_x[x0:(x1+1)],diff_r[rr0:(rr1+1)],method='cubic') - \
                (np.arange(rr0,rr1+1)))            
            drcof_d, residuals = np.polyfit(np.arange(rr0,rr1+1),dr_c,1,full=True)[:2]           
            dRcomp_fbp[f,b,p] = np.polyval(drcof_d,rzero_fbp[f,b,p]) +  \
                (R01diff_dfbp[0,f,b,p] - R01diff_dfbp[0,fref,b,p])
            dRcomperr_fbp[f,b,p] =  np.sqrt(residuals[0])/(comps_fbp[f,b,p] - 2)
            dR_cList_fbp[f,b,p] = list(dr_c + R01diff_dfbp[0,f,b,p] - R01diff_dfbp[0,fref,b,p])

    diffRows_bp -= 4                                # rdiff has overlap only 
    comperrlim = 0.035                              # limit Rcomp guiding targets by differr/diffampl
    okcomp_b = (meandifferr_bp/meandiffampl_bp < comperrlim).all(axis=1)
    compids = okcomp_b.sum()

    printstdlog(("Rcomp ids: "+compids*"%3i " % tuple(tbest_b[okcomp_b])),logfile)
         
    dRcomp_fbp = dRcomp_fbp - dRcomp_fbp[fref,:,:]
 
    if debug:
        np.savetxt("tbest_b.txt",np.vstack((range(besttargs),tbest_b,okcomp_b.astype(int))).T,fmt="%3i ")     
        np.savetxt("isgoodprof3_tpRf.txt", np.vstack((np.indices((ids,2,Rows)).reshape((3,-1)),  \
            isgoodprof3_ftpR.transpose((1,2,3,0)).reshape((-1,files)).astype(int).T)).T,fmt="%3i")
        np.savetxt("isgoodmax_tpf.txt",np.vstack((np.indices((ids,2)).reshape((2,-1)),    \
            isgoodmax_ftp.transpose((1,2,0)).reshape((-1,files)).T)).T,fmt=2*"%4i "+files*"%3i ") 
        np.savetxt('dRcomp_fpb.txt',dRcomp_fbp.transpose((0,2,1)).reshape((files,-1)).T,fmt="%9.4f")               
        np.savetxt('dC_fpb.txt',dC_fbp.transpose((0,2,1)).reshape((files,-1)).T,fmt="%9.4f")       
        np.savetxt('smax_fpb.txt',smax_fbp.transpose((0,2,1)).reshape((files,-1)).T,fmt="%10.0f")

        Rzero_fbp = rzero_fbp + R01diff_dfbp[0]  
        np.savetxt('Rzero_fbp.txt',np.vstack((np.indices((files,besttargs,2)).reshape((3,-1)), \
            Redge_dfbp.reshape((2,-1)),R01diff_dfbp.reshape((2,-1)),badprofcount_fbp.flatten(), \
            rr01diff_dfbp.reshape((2,-1)),Rzero_fbp.flatten())).T,fmt=10*"%3i "+"%8.3f ")

        diff_rfpb = np.zeros((diffRows_bp.max(),files,2,besttargs))
        r_X = np.arange(rx01_dpb[0].min(),rx01_dpb[1].max()+0.25,0.25)
        X0_pb = (4.*(rx01_dpb[0] - r_X[0])).astype(int)
        diffknots_pb =  (4.*(rx01_dpb[1] - rx01_dpb[0])).astype(int) + 1      
        meandiff_Xpb = np.zeros((len(r_X),2,besttargs))         
        dR_cfbp = np.zeros((comps_fbp.max(),files,besttargs,2))                        
        for f,b,p in np.ndindex((files,besttargs,2)):
            diff_rfpb[:diffRows_bp[b,p],f,p,b] = np.array(diff_rList_fbp[f,b,p])
            meandiff_Xpb[X0_pb[p,b]:(X0_pb[p,b]+diffknots_pb[p,b]),p,b] = np.array(meandiff_xList_bp[b,p]) 
            dR_cfbp[:comps_fbp[f,b,p],f,b,p] = np.array(dR_cList_fbp[f,b,p])            
        np.savetxt('diff_fpbr.txt',np.vstack((np.indices((files,2,besttargs)).reshape((3,-1)),  \
            diff_rfpb.reshape((diffRows_bp.max(),-1)))).T,fmt=3*"%3i "+diffRows_bp.max()*"%8.3f ")
        np.savetxt('meandiff_pbX.txt',np.vstack((np.indices((2,besttargs)).reshape((2,-1)),  \
            x01diff_dpb.reshape((2,-1)), rx01_dpb.reshape((2,-1)), meandiffampl_bp.T.flatten(),   \
            meandifferr_bp.T.flatten(), meandiff_Xpb.reshape((len(r_X),-1)))).T,    \
            fmt=4*"%3i "+"%5.1f %5.1f %7.3f %7.4f "+len(r_X)*"%8.3f ")
        np.savetxt('dRcomp_fbp.txt',np.vstack((np.indices((files,besttargs,2)).reshape((3,-1)), \
            okcomp_fbp.flatten().astype(int), rr01diff_dfbp.reshape((2,-1)), rzero_fbp.flatten(), \
            dRcomp_fbp.flatten(), dRcomperr_fbp.flatten(), dR_cfbp.reshape((comps_fbp.max(),-1)) )).T,   \
            fmt=6*"%3i "+(3+comps_fbp.max())*"%9.4f")            
                                           
  # evaluate guiding errors relative to impol reference file for each file
  #   first assume that impol reference file has same reference as calfile
    YX_db = YX_dt[:,tbest_b]
    ri_bpR = ri_tpR[tbest_b] 
    dYX_df = np.zeros((2,files))   
    drot_f = np.zeros(files)
    dyxOEoff_df = np.zeros((2,files)) 
    dYXerr_df = np.zeros((2,files))   
    droterr_f = np.zeros(files)
    dyxOEofferr_df = np.zeros((2,files))
    wavmap_fbpR = np.zeros((files,besttargs,2,Rows),dtype='float32')
    dwavmap_fbpR = np.zeros_like(wavmap_fbpR)    
    yxfref_dpb = yxcat_dpt[:,:,tbest_b]
    dyxp_dpb = dyxp_dpt[:,:,tbest_b]
                 
    for f in range(files):
        if (f==fref): continue    
      # fit okcomp dRcomp_fpb, (dC_fpb-dC_fbp[fref]) to get relative dYX_df,drot_f,dyxEOoff_df
        yxf_dpb = yxfref_dpb +   \
            rcbin_d[:,None,None]*pixmm*np.array([dRcomp_fbp[f].T,(dC_fbp[f]-dC_fbp[fref]).T])
        dYX_df[:,f],drot_f[f],dyxOEoff_df[:,f],dYXerr_df[:,f],droterr_f[f],dyxOEofferr_df[:,f] =  \
            impolguide(YX_db,yxf_dpb,yxOEoff_d,imwav,coltem,camtem,debug=debug,name=name)
        YXf_db = YX_db + dYX_df[:,f,None] + np.array([1.,-1.])[:,None]*np.radians(drot_f[f])*YX_db[::-1]   
        YXf_ds = np.repeat(YXf_db,Wavs,axis=1)
        wav_s = np.tile(wav_W,besttargs)            
        yxf_dpbW = RSScolpolcam(YXf_ds,wav_s,coltem,camtem, \
            yxOEoff_d=yxOEoff_d+dyxOEoff_df[:,f]).reshape((2,2,besttargs,Wavs)) + dyxp_dpb[:,:,:,None]
        rcf_dpbW = (yxf_dpbW - yx0_d[:,None,None,None])    \
            /(rcbin_d[:,None,None,None]*pixmm) + rccenter_d[:,None,None,None]       
        rcpf_dpbW = rcf_dpbW + np.array([[-rshift,-rshift-prows],[0,0]])[:,:,None,None]
        for b,p in np.ndindex((besttargs,2)):         
            RList = list(np.where((ri_bpR[b,p] >= rcpf_dpbW[0,p,b].min()) &    \
                (ri_bpR[b,p] <= rcpf_dpbW[0,p,b].max()))[0])                
            wavmap_fbpR[f,b,p,RList] = interp1d(rcpf_dpbW[0,p,b],wav_W,kind='cubic')(ri_bpR[b,p,RList])
            dwavmap_fbpR[f,b,p,RList] =     \
                interp1d(rcpf_dpbW[0,p,b],dWdY_ptW[p][tbest_b[b]],kind='cubic')(ri_bpR[b,p,RList])
            
  # use O,E smax positions in impol file for besttargets to compute telescope fref guiding error dYfref_b 
  #   use nominal wavmap to compute smaxwav in O, E, then use dwavmap to compute dY that equalizes them
  #   This assumes no y BS flexure between cal reference and nearby impol reference file         
    wavsmax_fbp = np.zeros((files,besttargs,2)) 
    dwavsmax_fbp = np.zeros_like(wavsmax_fbp)    
    dYfref_b = np.zeros(besttargs)
    wavmap_fbpR[fref] = wavmap_bpR
    dwavmap_fbpR[fref] = dwavmap_bpR         
    for f,b,p in np.ndindex((files,besttargs,2)):          
        Rowok_i = np.where(isgoodprof_fbpR[f,b,p])[0]
        Rsmax = Rsmax_fbp[f,b,p] 
        wavsmax_fbp[f,b,p] = interp1d(Rowok_i, wavmap_fbpR[f,b,p,isgoodprof_fbpR[f,b,p]], kind='cubic')(Rsmax)
        dwavsmax_fbp[f,b,p] = interp1d(Rowok_i, dwavmap_fbpR[f,b,p,isgoodprof_fbpR[f,b,p]], kind='cubic')(Rsmax) 
    dYfref_fb =  (wavsmax_fbp[:,:,0] - wavsmax_fbp[:,:,1])/(dwavsmax_fbp[:,:,0] - dwavsmax_fbp[:,:,1])
    wavsmax_fb = wavsmax_fbp[:,:,0] - dYfref_fb*dwavsmax_fbp[:,:,0]
    dydY_bp = dydY_ptW[:,tbest_b,Wimref].T
    dYfref_b = dYfref_fb.mean(axis=0)

    if debug:
        np.savetxt("dYfref_fb.txt",np.vstack((np.indices((files,besttargs)).reshape((2,-1)), \
            wavsmax_fbp.reshape((-1,2)).T,dwavsmax_fbp.reshape((-1,2)).T,Rsmax_fbp.reshape((-1,2)).T,    \
            dYfref_fb.flatten())).T, fmt="%3i %3i "+6*"%8.2f "+"%8.5f ") 
    
    YXfref_db = YX_db +  np.array([dYfref_b,np.zeros(besttargs)])   # first put in dY_db     
    yxfref_dpb =   \
        RSScolpolcam(YXfref_db,imwav,coltem,camtem,yxOEoff_d=yxOEoff_d).reshape((2,2,besttargs))        
    yxfref_dpb[1] += rcbin_d[1]*pixmm*dC_fbp[fref].T                  # now put in dx_dpb
    dYXfref_d,drotfref,dyxOEofffref_d =  \
        impolguide(YX_db,yxfref_dpb,yxOEoff_d,imwav,coltem,camtem,debug=debug,name=name)[:3]    
    YXfref_db += dYXfref_d[:,None] + np.array([1.,-1.])[:,None]*np.radians(drotfref)*YXfref_db[::-1]
    yxOEofffref_d = yxOEoff_d + dyxOEofffref_d
    yxfref_dpb =   \
        RSScolpolcam(YXfref_db,imwav,coltem,camtem,yxOEoff_d=yxOEofffref_d).reshape((2,2,besttargs))
    dYXfref_d,drotfref,dyxOEofffref_d,dYXerr_df[:,fref],droterr_f[fref],dyxOEofferr_df[:,fref] =  \
        impolguide(YX_db,yxfref_dpb,yxOEofffref_d,imwav,coltem,camtem,debug=debug,name=name)  

  # add in reference guide error so dYX, drot, dyxOEoff is relative to cal filter 
    dYX_df += dYXfref_d[:,None]
    drot_f += drotfref
    dyxOEoff_df += dyxOEofffref_d[:,None]
    yxOEoff_df = yxOEoff_d[:,None] + dyxOEoff_df
    
  # recompute wavmap, cmap for each file and all targets with guider corrected YX, rot, OEflex
    wavmap_ftpR = np.zeros((files,ids,2,Rows),dtype='float32')   
    Cmap_ftpR = np.zeros_like(wavmap_ftpR)
    yxcat_dfpt = np.zeros((2,files,2,ids))
    for f in range(files):
        YXf_dt = YX_dt + dYX_df[:,f,None] + np.array([1.,-1.])[:,None]*np.radians(drot_f[f])*YX_dt[::-1]   
        YXf_ds = np.repeat(YXf_dt,Wavs,axis=1)
        wav_s = np.tile(wav_W,ids)             
        yxf_dptW =   \
            RSScolpolcam(YXf_ds,wav_s,coltem,camtem,yxOEoff_d=yxOEoff_df[:,f]).reshape((2,2,ids,Wavs))
        yxcat_dfpt[:,f] = yxf_dptW[:,:,:,Wimref]
        rcf_dptW = (yxf_dptW - yx0_d[:,None,None,None])    \
            /(rcbin_d[:,None,None,None]*pixmm) + rccenter_d[:,None,None,None]       
        rcpf_dptW = rcf_dptW + np.array([[-rshift,-rshift-prows],[0,0]])[:,:,None,None]
        for t,p in np.ndindex((ids,2)):                             
            RList = list(np.where((ri_tpR[t,p] >= rcpf_dptW[0,p,t].min()) &    \
                (ri_tpR[t,p] <= rcpf_dptW[0,p,t].max()))[0])                
            wavmap_ftpR[f,t,p,RList] = interp1d(rcpf_dptW[0,p,t],wav_W,kind='cubic')(ri_tpR[t,p,RList])
            Cmap_ftpR[f,t,p,RList] = (interp1d(rcpf_dptW[0,p,t],rcpf_dptW[1,p,t],   \
                kind='cubic')(ri_tpR[t,p,RList]) - ci_tpC[t,p,0]).clip(min=0.)
    yxcat_dfpb = yxcat_dfpt[:,:,:,isbest_t]

  #  recull goodprofs to be within new ddC outer quartile limits, evaluate X guiding and cmap
    dC_ftp = np.ma.median(np.ma.masked_array(dC_ftpR,mask=~isgoodprof3_ftpR),axis=3).data
    ddC_ftpR = dC_ftpR - dC_ftp[:,:,:,None]
    outerddC_tpq = np.zeros((ids,2,2))   
    ddChist_tph = np.zeros((ids,2,80)).astype(int)
    ddCbin_h = np.arange(-2.,2.05,0.05)
    ddCbin_h[[0,-1]] = -20.,20.   
    for t,p in np.ndindex((ids,2)):
        if ~(okfov_t & okovlap_t & okbkg_t)[t]: continue
        if ~(isgoodprof3_ftpR[:,t,p].any()): continue 
        q1,q3 = np.percentile(ddC_ftpR[:,t,p][isgoodprof3_ftpR[:,t,p]],(25,75))
        outerddC_tpq[t,p] = (q1 - 3.*(q3-q1)),(q3 + 3.*(q3-q1))
        ddChist_tph[t,p] = np.histogram(ddC_ftpR[:,t,p][isgoodprof3_ftpR[:,t,p]],bins=ddCbin_h)[0]
    isgoodprof4_ftpR = (isgoodprof2_ftpR & ((ddC_ftpR > outerddC_tpq[None,:,:,None,0]) &    \
                        (ddC_ftpR < outerddC_tpq[None,:,:,None,1])))

  # evaluate seeing sigma for all goodprof profiles   
    sigma_f = np.ma.median(np.ma.masked_array(sigma_ftpR,mask=~isgoodprof4_ftpR),axis=(1,2,3)).data
    
    if debug:
#        np.savetxt("yxfit_fpb.txt",np.vstack((np.indices((files,2,besttargs)).reshape((3,-1)),  \
#            yxcat_dfpb[0].flatten(),yx_dfpb[0].flatten(),yxcat_dfpb[1].flatten(),yx_dfpb[1].flatten())).T,  \
#            fmt=3*"%3i "+4*"%9.4f ")       
        np.savetxt("guiderr_f.txt",np.vstack((imgno_f,dYX_df,dYXerr_df,drot_f,  \
            droterr_f,dyxOEoff_df,dyxOEofferr_df,sigma_f)).T,fmt="%3i "+6*"%8.4f "+4*"%8.5f "+"%6.3f")             
        np.savetxt("ddC_tpRf.txt", np.vstack((np.indices((ids,2,Rows)).reshape((3,-1)),   \
            ddC_ftpR.transpose((1,2,3,0)).reshape((-1,files)).T)).T, fmt=3*"%3i "+files*"%8.3f ")
        np.savetxt("dC_tpf.txt",dC_ftp.reshape((files,-1)).T,fmt="%8.3f")
        np.savetxt("ddChist_tph.txt",ddChist_tph.reshape((2*ids,-1)),fmt="%4i") 
        np.savetxt("isgoodprof4_tpRf.txt", np.vstack((np.indices((ids,2,Rows)).reshape((3,-1)),  \
            isgoodprof4_ftpR.transpose((1,2,3,0)).reshape((-1,files)).astype(int).T)).T,fmt="%3i")
    exit()

  # compute moffat column profiles for all ids
    x_iC = (np.arange(Cols)-Cmap_ftpR[:,:,:,:,None]).reshape((-1,Cols))
    moffat_ftpRC = moffat(np.ones(files*ids*2*Rows),np.repeat(sigma_f,ids*2*Rows),  \
        x_iC,beta=2.5).reshape((files,ids,2,Rows,Cols))
    fluxtot_ftpR = moffat_ftpRC.sum(axis=4)

  # cull target rows with flux/tot < 85%  due to crowded target boxes  
    flux_ftpR = (ok1_ftpRC.astype(int)*moffat_ftpRC).sum(axis=4)
    flux_ftpR[(Cmap_ftpR == 0.)] = 0.              # no flux in rows with no Cmap
    fluxfrac_ftpR = flux_ftpR/fluxtot_ftpR
    iscullfrac_ftpR = (fluxfrac_ftpR < .85)
    
    ok2_ftpRC = (ok1_ftpRC & ~iscullfrac_ftpR[:,:,:,:,None])   # cull removing missing signal
        
  # find CR's, if there are enough images (at least 6) to make a median for each file/target/beam
  #   corrected for guiding and seeing for each file
    crsignal_ftpRC = np.zeros((files,ids,2,Rows,Cols))
    crvar_ftpRC = np.zeros_like(crsignal_ftpRC)
    okcr_ftpRC = np.zeros_like(crsignal_ftpRC).astype(bool)
    sigerr_ftpRC = np.zeros_like(crsignal_ftpRC)    
    moffatmean_ftpRC = moffat(np.ones(files*ids*2*Rows),sigma_f.mean()*np.ones(files*ids*2*Rows),  \
        x_iC,beta=2.5).reshape(files,ids,2,Rows,Cols)           # profile in mean seeeing
    dRdY_p = np.array([0.5214,0.5274])/(rcbin_d[0]*pixmm)       # mean over FOV, sufficient for this correction
    dCdX = 0.5235/(rcbin_d[1]*pixmm)

    for f,p in np.ndindex((files,2)):
        fmaxf_tR = np.tile(sigma_f.mean()/sigma_f[f],(ids,Rows))
        seecornp_tRC = (fmaxf_tR[:,:,None]*moffat_ftpRC[f,:,p]/moffatmean_ftpRC[f,:,p])
        dRp = dRdY_p[p] * dY_f[f]
        crsignal_ftpRC[f,:,p] = shift(signal_ftpRC[f,:,p]/seecornp_tRC, (0,-dRp,-dC_f[f]),order=1)
        crvar_ftpRC[f,:,p] = shift(var_ftpRC[f,:,p]/seecornp_tRC**2, (0,-dRp,-dC_f[f]),order=1)
        okcr_ftpRC[f,:,p] = (shift(ok2_ftpRC[f,:,p].astype(int),(0,-dRp,-dC_f[f]),order=1) == 1) 

    okcr_ftpRC[:,okcr_ftpRC.sum(axis=0)<6] = False
    medsignal_tpRC = np.ma.median(np.ma.masked_array(crsignal_ftpRC,mask=~okcr_ftpRC),axis=0).data
    medvar_tpRC = np.ma.median(np.ma.masked_array(crvar_ftpRC,mask=~okcr_ftpRC),axis=0).data 

    crsignal_ftp = (crsignal_ftpRC*okcr_ftpRC).sum(axis=(3,4))
    medsignal_ftp = (medsignal_tpRC[None,:,:,:,:]*okcr_ftpRC).sum(axis=(3,4))
    okcr_ftpRC[medsignal_ftp==0.,:,:] = False
    norm_ftpRC = np.zeros((files,ids,2,Rows,Cols))
    norm_ftpRC[medsignal_ftp>0.,:,:] = (crsignal_ftp[medsignal_ftp>0.] /    \
        medsignal_ftp[medsignal_ftp>0.])[:,None,None]
    medvar_ftpRC = np.repeat(medvar_tpRC[None,:,:,:,:],files,axis=0)          
        
    crsignal_ftpRC[okcr_ftpRC] = crsignal_ftpRC[okcr_ftpRC]/norm_ftpRC[okcr_ftpRC]
    sigerr_ftpRC[okcr_ftpRC] = (crsignal_ftpRC - medsignal_tpRC[None,:,:,:,:])[okcr_ftpRC] \
        / np.sqrt(medvar_ftpRC[okcr_ftpRC]) 
        
    if debug:
        np.savetxt("norm_fpt.txt",  \
            np.vstack((norm_ftpRC[:,:,:,0,0].transpose((0,2,1)).reshape((-1,ids)))).T, fmt="%8.4f")
        dR = 45
        np.savetxt("crsignal_f_"+str(dt)+"_"+str(dp)+"_"+str(dR)+"_C.txt",  \
            np.vstack((signal_ftpRC[:,dt,dp,dR,:], crsignal_ftpRC[:,dt,dp,dR,:],    \
            medsignal_tpRC[dt,dp,dR,:], moffat_ftpRC[:,dt,dp,dR,:],     \
            moffatmean_ftpRC[:,dt,dp,dR,:])).T,fmt=(2*files+1)*"%10.1f "+2*files*"%8.4f ")

    sigerr_s = sigerr_ftpRC[okcr_ftpRC]
    var_s = medvar_ftpRC[okcr_ftpRC] 
    varbin_S = np.logspace(np.log10(var_s.max())-2.,np.log10(var_s.max()),17,endpoint=True)
    varbin_S[0] = 0.
    Qvar_qS = np.zeros((2,16))
    varcnt_S =  np.zeros(16)  
    for S in range(16):
        invarbin_s = ((var_s >= varbin_S[S]) & (var_s < varbin_S[S+1]))
        varcnt_S[S] = invarbin_s.sum()
        if varcnt_S[S]: Qvar_qS[:,S] = np.percentile(sigerr_s[invarbin_s],(25.,75.))
    Qvar_qS[0] = np.minimum(Qvar_qS[0],-0.67448)        # do not let quartiles be inside photon sigma
    Qvar_qS[1] = np.maximum(Qvar_qS[1],0.67448)

  # identify cr's with sigerr > upperfence Qfence = Q3 + 5*(Q3-Q1)
    fencefac = 5.
    var_S = varbin_S[:-1]
    var_S[1:-1] = (varbin_S[1:-2] + varbin_S[2:-1])/2.
    var_S[-1] = varbin_S[-1] + 0.01                     # keep interp range big enough
    Qfenceinterp = interp1d(var_S,Qvar_qS[1]+ fencefac*(Qvar_qS[1]-Qvar_qS[0]), kind='linear')
    iscr_ftpRC = np.zeros_like(okcr_ftpRC)
    iscr_ftpRC[okcr_ftpRC] = (sigerr_ftpRC[okcr_ftpRC] > Qfenceinterp(medvar_ftpRC[okcr_ftpRC]))
    crs_f = iscr_ftpRC.sum(axis=(1,2,3,4))
    ok3_ftpRC= (ok2_ftpRC & ~iscr_ftpRC)
    ok3_t = (okfov_t & okovlap_t & okbkg_t)
    
    if debug:
        np.savetxt("Qvar_qS.txt",np.vstack((varcnt_S,var_S,Qvar_qS)).T, \
            fmt="%6i "+"%8.0f "+2*"%8.3f ") 
        np.savetxt("sigerr.txt",np.vstack((np.indices((ids,2,Rows,Cols))[:,okcr_ftpRC[0]],    \
            1./np.sqrt(medvar_tpRC[okcr_ftpRC[0]]),sigerr_ftpRC[:,okcr_ftpRC[0]])).T, \
            fmt=4*"%4i "+"%8.4f "+files*"%8.2f ") 
        np.savetxt("cr.txt",np.vstack((np.indices((files,ids,2,Rows,Cols))[:,iscr_ftpRC],    \
            sigerr_ftpRC[iscr_ftpRC])).T, fmt=5*"%4i "+"%8.2f ")  

  # do extraction in column direction using moffat fits as weights
    xtrwt_ftpRC = moffat_ftpRC
    xtrnorm_ftpRC = np.repeat(xtrwt_ftpRC.sum(axis=4)[:,:,:,:,None],Cols,axis=4)
    xtrwt_ftpRC[xtrnorm_ftpRC>0] /= xtrnorm_ftpRC[xtrnorm_ftpRC>0]

  # Get standard imsppol output wavelength grid 
  #   _W edges of unbinned spectrum pixels, pix 0 left edge at 3200
  #   _w output spectrum (columns) binned at rbin  
    wav_l = np.arange(3100.,10600.,100.)
    wbin = rcbin_d[0]
    YX_dl = np.array([[0.,],[0.,]])   
    dyx_dpl = RSScolpolcam(YX_dl,wav_l,7.5,7.5)
    dy_l = (dyx_dpl[0,0] - dyx_dpl[0,1])/2.       # mean E,O unbinned bin edges (mm)
    dy_l -= dy_l[1]                               # relative to 3200
    Wavs = int(np.floor(dy_l[-1]/pixmm))+1        # unbinned bins    
    wedge_W =  interp1d(dy_l/pixmm,wav_l,kind='cubic')(np.arange(Wavs))
    wedge_w = wedge_W[np.arange(0,Wavs,wbin)]
    wavs = wedge_w.shape[0]-1
    wav_w = (wedge_w[:-1] + wedge_w[1:])/2. 

    fopt_ftpw = np.zeros((files,ids,2,wavs))
    vopt_ftpw = np.zeros_like(fopt_ftpw) 
    covar_ftpw = np.zeros_like(fopt_ftpw) 
    ok_ftpw = np.zeros_like(fopt_ftpw,dtype=bool) 

    for f,file in enumerate(fileList):
        fstd_tpR = np.zeros((ids,2,Rows))
        vstd_tpR = np.zeros((ids,2,Rows))
        ok_tpR = (ok3_ftpRC[f].sum(axis=3) > 0) 
        fstd_tpR[ok_tpR] = signal_ftpRC[f].sum(axis=3)[ok_tpR]/xtrwt_ftpRC[f].sum(axis=3)[ok_tpR]
        vstd_tpR[ok_tpR] = var_ftpRC[f].sum(axis=3)[ok_tpR] / xtrwt_ftpRC[f].sum(axis=3)[ok_tpR]**2
        vopt_tpRC = fstd_tpR[:,:,:,None]*xtrwt_ftpRC[f] +     \
                    bkg_ftpRC[f,:,:,Rows/2,Cols/2][:,:,None,None]
        norm_tpRC = np.zeros((ids,2,Rows,Cols))
        norm_tpRC[ok3_ftpRC[f]] =     \
                    (xtrwt_ftpRC[f][ok3_ftpRC[f]]**2/vopt_tpRC[ok3_ftpRC[f]])
        norm_tpR = norm_tpRC.sum(axis=3) 
        fopt_tpRC = np.zeros((ids,2,Rows,Cols))
        fopt_tpRC[vopt_tpRC > 0.] =     \
            (xtrwt_ftpRC[f]*signal_ftpRC[f])[vopt_tpRC > 0.]/vopt_tpRC[vopt_tpRC > 0.]
        fopt_tpR = np.zeros((ids,2,Rows))
        vopt_tpR = np.zeros((ids,2,Rows))
        fopt_tpR[ok_tpR] = (ok3_ftpRC[f]*fopt_tpRC).sum(axis=3)[ok_tpR]/norm_tpR[ok_tpR]
        vopt_tpR[ok_tpR] = (ok3_ftpRC[f]*xtrwt_ftpRC[f]).sum(axis=3)[ok_tpR]/norm_tpR[ok_tpR]
        
        np.savetxt("fopt_bpR_"+str(f)+".txt",fopt_tpR[tbest_b].reshape((-1,Rows)),fmt="%12.2f")

    #   scrunch onto wavelength grid after correcting wmap for telescope Y guiding
        wavmapf_tpR = wavmap_tpR - dY_f[f]*dwavmap_tpR
        for t,p in np.ndindex((ids,2)):
            if ~ok3_t[t]: continue
            dirn = [1,-1][p]                      # scrunch needs increasing indices
            Rowuse = np.where(wavmapf_tpR[t,p] > 0)[0]
            wedgeuse = np.where((wedge_w >= wavmapf_tpR[t,p,Rowuse].min()) &    \
                              (wedge_w < wavmapf_tpR[t,p,Rowuse].max()))[0]
            wavout = wedgeuse[:-1]                # scrunch has input bins starting at bin, not bin-1/2: 
            Rbinedge_W = (interp1d(wavmapf_tpR[t,p,Rowuse][::dirn],(Rowuse[::dirn]+0.5))    \
               (wedge_w[wedgeuse]))[::dirn] 
            oktp_w = (scrunch1d(ok_tpR[t,p].astype(int),Rbinedge_W)   \
               == (Rbinedge_W[1:] - Rbinedge_W[:-1]))[::dirn] 
            ok_ftpw[f,t,p,wavout] = oktp_w
           
            fopt_ftpw[f,t,p,wavout] = scrunch1d(fopt_tpR[t,p],Rbinedge_W)[::dirn]*oktp_w 
            vopt_ftpw[f,t,p,wavout],covar_ftpw[f,t,p,wavout] =   \
               scrunchvar1d(vopt_tpR[t,p],Rbinedge_W)[::dirn]*oktp_w
            if debug:
                if ((f == df) & (t == dt)):      
                    np.savetxt("Rbinedge_W_"+str(f)+"_"+str(t)+"_"+str(p)+".txt",Rbinedge_W.T,fmt="%9.5f")

        if debug:
            if (f == df):
                np.savetxt("wedge_w.txt",wedge_w.T,fmt="%8.3f")
                np.savetxt("wavmapf_tpR_"+str(df)+'_'+str(dt)+".txt",wavmapf_tpR[dt].T,fmt="%8.2f")
                np.savetxt("fstdf_tpR_"+str(f)+"_"+str(dt)+".txt",fstd_tpR[dt].T,fmt="%10.2f")
                np.savetxt("normf_tpR_"+str(f)+"_"+str(dt)+".txt",norm_tpR[dt].T,fmt="%10.2e")
                np.savetxt("okf_tpR_"+str(f)+"_"+str(dt)+".txt",ok_tpR[dt].astype(int).T,fmt="%3i")
                np.savetxt("xtrwt_ftpRC_"+dlblt+".txt",xtrwt_ftpRC[f,dt,dp],fmt="%10.4f")
                np.savetxt("xtrnorm_ftpRC_"+dlblt+".txt",xtrnorm_ftpRC[f,dt,dp],fmt="%10.4f")
                np.savetxt("image_ftpRC_"+dlblt+".txt",image_ftpRC[f,dt,dp],fmt="%10.2f")
                np.savetxt("bkg_ftpRC_"+str(f)+"_"+str(dt)+"_"+str(dp)+".txt",bkg_ftpRC[f,dt,dp],fmt="%10.2f") 
                np.savetxt("normf_tpRC_"+str(f)+"_"+str(dt)+"_"+str(dp)+".txt",norm_tpRC[dt,dp],fmt="%10.2e")
                np.savetxt("foptf_tpRC_"+str(f)+"_"+str(dt)+"_"+str(dp)+".txt",fopt_tpRC[dt,dp],fmt="%10.4f")
                np.savetxt("voptf_tpRC_"+str(f)+"_"+str(dt)+"_"+str(dp)+".txt",vopt_tpRC[dt,dp],fmt="%10.2f")
                np.savetxt("foptf_tpR_"+str(f)+"_"+str(dt)+".txt",fopt_tpR[dt].T,fmt="%10.2f")
                np.savetxt("fopt_ftpw_"+str(f)+"_"+str(dt)+".txt",fopt_ftpw[f,dt].T,fmt="%10.2f")
                np.savetxt("voptf_tpR_"+str(f)+"_"+str(dt)+".txt",vopt_tpR[dt].T,fmt="%10.2f")
                np.savetxt("ok_ftpw_"+str(f)+"_"+str(dt)+".txt",ok_ftpw[f,dt].astype(int).T,fmt="%3i")
                
  # cull targets based on adequate O/E wavelength overlap match
    oematchwavs_t = ok_ftpw.all(axis=(0,2)).sum(axis=1)
    okoematch_t = (oematchwavs_t > 3)
    
    oecullList = list(np.where(~okoematch_t & okfov_t & okovlap_t & okbkg_t)[0])
    if len(oecullList): printstdlog((("%3i ids match culled: "+len(oecullList)*"%2i ") %     \
                tuple([len(oecullList),]+oecullList)),logfile)       

    ok4_t = (okfov_t & okovlap_t & okbkg_t & okoematch_t)
        
  # cull results based on O/E ratio, 0.7 - 1.3
    oeratmin = 0.7
    oeratmax = 1.3
    okrat_ftw = ok_ftpw.all(axis=2)
    okrat_ftpw = okrat_ftw[:,:,None,:].repeat(2,axis=2)
    foptsum_pw = np.ma.sum(np.ma.masked_array(fopt_ftpw,mask=~okrat_ftpw),axis=(0,1)).data
    foptsum_w = foptsum_pw.mean(axis=0)
    OErat_w = np.zeros(wavs)
    OErat_w[foptsum_w>0.] = foptsum_pw[0,foptsum_w>0.]/foptsum_pw[1,foptsum_w>0.]
    OErat_tw = np.zeros((ids,wavs))
    foptmean_tpw = np.ma.mean(np.ma.masked_array(fopt_ftpw,mask=~okrat_ftpw),axis=0).data
    okrat_tw = (foptmean_tpw > 0.).all(axis=1)
    OErat_tw[okrat_tw] = foptmean_tpw[:,0][okrat_tw]/foptmean_tpw[:,1][okrat_tw]
    okrat_tw = (OErat_tw > oeratmin) & (OErat_tw < oeratmax)
    ok_ftpw &= okrat_tw[None,:,:].repeat(files,axis=0)[:,:,None,:].repeat(2,axis=2)
    fopt_ftpw *= ok_ftpw 
    vopt_ftpw *= ok_ftpw 
    covar_ftpw *= ok_ftpw 
                
    oeratwavs_t = ok_ftpw.all(axis=(0,2)).sum(axis=1)
    okoerat_t = (oeratwavs_t > 3)    
    ratcullList = list(np.where(~okoerat_t & okoematch_t & okfov_t & okovlap_t & okbkg_t)[0])
    if len(ratcullList): printstdlog((("%3i ids ratio culled: "+len(ratcullList)*"%2i ") %     \
                tuple([len(ratcullList),]+ratcullList)),logfile)     
    ok5_t = (okoerat_t & ok4_t)
      
    id_T = np.where(ok5_t)[0]
    Targets = id_T.shape[0]
    TargetmapTab = mapTab[id_T]
    if debug:
        np.savetxt("id_T.txt",np.vstack((np.arange(Targets),id_T)).T,fmt="%3i")    
        np.savetxt("oewavs_t.txt",np.vstack((np.arange(ids),oematchwavs_t,oeratwavs_t)).T,fmt="%3i") 
        np.savetxt("foptsum_pw.txt",np.vstack((wav_w,foptsum_pw)).T,  \
            fmt="%8.2f "+2*"%12.2f ")
        np.savetxt("OErat_w.txt", np.vstack((wav_w,OErat_w,OErat_tw)).T,fmt="%8.2f "+(ids+1)*"%8.4f ")
        TargetmapTab.write(name+"_TargetmapTab.txt",format='ascii.fixed_width',   \
                    bookend=False, delimiter=None, overwrite=True)
            
    printstdlog(("\nTargets:  "+Targets*"%2i ") % tuple(np.arange(Targets)),logfile)
    printstdlog(("from ids: "+Targets*"%2i ") % tuple(id_T),logfile) 
        
    printstdlog ("\n        Output file                dY (asec) dX     sigma   crs", logfile)
            
  # save the result
    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)
        outfile = 'e'+file
        del(hdul['CMAP'])
        del(hdul['WMAP'])
        del(hdul['TMAP'])
        hdul['SCI'].data = fopt_ftpw[f,id_T].astype('float32').transpose((1,0,2))
        for ext in [0,'SCI']:                       # specpolrawstokes wants them in both ext's
            hdul[ext].header['CRVAL1'] = 0
            hdul[ext].header['CDELT1'] = wbin 
            hdul[ext].header['CTYPE1'] = 'Angstroms'
            hdul[ext].header['CRVAL2'] = 0
            hdul[ext].header['CDELT2'] = 1 
            hdul[ext].header['CTYPE2'] = 'Target'
        hdr1 = hdul['SCI'].header
        hdul['VAR'].data = vopt_ftpw[f,id_T].astype('float32').transpose((1,0,2))
        hdul.append(pyfits.ImageHDU(data=covar_ftpw[f,id_T].astype('float32').transpose((1,0,2)), \
          header=hdr1, name='COV'))
        hdul['BPM'].data = (~ok_ftpw[f,id_T]).astype('uint8').transpose((1,0,2)) 
        hdul['TGT'] = pyfits.table_to_hdu(TargetmapTab)
        hdul.append(pyfits.ImageHDU(data=wedge_W.astype('float32'), header=hdr1, name='WCSDVARR'))
        hdul['WCSDVARR'].header['CDELT1'] = 1        
        hdul.writeto(outfile,overwrite=True)

        outfile = 'e'+file
        dvert,dhor = dY_f[f]*1000/saltfps , (dC_f[f]/dCdX)*1000/saltfps
        printstdlog ("%30s %8.3f %8.3f %8.3f  %4i" % (outfile,dvert,dhor,sigma_f[f],crs_f[f]), logfile)

    return

# ----------------------------------------------------------
def immospolextract(fileList,name,logfile='salt.log',debug=False):
    """derive extracted target data vs target and wavelength for MOS imaging spectropolarimetry
    """
    return

# ----------------------------------------------------------
def moffat(fmax_i,sigma_i,x_ib,beta=3.):
    f_ib = fmax_i[:,None]*(1. + (x_ib/sigma_i[:,None])**2)**(-beta)
    return f_ib

# ----------------------------------------------------------
def moffat2dfit(image_irc,rad_irc,ok_irc,beta=3.,debug=False):
    """fit moffat 2d profile to stack of images
    Parameters 
    ----------
    image_irc: 2d float image, higher dimensions flattened into i
    rad_irc: 2d float radial offset in bins from center of profile
    ok_irc: boolean array mask

    Returns: Moffat sigma, fmax for each image (flattened into i)

    """
    rows = image_irc.shape[-2]
    cols = image_irc.shape[-1]
    bins = rows*cols
    images = image_irc.size/bins
    image_ib = image_irc.reshape((images,bins))
    rad_ib = rad_irc.reshape((images,bins))
    ok_ib = ok_irc.reshape((images,bins))
    bcenter_i = np.argmin(rad_ib,axis=1)
  # for fmax, use 3x3 bins nearest the center, least-square fit quadratic surface
    rcenter_i, ccenter_i = np.unravel_index(bcenter_i,(rows,cols))
    iscenter_irc = ((np.abs(np.arange(rows)[None,:,None] - rcenter_i[:,None,None]) <=1) & \
                    (np.abs(np.arange(cols)[None,None,:] - ccenter_i[:,None,None]) <=1))
    iscenter_ib = (iscenter_irc.reshape((-1,rows*cols)) & ok_ib)
    xmean_i = (rad_ib**2*iscenter_ib).sum(axis=1)/iscenter_ib.sum(axis=1)
    x2mean_i = (rad_ib**4*iscenter_ib).sum(axis=1)/iscenter_ib.sum(axis=1)
    fmean_i = (image_ib*iscenter_ib).sum(axis=1)/iscenter_ib.sum(axis=1)
    fxmean_i = (image_ib*rad_ib**2*iscenter_ib).sum(axis=1)/iscenter_ib.sum(axis=1)
    fmax_i = (x2mean_i*fmean_i - xmean_i*fxmean_i)/(x2mean_i - xmean_i**2)

  # for sigma, use 17 bins closest to half-max
    sigma_ib = np.zeros((images,bins))
    oksig_ib = (ok_ib & (image_ib < fmax_i[:,None]) & (image_ib > 0.))
    sigma_ib[oksig_ib] = rad_ib[oksig_ib]/      \
            np.sqrt(np.power((image_ib/fmax_i[:,None])[oksig_ib],(-1./beta)) - 1)
    bsort_iB = np.argsort(image_ib/fmax_i[:,None], axis=1)
    idxbsort_iB = np.ogrid[:images,:bins]
    idxbsort_iB[1] = np.copy(bsort_iB)
    Bhmax_i = np.argmin(np.abs(image_ib[idxbsort_iB]/fmax_i[:,None] - 0.5),axis=1)

    idxbsort_iB = np.ogrid[:images,:9]
    idxbsort_iB[1] = bsort_iB[np.arange(images)[:,None],    \
            np.clip(np.arange(-4,5)[None,:]+Bhmax_i[:,None],0,bins-1)]

    sigma_i = np.median(sigma_ib[idxbsort_iB].reshape((images,9)),axis=1)
    fiterr_ib = image_ib - moffat(fmax_i,sigma_i,rad_ib,beta=beta)
    
    return sigma_i, fmax_i, fiterr_ib

# ----------------------------------------------------------
def moffat1dfit(prof_ix,ok_ix,beta=2.5,idebug=None):
    """fit moffat 1d profile to stack of 1D profiles
    Parameters 
    ----------
    prof_ix: 1d float profile, higher dimensions flattened into i
    ok_ix: boolean array mask

    Returns: Moffat sigma, fmax, x0 for each image (flattened into i)

    """
    bins = prof_ix.shape[-1]
    profs = prof_ix.size/bins
    prof_ib = prof_ix.reshape((profs,bins))
    ok_ib = ok_ix.reshape((profs,bins))
    x_b = np.arange(bins)
    if idebug is not None:
        print "\nprofile fit idebug = ", idebug
        print ("prof_x "+bins*"%6.3f " % tuple(prof_ix[idebug]))
        print ("ok_x   "+bins*"%6i " % tuple(ok_ix[idebug].astype(int)))         

  # for fmax, x0 first guess, use 3 highest ok neighbor bins, fit quadratic                   
    fmax_i = np.ma.max(np.ma.masked_array(prof_ib,mask=~ok_ib),axis=1).data         
    idx0_i = np.ma.argmax(np.ma.masked_array(prof_ib,mask=~ok_ib),axis=1).clip(1,bins-2)
    sigma_i = bins*np.ones(profs)                   # default 
    i_i = range(profs)             
    okmax_i = (ok_ib[i_i,idx0_i-1] & ok_ib[i_i,idx0_i+1])
    fiterr_ib = np.zeros((profs,bins))
         
    pcof0_i = (prof_ib[i_i,idx0_i-1] -2.*prof_ib[i_i,idx0_i] + prof_ib[i_i,idx0_i+1])/2.
    pcof1_i = (prof_ib[i_i,idx0_i] - prof_ib[i_i,idx0_i-1]) - pcof0_i*(2.*idx0_i - 1.)
    pcof2_i = prof_ib[i_i,idx0_i] - pcof0_i*idx0_i**2 - pcof1_i*idx0_i      
    fmax_i[okmax_i] = pcof2_i[okmax_i] - pcof1_i[okmax_i]**2/(4.*pcof0_i[okmax_i])
    okprof_i = (okmax_i & (fmax_i > 0.))
    x0_i = idx0_i.astype(float)                     # default
    x0_i[okprof_i] = -pcof1_i[okprof_i]/(2.*pcof0_i[okprof_i])
    if idebug is not None:
        print (("okprof, pcof0, pcof1, pcof2 %2i "+3*"%8.3f ") %   \
            (okprof_i[idebug].astype(int), pcof0_i[idebug], pcof1_i[idebug], pcof2_i[idebug]))
        print ("fmax, x0 %8.3f %8.3f" % (fmax_i[idebug], x0_i[idebug]))

  # for sigma, use 6 side bins around 0.75-max. 
  # require >0, at least 2 on a side.  Default sigma = bins
    Bins = 6
    okside_ib = (okprof_i[:,None] & ok_ib & (prof_ib < 0.75*fmax_i[:,None]) & (prof_ib > 0.))
    sidebinsleft_i = (okside_ib & (np.arange(bins)[None,:] < x0_i[:,None])).sum(axis=1)
    sidebinsright_i = (okside_ib & (np.arange(bins)[None,:] > x0_i[:,None])).sum(axis=1)    
    okprof_i = ((np.minimum(sidebinsleft_i,sidebinsright_i) >= 2) &   \
                ((sidebinsleft_i + sidebinsright_i) >= Bins)) 
    i_I = np.where(okprof_i)[0]
    okprofs = len(i_I)
    whereIleft = np.where((okside_ib & (np.arange(bins)[None,:] < x0_i[:,None]))[i_I])
    whereIright = np.where((okside_ib & (np.arange(bins)[None,:] > x0_i[:,None]))[i_I])
    dum,wleft_I = np.unique(whereIleft[0],return_index=True)
    dum,wright_I = np.unique(whereIright[0],return_index=True)                      
    binsleft_I = 3*np.ones(okprofs).astype(int)
    binsleft_I[sidebinsleft_i[i_I] < 3] = 2
    binsleft_I[sidebinsright_i[i_I] < 3] = 4
    binsright_I = 3 - (binsleft_I - 3)    
    wleft_I = wleft_I + sidebinsleft_i[i_I] - binsleft_I

    xside_IB = np.zeros((okprofs,Bins)).astype(int)
    fside_IB = np.zeros((okprofs,Bins))
    prof_Ib = prof_ib[okprof_i].reshape((okprofs,bins))
    fmax_I = fmax_i[okprof_i]          
    for I in range(okprofs):       
        xside_IB[I] = np.hstack((whereIleft[1][wleft_I[I]:(wleft_I[I]+binsleft_I[I])], \
                            whereIright[1][wright_I[I]:(wright_I[I]+binsright_I[I])]))                           
        fside_IB[I] = prof_Ib[I][xside_IB[I]]  
      
    q_IB = np.sign(xside_IB - x0_i[i_I][:,None])*    \
        np.sqrt(np.power((fside_IB/fmax_I[:,None]),(-1./beta)) - 1.)  

    x_I = xside_IB.mean(axis=1)                         # vectorize 1D polyfit
    x2_I = (xside_IB**2).mean(axis=1)
    y_I = q_IB.mean(axis=1)
    xy_I = (xside_IB*q_IB).mean(axis=1)    
    pcof1_I = (x2_I*y_I -  x_I*xy_I)/(x2_I - x_I**2)    # intercept
    pcof0_I = (xy_I - x_I*y_I)/(x2_I - x_I**2)          # slope
    x0_i[i_I] = -pcof1_I/pcof0_I
    sigma_i[i_I] = 1./pcof0_I 

    fiterr_ib[i_I] = prof_ib[i_I] -     \
        moffat(fmax_i[i_I],sigma_i[i_I],(x0_i[i_I,None]-x_b[None,:]),beta=beta)
    if idebug is not None:
        Idebug = np.where(i_I == idebug)[0][0]
        print (("fside_B "+Bins*"%6.3f ") % tuple(fside_IB[Idebug]))
        print (("xside_B "+Bins*"%6.3f ") % tuple(xside_IB[Idebug]))        
        print (("q_B     "+Bins*"%6.3f ") % tuple(q_IB[Idebug]))
        print ("pcof %8.3f %8.3f" % (pcof0_I[Idebug], pcof1_I[Idebug]))
        print (("fiterr "+bins*"%6.3f ") % tuple(fiterr_ib[idebug]))
        print (("sigma, fmax, x0 "+3*"%6.3f "+"\n") %  \
            (sigma_i[idebug], fmax_i[idebug], x0_i[idebug]))
        
    return sigma_i, fmax_i, x0_i, fiterr_ib, okprof_i

# ----------------------------------------------------------
def findpair(fileList,logfile='salt.log'):
  # h = (0,1..) filter-pair counter, i = (0,1) index within pair 
    patternfile = open(datadir + 'wppaterns.txt', 'r')
    patternlist = patternfile.readlines()

    obsDict = obslog(fileList)
    files= len(fileList)
    hsta_f = np.array([int(round(s/11.25)) for s in np.array(obsDict['HWP-ANG']).astype(float)])
    qsta_f = np.array([int(round(s)) for s in np.array(obsDict['QWP-STA'])])
    img_f = np.array([int(os.path.basename(s).split('.')[0][-4:]) for s in fileList])
    wpstate_f = [['unknown', 'out', 'qbl', 'hw', 'hqw']
                     [int(s[1])] for s in np.array(obsDict['WP-STATE'])]
    wppat_f = np.array(obsDict['WPPATERN'])
    hi_df = -np.ones((2,files),dtype=int)

    if wppat_f[0].count('NONE'):
        printstdlog((('\n     %s  WP Pattern: NONE. Calibration data, skipped') % fileList[0]),  \
            logfile)
        return hi_df
    elif wppat_f[0].count('UNKNOWN'):
        if (hsta_f % 2).max() == 0:
            wppat = "LINEAR"
        else:
            wppat = "LINEAR-HI"
        wppat_f = wppat

    if not(((wpstate_f[0] == 'hw') & (wppat_f[0] in ('LINEAR', 'LINEAR-HI'))
            | (wpstate_f[0] == 'hqw') & (wppat_f[0] in ('CIRCULAR', 'CIRCULAR-HI', 'ALL-STOKES')))):
        print "Image", img_f[0], ": wpstate ", wpstate_f[0], \
            " and wppattern ", wppat_f[0], "not consistent"
        return hi_df
    for p in patternlist:
        if (p.split()[0] == wppat_f[0]) & (p.split()[2] == 'hwp'):
            wpat_p = np.array(p.split()[3:]).astype(int)
        if (p.split()[0] == wppat_f[0]) & (p.split()[2] == 'qwp'):
            wpat_dp = np.vstack(wpat_p, np.array(p.split()[3:]))

    h = 0  
    for f in range(files):
        if (wppat_f[f] != wppat_f[0]): continue
        hList = (np.where(hi_df[0]==h)[0]).tolist()
        
        if (len(hList) == 0):
            if (np.where(wpat_p[0::2] == hsta_f[f])[0].size > 0):
                idxp = np.where(wpat_p == hsta_f[f])[0][0]
                hi_df[:,f] = np.array([h,0])
            continue

        if (len(np.where(hi_df[1,hList] == 0)[0]) > 0):
            if (hsta_f[f] == wpat_p[idxp]):
                hi_df[:,f] = np.array([h,0])
                continue
            elif (hsta_f[f] == wpat_p[idxp+1]):
                hi_df[:,f] = np.array([h,1])
        elif (hsta_f[f] == wpat_p[idxp+1]):
            hi_df[:,f] = np.array([h,1])
        hList = (np.where(hi_df[0]==h)[0]).tolist()
        
        if (len(np.where(hi_df[1,hList] == 1)[0]) > 0):
            if (f < (files-1)):
                if (hsta_f[f+1] == wpat_p[idxp+1]):
                    continue
                else:
                    h += 1

    return hi_df

# ----------------------------------------------------------
def printstdlog(string,logfile):
    print string
    print >>open(logfile,'a'), string
    return 

# ------------------------------------

if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)   
    polextract(infileList,**kwargs)

# debug:
# M30
# cd /d/pfis/khn/20161023/sci
# python polsalt.py polextract.py t*089.fits t*09?.fits t*010[0-4].fits debug=True  (4820 filter)
# python polsalt.py polextract.py t*010[5-9].fits t*011[0-2].fits debug=True  (7005 filter)
# python polsalt.py polextract.py t*006[5-9].fits t*008[0-8].fits debug=True  (1st imsp)
# python polsalt.py polextract.py t*011[3-9].fits t*0120.fits debug=True      (2nd imsp)
