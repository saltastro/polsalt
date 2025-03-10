
"""
immospolextract

Optimal extraction for MOS imaging polarimetric data
Write out extracted data fits (etm*) dimensions wavelength,target #

"""

import os, sys, glob, shutil, inspect, pprint

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
from specpolutils import datedline, rssdtralign
from rssmaptools import ccdcenter,boxsmooth1d,impolguide,rotate2d,Tableinterp,fence,fracmax
from rssoptics import RSScolpolcam,RSSpolgeom

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
    dfile,dtarget,dbeam = 0,59,0
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
def immospolextract(fileList,name,logfile='salt.log',debug=False):
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
    _d dimension index r,c = 0,1
    _f file index
    _i mask slits (entries in xml and tgtTab)    
    _t target for extraction
    _p pol beam = 0,1 for O,E
    _R, _C bin coordinate within target box
    _a amplifier = 0,1,2,3,4,5

    """

    if (debug):
        df, dt, dp, dR, dC = 5,0,0,0,0
        dlblftp = "_"+str(df)+"_"+str(dt)+"_"+str(dp)+".txt"

  # data in common, from first file
    pixmm = 0.015
    files = len(fileList) 
    imgno_f = np.array([f.split('.')[0][-4:] for f in fileList]).astype(int)         
    hdul0 = pyfits.open(fileList[0])
    hdr0 = hdul0[0].header
    exptime = hdr0['EXPTIME']
    lampid = hdr0['LAMPID'].strip().upper()
    rcbin_d = np.array(hdr0['CCDSUM'].split(' ')).astype(int)[::-1]
    prows, cols = hdul0[1].data.shape[1:]    
    camtem = hdr0['CAMTEM']
    coltem = hdr0['COLTEM']
    calimg = hdr0['CALIMG'].strip()
    refwav = float(hdr0['REFWAV'])
    filter = hdr0['FILTER']
    dateobs =  hdr0['DATE-OBS'].replace('-','')
    RAd0 = Longitude(hdr0['RA0']+' hours').degree
    DECd0 = Latitude(hdr0['DEC0']+' degrees').degree
    PAd0 = hdr0['PA0']                                   

    tgt_prc = hdul0['TMAP'].data
    tgtTab = Table.read(hdul0['TGT'])    
    entries = len(tgtTab['CATID'])
    oktgt_i = (tgtTab['CULL'] == '')
    i_t = np.where(oktgt_i)[0]   
    targets = oktgt_i.sum()
    istarget_ptrc = np.zeros((2,targets,prows,cols),dtype=bool)    
    for p,t in np.ndindex(2,targets):      
        istarget_ptrc[p,t] = (tgt_prc[p] == i_t[t]+1)        
    tgtname_t = np.array(tgtTab['CATID'])[oktgt_i]    
    YX_dt = np.array([tgtTab['YCE'],tgtTab['XCE']])[:,oktgt_i] 

  # data from cal file      
    hdulcal = pyfits.open(glob.glob('tm*'+calimg+".fits")[0])
    calimgno = int(calimg)
    yx0nom_dp,rshift,yxp0nom_dp,dum = RSSpolgeom(hdulcal,refwav)  # nominal dtr geom for arc image     
    yxp0ref_dp = np.array([map(float,hdr0['YXAXISO'].split()),map(float,hdr0['YXAXISE'].split())]).T       
    dyxp_dp = yxp0ref_dp - yxp0nom_dp                             # arc alignment offset
    trkrhocal = hdulcal[0].header['TRKRHO']
    ur0cal,uc0cal,saltfps = rssdtralign(dateobs,trkrhocal)
    yx0cal_d = -0.015*np.array([ur0cal,uc0cal])
    dtrfps = saltfps*np.diff(RSScolpolcam(np.array([[0.,0.],[0.,1.]]),refwav,coltem,camtem)[1,0])[0]           

  # fixed geometry info.   
    rccenter_d, cgap_c = ccdcenter(hdul0[1].data[0])        
    rccenter_d[0] = prows                                       # gets center of unsplit data            
    c0_a = (cgap_c[0]-2048/rcbin_d[1])*np.ones(6,dtype=int)    
    c0_a[[2,4]] = cgap_c[[1,3]]
    c0_a[[1,3,5]] = c0_a[[0,2,4]]+1024/rcbin_d[1]               # gets first column in each amplifier

  # target box info
    rckey_pd = np.array([['R0O','C0O'],['R0E','C0E']])
    Rows,Cols = np.array(hdul0[0].header['BOXRC'].split()).astype(int)
    ri_tpR = np.zeros((targets,2,Rows),dtype=int)
    ci_tpC = np.zeros((targets,2,Cols),dtype=int) 
    amp_Atp = np.zeros((2,targets,2),dtype=int)              # amplifier on A=left, right side of target box  
    for p in (0,1):
        ri_tpR[:,p] = np.clip(tgtTab[rckey_pd[p,0]][oktgt_i,None] + np.arange(Rows)[None,:], 0,prows-1)
        ci_tpC[:,p] = np.clip(tgtTab[rckey_pd[p,1]][oktgt_i,None] + np.arange(Cols)[None,:], 0,cols-1)
        amp_Atp[0,:,p] = np.argmax((c0_a[:,None] >= ci_tpC[None,:,p,0]),axis=0)-1
        amp_Atp[1,:,p] = np.argmax((c0_a[:,None] >= ci_tpC[None,:,p,-1]),axis=0)-1

  # target position prediction from cal image
    wav_W = np.arange(3200.,10600.,100.)
    Wavs = wav_W.shape[0]
    wav_s = np.tile(wav_W,targets)
    YX_ds = np.repeat(YX_dt,Wavs,axis=1)
    YX1_ds = np.copy(YX_ds)
    dY = 0.05      
    YX1_ds[0] += dY*np.ones(targets*Wavs)      
    yx_dptW = RSScolpolcam(YX_ds,wav_s,coltem,camtem).reshape((2,2,targets,Wavs)) + dyxp_dp[:,:,None,None]    
    yx1_dptW = RSScolpolcam(YX1_ds,wav_s,coltem,camtem).reshape((2,2,targets,Wavs)) + dyxp_dp[:,:,None,None]  
    dydY_ptW = (yx1_dptW[0] - yx_dptW[0])/dY
    dydW_ptW = np.zeros((2,targets,Wavs)) 
    dWdY_ptW = np.zeros_like(dydW_ptW)        
    for p,t in np.ndindex(2,targets):   
        dydW_ptW[p,t] = UnivariateSpline(wav_W,yx_dptW[0,p,t],s=0).derivative()(wav_W)
    dWdY_ptW = dydY_ptW/dydW_ptW                                                
    wavmap_tpR,dwavmap_tpR,Cmap_tpR = specmap(yx_dptW,wav_W,ri_tpR,ci_tpC,dWdY_ptW,hdulcal)            
    wavNIR = 9000.                                        # for debug evaluation of wavmap
    WNIR = np.where(wav_W==wavNIR)[0][0]
    RNIRmap_tp = np.zeros((targets,2))
    for t,p in np.ndindex(targets,2):
        okmap_R = (wavmap_tpR[t,p] > 0.)
        rootList = UnivariateSpline(np.arange(Rows)[okmap_R],wavmap_tpR[t,p,okmap_R]-wavNIR,s=0).roots()
        if len(rootList): RNIRmap_tp[t,p] = rootList[0]

  # assemble target data from all files
    print "Assembling target data"
    fref = np.argmin(np.abs(imgno_f - calimgno))        # fref, closest to wavcal, is reference file
    image_fprc = np.zeros((files,2,prows,cols))
    var_fprc = np.zeros_like(image_fprc) 
    okbin_fprc =  np.zeros_like(image_fprc).astype(bool)
    trkrho_f = np.zeros(files)         
    hsta_f = np.zeros(files,dtype=int) 
    qsta_f = np.zeros(files,dtype=int) 
    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)
        image_fprc[f] = hdul['SCI'].data
        var_fprc[f] = hdul['VAR'].data
        okbin_fprc[f] = (hdul['BPM'].data==0) 
        trkrho_f[f] = hdul[0].header['TRKRHO']
        hsta_f[f] = int(round(float(hdul[0].header['HWP-ANG'])/11.25))
        qsta_f[f] = int(round(float(hdul[0].header['QWP-ANG'])/45.))         
        
        if debug:
            if (f==df):
                rc_dptW = (yx_dptW - yx0cal_d[:,None,None,None])/(pixmm*rcbin_d[:,None,None,None]) +    \
                    rccenter_d[:,None,None,None]
                rcp_dptW = np.zeros((2,2,targets,Wavs))                
                for p,t in np.ndindex(2,targets):
                    rcp_dptW[0,p,t] = np.round(rc_dptW[0,p,t] - rshift-p*prows).astype(int)  
               
                np.savetxt("rtgt_tpW_"+str(f)+".txt",np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)),   \
                    rcp_dptW[0].transpose((2,1,0)).reshape((-1,2*targets)))).T,fmt=3*" %2i"+Wavs*" %4i ")      
       
    okbin_prc = okbin_fprc.all(axis=0)
    image_prc = image_fprc.mean(axis=0)
    var_prc = var_fprc.mean(axis=0)/np.sqrt(files)
                    
  # find saturated pixels in target boxes
  # cull targets with saturated bins in most files in either beam
    issat_fprc = np.zeros_like(okbin_fprc)
    for a in range(6):          # sat level may depend on amp due to different gains
        image_fpra = image_fprc[:,:,:,c0_a[a]:c0_a[a]+1024/rcbin_d[1]]
        satlevel = image_fpra.max()
        if ((image_fpra > 0.98*satlevel).sum() < 3): satlevel = 1.e9
        issat_fprc[:,:,:,c0_a[a]:c0_a[a]+1024/rcbin_d[1]] = (image_fpra > 0.98*satlevel)
    satbins_ftp = np.zeros((files,targets,2))
    for f,t,p in np.ndindex(files,targets,2):
        satbins_ftp[f,t,p] = (issat_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]).sum()
    oksat_t = (2*(satbins_ftp>0).sum(axis=0)/files < 1).all(axis=1)
    satcullList = list(np.where(~oksat_t)[0])

    if len(satcullList): printstdlog((("%3i targets saturation culled: "+len(satcullList)*"%2i ") %     \
                tuple([len(satcullList),]+satcullList)),logfile)
    oktarget_t = oksat_t
    if debug:
        np.savetxt("satbins_ftp.txt",np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)),    \
            satbins_ftp.reshape((files,-1)))).T,fmt=" %3i")
        np.savetxt("amp_Atp.txt",np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)),    \
            amp_Atp.reshape((2,-1)))).T,fmt=" %4i")

  # put targets into boxes
    image_ftpRC = np.zeros((files,targets,2,Rows,Cols))
    var_ftpRC = np.zeros_like(image_ftpRC)
    bkg_ftpRC = np.zeros_like(image_ftpRC)    
    okbin_ftpRC = np.zeros((files,targets,2,Rows,Cols),dtype=bool)
    oktgt_ftpRC = np.zeros_like(okbin_ftpRC)
    isbkg_ftpRC = np.zeros_like(okbin_ftpRC)      
    for f,t,p in np.ndindex(files,targets,2): 
        if ~(oktarget_t)[t]: continue    
        image_ftpRC[f,t,p] = image_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]        
        var_ftpRC[f,t,p] = var_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]] 
        okbin_ftpRC[f,t,p] = okbin_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
        oktgt_ftpRC[f,t,p] = (tgt_prc==i_t[t]+1)[p][ri_tpR[t,p],:][:,ci_tpC[t,p]] 
        isbkg_ftpRC[f,t,p] = (tgt_prc==255)[p][ri_tpR[t,p],:][:,ci_tpC[t,p]]         

  # process background        
    if (lampid != 'NONE'):
      # process background for lamp data: TO BE UPDATED USING targetmap  
      # use non-slit for lamp data
      # first find locations near each target to use for background, using mean over all images
      # process by amplifier     
        print "Processing lamp background"        
        isbkgarea_prc = (okbin_prc & (tgt_prc == 255))
        bkgimage_prc = image_prc*(isbkgarea_prc.astype(float))
        bkgvar_prc = var_prc*(isbkgarea_prc.astype(float))        
        isbkg_prc = np.zeros((2,prows,cols),dtype=bool)                          
        for a,p in np.ndindex(6,2):
            if (a not in amp_Atp[:,:,p]): continue
            t_i = np.where((amp_Atp[:,:,p] == a).any(axis=0))[0]
            for t in t_i:                                
                c0 =  (ci_tpC[t,p,0]-2*Cols).clip(c0_a[a],c0_a[a]+1024/rcbin_d[1])
                c1 =  (ci_tpC[t,p,-1]+2*Cols).clip(c0_a[a],c0_a[a]+1024/rcbin_d[1])
                istgtarea_rc = np.zeros((prows,cols)).astype(bool)                
                istgtarea_rc[ri_tpR[t,p,0]:ri_tpR[t,p,0]+Rows,c0:c1] = True
                isbkgarea_rc = (istgtarea_rc & isbkgarea_prc[p])            
                bkgimage_s = bkgimage_prc[p,isbkgarea_rc]
                bkgvar_s = bkgvar_prc[p,isbkgarea_rc] 
                r_s,c_s = np.where(isbkgarea_rc) 
                binwidth = np.sqrt(np.median(bkgvar_s))
                bkghist_b,binedge_B = np.histogram(bkgimage_s, bins=int(bkgimage_s.ptp()/binwidth))[:2]            
                bmin = np.argmax(bkghist_b)
                isbkg_s = ((bkgimage_s > binedge_B[bmin-1]) & (bkgimage_s < binedge_B[bmin+2]))
                isbkg_prc[p][r_s[isbkg_s],c_s[isbkg_s]] = True          
        if (debug): 
            hdul['SCI'].data = isbkg_prc.astype(float)*bkgimage_prc
            hdul['VAR'].data = isbkg_prc.astype(float)*bkgvar_prc                            
            hdul['BPM'].data = (~isbkg_prc).astype(int)
            hdul.writeto(name+'_meanbkg.fits',overwrite=True)
            hdul['SCI'].data = isbkg_prc.astype(float)*image_fprc[df]*(isbkgarea_prc.astype(float))
            hdul['VAR'].data = isbkg_prc.astype(float)*var_fprc[df]*(isbkgarea_prc.astype(float))                            
            hdul['BPM'].data = (~isbkg_prc).astype(int)
            hdul.writeto(name+'_'+str(df)+'_'+str(dp)+'_bkg.fits',overwrite=True)             

        r_s,C_s = np.where(np.ones((prows,1024/rcbin_d[1])))    # indices inside amplifier samples s        
        aList_sd = []                                           # set up polynomials to use for each amp
        for a in range(6):
            if (a not in amp_Atp): 
                aList_sd.append(0)
                continue
            cSlice = slice(c0_a[a], c0_a[a]+1024/rcbin_d[1]) 
            rorder,corder = 1,1
            for p in (0,1):                                     # ensure that same bkg polynomial used for O,E
                r_S,C_S = np.where(isbkg_prc[p,:,cSlice])       # indices of background samples S
                rorder = max(rorder,1 + int(3*r_S.ptp()/prows))
                corder = max(corder,1 + int((C_S.ptp()/(1024./rcbin_d[1]))>0.5))
            a_ds = np.ones((1,r_s.shape[0]))
            if rorder > 1: a_ds = np.vstack((a_ds,r_s))
            if rorder > 2: a_ds = np.vstack((a_ds,r_s**2))
            if corder > 1: a_ds = np.vstack((a_ds,C_s))                        
            aList_sd.append(a_ds.T)  

        bkg_ftpRC = np.zeros((files,targets,2,Rows,Cols))
        okbkg_ftpR = np.ones((files,targets,2,Rows),dtype=bool)
        okbkg_tpR = okbkg_ftpR.all(axis=0)        
        okbkg_t = okbkg_tpR.all(axis=(1,2))       
        for f,p in np.ndindex(files,2):
            bkg_rc = np.zeros((prows,cols))
            for a in range(6):
                if (a not in amp_Atp[:,:,p]): continue                  
                cSlice = slice(c0_a[a], c0_a[a]+1024/rcbin_d[1])           
                s_S = np.where(isbkg_prc[p,:,cSlice].flatten())[0]
                a_Sd = aList_sd[a][s_S]                                 
                cof_d,sumsqerr = lstsq(a_Sd,image_fprc[f,p,:,cSlice][isbkg_prc[p,:,cSlice]])[0:2]                              
                bkg_rc[:,cSlice] = np.matmul(aList_sd[a],cof_d).reshape((prows,1024/rcbin_d[1]))                
            for t in range(targets):
                bkg_ftpRC[f,t,p] = bkg_rc[ri_tpR[t,p],:][:,ci_tpC[t,p]]
            if (debug): 
                if ((f==df)&(p==dp)):
                    hdul['SCI'].data = bkg_rc
                    del(hdul['VAR'])
                    del(hdul['BPM'])
                    del(hdul['TMAP'])                                                            
                    hdul.writeto(name+'_'+str(f)+'_'+str(p)+'_fitbkg.fits',overwrite=True) 
  # for sky, use targetmap
    else:
        bkg_ftpR = np.ma.median(np.ma.masked_array(image_ftpRC,mask=~isbkg_ftpRC),axis=4).data
        bkg_ftpRC = bkg_ftpR[:,:,:,:,None]*(oktgt_ftpRC+isbkg_ftpRC)
    
    signal_ftpRC = image_ftpRC - bkg_ftpRC               

  # fit column profiles for each target to 1D Moffat for culling and guiding
    ok1_ftpRC = (okbin_ftpRC & oktgt_ftpRC & (~isbkg_ftpRC))
    cols_ftpR = ok1_ftpRC.sum(axis=4)
    prof_ftpRC = ok1_ftpRC*signal_ftpRC
    norm_ftpR = prof_ftpRC.max(axis=4)
    prof_ftpRC[cols_ftpR>0,:] /= norm_ftpR[cols_ftpR>0,None]    
    doprof_ftpR = (cols_ftpR == cols_ftpR.max(axis=3)[:,:,:,None]) & (Cmap_tpR>0.)[None,:,:,:]    
    doprof_ftpRC = (oktgt_ftpRC & (~isbkg_ftpRC) & doprof_ftpR[:,:,:,:,None])

    if debug:
        hdul0['SCI'].data = image_ftpRC[0]
        hdul0.writeto('file0image.fits',overwrite=True) 
        hdul0['SCI'].data = bkg_ftpRC[0]
        hdul0.writeto('file0bkg.fits',overwrite=True)             
        hdul0['SCI'].data = signal_ftpRC[0]
        hdul0.writeto('file0signal.fits',overwrite=True) 
        hdul0['SCI'].data = prof_ftpRC[0]
        hdul0.writeto('file0prof.fits',overwrite=True)
        hdul0['SCI'].data = doprof_ftpRC[0].astype(int)
        hdul0.writeto('file0doprof.fits',overwrite=True)                 

    i_ftpR = -np.ones((files,targets,2,Rows),dtype=int)
    i_ftpR[doprof_ftpR] = np.arange(doprof_ftpR.sum()) 
        
    sigma_i, fCmax_i, C0_i, fiterr_iC, okprof_i =    \
        moffat1dfit(prof_ftpRC[doprof_ftpR,:].reshape((-1,Cols)),    \
            doprof_ftpRC[doprof_ftpR,:].reshape((-1,Cols)), \
            beta=2.5)

    rmsmax = 0.2
    use_i = (sigma_i != Cols)
    rmserr_i = np.zeros_like(sigma_i)
    rmserr_i[use_i] =  np.sqrt((fiterr_iC[use_i]**2).mean(axis=1))
    use_i &= (rmserr_i < rmsmax)    
    notuse_i = np.logical_not(use_i)
    badprof_ftpR = np.in1d(i_ftpR.flatten(),np.where(notuse_i)[0]).reshape((files,targets,2,-1))
    okprof_ftpR = (doprof_ftpR & ~badprof_ftpR)
    okprof_ftp = (okprof_ftpR.sum(axis=3) > 3)

    C0_ftpR = np.zeros((files,targets,2,Rows))
    C0_ftpR[okprof_ftpR] = C0_i[i_ftpR[okprof_ftpR]]
    dC_ftpR = np.zeros((files,targets,2,Rows)) 
    dC_ftpR[okprof_ftpR] = (C0_ftpR - Cmap_tpR[None,:,:,:])[okprof_ftpR]
#    dCmedian_ftp = np.ma.median(np.ma.masked_array(dC_ftpR,mask=~okprof_ftpR),axis=3).data
    dClinpoly_dftp = np.zeros((2,files,targets,2))
    for f,t,p in np.ndindex(files,targets,2):
        if ~okprof_ftp[f,t,p]: continue 
        dClinpoly_dftp[:,f,t,p] = np.polyfit(np.arange(Rows)[okprof_ftpR[f,t,p]],        \
            dC_ftpR[f,t,p][okprof_ftpR[f,t,p]],1)
    ddC_ftpR = dC_ftpR - okprof_ftpR*(dClinpoly_dftp[1,:,:,:,None] +    \
        np.arange(Rows)[None,None,None,:]*dClinpoly_dftp[0,:,:,:,None])
    ddCLower,ddClower,ddCupper,ddCUpper = fence(ddC_ftpR[okprof_ftpR])
    baddC_ftpR = (((ddC_ftpR < ddCLower) | (ddC_ftpR > ddCUpper)) & okprof_ftpR)

    sigma_ftpR = np.zeros((files,targets,2,Rows))
    sigma_ftpR[okprof_ftpR] = sigma_i[i_ftpR[okprof_ftpR]]
    sigmamedian_ftp = np.ma.median(np.ma.masked_array(sigma_ftpR,mask=~okprof_ftpR),axis=3).data
    dsigma_ftpR = sigma_ftpR - sigmamedian_ftp[:,:,:,None]
    dsigLower,dsiglower,dsigupper,dsigUpper = fence(dsigma_ftpR[okprof_ftpR])            
    badsigma_ftpR = (((dsigma_ftpR < dsigLower) | (dsigma_ftpR > dsigUpper)) & okprof_ftpR)
    okprof_ftpR = (okprof_ftpR & ~baddC_ftpR & ~badsigma_ftpR)
    okprof_ftp = (okprof_ftpR.sum(axis=3) > 3)
        
    fCmax_ftpR = np.zeros((files,targets,2,Rows))                                    
    fCmax_ftpR[okprof_ftpR] = fCmax_i[i_ftpR[okprof_ftpR]]        
    fmax_ftpR = np.zeros((files,targets,2,Rows)) 
    fmax_ftpR[okprof_ftpR] = (fCmax_ftpR*norm_ftpR)[okprof_ftpR]
    rmserr_ftpR = np.zeros((files,targets,2,Rows))         
    rmserr_ftpR[okprof_ftpR] = np.sqrt((fiterr_iC[i_ftpR[okprof_ftpR]]**2).mean(axis=1))
    
    if debug:
        np.savetxt("doprof_ftpR.txt",np.vstack((np.indices((files,targets,2)).reshape((3,-1)),  \
            doprof_ftpR.astype(int).reshape((-1,Rows)).T)).T,fmt="%3i %3i %3i  "+Rows*"%3i ")     
        np.savetxt("okprof_ftpR.txt",np.vstack((np.indices((files,targets,2)).reshape((3,-1)),  \
            okprof_ftpR.astype(int).reshape((-1,Rows)).T)).T,fmt="%3i %3i %3i  "+Rows*"%3i ")            
        np.savetxt("proffit.txt",np.vstack((i_ftpR[okprof_ftpR], np.where(okprof_ftpR),  \
            sigma_ftpR[okprof_ftpR], fCmax_ftpR[okprof_ftpR],dC_ftpR[okprof_ftpR],  \
            rmserr_ftpR[okprof_ftpR],fmax_ftpR[okprof_ftpR])).T,    \
            fmt="%6i "+4*"%4i "+4*"%9.3f "+"%9.0f ") 
        np.savetxt("Cmap_tpR.txt", np.vstack((np.indices((targets,2)).reshape((2,-1)), \
            Cmap_tpR.reshape((-1,Rows)).T)).T, fmt=2*"%4i "+Rows*"%8.3f ")
        np.savetxt("badprofs_ftp.txt",np.vstack((np.indices((targets,2)).reshape((2,-1)),   \
            (baddC_ftpR | badsigma_ftpR).sum(axis=3).reshape((files,-1)))).T,fmt=(files+2)*"%3i ") 
        np.savetxt("ddCdsig_R"+dlblftp,np.vstack((okprof_ftpR[df,dt,dp],ddC_ftpR[df,dt,dp],   \
            dsigma_ftpR[df,dt,dp])).T,fmt="%2i %8.3f %8.3f")                                        

  # compute mean column position relative to slit-center Cmap for each target and sigma using inverse-variance weight
    wt_ftpR = np.ones((files,targets,2,Rows))
    wt_ftpR[rmserr_ftpR>0] = 1./rmserr_ftpR[rmserr_ftpR>0]**2
    okprof_ftp = ((wt_ftpR != 1.).sum(axis=3) > 2)          # want at least 3 good rows
    wt_ftp = np.zeros((files,targets,2))
    wt_ftp = (wt_ftpR*(wt_ftpR != 1.)).sum(axis=3)
    
    dC_ftp = np.average(dC_ftpR, axis=3, weights=wt_ftpR)
    dC_f = np.zeros(files)
    dC_f[1:] = np.average((dC_ftp[1:]-dC_ftp[0]).reshape((-1,2*targets)), axis=1,   \
        weights=wt_ftp[1:].reshape((-1,2*targets)))
        
    if debug:
        np.savetxt("wt_ftp.txt", np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)), \
            wt_ftp.reshape((files,-1)))).T, fmt=" %2i %2i %2i  "+files*"%10.3e ")   
        
    dC_tp = np.average((dC_ftp - dC_f[:,None,None]), axis=0, weights=wt_ftp) 

    sigma_ftp = np.average(sigma_ftpR, axis=3, weights=wt_ftpR)    
    sigma_f = np.sqrt(np.average(sigma_ftp.reshape((-1,2*targets))**2 -     \
        sigma_ftp[okprof_ftp].min()**2, axis=1,weights=wt_ftp.reshape((-1,2*targets))))
    sigma_tp = np.sqrt(np.average((sigma_ftp**2 - sigma_f[:,None,None]**2), axis=0, \
        weights=wt_ftp))

    if debug:       
        np.savetxt("dC_ftp.txt", np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)), \
            dC_ftp.reshape((files,-1)))).T, fmt=" %2i %2i %2i "+files*"%8.3f ")
        np.savetxt("sigma_ftp.txt", np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)), \
            sigma_ftp.reshape((files,-1)))).T, fmt=" %2i %2i %2i "+files*"%8.3f ")                     
        np.savetxt("sigmadC_f.txt", np.vstack((range(files),sigma_f,dC_f)).T,   \
            fmt="%2i %8.3f %8.3f")
        np.savetxt("sigmadC_tp.txt", np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)), \
            sigma_tp.flatten(),dC_tp.flatten())).T,fmt=" %2i %2i %2i %8.3f %8.3f")         

  # compute moffat column profiles for best-fit corrected C0, sigma for all targets
    sigma_ftp = np.sqrt(sigma_f[:,None,None]**2 + sigma_tp[None,:,:]**2)
    C0_ftpR = Cmap_tpR[None,:,:,:] + dC_f[:,None,None,None] + dC_tp[None,:,:,None]
    x_iC = (np.arange(Cols)-C0_ftpR[:,:,:,:,None]).reshape((-1,Cols))
    okmoffat_ftpR = (C0_ftpR != 0)

    moffat_ftpRC = moffat(np.ones(files*targets*2*Rows),np.repeat(sigma_ftp.flatten(),Rows),  \
        x_iC,beta=2.5).reshape((files,targets,2,Rows,Cols))
    fluxtot_ftpR = moffat_ftpRC.sum(axis=4)
        
  # compare actual profile to moffat profile fit (only for good profiles)
  # first cull bad edge columns in target caused by intruding neighbors
    compsigma = 6.
    compslope = 2.
    okcomp_ftpRC = ok1_ftpRC & okprof_ftpR[:,:,:,:,None] & okmoffat_ftpR[:,:,:,:,None]       
    signalfit_ftpRC = okmoffat_ftpR[:,:,:,:,None]*norm_ftpR[:,:,:,:,None]*moffat_ftpRC
    signalerr_ftpRC = signal_ftpRC - signalfit_ftpRC
    photerr_ftpRC = np.sqrt(var_ftpRC)
    sigslope_ftpRC = np.zeros((files,targets,2,Rows,Cols))    
    sigslope_ftpRC[:,:,:,:,1:-1] = (np.abs(np.diff(signalfit_ftpRC[:,:,:,:,1:],axis=4)) + \
        np.abs(np.diff(signalfit_ftpRC[:,:,:,:,:-1],axis=4)))/2.
    badC_ftpRC = (okcomp_ftpRC &(signalerr_ftpRC > compsigma*photerr_ftpRC) &   \
        (compslope*sigslope_ftpRC < compsigma*photerr_ftpRC))
    badCs_tpC = badC_ftpRC.sum(axis=(0,3))
    isbadC_tpC = ((badC_ftpRC.sum(axis=0) > 2).sum(axis=2) > 2)
    
    if isbadC_tpC.sum():
        targetswithcull = np.unique(np.where(isbadC_tpC)[0])        
        printstdlog((("%3i columns culled for contamination in targets: "+len(targetswithcull)*"%2i ") %     \
            ((isbadC_tpC.sum(),)+tuple(targetswithcull))),logfile)    
        ok1_ftpRC = ok1_ftpRC & ~isbadC_tpC[None,:,:,None,:]
        okcomp_ftpRC = okcomp_ftpRC & ~isbadC_tpC[None,:,:,None,:]
            
  # now find CR's                     
    iscr_ftpRC = np.zeros((files,targets,2,Rows,Cols)).astype(bool)
    iscr_ftpRC[okcomp_ftpRC] = (signalerr_ftpRC >    \
        np.maximum(compsigma*photerr_ftpRC,compslope*sigslope_ftpRC))[okcomp_ftpRC]                        
    crs_f = iscr_ftpRC.sum(axis=(1,2,3,4))
    ok3_ftpRC= (ok1_ftpRC & ~iscr_ftpRC)

    if debug:
        print "Photon sigma max: ", compsigma
        print "Profile slope allowance: ", compslope
        np.savetxt("signal_RC"+dlblftp,(ok1_ftpRC*signal_ftpRC)[df,dt,dp],fmt="%8.1f ")
        np.savetxt("signalfit_RC"+dlblftp,signalfit_ftpRC[df,dt,dp],fmt="%8.1f ")          
        np.savetxt("var_RC"+dlblftp,var_ftpRC[df,dt,dp],fmt="%8.1f ")
        np.savetxt("okcomp_RC"+dlblftp,okcomp_ftpRC[df,dt,dp],fmt="%3i ")        
        np.savetxt("iscr_RC"+dlblftp,iscr_ftpRC[df,dt,dp],fmt="%3i ")
        np.savetxt("badCs_tpC.txt", np.vstack((np.indices((targets,2)).reshape((2,-1)),   \
            badCs_tpC.reshape((-1,Cols)).T)).T,fmt="%2i %2i "+Cols*"%3i ")
        np.savetxt("crs_ftp.txt",np.vstack((np.indices((targets,2)).reshape((2,-1)),   \
            iscr_ftpRC.sum(axis=(3,4)).reshape((files,-1)))).T,fmt=(files+2)*"%3i ")           
                
  # do extraction in column direction using moffat fits as weights
    print "Do extraction"
    xtrwt_ftpRC = np.copy(moffat_ftpRC)
    xtrnorm_ftpRC = np.repeat(xtrwt_ftpRC.sum(axis=4)[:,:,:,:,None],Cols,axis=4)
    xtrwt_ftpRC[xtrnorm_ftpRC>0] /= xtrnorm_ftpRC[xtrnorm_ftpRC>0]
    fopt_ftpR = np.zeros((files,targets,2,Rows))
    vopt_ftpR = np.zeros((files,targets,2,Rows))
    ok_ftpR = ((ok3_ftpRC.sum(axis=4) > 0) & okprof_ftpR)    

    for f in range(files):
        fstd_tpR = np.zeros((targets,2,Rows))
        vstd_tpR = np.zeros((targets,2,Rows))
        ok_tpR = ok_ftpR[f]
        fstd_tpR[ok_tpR] = signal_ftpRC[f].sum(axis=3)[ok_tpR]/xtrwt_ftpRC[f].sum(axis=3)[ok_tpR]
        vstd_tpR[ok_tpR] = var_ftpRC[f].sum(axis=3)[ok_tpR] / xtrwt_ftpRC[f].sum(axis=3)[ok_tpR]**2
        vopt_tpRC = fstd_tpR[:,:,:,None]*xtrwt_ftpRC[f] +     \
                    bkg_ftpRC[f,:,:,Rows/2,Cols/2].clip(min=0.)[:,:,None,None]
        norm_tpRC = np.zeros((targets,2,Rows,Cols))                
        norm_tpRC[vopt_tpRC > 0.] =     \
                    (xtrwt_ftpRC[f][vopt_tpRC > 0.]**2/vopt_tpRC[vopt_tpRC > 0.])
        norm_tpR = norm_tpRC.sum(axis=3) 
        fopt_tpRC = np.zeros((targets,2,Rows,Cols))
        fopt_tpRC[vopt_tpRC > 0.] =     \
            (xtrwt_ftpRC[f]*signal_ftpRC[f])[vopt_tpRC > 0.]/vopt_tpRC[vopt_tpRC > 0.]
        fopt_ftpR[f][ok_tpR] = (ok3_ftpRC[f]*fopt_tpRC).sum(axis=3)[ok_tpR]/norm_tpR[ok_tpR]
        vopt_ftpR[f][ok_tpR] = (ok3_ftpRC[f]*xtrwt_ftpRC[f]).sum(axis=3)[ok_tpR]/norm_tpR[ok_tpR]
        
  # compute row-direction image motion relative to reference file using NIR half-point in optimal row exraction
    fnorm_ftpR = fopt_ftpR/fopt_ftpR.max(axis=3)[:,:,:,None]
    RNIR_ftp = -np.ones((files,targets,2))
    for f,t,p in np.ndindex(files,targets,2):    
        if (fopt_ftpR.max(axis=3)[f,t,p]==0.): continue          # NIR is right for O, left for E 
        RNIR_ftp[f,t,p] = fracmax(fnorm_ftpR[f,t,p],points=3,ok_x=ok_ftpR[f,t,p])[1-p]
    drtot_ftp = RNIR_ftp - RNIR_ftp[fref][None,:,:]             # change from ref image
    okdr_ft = (RNIR_ftp != -1.).all(axis=2)
    drmean_ft = okdr_ft*drtot_ftp.mean(axis=2)
    xmean_t = yx_dptW[1].mean(axis=0)[:,WNIR]
    drsplit_ft = okdr_ft*(drtot_ftp[:,:,1] - drtot_ftp[:,:,0])

  # fit drmean to column position to give intercept and slope, then dY and dPA
    dy0_f = np.zeros(files)
    drot_f = np.zeros(files)
    for f in range(files):
        okdr_t = okdr_ft[f]     
        drot_f[f],dy0_f[f] = np.polyfit(xmean_t[okdr_t],drmean_ft[f][okdr_t]*rcbin_d[0]*pixmm,1)
    dY_f = dy0_f*saltfps/dtrfps
    dPA_f = np.degrees(drot_f)
    
  # add in rho flexure in PA and dr from cal to ref from flexure models     
    ur0ref,dum,dum = rssdtralign(dateobs,trkrho_f[fref])        
    dY_f += -(ur0ref - ur0cal)*(pixmm*rcbin_d[0])*saltfps/dtrfps
      
  # finally, compute bsflex-corrected wavmap_ftpR, 
  #   then add in dtr flexure correction to cal image by drflex shift (assumes guiding is small effect)
    wavmap_ftpR = np.zeros((files,targets,2,Rows))
    RNIRmap_ftp = np.zeros((files,targets,2)) 
      
    for f in range(files):
        YXf_ds = rotate2d(YX_ds, dPA_f[f], center=np.zeros(2)) + np.array([dY_f[f],0.])[:,None]
        yxf_dptW = RSScolpolcam(YXf_ds,wav_s,coltem,camtem).reshape((2,2,targets,Wavs)) \
            + dyxp_dp[:,:,None,None]    
        wavmap_ftpR[f],dum,dum = specmap(yxf_dptW,wav_W,ri_tpR,ci_tpC,dWdY_ptW,hdulcal)
        for t,p in np.ndindex((targets,2)):
            okmap_R = (wavmap_ftpR[f,t,p] > 0.)        
            rootList = UnivariateSpline(np.arange(Rows)[okmap_R],wavmap_ftpR[f,t,p,okmap_R]-wavNIR,s=0).roots()
            if len(rootList): RNIRmap_ftp[f,t,p] = rootList[0]
    
    if debug:
        np.savetxt("RNIR_fpt.txt",RNIR_ftp.transpose((0,2,1)).reshape(files,-1).T,fmt=" %9.4f")
        np.savetxt("drtot_ftp.txt",np.vstack((np.indices((targets,2)).reshape((2,-1)),  \
            (okdr_ft[:,:,None]*drtot_ftp).reshape((files,-1)))).T, fmt=" %3i %3i"+files*" %9.4f")         
        np.savetxt("dY_f.txt",np.vstack((np.arange(files),dY_f,dPA_f,trkrho_f)).T, \
            fmt=" %3i %9.5f %8.4f %8.3f")        
        np.savetxt("dr_ft.txt",np.vstack((np.indices((files,targets)).reshape((2,-1)),  \
            drmean_ft.flatten(),drsplit_ft.flatten())).T, fmt=" %3i %3i %9.4f %9.4f")
        np.savetxt("drsplit_ft.txt",np.vstack((np.arange(targets),  \
            drsplit_ft)).T, fmt=" %3i"+files*" %9.4f")        
        np.savetxt("RNIRmapcal_tp.txt",np.vstack((np.indices((targets,2)).reshape((2,-1)),  \
            RNIRmap_tp.flatten())).T,fmt=" %3i %3i %8.2f")
        np.savetxt("RNIRmap_ftp.txt",np.vstack((np.indices((targets,2)).reshape((2,-1)),    \
            RNIRmap_ftp.reshape((files,-2)))).T, fmt=" %3i %3i"+files*" %8.2f")

  # scrunch onto wavelength grid
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
     
    fopt_ftpw = np.zeros((files,targets,2,wavs))
    vopt_ftpw = np.zeros_like(fopt_ftpw) 
    covar_ftpw = np.zeros_like(fopt_ftpw) 
    ok_ftpw = np.zeros_like(fopt_ftpw,dtype=bool)
          
    for f,t,p in np.ndindex((files,targets,2)):
        dirn = [1,-1][p]                      # scrunch needs increasing indices
        Rowuse = np.where(wavmap_ftpR[f,t,p] > 0)[0]
        wedgeuse = np.where((wedge_w >= wavmap_ftpR[f,t,p,Rowuse].min()) &    \
                            (wedge_w < wavmap_ftpR[f,t,p,Rowuse].max()))[0]
        wavout = wedgeuse[:-1]                # scrunch has input bins starting at bin, not bin-1/2: 
        Rbinedge_W = (interp1d(wavmap_ftpR[f,t,p,Rowuse][::dirn],(Rowuse[::dirn]+0.5))    \
            (wedge_w[wedgeuse]))[::dirn]             
        oktp_w = (scrunch1d(ok_ftpR[f,t,p].astype(int),Rbinedge_W)   \
               == (Rbinedge_W[1:] - Rbinedge_W[:-1]))[::dirn] 
        ok_ftpw[f,t,p,wavout] = oktp_w
           
        fopt_ftpw[f,t,p,wavout] = scrunch1d(fopt_ftpR[f,t,p],Rbinedge_W)[::dirn]*oktp_w 
        vopt_W,covar_W = scrunchvar1d(vopt_ftpR[f,t,p],Rbinedge_W)
        vopt_ftpw[f,t,p,wavout] = vopt_W[::dirn]*oktp_w
        covar_ftpw[f,t,p,wavout] = covar_W[::dirn]*oktp_w               

  # check row-direction wavelength stability relative to reference file using NIR half-point in scrunched data
    fnorm_ftpw = fopt_ftpw/fopt_ftpw.max(axis=3)[:,:,:,None]
    wNIR_ftp = -np.ones((files,targets,2))
    for f,t,p in np.ndindex(files,targets,2):    
        if (fopt_ftpw.max(axis=3)[f,t,p]==0.): continue          # NIR is right for O, left for E 
        wNIR_ftp[f,t,p] = fracmax(fnorm_ftpw[f,t,p],points=3,ok_x=ok_ftpw[f,t,p])[1-p]
    dwtot_ftp = wNIR_ftp - wNIR_ftp[fref][None,:,:]             # change from ref image

    if debug:
        tgtTab.write(name+"_extractTab.txt",format='ascii.fixed_width',   \
            bookend=False, delimiter=None, overwrite=True)          
        np.savetxt("wedge_w.txt",wedge_w.T,fmt="%8.3f")
        np.savetxt("wavmap_ftpR_"+str(df)+"_"+str(dt)+".txt",wavmap_ftpR[df,dt].T,fmt="%10.2f")                
        np.savetxt("ok_ftpR_"+str(df)+"_"+str(dt)+".txt",ok_ftpR[df,dt].astype(int).T,fmt="%3i")
        np.savetxt("xtrwt_ftpRC_"+dlblftp+".txt",xtrwt_ftpRC[df,dt,dp],fmt="%10.4f")
        np.savetxt("xtrnorm_ftpRC_"+dlblftp+".txt",xtrnorm_ftpRC[df,dt,dp],fmt="%10.4f")
        np.savetxt("image_ftpRC_"+dlblftp+".txt",image_ftpRC[df,dt,dp],fmt="%10.2f")
        np.savetxt("bkg_ftpRC_"+str(df)+"_"+str(dt)+"_"+str(dp)+".txt",bkg_ftpRC[df,dt,dp],fmt="%10.2f") 
        np.savetxt("fopt_ftpR_"+str(df)+"_"+str(dt)+".txt",fopt_ftpR[df,dt].T,fmt="%10.2f")
        np.savetxt("fopt_ftpw_"+str(df)+"_"+str(dt)+".txt",fopt_ftpw[df,dt].T,fmt="%10.2f")
        np.savetxt("vopt_ftpR_"+str(df)+"_"+str(dt)+".txt",vopt_ftpR[df,dt].T,fmt="%10.2f")
        np.savetxt("vopt_ftpw_"+str(df)+"_"+str(dt)+".txt",vopt_ftpw[df,dt].T,fmt="%10.2f")
        np.savetxt("covar_ftpw_"+str(df)+"_"+str(dt)+".txt",covar_ftpw[df,dt].T,fmt="%10.2f")                                
        np.savetxt("ok_ftpw_"+str(df)+"_"+str(dt)+".txt",ok_ftpw[df,dt].astype(int).T,fmt="%3i")
        np.savetxt("dwtot_ftp.txt",np.vstack((np.indices((targets,2)).reshape((2,-1)),  \
            (okdr_ft[:,:,None]*dwtot_ftp).reshape((files,-1)))).T, fmt=" %3i %3i"+files*" %9.4f")        
        
    printstdlog     \
        ("\n        Output file               dY (\") dX (\") dPA (deg) sigma (\") relflux crs", logfile)
            
  # save the result
    signal_f = norm_ftpR.max(axis=3).sum(axis=(1,2))
    relflux_f = signal_f/signal_f.max()  
    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)
        outfile = 'e'+file
        del(hdul['TMAP'])
        hdul['SCI'].data = fopt_ftpw[f].astype('float32').transpose((1,0,2))
        for ext in [0,'SCI']:                       # specpolrawstokes wants them in both ext's
            hdul[ext].header['CRVAL1'] = 0
            hdul[ext].header['CDELT1'] = wbin 
            hdul[ext].header['CTYPE1'] = 'Angstroms'
            hdul[ext].header['CRVAL2'] = 0
            hdul[ext].header['CDELT2'] = 1 
            hdul[ext].header['CTYPE2'] = 'Target'
        hdr1 = hdul['SCI'].header
        hdul['VAR'].data = vopt_ftpw[f].astype('float32').transpose((1,0,2))
        hdul.append(pyfits.ImageHDU(data=covar_ftpw[f].astype('float32').transpose((1,0,2)), \
          header=hdr1, name='COV'))
        hdul['BPM'].data = (~ok_ftpw[f]).astype('uint8').transpose((1,0,2)) 
        hdul['TGT'] = pyfits.table_to_hdu(tgtTab)
        hdul.append(pyfits.ImageHDU(data=wedge_W.astype('float32'), header=hdr1, name='WCSDVARR'))
        hdul['WCSDVARR'].header['CDELT1'] = 1        
        hdul.writeto(outfile,overwrite=True)

        outfile = 'e'+file

        ur0,uc0,saltfps = rssdtralign(dateobs,trkrhocal)       # saltfps =micr/arcsec    

        dver = dY_f[f]*1000./saltfps                            
        dhor = dC_f[f]*1000.*pixmm*rcbin_d[1]/dtrfps
        sigma = sigma_f[f]*1000.*pixmm*rcbin_d[0]/dtrfps
        printstdlog ("%30s %8.3f %8.3f %9.4f %8.3f %7.2f %4i" %     \
            (outfile,dver,dhor,dPA_f[f],sigma,relflux_f[f],crs_f[f]), logfile)

    return

# ----------------------------------------------------------
def specmap(yx_dptW,wav_W,ri_tpR,ci_tpC,dWdY_ptW,hdul,drflex=0.):
    """compute target spectrum maps for imaging spectropolarimetry
    """
    ids = yx_dptW.shape[2]
    Rows = ri_tpR.shape[2]
    hdr = hdul[0].header
    rcbin_d = np.array(hdr['CCDSUM'].split(' ')).astype(int)[::-1]
    prows, cols = hdul[1].data.shape[1:] 
    dateobs =  hdr['DATE-OBS'].replace('-','')
    trkrho = hdr['TRKRHO'] 
    pixmm = 0.015
    imwav = 5000.      
    ur0,uc0,saltfps = rssdtralign(dateobs,trkrho)       # ur, uc =unbinned pixels, saltfps =micr/arcsec
    yx0_d = -0.015*np.array([ur0,uc0])                  # center of CCD relative to optical axis in mm          
    yx0_dp,rshift = RSSpolgeom(hdul,imwav)[:2]          # use pol image shift for 5000 Ang              
    rccenter_d = ccdcenter(hdul[1].data[0])[0]*np.array([2,1])         # gets center of unsplit data        
    
    rc_dptW = (yx_dptW - yx0_d[:,None,None,None])    \
            /(rcbin_d[:,None,None,None]*pixmm) + rccenter_d[:,None,None,None] + drflex                   
    rcp_dptW = rc_dptW + np.array([[-rshift,-rshift-prows],[0,0]])[:,:,None,None]
    wavmap_tpR = np.zeros((ids,2,Rows))
    dwavmap_tpR = np.zeros_like(wavmap_tpR)
    Cmap_tpR = np.zeros_like(wavmap_tpR)    
    for t,p in np.ndindex((ids,2)):                             
        RList = list(np.where((ri_tpR[t,p] >= rcp_dptW[0,p,t].min()) &    \
                (ri_tpR[t,p] <= rcp_dptW[0,p,t].max()))[0])                
        wavmap_tpR[t,p,RList] = interp1d(rcp_dptW[0,p,t],wav_W,kind='cubic')(ri_tpR[t,p,RList])
        dwavmap_tpR[t,p,RList] =     \
                interp1d(rcp_dptW[0,p,t],dWdY_ptW[p,t],kind='cubic')(ri_tpR[t,p,RList])        
        Cmap_tpR[t,p,RList] = (interp1d(rcp_dptW[0,p,t],rcp_dptW[1,p,t],   \
                kind='cubic')(ri_tpR[t,p,RList]) - ci_tpC[t,p,0]).clip(min=0.) 
                   
    return wavmap_tpR,dwavmap_tpR,Cmap_tpR
    
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
            wpat_dp = np.vstack((wpat_p, np.array(p.split()[3:])))

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

