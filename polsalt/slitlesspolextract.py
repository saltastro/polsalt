
"""
slitlesspolextract

Optimal extraction for slitless imaging polarimetric data
Contains corrections for guiding, seeing variations
Write out extracted data fits (etm*) dimensions wavelength,target #

"""

import os, sys, glob, shutil, inspect, pprint

import numpy as np
from scipy.interpolate import griddata, interp1d, UnivariateSpline, LSQUnivariateSpline
from scipy.ndimage.interpolation import shift
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import Latitude,Longitude,Angle

# this is pysalt-free

import rsslog
from obslog import create_obslog
from scrunch1d import scrunch1d, scrunchvar1d
from polutils import rssdtralign
from immospolextract import specmap, moffat, moffat1dfit, findpair 
from polmaptools import ccdcenter,boxsmooth1d,impolguide
from rssoptics import RSSpolgeom, RSScolpolcam

datadir = os.path.dirname(__file__) + '/data/'
keywordfile = datadir+"obslog_config.json"
np.set_printoptions(threshold=np.nan)

import warnings 
# warnings.filterwarnings("error")

# ----------------------------------------------------------
def slitlesspolextract(fileList,name,logfile='salt.log',debug=False):
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
    _a amplifier = 0,1,2,3,4,5

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
    fref = np.unravel_index(np.argmin(np.abs(imgno_f[:,None] - calimgno_F[None,:])),(files,calfilters))[0]
    
    rsslog.message(("image closest to cal: %4i" % imgno_f[fref]),logfile)            
    rsslog.message(("candidate ids:       %3i" % ids),logfile)
        
  # get id'd target positions in calfilter(s) at reference (imwav) wavelength
    dateobs =  hdr0['DATE-OBS'].replace('-','')
    trkrho = hdr0['TRKRHO']   
    ur0,uc0,saltfps = rssdtralign(dateobs,trkrho)       # ur, uc =unbinned pixels, saltfps =micr/arcsec
    yx0_d = -0.015*np.array([ur0,uc0])                  # center of CCD relative to optical axis in mm
    yxOEoff_d = -np.diff((yxp0_dp - RSSpolgeom(hdul0,imwav)[2]), axis=1)[:,0]            
    yx0_dp,rshift = RSSpolgeom(hdul0,imwav,yxOEoff_d=yxOEoff_d)[:2]     # use pol image shift for 5000 Ang
              
    rccenter_d, cgap_c = ccdcenter(hdul0[1].data[0])
    rccenter_d[0] *= 2                                  # gets center of unsplit data
    c0_a = (cgap_c[0]-2048/rcbin_d[1])*np.ones(6,dtype=int)
    c0_a[[2,4]] = cgap_c[[1,3]]
    c0_a[[1,3,5]] = c0_a[[0,2,4]]+1024/rcbin_d[1]       # gets first column in each amplifier               

    YX_dt = np.array([tgtTab['Y'],tgtTab['X']])        
    yxcat_dpt = RSScolpolcam(YX_dt,imwav,coltem,camtem,yxOEoff_d=yxOEoff_d)
    yxpcat_dpt = yxcat_dpt - yx0_dp[:,:,None]
    yxp_dpt = np.array([[tgtTab['YO'],tgtTab['YE']],[tgtTab['XO'],tgtTab['XE']]])
    dyxp_dpt = yxp_dpt - yxpcat_dpt                     # target offset from model
 
    Rfovlim = 45.0                                      # avoid waveplate vignetting at edge of SALT FOV 
    okfov_t = (np.sqrt((YX_dt**2).sum(axis=0)) < Rfovlim)
    fovcullList = list(np.where(~okfov_t)[0])
    if len(fovcullList): rsslog.message((("%3i ids FOV culled: "+len(fovcullList)*"%2i ") %     \
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
        
  # get target box info, form targetmaps
    if (debug):
        df, dt, db, dp = 2,12,7,1
        dlblt = "_"+str(df)+"_"+str(dt)+"_"+str(dp)+".txt"
        dlblb = "_"+str(df)+"_"+str(db)+"_"+str(dp)+".txt"
        open("bkgsum.txt","w")    # new empty file 
        sumfile = open("bkgsum.txt","a")

    rckey_pd = np.array([['R0O','C0O'],['R0E','C0E']])
    Rows,Cols = np.array(hdul0[0].header['BOXRC'].split()).astype(int)
    ri_tpR = np.zeros((ids,2,Rows),dtype=int)
    ci_tpC = np.zeros((ids,2,Cols),dtype=int)
    bkgwidth = 4 
    cbi_tpC = np.zeros((ids,2,Cols+2*bkgwidth),dtype=int)   # columns around box for background
    for p in (0,1):
        ri_tpR[:,p] = np.clip(tgtTab[rckey_pd[p,0]][:,None] + np.arange(Rows)[None,:], -1,prows)
        ci_tpC[:,p] = np.clip(tgtTab[rckey_pd[p,1]][:,None] + np.arange(Cols)[None,:], 0,cols-1)
        cbi_tpC[:,p] = np.clip(tgtTab[rckey_pd[p,1]][:,None] - bkgwidth + \
            np.arange(Cols+2*bkgwidth)[None,:], 0,cols-1) 
    cliprow_tpR = ((ri_tpR == -1) | (ri_tpR == prows))
    np.clip(ri_tpR,0,prows-1,out=ri_tpR)

    wavmap_tpR,dwavmap_tpR,Cmap_tpR =   \
        specmap(yx_dptW,wav_W,ri_tpR,ci_tpC,dWdY_ptW,hdul0) 
    wavmap_tpR[cliprow_tpR] = 0.

  # find good Rows with no target overlap.  If none, cull it.
    okovlap_tpR = np.zeros((ids,2,Rows),dtype=bool)
    for t,p in np.ndindex(ids,2):
        okovlap_tpR[t,p] = (tgt_prc[p][ri_tpR[t,p],:][:,ci_tpC[t,p]]==t+1).all(axis=1)
    okovlap_t = okovlap_tpR.any(axis=2).all(axis=1)
    ovlapcullList = list(np.where(~okovlap_t & okfov_t)[0])

    if len(ovlapcullList): rsslog.message((("%3i ids overlap culled: "+len(ovlapcullList)*"%2i ") %     \
                tuple([len(ovlapcullList),]+ovlapcullList)),logfile)
    oktarget_t = okovlap_t & okfov_t

  # assemble target data
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
        
  # find saturated pixels in target boxes
  # cull targets with saturated bins in most files in either beam
    issat_fprc = np.zeros_like(okbin_fprc)
    for a in range(6):          # sat level may depend on amp due to different gains
        image_fpra = image_fprc[:,:,:,c0_a[a]:c0_a[a]+1024/rcbin_d[1]]
        satlevel = image_fpra.max()
        if ((image_fpra > 0.98*satlevel).sum() < 3): satlevel = 1.e9
        issat_fprc[:,:,:,c0_a[a]:c0_a[a]+1024/rcbin_d[1]] = (image_fpra > 0.98*satlevel)
    satbins_ftp = np.zeros((files,ids,2))
    for f,t,p in np.ndindex(files,ids,2):
        satbins_ftp[f,t,p] = (issat_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]).sum()
    oksat_t = (2*(satbins_ftp>0).sum(axis=0)/files < 1).all(axis=1)
    satcullList = list(np.where(~oksat_t & oktarget_t)[0])

    if len(satcullList): rsslog.message((("%3i ids saturation culled: "+len(satcullList)*"%2i ") %     \
                tuple([len(satcullList),]+satcullList)),logfile)
    oktarget_t &= oksat_t
    if debug:
        np.savetxt("satbins_ftp.txt",np.vstack((np.indices((ids,2)).reshape((2,-1)),    \
            satbins_ftp.reshape((files,-1)))).T,fmt="%3i ")

  # process background
  # find bad rows based on bkg positive outliers (use median across bkgwidth rows)
  # require (med(bkg) - min(med(bkg))/maxsig < relbkglim              
    relbkglim = 1. 
    bkgneglim = -3.*np.sqrt(var_fprc[okbin_fprc].min())        # reject negative bkg spikes
    okbkg_ftpR = np.zeros((files,ids,2,Rows),dtype=bool)
    bkgleft_ftpR = np.zeros((files,ids,2,Rows))
    bkgright_ftpR = np.zeros((files,ids,2,Rows))
    for f in range(files): 
        okbin_fprc[f] = okbin_hprc[hi_df[0,f]]
        relxcess_tpR = np.zeros((ids,2,Rows))
        for t,p in np.ndindex(ids,2):
            if ~(oktarget_t[t]): continue
            bkg_RC = np.zeros((Rows,Cols+2*bkgwidth))
            okbkg_RC = np.zeros_like(bkg_RC,dtype=bool)
            bkg_RC = image_fprc[f,p][ri_tpR[t,p],:][:,cbi_tpC[t,p]]            
            okbkg_RC = ((bkg_RC != 0.)&(okovlap_tpR[t,p][:,None] &  \
                (okbin_fprc[f,p] & (tgt_prc[p]==255))[ri_tpR[t,p],:][:,cbi_tpC[t,p]])) 
            bkgleft_ftpR[f,t,p] =np.ma.min(np.ma.masked_array(bkg_RC[:,:bkgwidth], \
                mask=~okbkg_RC[:,:bkgwidth]),axis=1).data*okbkg_RC[:,:bkgwidth].any(axis=1) 
            bkgright_ftpR[f,t,p] =np.ma.min(np.ma.masked_array(bkg_RC[:,-bkgwidth:], \
                mask=~okbkg_RC[:,-bkgwidth:]),axis=1).data*okbkg_RC[:,-bkgwidth:].any(axis=1) 
            okbkg_ftpR[f,t,p] = ((bkgleft_ftpR[f,t,p] >bkgneglim)  & (bkgleft_ftpR[f,t,p] != 0.) &   \
                                 (bkgright_ftpR[f,t,p] >bkgneglim) & (bkgright_ftpR[f,t,p] != 0.))
            
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
                okbkg_ftpR[f,t,p] &= (relxcess_tpR[t,p] < relbkglim)
                                 
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
    bkgcullList = list(np.where(~okbkg_t & oktarget_t)[0])

    if len(bkgcullList): rsslog.message((("%3i ids background culled: "+len(bkgcullList)*"%2i ") %     \
                tuple([len(bkgcullList),]+bkgcullList)),logfile)
    oktarget_t &= okbkg_t   
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

  # extend Cmap one bin out to allow for guiding errors           
    for t,p in np.ndindex(ids,2):
        RC0,RC1 = np.where(Cmap_tpR[t,p] > 0.)[0][[0,-1]].clip(1,Rows-2)
        Cmap_tpR[t,p,RC0-1] = Cmap_tpR[t,p,RC0] +(Cmap_tpR[t,p,RC0]-Cmap_tpR[t,p,RC0+1])
        Cmap_tpR[t,p,RC1+1] = Cmap_tpR[t,p,RC1] +(Cmap_tpR[t,p,RC1]-Cmap_tpR[t,p,RC1-1])    
    ok1_ftpRC = (okbin_ftpRC & oktgt_ftpRC) & okbkg_ftpR[:,:,:,:,None]
    
  # fit mean column profiles for each file and target to 1D Moffat, using rows with all columns    
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
    dC_ftpR = np.zeros((files,ids,2,Rows)) 
    dC_ftpR[fitprof_ftpR] = (C0_ftpR - Cmap_tpR[None,:,:,:])[fitprof_ftpR]    
    rmserr_ftpR = np.sqrt((fiterr_iC[i_ftpR,:]**2).mean(axis=4))
    okprof_ftpR = np.zeros((files,ids,2,Rows)).astype(bool)
    okprof_ftpR[fitprof_ftpR] = okprof_i

  # identify good profiles, best targets to use for seeing, guiding
    goodprofrmsmax = 3.*np.median(rmserr_ftpR)          # khn mod 20191225
    isgoodprof1_ftpR = (okprof_ftpR & (rmserr_ftpR < goodprofrmsmax))
   
  # goodprof also requires sigma not above outer quartile limits 
    outersigma_fq = np.zeros((files,2)) 
    for f in range(files):  
        q1,q3 = np.percentile(sigma_ftpR[f][isgoodprof1_ftpR[f]],(25,75))
        outersigma_fq[f] = (q1 - 3.*(q3-q1)),(q3 + 3.*(q3-q1))
    isgoodprof2_ftpR = (isgoodprof1_ftpR & (sigma_ftpR < outersigma_fq[:,1,None,None,None]))
    
  # do a first cut on ddC: column errors relative to target median
    okmedian_pR = np.array([(np.arange(Rows) > Rows/5),(np.arange(Rows) < 4*Rows/5)])    
    dC2_ftp = np.ma.median(np.ma.masked_array(dC_ftpR,  \
        mask=~(isgoodprof2_ftpR & okmedian_pR)),axis=3).data 
    ddC2_ftpR = dC_ftpR - dC2_ftp[:,:,:,None]
    q1,q3 = np.percentile(ddC2_ftpR[isgoodprof2_ftpR],(25,75))    
    lower3,upper3 = (q1 - 5.*(q3-q1)),(q3 + 5.*(q3-q1))
    isgoodprof3_ftpR = (isgoodprof2_ftpR & ((ddC2_ftpR > lower3) & (ddC2_ftpR < upper3)))  
    quartileculls3_tp = (isgoodprof2_ftpR - isgoodprof3_ftpR).sum(axis=(0,3))

  # do further cuts for 3 histogram bins adjoining upper,lower quartile cuts:
  #   cut rows adjoining new row edges 
  #   cut profs on blue side (likely contaminating spectra)
    histentries = 32                # includes 2 outer bins outside quartile limits     
    ddC2hist_tph = np.zeros((ids,2,histentries)).astype(int)  
    histsum_tp = np.zeros((ids,2)).astype(int)
    histmax_tp = np.zeros((ids,2)).astype(int)
    edgebin_dtp = np.zeros((2,ids,2))
    edgeheight_dtp = np.zeros((2,ids,2))    
    Redgelim_dxtp = np.zeros((2,3,ids,2)).astype(int) 
    Rdatalim_dtp = np.zeros((2,ids,2))
    isgoodprof3start_ftpR = np.copy(isgoodprof3_ftpR)      
    histbin = (upper3-lower3)/(histentries - 2)
    ddC2bin_h = np.linspace(lower3-histbin,upper3+histbin,histentries+1)    
    minRows = 4         
    for t,p in np.ndindex((ids,2)):             
        if ((isgoodprof2_ftpR[:,t,p].sum(axis=0) > 0).sum() < minRows): continue
        Rdatalim_dtp[:,t,p] = np.where(isgoodprof3_ftpR[:,t,p].any(axis=0))[0][[0,-1]]
        ddC2tp_i = ddC2_ftpR[:,t,p][isgoodprof2_ftpR[:,t,p]]
        ddC2bin_h[[0,-1]] = min(lower3-histbin,ddC2tp_i.min()), max(upper3+histbin,ddC2tp_i.max())
        ddC2hist_tph[t,p] = np.histogram(ddC2tp_i,bins=ddC2bin_h)[0]
        histsum_tp[t,p] = ddC2hist_tph[t,p,1:-1].sum()
        histmax_tp[t,p] = ddC2hist_tph[t,p,1:-1].max()
        edgeheight_d = np.array([ddC2hist_tph[t,p,1]*(ddC2hist_tph[t,p][:2]>0).all(),    \
                                ddC2hist_tph[t,p,-2]*(ddC2hist_tph[t,p][-2:]>0).all()])
        for d in np.where(edgeheight_d > 0)[0]:     # there are occupied bins outside quartile lim 
            ddC2edge = ddC2hist_tph[t,p][1:histentries-1][::[1,-1][d]][:3]                               
            imax = np.argmax(ddC2edge[:-1])
            hmax = [imax+1,histentries-2-imax][d]            
            h1,h2 = [(1,hmax+1),(hmax-1,histentries-2)][d]
            R_i = np.where(isgoodprof2_ftpR[:,t,p])[1]          
            R_j = R_i[np.where((ddC2tp_i > ddC2bin_h[h1]) & \
                               (ddC2tp_i < ddC2bin_h[h2+1]))[0]] 
            edgebin_dtp[d,t,p] = hmax                               
            edgeheight_dtp[d,t,p] = ddC2hist_tph[t,p,hmax]                                                             
            Redgelim_dxtp[d,:,t,p,] = R_j.min(),np.median(R_j),R_j.max()
          # if edge bin is at Row extreme, kill all rows out to extreme (likely crowdung)
            if ((R_j.min() == Rdatalim_dtp[0,t,p]) | (R_j.max() == Rdatalim_dtp[1,t,p])):
                if (R_j.min() == Rdatalim_dtp[0,t,p]):
                    isgoodprof3_ftpR[:,t,p,R_j.min():(R_j[R_j<Rows/2]).max()+1] = False
                if (R_j.max() == Rdatalim_dtp[1,t,p]):
                    isgoodprof3_ftpR[:,t,p,(R_j[R_j>Rows/2]).min():R_j.max()+1] = False
          # otherwise, if they are on blue side, just kill those rows
            elif ((np.median(R_j) > Rows/2)==bool(p)):
                isbadRow_R = np.in1d(range(Rows),R_j)
                isgoodprof3_ftpR[:,t,p,isbadRow_R] = False

    edgeculls3_tp = (isgoodprof3start_ftpR - isgoodprof3_ftpR).sum(axis=(0,3))                   
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
        np.savetxt("Cmap_tpR.txt", np.vstack((np.indices((ids,2)).reshape((2,-1)), \
            Cmap_tpR.reshape((-1,Rows)).T)).T, fmt=2*"%4i "+Rows*"%8.3f ")              
        np.savetxt("C0_ftpR.txt", np.vstack((np.where(fitprof_ftpR), isgoodprof2_ftpR[fitprof_ftpR], \
            isgoodprof3_ftpR[fitprof_ftpR], C0_ftpR[fitprof_ftpR], dC_ftpR[fitprof_ftpR])).T, \
            fmt=6*"%4i "+2*"%8.3f ")
        np.savetxt("dC2_tpf.txt",np.vstack((np.indices((ids,2)).reshape((2,-1)),  \
            dC2_ftp.transpose((1,2,0)).reshape((-1,files)).T)).T,fmt=2*"%3i "+files*"%8.3f ")         
        np.savetxt("Ridxsmax_tpf.txt", np.vstack((np.indices((ids,2)).reshape((2,-1)),  \
            Ridxsmax_ftp.transpose((1,2,0)).reshape((-1,files)).T)).T,fmt="%3i")
        np.savetxt("contRows_tpf.txt", np.vstack((np.indices((ids,2)).reshape((2,-1)),  \
            contRows_ftp.transpose((1,2,0)).reshape((-1,files)).T)).T,fmt="%3i") 
        np.savetxt("ddC2hist_tph.txt",ddC2hist_tph.reshape((2*ids,-1)),fmt="%4i")
        np.savetxt("ddC2histstats.txt",np.vstack((np.indices((ids,2)).reshape((2,-1)),  \
            Rdatalim_dtp.reshape((2,-1)),histsum_tp.flatten(),histmax_tp.flatten(), \
            quartileculls3_tp.flatten(),edgeculls3_tp.flatten(),edgebin_dtp.reshape((2,-1)),    \
            edgeheight_dtp.reshape((2,-1)),Redgelim_dxtp.reshape((6,-1)))).T,fmt="%4i ")        

  # best targets require good profs in at least Rows/3 continuous rows (file median), both O,E
  #  eliminate targets with >2 files with less than Rows/4 continuous

    isbest_t = (oktarget_t & (np.median(contRows_ftp, axis=0) > Rows/3).all(axis=1))
    isbest_t &= ((contRows_ftp < Rows/4).sum(axis=0) < 3).all(axis=1)
    tbest_b = np.where(isbest_t)[0]
    if debug: 
        besttargs = isbest_t.sum()
        print (("%2i Rows > %2i: "+besttargs*"%3i ") % ((besttargs,Rows/3,)+tuple(tbest_b)))

  # best targets also require good prof at max signal, all files, both O,E
    bestprof_ftpR = isgoodprof3_ftpR*(4./3.)*(fmax_ftpR*sigma_ftpR)  
    Ridxsmax_ftp = np.argmax(bestprof_ftpR,axis=3)      
    isgoodmax_ftp = isgoodprof3_ftpR.reshape((-1,Rows))  \
        [range(files*ids*2),Ridxsmax_ftp.flatten()].reshape((files,ids,2)) 
    isbest_t &= isgoodmax_ftp.all(axis=(0,2)) 
    tbest_b = np.where(isbest_t)[0]  
    if debug:
        besttargs = isbest_t.sum()
        print (("%2i good max:  "+besttargs*"%3i ") % ((besttargs,)+tuple(tbest_b)))

  # best targets require good fit to signal around max signal, both O,E
    Rsmax_ftp = np.zeros((files,ids,2))
    maxfiterr_ftp = np.zeros((files,ids,2))
    maxptp_ftp = np.zeros((files,ids,2))    
    dRmax = [5,4,3,2][rcbin_d[0]-1]
    for t in tbest_b:
        for f in range(files):
            for dR in range(-dRmax,dRmax+1):
                isgoodmax_ftp[f,t] &= isgoodprof3_ftpR[f,t][range(2),Ridxsmax_ftp[f,t]+dR] 
        if  (not isgoodmax_ftp[:,t].all()): continue
        for f,p in np.ndindex((files,2)):           
            Redge_d = np.where(isgoodprof3_ftpR[f,t,p])[0][[0,-1]]
            RmaxList = range(max(Redge_d[0],Ridxsmax_ftp[f,t,p]-dRmax), \
                min(Redge_d[1],Ridxsmax_ftp[f,t,p]+dRmax)+1)
            cof_x,residuals = np.polyfit(RmaxList,bestprof_ftpR[f,t,p,RmaxList],2,full=True)[:2]
            Rsmax_ftp[f,t,p] = -0.5*cof_x[1]/cof_x[0]
            maxfiterr_ftp[f,t,p] = np.sqrt(residuals[0]/len(RmaxList))
            maxptp_ftp[f,t,p] = np.ptp(bestprof_ftpR[f,t,p,RmaxList])
            isgoodmax_ftp[f,t,p] &= ((Rsmax_ftp[f,t,p] > RmaxList[0]) &     \
                (Rsmax_ftp[f,t,p] < RmaxList[-1]))
                                
    isbest_t &= isgoodmax_ftp.all(axis=(0,2)) 
    tbest_b = np.where(isbest_t)[0]        
    besttargs = isbest_t.sum()

    rsslog.message(("%2i best ids:  "+besttargs*"%3i ") % ((besttargs,)+tuple(tbest_b)),logfile)

    isgoodprof_fbpR = isgoodprof3_ftpR[:,isbest_t,:,:]       
    bestprof_fbpR = isgoodprof_fbpR*(4./3.)*(fmax_ftpR*sigma_ftpR)[:,tbest_b]
    bestvar_fbpR = bestprof_fbpR + isgoodprof_fbpR*bkg_ftpRC[:,isbest_t].mean(axis=4)
    Cmap_bpR = Cmap_tpR[isbest_t,:,:]    
    wavmap_bpR = wavmap_tpR[isbest_t,:,:]
    dwavmap_bpR = dwavmap_tpR[isbest_t,:,:] 
 
    if debug:
        np.savetxt("max_ftp.txt",np.vstack((np.indices((files,ids,2)).reshape((3,-1)),  \
            isgoodmax_ftp.flatten(), Ridxsmax_ftp.flatten(), Rsmax_ftp.flatten(),   \
            maxfiterr_ftp.flatten(), maxptp_ftp.flatten())).T,fmt=5*"%4i "+"%8.2f %9.1f %9.0f")           
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
        np.savetxt("tbest_b.txt",np.vstack((range(besttargs),tbest_b)).T,fmt="%3i ")     
        np.savetxt("isgoodprof3_tpRf.txt", np.vstack((np.indices((ids,2,Rows)).reshape((3,-1)),  \
            isgoodprof3_ftpR.transpose((1,2,3,0)).reshape((-1,files)).astype(int).T)).T,fmt="%3i")
        np.savetxt("isgoodmax_tpf.txt",np.vstack((np.indices((ids,2)).reshape((2,-1)),    \
            isgoodmax_ftp.transpose((1,2,0)).reshape((-1,files)).T)).T,fmt=2*"%4i "+files*"%3i ") 

  # for besttargs compute column motion relative to calfil reference
    dC_fbpR = dC_ftpR[:,isbest_t]
    isgoodprof0_fbpR = (isgoodprof_fbpR & isgoodprof_fbpR[0])
    dC_fbp = np.ma.median(np.ma.masked_array(dC_fbpR, mask=~isgoodprof0_fbpR),axis=3).data
    sigma_fbp = np.ma.median(np.ma.masked_array(sigma_ftpR[:,isbest_t], mask=~isgoodprof_fbpR),axis=3).data
    sigmabmed_fp = np.median(sigma_fbp,axis=1)

  # for besttargs compute relative row motion comparing difference spectrum with impol reference file 
  #   impol reference file is the image number closest to a cal reference file number
    fminblue = 0.50                                # only use blue rows > 0.5*smax
    fminred = 0.20                                 # only use red rows > 0.2*smax or min+0.05    

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
            Redge_d = np.array([dR0,dR1]) + Ridxsmax_fbp[f,b,p]  
                                                  # now extend red side
            min_d = np.array([f_R[:Redge_d[0]].min(),f_R[(Redge_d[1]+1):].min()])
            fminredp = max(fminred, min_d[1-p]+.05)
            ok_R = (isgoodprof_fbpR[f,b,p] & (f_R > fminredp))            
            if (p==0):                                  # O beam: red is on right (d=1)
                dR = (list(np.where(~ok_R[Redge_d[1]:-1])[0])+[Rows-Redge_d[1]])[0] -1
            else:
                dR = -(list(np.where(~ok_R[Redge_d[0]::-1])[0])+[Redge_d[0]])[0] +1
            Redge_d[1-p] = Redge_d[1-p] + dR          
            Redge_dfbp[:,f,b,p] = Redge_d
   
        diffRows_bp[b,p] = (Redge_dfbp[1,:,b,p]-Redge_dfbp[0,:,b,p] +1).min(axis=0).clip(0,Rows/3)
     
        for f in range(files):                      # compute diff, mean diff for target by smoothed spline
            f_R = bestprof_fbpR[f,b,p,:]/smax_fbp[f,b,p]        
            clipRows = Redge_dfbp[1,f,b,p]-Redge_dfbp[0,f,b,p] +1 - diffRows_bp[b,p]
                                                    # clip down to diffRows with best centering on max
            dRmax = Redge_dfbp[:,f,b,p].mean() - Ridxsmax_fbp[f,b,p]
            dR0 = np.clip(int([np.ceil,np.floor][p](clipRows/2. - dRmax)),0,    
                np.diff(Redge_dfbp[:,f,b,p])[0]+1-diffRows_bp[b,p])
            R0 = Redge_dfbp[0,f,b,p] + dR0
            R1 = R0 + diffRows_bp[b,p] - 1                                    
            badprofcount_fbp[f,b,p] = (~isgoodprof_fbpR[f,b,p,R0:(R1+1)]).sum()  
            diff_r = f_R[R0:(R1-3)] - f_R[(R0+4):(R1+1)] 
            r0idx = max((list(np.where(diff_r > 0.)[0])+[len(diff_r)-1])[0], 1)
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
            
        rx0 = int(np.floor(diff_Xd.min(axis=0)[0]+.001))+1   # avoid having knot fall right on max or min 
        rx1 = int(np.floor(diff_Xd.max(axis=0)[0]-.001))                 
        Xsort_X = np.argsort(diff_Xd[:,0])          # spline requires sorted x's 
               
        meandiffSpline = LSQUnivariateSpline(diff_Xd[Xsort_X,0],diff_Xd[Xsort_X,1],np.arange(rx0,rx1+1.))
        r_x = np.arange(rx0,rx1+0.25,0.25)          # oversample spline 4x for reverse interpolation            
        meandiff_x = meandiffSpline(r_x)        
        meandiff_xList_bp[b,p] = list(meandiff_x)
        knots = len(meandiff_x)
        rx01_dpb[:,p,b] = rx0,rx1                   # for debug 
        meandifferr_bp[b,p] =  np.sqrt(meandiffSpline.get_residual()/len(Xsort_X))
           
      # only use meandiff around zero cross with x slope > xdifflim (stop when not)
        xdifflim = 0.01
        badslope_x = (np.diff(meandiff_x) < xdifflim)     # note: the diff makes badslope length knots-1
        xidxzero = (list(np.where((meandiff_x[:-1] > 0.) & ~badslope_x)[0])+[len(meandiff_x)])[0]-1
        x0 = xidxzero - ((list(np.where(badslope_x[xidxzero::-1])[0])+[xidxzero+1])[0]-1)
        x1 = xidxzero + (list(np.where(badslope_x[xidxzero:])[0] +1 )+[knots-xidxzero])[0]-1 

        if debug:
            if ((b==db) & (p==dp)):
                np.savetxt('differr_'+str(b)+'_'+str(p)+'_X.txt',np.vstack((f_X[Xsort_X],   \
                    diff_Xd[Xsort_X,0], diff_Xd[Xsort_X,1],meandiffSpline(diff_Xd[Xsort_X,0]))).T,  \
                    fmt="%3i "+3*"%9.4f ")

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
    comperrlim = 1.5*np.median(meandifferr_bp/meandiffampl_bp)          # limit Rcomp guiding targets
    okcomp_b = (okcomp_fbp.all(axis=(0,2)) & (meandifferr_bp/meandiffampl_bp < comperrlim).all(axis=1))
    compids = okcomp_b.sum()

    rsslog.message((("%2i Rcomp ids: "+compids*"%3i ") % ((compids,)+tuple(tbest_b[okcomp_b]))),logfile)
         
    dRcomp_fbp = dRcomp_fbp - dRcomp_fbp[fref,:,:]
 
    if debug:
        np.savetxt('sigma_fbp.txt',np.vstack((sigma_fbp.reshape((files,-1)).T,sigmabmed_fp.T)),fmt="%8.3f ") 
        np.savetxt('dRcomp_fpb.txt',np.vstack((dRcomp_fbp.transpose((0,2,1)).reshape((files,-1)), \
            meandifferr_bp.T.flatten(),meandiffampl_bp.T.flatten())).T,fmt="%9.4f")               
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
    dYX_df = np.zeros((2,files))   
    drot_f = np.zeros(files)
    dyxOEoff_df = np.zeros((2,files)) 
    dYXerr_df = np.zeros((2,files))   
    droterr_f = np.zeros(files)
    dyxOEofferr_df = np.zeros((2,files))
    wavmap_fbpR = np.zeros((files,besttargs,2,Rows),dtype='float32')
    dwavmap_fbpR = np.zeros_like(wavmap_fbpR)
    Cmap_fbpR = np.zeros_like(wavmap_fbpR)         
    yxfref_dpb = yxcat_dpt[:,:,tbest_b]
    dyxp_dpb = dyxp_dpt[:,:,tbest_b]
                     
    for f in range(files):
        if (f==fref): continue    
      # fit okcomp dRcomp_fpb, (dC_fpb-dC_fbp[fref]) to get relative dYX_df,drot_f,dyxEOoff_df
        yxf_dpb = yxfref_dpb +   \
            rcbin_d[:,None,None]*pixmm*np.array([dRcomp_fbp[f].T,(dC_fbp[f]-dC_fbp[fref]).T])
        dYX_df[:,f],drot_f[f],dyxOEoff_df[:,f],dYXerr_df[:,f],droterr_f[f],dyxOEofferr_df[:,f] =  \
            impolguide(YX_db[:,okcomp_b],yxf_dpb[:,:,okcomp_b],yxOEoff_d, \
            imwav,coltem,camtem,fitOEoff=False,debug=debug,name=name)
        YXf_db = YX_db + dYX_df[:,f,None] + np.array([1.,-1.])[:,None]*np.radians(drot_f[f])*YX_db[::-1]   
        YXf_ds = np.repeat(YXf_db,Wavs,axis=1)
        wav_s = np.tile(wav_W,besttargs)            
        yxcatf_dpbW = RSScolpolcam(YXf_ds,wav_s,coltem,camtem,  \
            yxOEoff_d=yxOEoff_d+dyxOEoff_df[:,f]).reshape((2,2,besttargs,Wavs)) 
        wavmap_fbpR[f],dwavmap_fbpR[f],Cmap_fbpR[f] =   \
            specmap(yxcatf_dpbW + dyxp_dpb[:,:,:,None],wav_W,   \
            ri_tpR[tbest_b],ci_tpC[tbest_b],dWdY_ptW[:,tbest_b],hdul0)             

    if debug:  
        np.savetxt("relguiderr_f.txt",np.vstack((imgno_f,dYX_df,dYXerr_df,drot_f,  \
            droterr_f,dyxOEoff_df,dyxOEofferr_df)).T,fmt="%3i "+6*"%8.4f "+4*"%8.5f ")
            
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
    YXfref_db = YX_db +  np.array([dYfref_b,np.zeros(besttargs)])   # first put in dY_db     
    yxfref_dpb =   \
        RSScolpolcam(YXfref_db,imwav,coltem,camtem,yxOEoff_d=yxOEoff_d).reshape((2,2,besttargs))        
    yxfref_dpb[1] += rcbin_d[1]*pixmm*dC_fbp[fref].T                # now put in dx_dpb
    dYXfref_d,drotfref,dyxOEofffref_d,dYXerr_df[:,fref],droterr_f[fref],dyxOEofferr_df[:,fref] =  \
        impolguide(YX_db[:,okcomp_b],yxfref_dpb[:,:,okcomp_b],yxOEoff_d,\
        imwav,coltem,camtem,fitOEoff=False,debug=debug,name=name)           
            
  # add in reference guide error so dYX, drot, dyxOEoff is relative to cal filter 
    dYX_df += dYXfref_d[:,None]
    drot_f += drotfref
    dyxOEoff_df += dyxOEofffref_d[:,None]
    yxOEoff_df = yxOEoff_d[:,None] + dyxOEoff_df
    
  # recompute wavmap, cmap for each file and all targets with guider corrected YX, rot, OEflex
    wavmap_ftpR = np.zeros((files,ids,2,Rows),dtype='float32') 
    dwavmap_ftpR = np.zeros_like(wavmap_ftpR)      
    Cmap_ftpR = np.zeros_like(wavmap_ftpR)
    yxcat_dfpt = np.zeros((2,files,2,ids))
    for f in range(files):
        YXf_dt = YX_dt + dYX_df[:,f,None] + np.array([1.,-1.])[:,None]*np.radians(drot_f[f])*YX_dt[::-1]   
        YXf_ds = np.repeat(YXf_dt,Wavs,axis=1)
        wav_s = np.tile(wav_W,ids)             
        yxcatf_dptW =   \
            RSScolpolcam(YXf_ds,wav_s,coltem,camtem,yxOEoff_d=yxOEoff_df[:,f]).reshape((2,2,ids,Wavs))
        yxcat_dfpt[:,f] = yxcatf_dptW[:,:,:,Wimref]
        wavmap_ftpR[f],dwavmap_ftpR[f],Cmap_ftpR[f] =   \
            specmap(yxcatf_dptW + dyxp_dpt[:,:,:,None],wav_W,ri_tpR,ci_tpC,dWdY_ptW,hdul0)          

    wavmap_ftpR[:,cliprow_tpR] = 0.
    yxcat_dfpb = yxcat_dfpt[:,:,:,isbest_t]         # for debug

  # fit dC vs R for each t,p using all files and corrected Cmap
    dC_ftpR[fitprof_ftpR] = (C0_ftpR - Cmap_ftpR)[fitprof_ftpR]
    minRows = 4                                 # want at least minRows, if stack all files
    dR_fR = np.tile(np.arange(-Rows/2,Rows/2),(files,1))    
    dCcof_dtp = np.zeros((2,ids,2))
    dCrms_tp = np.zeros((ids,2))
    dCfit_tpR = np.zeros((ids,2,Rows))

  # recull goodprofs to be within outer quartile limits (all samples) to fit.  Iterate.
    isgoodprof4_ftpR = np.copy(isgoodprof3_ftpR)
    for iter in range(4):     
        for t,p in np.ndindex((ids,2)):    
            if ((isgoodprof3_ftpR[:,t,p].sum(axis=0) > 0).sum() < minRows): continue
            dCtp_i = dC_ftpR[:,t,p][isgoodprof4_ftpR[:,t,p]]
            dRtp_i = dR_fR[isgoodprof4_ftpR[:,t,p]]        
            dCcof_dtp[:,t,p],dCrmstp = np.polyfit(dRtp_i,dCtp_i,1,full=True)[0:2] 
            dCrms_tp[t,p] = np.sqrt(dCrmstp[0]/isgoodprof4_ftpR[:,t,p].sum())  # sumsqerr is 1D array                        
            dCfit_tpR[t,p] = np.polyval(dCcof_dtp[:,t,p],dR_fR[0])
        ddC4_ftpR = np.zeros((files,ids,2,Rows))
        ddC4_ftpR = (dC_ftpR - dCfit_tpR[None,:,:,:])   
        q1,q3 = np.percentile(ddC4_ftpR[isgoodprof4_ftpR],(25,75))        
        lower4,upper4 = (q1 - 3.*(q3-q1)),(q3 + 3.*(q3-q1)) 
           
        isgoodprof4_ftpR = isgoodprof3_ftpR & ((ddC4_ftpR >= lower4) & (ddC4_ftpR < upper4))
                       
    ddC3hist_tph = np.zeros((ids,2,histentries)).astype(int)    
    histbin = (upper4-lower4)/(histentries - 2)
    ddC3bin_h = np.linspace(lower4-histbin,upper4+histbin,histentries+1)  
    ddC3_ftpR = np.zeros((files,ids,2,Rows))
    ddC3_ftpR[isgoodprof3_ftpR] = (dC_ftpR - dCfit_tpR[None,:,:,:])[isgoodprof3_ftpR]           
    for t,p in np.ndindex((ids,2)):             # for debug reporting
        if (dCrms_tp[t,p]==0): continue
        ddC3tp_i = ddC3_ftpR[:,t,p][isgoodprof3_ftpR[:,t,p]]        
        ddC3bin_h[[0,-1]] = min(lower4-histbin,ddC3tp_i.min()), max(upper4+histbin,ddC3tp_i.max())                               
        ddC3hist_tph[t,p] = np.histogram(ddC3tp_i,bins=ddC3bin_h)[0]

    ddCculls_ftp = (isgoodprof3_ftpR & (~isgoodprof4_ftpR)).sum(axis=3)
    goodRows_ftp = isgoodprof4_ftpR.sum(axis=3)
    okgoodprof_t = (isgoodprof4_ftpR.sum(axis=3) > minRows).all(axis=(0,2))    
    profcullList = list(np.where(~okgoodprof_t & oktarget_t)[0])    
    if len(profcullList): rsslog.message((("%3i ids goodprof culled: "+len(profcullList)*"%2i ") %     \
                tuple([len(profcullList),]+profcullList)),logfile)
    oktarget_t &= okgoodprof_t

  # evaluate seeing sigma for all goodprof profiles   
    sigma_f = np.ma.median(np.ma.masked_array(sigma_ftpR,mask=~isgoodprof4_ftpR),axis=(1,2,3)).data
   
    if debug:
        dydY_pb = dydY_ptW[:,tbest_b,Wimref]
        np.savetxt("dYfref_fb.txt",np.vstack((np.indices((files,besttargs)).reshape((2,-1)), \
            wavsmax_fbp.reshape((-1,2)).T,dwavsmax_fbp.reshape((-1,2)).T,Rsmax_fbp.reshape((-1,2)).T,    \
            dYfref_fb.flatten(),np.tile(dydY_pb,files))).T, fmt="%3i %3i "+6*"%8.2f "+3*"%8.5f ")          
        np.savetxt("guiderr_f.txt",np.vstack((imgno_f,dYX_df,dYXerr_df,drot_f,  \
            droterr_f,dyxOEoff_df,dyxOEofferr_df,sigma_f)).T,fmt="%3i "+6*"%8.4f "+4*"%8.5f "+"%6.3f")
        ddC4_ftpR *= isgoodprof4_ftpR                                             
        np.savetxt("dC_ptRf.txt", np.vstack((np.indices((2,ids,Rows)).reshape((3,-1)),   \
            isgoodprof2_ftpR.astype(int).transpose((2,1,3,0)).reshape((-1,files)).T,    \
            isgoodprof3_ftpR.astype(int).transpose((2,1,3,0)).reshape((-1,files)).T, \
            isgoodprof4_ftpR.astype(int).transpose((2,1,3,0)).reshape((-1,files)).T, \
            dC_ftpR.transpose((2,1,3,0)).reshape((-1,files)).T,     \
            ddC4_ftpR.transpose((2,1,3,0)).reshape((-1,files)).T)).T, \
            fmt=3*"%3i "+3*("  "+files*"%1i ")+2*(" "+files*"%8.3f "))
        np.savetxt("dCfit_ptR.txt", np.vstack((np.indices((2,ids,Rows)).reshape((3,-1)),   \
            dCfit_tpR.transpose((1,0,2)).flatten())).T,fmt=3*"%3i "+" %8.3f ")             
        print ("outer ddC3 limits: %8.3f %8.3f" % (lower3, upper3))
        print ("outer ddC4 limits: %8.3f %8.3f" % (lower4, upper4))        
        np.savetxt("ddC_ftp.txt", np.vstack((np.indices((files,ids,2)).reshape((3,-1)), \
            goodRows_ftp.flatten(), ddCculls_ftp.flatten())).T, fmt="%3i ")
        np.savetxt("ddCfit_tp.txt", np.vstack((np.indices((ids,2)).reshape((2,-1)), \
            dCcof_dtp.reshape((2,-1)), dCrms_tp.flatten())).T, fmt=2*"%3i "+"%9.4f %8.3f %8.3f ")              
        np.savetxt("ddC3hist_tph.txt",ddC3hist_tph.reshape((2*ids,-1)),fmt="%4i")
        np.savetxt("wavmap_f_"+str(dt)+"_pR.txt",np.vstack((np.indices((2,files)).reshape((2,-1)), \
            wavmap_ftpR[:,dt].transpose((1,0,2)).reshape((-1,Rows)).T)).T,fmt="%3i %3i "+Rows*"%9.3f ") 

  # compute moffat column profiles for all ids, using Moffat C0 fit, not Cmap
    x_iC = (np.arange(Cols)-C0_ftpR[:,:,:,:,None]).reshape((-1,Cols))
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
    dRdY_p = dydY_ptW[:,:,Wimref].mean(axis=1)/(rcbin_d[0]*pixmm)   # mean over FOV, sufficient for this correction
    dCdX = 0.5235/(rcbin_d[1]*pixmm)

    for f,p in np.ndindex((files,2)):
        fmaxf_tR = np.tile(sigma_f.mean()/sigma_f[f],(ids,Rows))
        seecornp_tRC = (fmaxf_tR[:,:,None]*moffat_ftpRC[f,:,p]/moffatmean_ftpRC[f,:,p])
        dRp = dRdY_p[p] * dYX_df[0,f]
        dCp = dC_ftpR[f,:,p][isgoodprof4_ftpR[f,:,p]].mean()
        crsignal_ftpRC[f,:,p] = shift(signal_ftpRC[f,:,p]/seecornp_tRC, (0,-dRp,-dCp),order=1)
        crvar_ftpRC[f,:,p] = shift(var_ftpRC[f,:,p]/seecornp_tRC**2, (0,-dRp,-dCp),order=1)
        okcr_ftpRC[f,:,p] = (shift(ok2_ftpRC[f,:,p].astype(int),(0,-dRp,-dCp),order=1) == 1) 

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
        ok_tpR = ((ok3_ftpRC[f].sum(axis=3) > 0) & isgoodprof4_ftpR[f])
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

    #   scrunch onto wavelength grid
        for t,p in np.ndindex((ids,2)):
            if ~ok3_t[t]: continue
            dirn = [1,-1][p]                      # scrunch needs increasing indices
            Rowuse = np.where(wavmap_ftpR[f,t,p] > 0)[0]
            wedgeuse = np.where((wedge_w >= wavmap_ftpR[f,t,p,Rowuse].min()) &    \
                              (wedge_w < wavmap_ftpR[f,t,p,Rowuse].max()))[0]
            wavout = wedgeuse[:-1]                # scrunch has input bins starting at bin, not bin-1/2: 
            Rbinedge_W = (interp1d(wavmap_ftpR[f,t,p,Rowuse][::dirn],(Rowuse[::dirn]+0.5),kind='cubic')    \
               (wedge_w[wedgeuse]))[::dirn] 
            oktp_w = (scrunch1d(ok_tpR[t,p].astype(int),Rbinedge_W)   \
               == (Rbinedge_W[1:] - Rbinedge_W[:-1]))[::dirn] 
            ok_ftpw[f,t,p,wavout] = oktp_w
           
            fopt_ftpw[f,t,p,wavout] = scrunch1d(fopt_tpR[t,p],Rbinedge_W)[::dirn]*oktp_w 
            vopt_W,covar_W = scrunchvar1d(vopt_tpR[t,p],Rbinedge_W)
            vopt_ftpw[f,t,p,wavout] = vopt_W[::dirn]*oktp_w
            covar_ftpw[f,t,p,wavout] = covar_W[::dirn]*oktp_w               
            if debug:
                if ((f == df) & (t == dt) & (p == dp)):
                    np.savetxt("Rbininterp"+dlblt,np.vstack((wavmap_ftpR[f,t,p,Rowuse][::dirn],    \
                        (Rowuse[::dirn]+0.5))).T,fmt="%9.5f %8.1f") 
                    np.savetxt("wavmap_R"+dlblt,wavmap_ftpR[f,t,p].T,fmt="%10.6f")     
                    np.savetxt("Rbinedge_W"+dlblt,Rbinedge_W.T,fmt="%10.6f")

        if debug:
            np.savetxt("wedge_w.txt",wedge_w.T,fmt="%8.3f")
            if (f == df):
                np.savetxt("fstdf_tpR_"+str(f)+"_"+str(dt)+".txt",fstd_tpR[dt].T,fmt="%10.2f")
                np.savetxt("normf_tpR_"+str(f)+"_"+str(dt)+".txt",norm_tpR[dt].T,fmt="%10.2e")
                np.savetxt("okf_tpR_"+str(f)+"_"+str(dt)+".txt",ok_tpR[dt].astype(int).T,fmt="%3i")
                np.savetxt("xtrwt_ftpRC_"+dlblt,xtrwt_ftpRC[f,dt,dp],fmt="%10.4f")
                np.savetxt("xtrnorm_ftpRC_"+dlblt,xtrnorm_ftpRC[f,dt,dp],fmt="%10.4f")
                np.savetxt("image_ftpRC_"+dlblt,image_ftpRC[f,dt,dp],fmt="%10.2f")
                np.savetxt("bkg_ftpRC_"+str(f)+"_"+str(dt)+"_"+str(dp)+".txt",bkg_ftpRC[f,dt,dp],fmt="%10.2f") 
                np.savetxt("normf_tpRC_"+str(f)+"_"+str(dt)+"_"+str(dp)+".txt",norm_tpRC[dt,dp],fmt="%10.2e")
                np.savetxt("foptf_tpRC_"+str(f)+"_"+str(dt)+"_"+str(dp)+".txt",fopt_tpRC[dt,dp],fmt="%10.4f")
                np.savetxt("voptf_tpRC_"+str(f)+"_"+str(dt)+"_"+str(dp)+".txt",vopt_tpRC[dt,dp],fmt="%10.2f")
                np.savetxt("foptf_tpR_"+str(f)+"_"+str(dt)+".txt",fopt_tpR[dt].T,fmt="%10.2f")
                np.savetxt("fopt_ftpw_"+str(f)+"_"+str(dt)+".txt",fopt_ftpw[f,dt].T,fmt="%10.2f")
                np.savetxt("voptf_tpR_"+str(f)+"_"+str(dt)+".txt",vopt_tpR[dt].T,fmt="%10.2f")
                np.savetxt("vopt_ftpw_"+str(f)+"_"+str(dt)+".txt",vopt_ftpw[f,dt].T,fmt="%10.2f")
                np.savetxt("covar_ftpw_"+str(f)+"_"+str(dt)+".txt",covar_ftpw[f,dt].T,fmt="%10.2f")                                
                np.savetxt("ok_ftpw_"+str(f)+"_"+str(dt)+".txt",ok_ftpw[f,dt].astype(int).T,fmt="%3i")
                
  # cull targets based on adequate O/E wavelength overlap match
    oematchwavs_t = ok_ftpw.all(axis=(0,2)).sum(axis=1)
    okoematch_t = (oematchwavs_t > 3)
    
    oecullList = list(np.where(~okoematch_t & oktarget_t)[0])
    if len(oecullList): rsslog.message((("%3i ids match culled: "+len(oecullList)*"%2i ") %     \
                tuple([len(oecullList),]+oecullList)),logfile)
    oktarget_t &= okoematch_t       

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
    ratcullList = list(np.where(~okoerat_t & oktarget_t)[0])
    if len(ratcullList): rsslog.message((("%3i ids ratio culled: "+len(ratcullList)*"%2i ") %     \
                tuple([len(ratcullList),]+ratcullList)),logfile)
    oktarget_t &= okoerat_t     
      
    id_T = np.where(oktarget_t)[0]
    Targets = id_T.shape[0]
    TargetmapTab = tgtTab[id_T]
    if debug:
        np.savetxt("id_T.txt",np.vstack((np.arange(Targets),id_T)).T,fmt="%3i")    
        np.savetxt("oewavs_t.txt",np.vstack((np.arange(ids),oematchwavs_t,oeratwavs_t)).T,fmt="%3i") 
        np.savetxt("OErat_w.txt", np.vstack((wav_w,OErat_w,OErat_tw)).T,fmt="%8.2f "+(ids+1)*"%8.4f ")
        TargetmapTab.write(name+"_TargetmapTab.txt",format='ascii.fixed_width',   \
                    bookend=False, delimiter=None, overwrite=True)
        np.savetxt("idcullsum_t.txt",np.vstack((range(ids),okfov_t,okovlap_t,oksat_t,okbkg_t,okoerat_t, \
                    okgoodprof_t,okoematch_t)).T,fmt="%3i ")  
            
    rsslog.message(("\n%2i Targets:  "+Targets*"%2i ") % ((Targets,)+tuple(np.arange(Targets))),logfile)
    rsslog.message(("from ids   : "+Targets*"%2i ") % tuple(id_T),logfile) 
        
    rsslog.message ("\n        Output file                dY (asec) dX     sigma   crs", logfile)
            
  # save the result
    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)
        outfile = 'e'+file
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
        dvert,dhor = dYX_df[:,f]*1000/saltfps
        rsslog.message ("%30s %8.3f %8.3f %8.3f  %4i" % (outfile,dvert,dhor,sigma_f[f],crs_f[f]), logfile)

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
