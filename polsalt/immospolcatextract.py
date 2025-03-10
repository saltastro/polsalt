
"""
immospolcatextract

Optimal extraction for MOS imaging polarimetric data, using catalog and filter images
Write out extracted data fits (etm*) dimensions wavelength,target #

"""

import os, sys, glob, shutil, inspect, pprint

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.linalg import lstsq
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import Latitude,Longitude,Angle

# this is pysalt-free

import rsslog
from obslog import create_obslog
from scrunch1d import scrunch1d, scrunchvar1d
from polutils import datedline, rssdtralign
from polmaptools import ccdcenter,boxsmooth1d,impolguide
from rssoptics import RSSpolgeom, RSScolpolcam
from immospolextract import specmap, moffat, moffat1dfit, findpair

datadir = os.path.dirname(__file__) + '/data/'
keywordfile = datadir+"obslog_config.json"
np.set_printoptions(threshold=np.nan)

import warnings 
# warnings.filterwarnings("error")

# ----------------------------------------------------------
def immospolcatextract(fileList,name,logfile='salt.log',debug=False):
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
    lampid = hdr0['LAMPID'].strip().upper()
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
    tgtname_t = np.array(tgtTab['CATID'])    
    RAd0 = Longitude(hdr0['RA0']+' hours').degree
    DECd0 = Latitude(hdr0['DEC0']+' degrees').degree
    PAd0 = hdr0['PA0']    
    fref = np.unravel_index(np.argmin(np.abs(imgno_f[:,None] - calimgno_F[None,:])),(files,calfilters))[0]
             
    rsslog.message(("Candidate ids:       %3i" % ids),logfile)
        
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
 
    Rfovlim = 50.0                                      # avoid effects at edge of SALT FOV 
    okfov_t = (np.sqrt((YX_dt**2).sum(axis=0)) < Rfovlim)
    fovcullList = list(np.where(~okfov_t)[0])
    if len(fovcullList): rsslog.message((("%3i ids FOV culled: "+len(fovcullList)*"%2i ") %     \
                tuple([len(fovcullList),]+fovcullList)),logfile)
                
  # form reference wav, dwav, and col maps
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
        df, dt, db, dp = 0,93,70,0
        dlblt = "_"+str(df)+"_"+str(dt)+"_"+str(dp)+".txt"
        dlblb = "_"+str(df)+"_"+str(db)+"_"+str(dp)+".txt"

    rckey_pd = np.array([['R0O','C0O'],['R0E','C0E']])
    Rows,Cols = np.array(hdul0[0].header['BOXRC'].split()).astype(int)
    ri_tpR = np.zeros((ids,2,Rows),dtype=int)
    ci_tpC = np.zeros((ids,2,Cols),dtype=int) 
    amp_Atp = np.zeros((2,ids,2),dtype=int)              # amplifier on A=left, right side of target box
  
    for p in (0,1):
        ri_tpR[:,p] = np.clip(tgtTab[rckey_pd[p,0]][:,None] + np.arange(Rows)[None,:], 0,prows-1)
        ci_tpC[:,p] = np.clip(tgtTab[rckey_pd[p,1]][:,None] + np.arange(Cols)[None,:], 0,cols-1)
        amp_Atp[0,:,p] = np.argmax((c0_a[:,None] >= ci_tpC[None,:,p,0]),axis=0)-1
        amp_Atp[1,:,p] = np.argmax((c0_a[:,None] >= ci_tpC[None,:,p,-1]),axis=0)-1            
                            
    wavmap_tpR,dwavmap_tpR,Cmap_tpR =   \
        specmap(yx_dptW,wav_W,ri_tpR,ci_tpC,dWdY_ptW,hdul0) 

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
    print "Assembling target data"
    image_fprc = np.zeros((files,2,prows,cols))
    var_fprc = np.zeros_like(image_fprc) 
    okbin_fprc =  np.zeros_like(image_fprc).astype(bool) 
    hsta_f = np.zeros(files,dtype=int) 
    qsta_f = np.zeros(files,dtype=int)       
    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)
        image_fprc[f] = hdul['SCI'].data
        var_fprc[f] = hdul['VAR'].data
        okbin_fprc[f] = (hdul['BPM'].data==0) 
        hsta_f[f] = int(round(float(hdul[0].header['HWP-ANG'])/11.25))
        qsta_f[f] = int(round(float(hdul[0].header['QWP-ANG'])/45.))                  
    okbin_prc = okbin_fprc.all(axis=0)
    image_prc = image_fprc.mean(axis=0)
    var_prc = var_fprc.mean(axis=0)/np.sqrt(files)

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
        np.savetxt("amp_Atp.txt",np.vstack((np.indices((ids,2)).reshape((2,-1)),    \
            amp_Atp.reshape((2,-1)))).T,fmt="%4i")

  # put good targets into boxes
    image_ftpRC = np.zeros((files,ids,2,Rows,Cols))
    var_ftpRC = np.zeros_like(image_ftpRC)
    bkg_ftpRC = np.zeros_like(image_ftpRC)    
    okbin_ftpRC = np.zeros((files,ids,2,Rows,Cols),dtype=bool)
    oktgt_ftpRC = np.zeros_like(okbin_ftpRC) 
    for f,t,p in np.ndindex(files,ids,2): 
        if ~(oktarget_t)[t]: continue    
        image_ftpRC[f,t,p] = image_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
        var_ftpRC[f,t,p] = var_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]] 
        okbin_ftpRC[f,t,p] = okbin_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
        oktgt_ftpRC[f,t,p] = (tgt_prc==t+1)[p][ri_tpR[t,p],:][:,ci_tpC[t,p]] 
        
  # process background 
  
  # use non-slit for lamp data
  # first find locations near each target to use for background, using mean over all images
  # process by amplifier
    if (lampid != 'NONE'): 
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

        bkg_ftpRC = np.zeros((files,ids,2,Rows,Cols))
        okbkg_ftpR = np.ones((files,ids,2,Rows),dtype=bool)
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
            for t in range(ids):
                bkg_ftpRC[f,t,p] = bkg_rc[ri_tpR[t,p],:][:,ci_tpC[t,p]]
            if (debug): 
                if ((f==df)&(p==dp)):
                    hdul['SCI'].data = bkg_rc
                    del(hdul['VAR'])
                    del(hdul['BPM'])
                    del(hdul['TMAP'])                                                            
                    hdul.writeto(name+'_'+str(f)+'_'+str(p)+'_fitbkg.fits',overwrite=True)                                                             

  # use slit for sky data    
    else:
        print "Processing sky background"

    signal_ftpRC = image_ftpRC - bkg_ftpRC 

  # add up files for profile analysis
    print "Do profile analysis"
    signal_tpRC = signal_ftpRC.sum(axis=0)
    bkg_tpRC = bkg_ftpRC.sum(axis=0)
    okbin_tpRC = okbin_ftpRC.all(axis=0)   
    ok1_tpRC = (okbin_ftpRC & oktgt_ftpRC & okbkg_ftpR[:,:,:,:,None]).all(axis=0) 
    
  # fit mean column profiles for each target to 1D Moffat, using rows with all columns    
    cols_tpR = (ok1_tpRC & (signal_tpRC>0)).sum(axis=3)
    prof_tpRC = ok1_tpRC*signal_tpRC
    prof_tpRC[cols_tpR>0,:] /= cols_tpR[cols_tpR>0,None]
    norm_tpR = prof_tpRC.max(axis=3)    
    prof_tpRC[cols_tpR>0,:] /= norm_tpR[cols_tpR>0,None]    
    fitprof_tpR = ((cols_tpR == Cols) & (Cmap_tpR>0.))     
    fitprof_tpR[~(okfov_t & okovlap_t & okbkg_t)] = False

    i_tpR = np.zeros((ids,2,Rows),dtype=int)
    i_tpR[fitprof_tpR] = np.arange(fitprof_tpR.sum()) 
    if debug: 
       # idebug = i_tpR[dt,dp,28] 
        idebug=None 
        
    sigma_i, fCmax_i, C0_i, fiterr_iC, okprof_i =    \
        moffat1dfit(prof_tpRC[fitprof_tpR,:].reshape((-1,Cols)),    \
            np.ones((fitprof_tpR.sum(),Cols),dtype=bool),beta=2.5,idebug=idebug)
                    
    i_tpR = np.zeros((ids,2,Rows),dtype=int)
    i_tpR[fitprof_tpR] = np.arange(fitprof_tpR.sum()) 
    sigma_tpR = sigma_i[i_tpR]
    fCmax_tpR = fCmax_i[i_tpR]
    fmax_tpR = fCmax_i[i_tpR]*norm_tpR
    C0_tpR = C0_i[i_tpR]
    dC_tpR = np.zeros((ids,2,Rows)) 
    dC_tpR[fitprof_tpR] = (C0_tpR - Cmap_tpR)[fitprof_tpR]    
    rmserr_tpR = np.sqrt((fiterr_iC[i_tpR,:]**2).mean(axis=3))
    okprof_tpR = np.zeros((ids,2,Rows)).astype(bool)
    okprof_tpR[fitprof_tpR] = okprof_i

  # identify good profiles, best targets to use for wavcal
    goodprofrmsmax = 3.*np.median(rmserr_tpR)    
    isgoodprof1_tpR = (okprof_tpR & (rmserr_tpR < goodprofrmsmax))
   
  # goodprof also requires sigma not above outer quartile limits 
    outersigma_q = np.zeros(2) 
    q1,q3 = np.percentile(sigma_tpR[isgoodprof1_tpR],(25,75))
    outersigma_q = (q1 - 3.*(q3-q1)),(q3 + 3.*(q3-q1))
    isgoodprof2_tpR = (isgoodprof1_tpR & (sigma_tpR < outersigma_q[1]))
    
  # do a cut on ddC: column errors relative to target median
    okmedian_pR = np.array([(np.arange(Rows) > Rows/5),(np.arange(Rows) < 4*Rows/5)])    
    dC2_tp = np.ma.median(np.ma.masked_array(dC_tpR,  \
        mask=~(isgoodprof2_tpR & okmedian_pR)),axis=2).data 
    ddC2_tpR = dC_tpR - dC2_tp[:,:,None]
    q1,q3 = np.percentile(ddC2_tpR[isgoodprof2_tpR],(25,75))    
    lower3,upper3 = (q1 - 5.*(q3-q1)),(q3 + 5.*(q3-q1))
    isgoodprof3_tpR = (isgoodprof2_tpR & ((ddC2_tpR > lower3) & (ddC2_tpR < upper3)))  
    quartileculls3_tp = (isgoodprof2_tpR - isgoodprof3_tpR).sum(axis=2)
                  
    contRows_tp = np.zeros((ids,2))
    for t,p in np.ndindex((ids,2)):
        goodR = np.where(isgoodprof3_tpR[t,p])[0]
        contList = np.split(goodR,np.where(np.diff(goodR) != 1)[0]+1)
        contRows_tp[t,p] = max(map(len,contList))
    
    if debug:
        np.savetxt("goodprof_tpR.txt", np.vstack((np.indices((ids,2,Rows)).reshape((3,-1)),  \
            fitprof_tpR.flatten(),isgoodprof1_tpR.flatten(),isgoodprof2_tpR.flatten(),  \
            isgoodprof3_tpR.flatten())).T,fmt="%3i")  
        np.savetxt("fitprof_tpR.txt",np.vstack((np.indices((ids,2)).reshape((2,-1)),  \
            fitprof_tpR.astype(int).reshape((-1,Rows)).T)).T,fmt="%3i %3i  "+Rows*"%3i ") 
        np.savetxt("proffit.txt",np.vstack((np.where(fitprof_tpR), isgoodprof2_tpR[fitprof_tpR], \
            isgoodprof3_tpR[fitprof_tpR].astype(int),sigma_i,fCmax_i,dC_tpR[fitprof_tpR],   \
            rmserr_tpR[fitprof_tpR],fmax_tpR[fitprof_tpR])).T, fmt=5*"%4i "+4*"%9.3f "+"%9.0f ") 
        np.savetxt("Cmap_tpR.txt", np.vstack((np.indices((ids,2)).reshape((2,-1)), \
            Cmap_tpR.reshape((-1,Rows)).T)).T, fmt=2*"%4i "+Rows*"%8.3f ")
        np.savetxt("contRows_tp.txt", np.vstack((np.indices((ids,2)).reshape((2,-1)), \
            contRows_tp.flatten())).T, fmt="%4i")                                 

  # best targets require good profs in at least Rows/3 continuous rows, both O,E
  #   also eliminate targets with less than Rows/4 continuous
    print "Analyse image motion"
    isbest_t = (oktarget_t & (contRows_tp > Rows/3).all(axis=1))
    isbest_t &= ((contRows_tp < Rows/4) < 3).all(axis=1)
    tbest_b = np.where(isbest_t)[0]
    if debug: print (("Rows > %2i: "+len(tbest_b)*"%3i ") % ((Rows/3,)+tuple(tbest_b)))

  # best targets also require good prof at max signal, all files, both O,E 
    bestprof_tpR = isgoodprof3_tpR*(4./3.)*(fmax_tpR*sigma_tpR)  
    Ridxsmax_tp = np.argmax(bestprof_tpR,axis=2)      
    isgoodmax_tp = isgoodprof3_tpR.reshape((-1,Rows))  \
        [range(ids*2),Ridxsmax_tp.flatten()].reshape((ids,2)) 
    isbest_t &= isgoodmax_tp.all(axis=1) 
    tbest_b = np.where(isbest_t)[0]  
   
  # best targets require good fit to signal around max signal, both O,E
    Rsmax_tp = np.zeros((ids,2))
    maxfiterr_tp = np.zeros((ids,2))
    maxptp_tp = np.zeros((ids,2))    
    dRmax = [5,4,3,2][rcbin_d[0]-1]
    for t in tbest_b:
        for dR in range(-dRmax,dRmax+1):
            isgoodmax_tp[t] &= isgoodprof3_tpR[t][range(2),Ridxsmax_tp[t]+dR] 
        if  (not isgoodmax_tp[t].all()): continue
        for p in range(2):           
            Redge_d = np.where(isgoodprof3_tpR[t,p])[0][[0,-1]]
            RmaxList = range(max(Redge_d[0],Ridxsmax_tp[t,p]-dRmax),min(Redge_d[1],Ridxsmax_tp[t,p]+dRmax)+1)
            cof_x,residuals = np.polyfit(RmaxList,bestprof_tpR[t,p,RmaxList],2,full=True)[:2]
            Rsmax_tp[t,p] = -0.5*cof_x[1]/cof_x[0]
            maxfiterr_tp[t,p] = np.sqrt(residuals[0]/len(RmaxList))
            maxptp_tp[t,p] = np.ptp(bestprof_tpR[t,p,RmaxList])
            isgoodmax_tp[t,p] &= ((Rsmax_tp[t,p] > RmaxList[0]) & (Rsmax_tp[t,p] < RmaxList[-1]))
                                
    isbest_t &= isgoodmax_tp.all(axis=1) 
    tbest_b = np.where(isbest_t)[0]        
    besttargs = isbest_t.sum()

    rsslog.message(("best ids:  "+besttargs*"%3i " % tuple(tbest_b)),logfile)

    isgoodprof_bpR = isgoodprof3_tpR[isbest_t,:,:]       
    bestprof_bpR = isgoodprof_bpR*(4./3.)*(fmax_tpR*sigma_tpR)[tbest_b]
    bestvar_bpR = bestprof_bpR + isgoodprof_bpR*bkg_tpRC[isbest_t].mean(axis=3)
    Cmap_bpR = Cmap_tpR[isbest_t,:,:]    
    wavmap_bpR = wavmap_tpR[isbest_t,:,:]
    dwavmap_bpR = dwavmap_tpR[isbest_t,:,:] 
 
    if debug:
        np.savetxt("max_tp.txt",np.vstack((np.indices((ids,2)).reshape((2,-1)), isgoodmax_tp.flatten(), \
            Ridxsmax_tp.flatten(), Rsmax_tp.flatten(),maxfiterr_tp.flatten(),   \
            maxptp_tp.flatten())).T,fmt=4*"%4i "+"%8.2f %9.1f %9.0f")                
        rmserr_bpR = rmserr_tpR[isbest_t]
        sigma_bpR = sigma_tpR[isbest_t]
        dC_bpR = dC_tpR[isbest_t]
        C0_bpR = C0_tpR[isbest_t]
        okbkg_bpR = okbkg_tpR[isbest_t]       
        fitprof_bpR = fitprof_tpR[isbest_t]    
        np.savetxt("contRows_bp.txt",contRows_tp[isbest_t],fmt="%4i ") 
        np.savetxt("Cmap_bpR.txt",np.vstack((np.indices((besttargs,2)).reshape((2,-1)),   \
                    Cmap_bpR.reshape((-1,Rows)).T)).T,fmt="%4i %4i "+Rows*"%8.3f ")         
        np.savetxt("wavmap_bpR.txt",np.vstack((np.indices((besttargs,2)).reshape((2,-1)),   \
                    wavmap_bpR.reshape((-1,Rows)).T)).T,fmt="%4i %4i "+Rows*"%8.2f ") 
        np.savetxt("bestprof_bpR.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    bestprof_bpR.flatten())).T,fmt="%4i %4i %4i %8.1f ")    
        np.savetxt("rmserr_bpR.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    rmserr_bpR.flatten())).T,fmt="%4i %4i %4i %8.3f ")  
        np.savetxt("sigma_bpR.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    sigma_bpR.flatten())).T,fmt="%4i %4i %4i %10.3f ")
        np.savetxt("dC_bpR.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    dC_bpR.flatten())).T,fmt="%4i %4i %4i %10.3f ") 
        np.savetxt("C0_bpR.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    C0_bpR.flatten())).T,fmt="%4i %4i %4i %8.3f ")
        np.savetxt("okbkg_bpR.txt",np.vstack((np.indices((besttargs,2)).reshape((2,-1)),    \
                    okbkg_bpR.reshape((-1,Rows)).astype(int).T)).T, fmt="%4i %4i "+Rows*"%3i ")
        np.savetxt("fitprof_bpR.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    fitprof_bpR.flatten().astype(int))).T, fmt="%4i %4i %4i %3i ")
        np.savetxt("isgoodprof_bpR.txt",np.vstack((np.indices((besttargs,2,Rows)).reshape((3,-1)),    \
                    isgoodprof_bpR.flatten().astype(int).T)).T, fmt="%4i %4i %4i %3i ")
        np.savetxt("tbest_b.txt",np.vstack((range(besttargs),tbest_b)).T,fmt="%3i ")     

  # for besttargs compute column motion relative to calfil reference
    dC_bpR = dC_tpR[isbest_t]
    dC_bp = np.ma.median(np.ma.masked_array(dC_bpR, mask=~isgoodprof_bpR),axis=2).data
            
  # use O,E smax positions in impol file for besttargets to compute telescope fref guiding error dYfref_b 
  #   use nominal wavmap to compute smaxwav in O, E, then use dwavmap to compute dY that equalizes them
  #   This assumes no y BS flexure between cal reference and nearby impol reference file         
    wavsmax_bp = np.zeros((besttargs,2)) 
    dwavsmax_bp = np.zeros_like(wavsmax_bp)
    Rsmax_bp = Rsmax_tp[isbest_t] 
         
    for b,p in np.ndindex((besttargs,2)):
        Rowok_i = np.where(isgoodprof_bpR[b,p])[0]        
        wavsmax_bp[b,p] = interp1d(Rowok_i, wavmap_bpR[b,p,isgoodprof_bpR[b,p]],    \
            kind='cubic')(Rsmax_bp[b,p])
        dwavsmax_bp[b,p] = interp1d(Rowok_i, dwavmap_bpR[b,p,isgoodprof_bpR[b,p]],  \
            kind='cubic')(Rsmax_bp[b,p])
         
    dY_b =  (wavsmax_bp[:,0] - wavsmax_bp[:,1])/(dwavsmax_bp[:,0] - dwavsmax_bp[:,1])
    wavsmax_b = wavsmax_bp[:,0] - dY_b*dwavsmax_bp[:,0]
    dydY_bp = dydY_ptW[:,tbest_b,Wimref].T  
    YX_db = YX_dt[:,isbest_t] +  np.array([dY_b,np.zeros(besttargs)])   # first put in dY_db     
    yx_dpb =   \
        RSScolpolcam(YX_db,imwav,coltem,camtem,yxOEoff_d=yxOEoff_d).reshape((2,2,besttargs))        
    yx_dpb[1] += rcbin_d[1]*pixmm*dC_bp.T                # now put in dx_dpb
    dYX_d,drot,dyxOEoff_d,dYXerr_d,droterr,dyxOEofferr_d =  \
        impolguide(YX_dt[:,isbest_t],yx_dpb,yxOEoff_d,imwav,coltem,camtem,debug=debug,name=name)           
            
  # add in reference guide error so dYX, drot, dyxOEoff is relative to cal filter 
    yxOEoff_d = yxOEoff_d + dyxOEoff_d
    
  # recompute wavmap, cmap for all targets with corrected YX, rot, OEflex
    yxcat_dpt = np.zeros((2,2,ids))

    YX_dt = YX_dt + dYX_d[:,None] + np.array([1.,-1.])[:,None]*np.radians(drot)*YX_dt[::-1]   
    YX_ds = np.repeat(YX_dt,Wavs,axis=1)
    wav_s = np.tile(wav_W,ids)             
    yxcat_dptW =   \
            RSScolpolcam(YX_ds,wav_s,coltem,camtem,yxOEoff_d=yxOEoff_d).reshape((2,2,ids,Wavs))
    yxcat_dpt = yxcat_dptW[:,:,:,Wimref]
    wavmap_tpR,dwavmap_tpR,Cmap_tpR =   \
        specmap(yxcat_dptW + dyxp_dpt[:,:,:,None],wav_W,ri_tpR,ci_tpC,dWdY_ptW,hdul0)          
    yxcat_dpb = yxcat_dpt[:,:,isbest_t]         # for debug

  # fit dC vs R for each t,p and corrected Cmap
    dC_tpR[fitprof_tpR] = (C0_tpR - Cmap_tpR)[fitprof_tpR]
    minRows = 4                                 # want at least minRows, if stack all files
    dR_R = np.tile(np.arange(-Rows/2,Rows/2),1)    
    dCcof_dtp = np.zeros((2,ids,2))
    dCrms_tp = np.zeros((ids,2))
    dCfit_tpR = np.zeros((ids,2,Rows))           

  # evaluate seeing sigma for all goodprof profiles   
    sigma = np.ma.median(np.ma.masked_array(sigma_tpR,mask=~isgoodprof3_tpR))
   
    if debug:
        dydY_pb = dydY_ptW[:,tbest_b,Wimref]
        np.savetxt("dY_b.txt",np.vstack((dwavsmax_bp.T,  \
            Rsmax_bp.T, dY_b,dydY_pb)).T, fmt=4*"%8.2f "+3*"%8.5f ")                                                     
        np.savetxt("dCfit_ptR.txt", np.vstack((np.indices((2,ids,Rows)).reshape((3,-1)),   \
            dCfit_tpR.transpose((1,0,2)).flatten())).T,fmt=3*"%3i "+" %8.3f ")             
        print ("outer ddC3 limits: %8.3f %8.3f" % (lower3, upper3))     
        np.savetxt("ddCfit_tp.txt", np.vstack((np.indices((ids,2)).reshape((2,-1)), \
            dCcof_dtp.reshape((2,-1)), dCrms_tp.flatten())).T, fmt=2*"%3i "+"%9.4f %8.3f %8.3f ")              

  # compute moffat column profiles for all ids, using Moffat C0 fit, not Cmap
    x_iC = (np.arange(Cols)-C0_tpR[:,:,:,None]).reshape((-1,Cols))    
    moffat_tpRC = moffat(np.ones(ids*2*Rows),sigma*np.ones(ids*2*Rows),  \
        x_iC,beta=2.5).reshape((ids,2,Rows,Cols))
    fluxtot_tpR = moffat_tpRC.sum(axis=3)
        
  # find CR's, if there are enough cycles (at least 3) to make a median over cycles for each hw,qw position
    iscr_ftpRC = np.zeros((files,ids,2,Rows,Cols)).astype(bool)
    crsigma = 15.    

    for h,q in np.array(np.meshgrid(np.unique(hsta_f),np.unique(qsta_f))).T.reshape(-1,2):
        fileListhq = np.where((hsta_f==h)&(qsta_f==q))[0]
        usebinhq_tpRC = (okbin_ftpRC[fileListhq].sum(axis=0) > 2)             
        if (usebinhq_tpRC.sum() ==0): continue        
        usebinhq_ftpRC = (((hsta_f==h)&(qsta_f==q))[:,None,None,None,None] & usebinhq_tpRC[None,:,:,:,:])
        medsignalhq_tpRC = np.ma.median(np.ma.masked_array(signal_ftpRC,mask=~usebinhq_ftpRC),axis=0).data
        medvarhq_tpRC = np.ma.median(np.ma.masked_array(var_ftpRC,mask=~usebinhq_ftpRC),axis=0).data 
        medsignalhq_ftpRC = np.tile(medsignalhq_tpRC,(files,1,1,1,1))
        medvarhq_ftpRC = np.tile(medvarhq_tpRC,(files,1,1,1,1))      
        iscr_ftpRC[usebinhq_ftpRC] = ((signal_ftpRC[usebinhq_ftpRC] -   \
            medsignalhq_ftpRC[usebinh_ftpRC]) > crsigma*np.sqrt(medvarhq_ftpRC[usebinhq_ftpRC]))

    crs_f = iscr_ftpRC.sum(axis=(1,2,3,4))
    ok3_ftpRC= (ok1_tpRC[None,:,:,:,:] & ~iscr_ftpRC)
    ok3_t = (okfov_t & okovlap_t & okbkg_t)

  # do extraction in column direction using moffat fits as weights
    print "Do extraction"
    xtrwt_ftpRC = np.tile(moffat_tpRC,(files,1,1,1,1))
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

    for f in range(files):
        fstd_tpR = np.zeros((ids,2,Rows))
        vstd_tpR = np.zeros((ids,2,Rows))
        ok_tpR = ((ok3_ftpRC[f].sum(axis=3) > 0) & isgoodprof3_tpR)
        fstd_tpR[ok_tpR] = signal_ftpRC[f].sum(axis=3)[ok_tpR]/xtrwt_ftpRC[f].sum(axis=3)[ok_tpR]
        vstd_tpR[ok_tpR] = var_ftpRC[f].sum(axis=3)[ok_tpR] / xtrwt_ftpRC[f].sum(axis=3)[ok_tpR]**2
        vopt_tpRC = fstd_tpR[:,:,:,None]*xtrwt_ftpRC[f] +     \
                    bkg_ftpRC[f,:,:,Rows/2,Cols/2].clip(min=0.)[:,:,None,None]
        norm_tpRC = np.zeros((ids,2,Rows,Cols))                
        norm_tpRC[vopt_tpRC > 0.] =     \
                    (xtrwt_ftpRC[f][vopt_tpRC > 0.]**2/vopt_tpRC[vopt_tpRC > 0.])
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
            Rowuse = np.where(wavmap_tpR[t,p] > 0)[0]
            wedgeuse = np.where((wedge_w >= wavmap_tpR[t,p,Rowuse].min()) &    \
                              (wedge_w < wavmap_tpR[t,p,Rowuse].max()))[0]
            wavout = wedgeuse[:-1]                # scrunch has input bins starting at bin, not bin-1/2: 
            Rbinedge_W = (interp1d(wavmap_tpR[t,p,Rowuse][::dirn],(Rowuse[::dirn]+0.5))    \
               (wedge_w[wedgeuse]))[::dirn] 
            oktp_w = (scrunch1d(ok_tpR[t,p].astype(int),Rbinedge_W)   \
               == (Rbinedge_W[1:] - Rbinedge_W[:-1]))[::dirn] 
            ok_ftpw[f,t,p,wavout] = oktp_w
           
            fopt_ftpw[f,t,p,wavout] = scrunch1d(fopt_tpR[t,p],Rbinedge_W)[::dirn]*oktp_w 
            vopt_W,covar_W = scrunchvar1d(vopt_tpR[t,p],Rbinedge_W)
            vopt_ftpw[f,t,p,wavout] = vopt_W[::dirn]*oktp_w
            covar_ftpw[f,t,p,wavout] = covar_W[::dirn]*oktp_w               
            if debug:
                if ((f == df) & (t == dt)):      
                    np.savetxt("Rbinedge_W_"+str(f)+"_"+str(t)+"_"+str(p)+".txt",Rbinedge_W.T,fmt="%9.5f")

        if debug:
            if (f == df):
                np.savetxt("wedge_w.txt",wedge_w.T,fmt="%8.3f")
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
    id_T = np.where(oktarget_t)[0]
    Targets = id_T.shape[0]
    TargetmapTab = tgtTab[id_T]
    if debug:
        id_TFile = open("id_T.txt",'w')
        for T in range(Targets): print >> id_TFile, ("%3i %3i %s" % (T,id_T[T],tgtname_t[oktarget_t][T]))    
        np.savetxt("oewavs_t.txt",np.vstack((np.arange(ids),oematchwavs_t)).T,fmt="%3i") 
        TargetmapTab.write(name+"_TargetmapTab.txt",format='ascii.fixed_width',   \
                    bookend=False, delimiter=None, overwrite=True)
        np.savetxt("idcullsum_t.txt",np.vstack((range(ids),okfov_t,okovlap_t,oksat_t,okbkg_t, \
                    okoematch_t)).T,fmt="%3i ")  
            
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
        dvert,dhor = dYX_d*1000/saltfps
        rsslog.message ("%30s %8.3f %8.3f %8.3f  %4i" % (outfile,dvert,dhor,sigma,crs_f[f]), logfile)

    return

# ------------------------------------

