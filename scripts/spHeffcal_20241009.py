#! /usr/bin/env python2.7

"""
spHeffcal

Fit HeffCal Moon data  
  evaluate pupil illum dependence of HW_Calibration
  use polmapcor file from OOF images 

"""

import os, sys, glob, shutil, inspect, datetime, operator
import numpy as np

import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter("error")

from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy import linalg as la
from scipy.interpolate import LSQUnivariateSpline, interp1d
from scipy.ndimage.interpolation import shift
from zipfile import ZipFile
from json import load
polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

import rsslog
from obslog import create_obslog
from scrunch1d import scrunch1d, scrunchvar1d
from polutils import skypupillum, viewstokes, rotate2d
keywordfile = datadir+"obslog_config.json"
np.set_printoptions(threshold=np.nan) 

#---------------------------------------------------------------------------------------------
def spHeffcal(DirList, fitcofnameList, **kwargs):
    """Compute pupil illumination bin dependence of calibration

    Parameters
    ----------
    DirList [dir1,dir2,..] obs directories (eg 20230829/red,20230829/blue) 
    debug=: False (default)
            True (debug output)
    _D dirs
    _f file index in dir
    _d YX target dim
    _t all targets in mask    
    _g grang index for multi-grang obs
    _a PA index for multi-PA obs
    _o obs index 
    _v wavelength index of input files
    _w wavelength index for 50-Ang binned data (same as zemax fit binning) and fit
    _W wavelength calibration output grid index
    _V fit variable index
    _p (0,1) calibration file (P,PA, or Q,U)
    _S (unnormalized) stokes: (I,Q,U)

    """
    
    print (sys. version) 
    logfile= kwargs.pop('logfile','spHeffcal.log')            
    rsslog.history(logfile)
    rsslog.message(str(kwargs), logfile)

    pupoffcor = float(kwargs.pop('pupoffcor','1.'))       
    paramDict = load(open(datadir+'/skypupilparams.json'))
    primwidth = paramDict["primwidth"]
    sacpupilrad = primwidth/np.sqrt(3.)    
    calPA_D = kwargs.pop('calPA',[])
    useshuttercor =  kwargs.pop('useshuttercor',False) 
    centeronly =  kwargs.pop('centeronly',False)        
    polmapcor = kwargs.pop('polmapcor','None')
    debug = (kwargs.pop('debug','False') == 'True')
    logfile= kwargs.pop('logfile','spHeffcal.log')    

  # get polmapcor oofcor file, if specified, from data directory
#   if (polmapcor != 'None'):
#       file is zip of fits files, each with Pdev, PAdev 108^2 RC maps for et21b targets _t for one wavelength _x
#       vcor_xtpRC
#       x==0 is at wvplt perfect (6700), cor = 0 Possible: above this, use cy, cx.
#       add cof 'cor' which interpolates between x files (0==no cor, 1=first wav, ..) in make_PAPol
  
  # first get Dir-specific data
    basedir = "/d/pfis/khn/"
    Dirs = len(DirList)
    calPA = 90.
    if len(calPA_D)==0:
        calPA_D = np.repeat([calPA],Dirs)
    else:
        calPA_D = np.array(map(float,calPA_D.split(',')))
         
    files_D = np.zeros(Dirs,dtype=int)
    wavs_D = np.zeros(Dirs,dtype=int) 
    wav0_D = np.zeros(Dirs,dtype=int)
    maskid_D =  np.zeros(Dirs,dtype=str)
    PA_D = np.zeros(Dirs)                   # proposed PA    
    telPA_D = np.zeros(Dirs)                # actual PA, may be 180 different from PA
    grtilt_D = np.zeros(Dirs)                 

    f_DList = []   
    f0 = 0
    filenameList = []
                  
    for D,Dir in enumerate(DirList):    
        stokesList = sorted(glob.glob(basedir+DirList[D]+"/Moon-*rawstokes.fits"))
        filenameList += stokesList      
        files_D[D] = len(stokesList)
        f_DList.append(f0 + np.arange(files_D[D]))
        f0 = f_DList[D][-1] + 1                   
        hdul0 = pyfits.open(stokesList[0])
        hdr0 = hdul0[0].header         
        tgtTab = Table.read(hdul0['TGT'])                    
        if (D==0):
            YX_dt = np.array([tgtTab['YCE'],tgtTab['XCE']])
            tcenter = np.argmin((YX_dt**2).sum(axis=0))          
            stokess = hdul0['SCI'].data.shape[0]
            grating =  hdr0['GRATING'].strip()                                    
            targets = len(tgtTab['CATID'])
            dwav = hdul0['SCI'].header['CDELT1']            
            oktgt_Dt = np.zeros((Dirs,targets),dtype=bool)                    
        oktgt_Dt[D] = (tgtTab['CULL'] == '')              
        wav0_D[D] = hdul0['SCI'].header['CRVAL1']        
        wavs_D[D] = hdul0['SCI'].data.shape[-1]
        PA_D[D] = hdr0['PA']                                   
        telPA_D[D] = hdr0['TELPA']
        maskid_D[D] =  hdr0['MASKID'].strip()
        grtilt_D[D] = hdr0['GRTILT']        # this is proposed grang                                                  

    files = files_D.sum() 
    t_T_DList = [np.where(oktgt_Dt[D])[0] for D in range(Dirs)]
    
  # now get file-dependent data
    MBYX_df = np.zeros((2,files))               # SALT coords
    MByx_df = np.zeros((2,files))               # RSS coords    
    hdul_f = np.empty(files,dtype=object)    
    obsDict = create_obslog(filenameList,keywordfile) 
    rho_f = obsDict['TRKRHO'] 
    imgtime_f = np.array(obsDict['EXPTIME'])/2.           # assuming cal data does not use dtr reps      
    files_tv_DList = [np.zeros((targets,wavs_D[D])) for D in range(Dirs)]
    wav_v_DList = []
                       
    for D,Dir in enumerate(DirList):
        rsslog.message(" Inputting data from: %s "% Dir, logfile)               
        for f in f_DList[D]:
            hdul_f[f] = pyfits.open(filenameList[f])
            MBYX_df[:,f] = np.array([obsDict['MBY'][f],obsDict['MBX'][f]]).astype(float)
            MByx_df[:,f] = rotate2d(MBYX_df[:,f], rho_f[f])
            files_tv_DList[D][oktgt_Dt[D]] += (hdul_f[f]['BPM'].data == 0).all(axis=0)                                                                            
        wav_v_DList.append(wav0_D[D] + dwav*np.arange(wavs_D[D]))                                            

    if (grating=="N/A"):
        config = "Imaging"
    else:
        config=grating  

  # tabulate grangs and PAs into separate obs, for pa vs grang computation                          
    PAList = sorted(list(set(PA_D)))
    PAs = len(PAList)
    tiltList = sorted(list(set(grtilt_D)))
    grangs = len(tiltList)
    grang0 = (grtilt_D.min() + grtilt_D.max())/2.
    dgrang = grtilt_D.max() - grang0
    dograng = (grangs>1)
    ag_dD = np.vstack((PA_D,grtilt_D))
    ag_do = np.unique(ag_dD,axis=1)
    obss = ag_do.shape[1]
    obs_D = -np.ones(Dirs,dtype=int)
    
    for obs in range(obss):
        DobsList = list(np.where(np.all(ag_dD.T==ag_do[:,obs], axis=1))[0])
        obs_D[DobsList] = obs

  # input zemax fit coefficients vs wavelength (50 Ang binning) for each target
    zcofnameList = ['c0','cy2','cx2','cxy','cr3']
    zcofs = len(zcofnameList)    
    zfitZip = ZipFile(datadir+"/et21zemaxfit.zip",mode='r')
    zipzList = zfitZip.namelist()
    ztargets = len(zipzList)

  # assume that targets=ztargets, and they are the same range(targets) set
    pupoffyx_dt = np.zeros((2,targets))
    YXz_dz = np.zeros((2,ztargets))
    t_z = np.array([z.split('.')[0].split('_')[-1] for z in zipzList]).astype(int)
    z_t = t_z[t_z]
                                 
    for z in range(ztargets):
        datalineList = list(filter(None,zfitZip.read(zipzList[z]).split('\n')))
        hdrs = [x[0] for x in datalineList].count('#')    
        if (z==0):
            wavz_pw = np.array([x.split()[0] for x in datalineList[hdrs:]]).astype(float).reshape((2,-1))
            wavs = wavz_pw.shape[1]
            wav_w = wavz_pw[0]
            zcofs = len(np.fromstring(datalineList[hdrs],dtype=float,sep=" ")) - 1          
            zcof_pzcw = np.zeros((2,ztargets,zcofs,wavs))             
        hdrlineList = datalineList[:hdrs]
        hdrtitleList = [hdrlineList[x][1:].split()[0] for x in range(hdrs)]
        YXline = hdrtitleList.index("pupoffyx")     
        YXz_dz[:,z] = np.fromstring(hdrlineList[YXline][12:],dtype=float,sep=" ")[:2]                             
        pupoffyxline = hdrtitleList.index("pupoffyx")     
        pupoffyx_dt[:,t_z[z]] = np.fromstring(hdrlineList[pupoffyxline][12:],dtype=float,sep=" ")[:2]      
        zcof_pzcw[:,z] = np.array([x.split() for x in datalineList[hdrs:]]).astype(float)[:,1:] \
            .reshape((2,wavs,-1)).transpose((0,2,1))
        zcof_pzcw[1,z,0] = np.mod(zcof_pzcw[1,z,0], 180.) 

    pupoffyx_dt *= pupoffcor        # optionally, not 1, to correct for zemax model fov error        
    okz_w = (zcof_pzcw[0] != 0.).all(axis=(0,1))       
                   
  # add new cofs and decide what to fit
    newcofnameList = ['cy1','cx1','cy3','cx3']
    cofnameList = zcofnameList+newcofnameList
    cofs = len(cofnameList)
    
    var_V = np.array(["y","x","r"])
    pow_Vc = np.array([[0,2,0,1,0,1,0,3,0],[0,0,2,1,0,0,1,0,3],[0,0,0,0,3,0,0,0,0]])
    vars = pow_Vc.shape[0]
                
    fitcofs = len(fitcofnameList)
#    cfitList = [cofnameList.index(name) for name in fitcofnameList]
    calnameList = list(set((zcofnameList + fitcofnameList)))    # zcofs plus new fit cofs
    calcofs = len(calnameList)
    ccalList = sorted([cofnameList.index(name) for name in calnameList])
    calnameList = [cofnameList[cof] for cof in ccalList]
    
    rsslog.message(('\n Fit Coefficients: '+fitcofs*'%s ') % tuple(fitcofnameList), logfile)

  # obtain illumfiles
    model0_RC = skypupillum(np.zeros(2), np.zeros(2), "", 0., 0.)[0]   # get RC dims
    Rows,Cols = model0_RC.shape
    rcmax_d = np.array([Rows-1,Cols-1])/2.
    pupyx_dRC = (np.indices((Rows,Cols)) - rcmax_d[:,None,None])/sacpupilrad
    if debug:
        np.savetxt("pupyx_dRC.txt",pupyx_dRC.reshape((2*Rows,-1)),fmt = " %9.5f")
        
    ispup_RC = ((pupyx_dRC**2).sum(axis=0)<1.)        
    illum_ftRC = np.zeros((files,targets,Rows,Cols),dtype=float)    
    for f,t in np.ndindex(files,targets):                           
        model_RC, modelarea, modelareanom = skypupillum(YX_dt[:,t], MBYX_df[:,f], "", rho_f[f], imgtime_f[f],   \
            useshuttercor=useshuttercor, debug=debug)
        illum_ftRC[f,t] = ispup_RC.astype(float)[None,None,:,:]*model_RC
        
        if (debug & (f in (0,20,39,80,100,119)) & (t==tcenter)): 
            np.savetxt("illumRC_"+str(f).zfill(2)+"_"+str(t)+".txt", illum_ftRC[f,t],fmt=" %6.3f")      

  # for grang analysis, get wavelength overlap for fit
    oktgt_t = oktgt_Dt.all(axis=0)
    vshift_D = np.zeros(Dirs)
    dcofg_ftpc = []  
    if dograng:
        oksigs_tv_DList = [(files_tv_DList[D] == files_D[D,None]) for D in range(Dirs)]    # signal in all files
        maxwavs = wavs_D.max()
        oksigs_DtW = np.zeros((Dirs,targets,maxwavs),dtype=bool)
        okcomp_DtW = np.zeros((Dirs,targets,maxwavs),dtype=bool)    
        vshift_D[1:] = ((wav0_D[1:]-wav0_D[0])/dwav).astype(int)
        
        for D in range(Dirs):
            oksigs_DtW[D,:,:wavs_D[D]] = oksigs_tv_DList[D]
        okcomp_DtW[0] = oksigs_DtW[0]        
        for D in range(1,Dirs):        
            okcomp_DtW[0] &= (shift(oksigs_DtW[D],(0,vshift_D[D]),order=0))                    
        for D in range(1,Dirs):        
            okcomp_DtW[D] = (shift(okcomp_DtW[0],(0,-vshift_D[D]),order=0))                        
        okcomp_tv_DList = [okcomp_DtW[D,:,:wavs_D[D]] for D in range(Dirs)]
        okcomp_t = (okcomp_tv_DList[0].sum(axis=1) > 0)
        for D in range(Dirs): 
            np.savetxt("okcomp_tv_"+str(D)+".txt",np.vstack((wav_v_DList[D],okcomp_tv_DList[D])).T,fmt=(" %8.0f "+targets*" %2i"))
      
        rsslog.message(("\n Grating Angle matching bins: %4i \n target"+targets*" %4i"+" \n bins  "+targets*" %4i") %     \
            ((okcomp_tv_DList[0].sum(),)+tuple(range(targets))+tuple(okcomp_tv_DList[0].sum(axis=1))), logfile)

      # for each usable target, difference fit between the two grangs
        grangfitcofnameList = ['c0','cy1','cx1']
        cfitList = [cofnameList.index(name) for name in grangfitcofnameList]    
        cof_tgpc = np.zeros((targets,2,2,cofs))
        err_tgpc = np.zeros_like(cof_tgpc)
        vstokes_ftp = np.zeros((files,targets,2))
        for t in np.where(okcomp_t)[0]:   
            compwav = wav_v_DList[0][okcomp_tv_DList[0][t]].mean()
            zcof_pc = interp1d(wav_w,zcof_pzcw[:,t],kind='cubic')(compwav)
            for g in (0,1):
                obsDList = obs_D[[g,g+2]]
                fidxList = list(np.concatenate((f_DList[obs_D[g]],f_DList[obs_D[g+2]])))                             
                for a,D in enumerate(obsDList):
                    T = np.where(t_T_DList[D]==t)[0][0]            
                    for f in f_DList[D]:
                        hdul = hdul_f[f]                           
                        stokes_Sv = hdul['SCI'].data[:,T]
                        var_Sv = hdul['VAR'].data[:,T]                        
                        vstokes_ftp[f,t] = viewstokes(stokes_Sv[:,okcomp_tv_DList[D][t]],    \
                            var_Sv[:,okcomp_tv_DList[D][t]])[0].mean(axis=1)                             
                for p in (0,1):               
                    cof0_c = np.zeros(cofs)
                    cof0_c[:zcofs] = zcof_pc[p]                                                                                             
                    cof_C, covar_CC = curve_fit(make_PAPol(illum_ftRC[:,t],pupyx_dRC,pupoffyx_dt[:,t],cof0_c,cfitList), \
                        fidxList, vstokes_ftp[fidxList,t,p],p0=tuple(cof0_c[cfitList]))
                    cof_c = np.copy(cof0_c)
                    cof_c[cfitList] = cof_C
                    err_c = np.zeros(cofs)
                    err_c[cfitList] = np.sqrt(np.diagonal(covar_CC))                        
                    cof_tgpc[t,g,p] = cof_c
                    err_tgpc[t,g,p] = err_c
                    
        calPAisoff_D = (calPA_D <> calPA)         
        dcalPA = 0.                               
        if calPAisoff_D.sum():
            goffpol = np.where(tiltList==grtilt_D[calPAisoff_D])[0][0]    # assume offset polaroid is for one grating setting          
            dcalPA = cof_tgpc[tcenter,goffpol,1,0] - cof_tgpc[tcenter,1-goffpol,1,0]            
            cof_tgpc[:,goffpol,1,0] -= dcalPA
            rsslog.message(("\n Cal polaroid PA offset for Dirs "+calPAisoff_D.sum()*" %2i"+":  %8.3f degs") %  \
                (tuple(np.where(calPAisoff_D)[0])+(dcalPA,)), logfile)
                                    
        dcalPA_f = np.zeros(0)
        for D in range(Dirs):      
            dcalPA_f = np.concatenate((dcalPA_f,np.repeat(calPAisoff_D[D]*dcalPA, files_D[D])))        
            np.savetxt("vstokes_ftp_"+str(D)+".txt",np.vstack((np.tile(MByx_df[:,f_DList[D]],(1,2)),  \
                vstokes_ftp[f_DList[D]].transpose((1,2,0)).reshape((targets,-1)))).T,fmt=2*" %10.5f"+targets*" %10.4f")
                             
        dcofg_tpc = np.diff(cof_tgpc,axis=1)[:,0]/2.
        dcofg_ftpc = np.tile(-dcofg_tpc,(files_D[0],1,1,1))
        for D in range(1,Dirs):
            dcofg_ftpc = np.vstack((dcofg_ftpc,np.tile(-dcofg_tpc*(-1.)**D,(files_D[D],1,1,1))))           
        oktgt_t &= okcomp_t
        np.savetxt("cof_tgpc.txt",cof_tgpc.reshape((-1,cofs)),fmt=" %10.4f")
        np.savetxt("dcofg_tpc.txt",np.vstack((dcofg_tpc.reshape((-1,cofs)),dcofg_ftpc.reshape((-1,cofs)))),fmt=" %10.4f")        
        np.savetxt("err_tgpc.txt",err_tgpc.reshape((-1,cofs)),fmt=" %10.4f")     
    
  # get calibration data (wav_v), and bin down to 50 Ang (wav_w) (same as zemax file)
    wbin = 50.
    wedge_w = np.append(wav_w - wbin/2., wav_w.max()+wbin/2.)
    rsslog.message("\n Input files: \n   f      MBy        MBx      rho     telpa    grtilt    file", logfile)
    D_f =  np.concatenate([np.repeat(D,files_D[D]) for D in range(Dirs)])
    telPA_f = np.concatenate([np.repeat(telPA_D[D],files_D[D]) for D in range(Dirs)])
    grtilt_f =  np.concatenate([np.repeat(grtilt_D[D],files_D[D]) for D in range(Dirs)])    
    nameList = [ "/".join(f.split('/')[-3:]) for f in filenameList ]
    oktgt_t = np.ones(targets,dtype=bool)     
    if debug:        
        oktgt_t = np.zeros(targets,dtype=bool)
        oktgt_t[12] = True         
        wmap = np.where(wav_w == 4000.)[0][0]     
    vstokes_fptw = np.zeros((files,2,targets,wavs))
    ok_ftw = np.ones((files,targets,wavs),dtype=bool)
    medint_ft = np.zeros((files,targets))                
    for f,hdul in enumerate(hdul_f):       
        ok_Tv = (hdul['BPM'].data==0).all(axis=0)
        stokes_STv = hdul['SCI'].data*ok_Tv[None,:,:]
        var_STv = hdul['VAR'].data*ok_Tv[None,:,:]                   
        wav_v = wav_v_DList[D_f[f]]
        colArray = np.where(wav_v > 0.)[0]
        wedgeArray = np.where((wedge_w >= wav_v[colArray].min()) &    \
                              (wedge_w < wav_v[colArray].max()))[0]
        wavout = wedgeArray[:-1]   
        cbinedge_W = interp1d(wav_v[colArray],(colArray+0.5))(wedge_w[wedgeArray])

        ok_ftw[f] *= oktgt_Dt[D_f[f]][:,None]                   
        for t in np.where(oktgt_Dt[D_f[f]])[0]:        
            T = np.where(t_T_DList[D_f[f]]==t)[0][0]
            medint_ft[f,t] = np.median((stokes_STv[0]*ok_Tv)[T])           
            bins_W = scrunch1d(ok_Tv[T].astype(int),cbinedge_W)                                
            ok_W = (bins_W  > (cbinedge_W[1:] - cbinedge_W[:-1])/2.)
            stokes_Sw = np.zeros((3,wavs))
            var_Sw = np.zeros((3,wavs))
            ok_w = np.zeros(wavs,dtype=bool)
            ok_w[wavout] = ok_W
                                                      
            for S in (0,1,2):
                stokes_Sw[S,wavout[ok_W]] = scrunch1d(stokes_STv[S,T],cbinedge_W)[ok_W]/bins_W[ok_W]
                var_Sw[S,wavout[ok_W]] = scrunchvar1d(var_STv[S,T],cbinedge_W)[0][ok_W]/bins_W[ok_W]**2
                                                       
            vstokes_fptw[f,0,t,ok_w],vstokes_fptw[f,1,t,ok_w] =     \
                viewstokes(stokes_Sw[:,ok_w],var_Sw[:,ok_w])[0]           
            ok_ftw[f,t] = ok_w                                                                
                                        
        rsslog.message(" %3i %10.5f %10.5f %8.3f %8.3f %8.3f   %s " %     \
            ((f,)+tuple(MByx_df[:,f])+(rho_f[f],telPA_f[f],grtilt_f[f],nameList[f])), logfile)   
       
      
  # for each target, compute fit_pw fit file
    cof_tcpw = np.zeros((targets,cofs,2,wavs))
    fit_ftpw = np.zeros((files,targets,2,wavs))
    PAPol_pRC = np.zeros((2,Rows,Cols))
    PAPol_pdRC = np.zeros((2,5,Rows,Cols))
    fstd_tpw = np.zeros((targets,2,wavs))                 
    cfitList = [cofnameList.index(name) for name in fitcofnameList]
    ok_tw = np.zeros((targets,wavs),dtype=bool)        

  # cull spectra to remove very low intensities
    intcull = 0.5
    hdr = ((oktgt_t.sum()*"%8i ") % tuple(np.where(oktgt_t)[0]))        
    np.savetxt("medint_ft.txt",medint_ft,header=hdr,fmt=oktgt_t.sum()*" %8.0f")
    iscull_ft = np.zeros((files,targets),dtype=bool)
    medint_f = np.ma.median(np.ma.masked_array(medint_ft,mask=(medint_ft==0)),axis=1).data
    for D in range(Dirs):
        fList = f_DList[D]
        medintD_t = np.ma.median(np.ma.masked_array(medint_ft[fList],mask=(medint_ft[fList]==0)),axis=0).data
        medintD = np.median(medintD_t[medintD_t<>0.])
        tList = list(np.where(medintD_t<>0.)[0])
        iscull_Ft = np.zeros((files_D[D],targets),dtype=bool)        
        iscull_Ft[:,tList] =        \
            (medint_ft[fList][:,tList]/(medintD_t[None,tList]*medint_f[fList,None]/medintD) < intcull)
        iscull_Ft &= (medint_ft[fList] <>0.)            
        iscull_ft[fList] = iscull_Ft

    rsslog.message('\n Cull low spectra: \n  t  files', logfile)
    
    for t in np.where(iscull_ft.any(axis=0))[0]:    
        rsslog.message((' %2i '+iscull_ft[:,t].sum()*' %3i') % ((t,)+tuple(np.where(iscull_ft[:,t])[0])), logfile)
    ok_ftw &= np.logical_not(iscull_ft)[:,:,None]
        
    rsslog.message('\n Fit summary\n target file  %P Wstd %P Wrms   PA Wstd  PA Wrms', logfile)  

    if centeronly:        
        oktgt_t = np.zeros(targets,dtype=bool)
        oktgt_t[tcenter] = True
                       
    for t in np.where(oktgt_t)[0]:    
        wmean_pf = np.zeros((2,files))
        wstd_pf = np.zeros((2,files))
        diff_pfw = np.zeros((2,files,wavs))     
        ok_tw[t] = ok_ftw[:,t].sum(axis=0) > 0.9*files/2
        ffitList = list(np.where(ok_ftw[:,t].any(axis=1))[0])
       
        for p in (0,1):
            cof0_wc = np.zeros((wavs,cofs))
            cof0_wc[:,:zcofs] = zcof_pzcw[p,z_t[t]].T                            
            for w in np.where(ok_tw[t])[0]:
                ok_f = ok_ftw[:,t,w]
                fidxList = np.where(ok_f)[0]
                cof0_c = cof0_wc[w]                                          
                cof_C, covar_CC = curve_fit(make_PAPol(illum_ftRC[:,t],pupyx_dRC,pupoffyx_dt[:,t],cof0_c,   \
                    cfitList,dcofg_fc=dcofg_ftpc[:,t,p],dcal_f=p*dcalPA_f),             \
                    fidxList, vstokes_fptw[fidxList,p,t,w],p0=tuple(cof0_c[cfitList]))
                cof_c = np.copy(cof0_c)
                cof_c[cfitList] = cof_C
                err_c = np.zeros(cofs)
                err_c[cfitList] = np.sqrt(np.diagonal(covar_CC))
                fit_F =make_PAPol(illum_ftRC[:,t],pupyx_dRC,pupoffyx_dt[:,t],cof0_c,    \
                    cfitList,dcofg_fc=dcofg_ftpc[:,t,p],dcal_f=p*dcalPA_f)(fidxList,*tuple(cof_c[cfitList]))
                cof_tcpw[t,:,p,w] = cof_c
                fit_ftpw[ok_f,t,p,w] = fit_F
                if debug:
                    if (w != wmap): continue
                    PAPol_pRC[p], PAPol_pdRC[p] =make_PAPol_map(illum_ftRC[:,t],pupyx_dRC,pupoffyx_dt[:,t],cof0_c,    \
                        cfitList,dcofg_fc=dcofg_ftpc[:,t,p],dcal_f=p*dcalPA_f)(fidxList,*tuple(cof_c[cfitList]))                
            for f in ffitList:
                diff_pfw[p,f] = ok_tw[t]*(vstokes_fptw[f,p,t] - fit_ftpw[f,t,p])      
                wmean_pf[p,f] = diff_pfw[p,f][ok_tw[t]].mean()
                wstd_pf[p,f] =  diff_pfw[p,f][ok_tw[t]].std()                
        fstd_tpw[t] = diff_pfw.std(axis=1)
           
        for f in ffitList:
            rsslog.message(' %3i %3i %8.3f %8.3f %8.3f %8.3f' %     \
                (t, f, wmean_pf[0,f], wstd_pf[0,f],wmean_pf[1,f],wstd_pf[1,f]), logfile)

    hdr = (("  Wavl "+oktgt_t.sum()*"  %8i        ") % tuple(np.where(oktgt_t)[0]))               
    np.savetxt("fdiffstd_tpw.txt",np.vstack((wav_w,fstd_tpw[oktgt_t].reshape((-1,wavs)))).T,  \
        header=hdr,fmt=" %7.0f "+2*oktgt_t.sum()*"%8.3f ")
    hdr = (("  Wavl "+oktgt_t.sum()*" %4i") % tuple(np.where(oktgt_t)[0]))
    np.savetxt("filecount_tw.txt",np.vstack((wav_w,ok_ftw[:,oktgt_t].sum(axis=0))).T,   \
        header=hdr,fmt=" %7.0f "+oktgt_t.sum()*" %4i")

    if debug:
        np.savetxt("PAPol_RC.txt", PAPol_pRC.reshape((2*Rows,-1)), fmt=" %9.4f")
        np.savetxt("PAPol_dRC.txt", PAPol_pdRC.reshape((10*Rows,-1)), fmt=" %9.4f")        

  # save cal cof file after scrunching _w down to cal _W 
    fitstring = ','.join(fitcofnameList)
    if useshuttercor: fitstring += '_shtrcor'
    if (oktgt_t.sum()>1): fitstring+= '_'+str(int(1000.*pupoffcor))        
    heffcalzip = 'RSSpol_Heff_Moon0_'+fitstring+'.zip' 
    calzip = ZipFile(heffcalzip,mode='w')          
    wav_W = np.array(range(3200,3500,100)+range(3500,5000,50)+  \
        range(5000,7000,100)+range(7000,10000,200)).astype(float)
    Wavs = len(wav_W)
    wedge_W = (wav_W[:-1] + wav_W[1:])/2.
    wedge_W = np.concatenate(([wav_W[0]-(wav_W[1]-wav_W[0])/2.],wedge_W,[wav_W[-1]+(wav_W[-1]-wav_W[-2])/2.]))
    wbin_W = np.diff(wedge_W)   
    colArray = np.where(wav_w > 0.)[0]
    WedgeArray = np.where((wedge_W >= wav_w[colArray].min()) &    \
                          (wedge_W < wav_w[colArray].max()))[0]
    Wavout = WedgeArray[:-1]          
    cbinedge_W = interp1d(wav_w[colArray],(colArray+0.5))(wedge_W[WedgeArray])
    wav_W = (wedge_W[:-1] + wedge_W[1:])/2.
    
    lbl_l = np.tile(wav_W[:Wavs],2)   
    if dograng:        
        lbl_l = np.concatenate(([dgrang],wav_W,[dgrang],wav_W))
     
    for t in np.where(oktgt_t)[0]: 
        cof_cpW = np.zeros((cofs,2,Wavs))
        bins_W = scrunch1d(ok_tw[t].astype(int),cbinedge_W)                                
        ok_W = (bins_W  > (cbinedge_W[1:] - cbinedge_W[:-1])/2.) 
               
        for c in ccalList:     
            for p in (0,1):
                cof_cpW[c,p,Wavout[ok_W]] = scrunch1d(cof_tcpw[t,c,p],cbinedge_W)[ok_W]/bins_W[ok_W]                  
        if dograng:
            cof_cl = np.vstack((dcofg_tpc[t,0][None,:],cof_cpW[:,0].T,   \
                    dcofg_tpc[t,1][None,:],cof_cpW[:,1].T)).T
        else:
            cof_cl = cof_cpW.reshape((cofs,-1))
              
        hdr = '/'.join(os.getcwd().split('/')[-2:])+' et21test '+fitstring+' '+  \
            str(datetime.date.today()).replace('-','')
        hdr += ("\n calPA  %6.2f" % calPA)
        if useshuttercor:
            hdr += "\n useshuttercor"            
        hdr += ("\n grang0 %6.3f" % grang0)                     
        hdr += ("\n fovYX  %6.2f %6.2f" % tuple(YX_dt[:,t]))
        hdr += ("\n offyx  %8.4f %8.4f" % tuple(pupoffyx_dt[:,t]))          

        for v in range(vars): hdr += ("\n %5s  "+calcofs*"%10i ") % ((var_V[v],)+tuple(pow_Vc[v][ccalList]))
            
        targetfile = "et21b_target_"+str(t).zfill(2)+".txt"
        np.savetxt(targetfile, np.vstack((lbl_l,cof_cl[ccalList].reshape((calcofs,-1)))).T,    \
            header=hdr, fmt="%9.2f "+calcofs*"%10.3f ")
        calzip.write(targetfile)
        os.remove(targetfile)
    calzip.close()

  # save et21 fit files
    PAPolrawzip = 'ET21B_PAPol_raw_'+fitstring+'.zip' 
    rawzip = ZipFile(PAPolrawzip,mode='w')   
    PAPolfitzip = 'ET21B_PAPol_fit_'+fitstring+'.zip' 
    fitzip = ZipFile(PAPolfitzip,mode='w')   

    for t in np.where(oktgt_t)[0]:
        hdr = '/'.join(os.getcwd().split('/')[-2:])+' et21skytest '+fitstring+' '+  \
            str(datetime.date.today()).replace('-','')                
        hdr += ("\n YX   %6.2f %6.2f" % tuple(YX_dt[:,t]))
        rawtargetfile = "et21b_rawtarget_"+str(t).zfill(2)+".txt"                    
        fittargetfile = "et21b_fittarget_"+str(t).zfill(2)+".txt"  

        np.savetxt(rawtargetfile, np.vstack((np.tile(wav_w,2),vstokes_fptw[:,:,t].reshape((files,-1)))).T,    \
            header=hdr, fmt="%9.2f "+files*"%10.3f ")
        rawzip.write(rawtargetfile)
        os.remove(rawtargetfile)
                                            
        np.savetxt(fittargetfile, np.vstack((np.tile(wav_w,2),fit_ftpw[:,t].reshape((files,-1)))).T,    \
            header=hdr, fmt="%9.2f "+files*"%10.3f ")
        fitzip.write(fittargetfile)
        os.remove(fittargetfile)    

    rawzip.close()
    fitzip.close()                                 
    return

#--------------------------------------
def make_PAPol(illum_fRC,pupyx_dRC,pupoffyx_d,cof0_c,cfitList,dcofg_fc=[],dcal_f=[]):
    # cofnameList = ['c0','cy2','cx2','cxy','cr3','cy1','cx1','cy3','cx3']
    def fit(fList,*cof):
        cof_Fc = np.tile(cof0_c,(len(fList),1))
        cof_Fc[:,cfitList] = cof
        if len(dcofg_fc):
            cof_Fc = cof_Fc + dcofg_fc[fList]
        if len(dcal_f):
            cof_Fc[:,0] = cof_Fc[:,0] + dcal_f[fList]
        Rows,Cols = pupyx_dRC.shape[1:]
        ispup_RC = ((pupyx_dRC**2).sum(axis=0)<1.)
        pupoffyx_dRC = pupyx_dRC - pupoffyx_d[:,None,None]
        
      # reverse zemax X sign: zemax frame is down beam
      # don't: pupoffyx_dRC = np.array([1.,-1.])[:,None,None]*pupoffyx_dRC              
        PAPol_FRC =  ispup_RC[None,:,:].astype(float)*  \
            (cof_Fc[:,0,None,None] +  \
            (cof_Fc[:,1:3,None,None]*pupoffyx_dRC[None,:,:,:]**2).sum(axis=1) +    \
            cof_Fc[:,3,None,None]*(pupoffyx_dRC[None,:,:,:].prod(axis=1)) +    \
            cof_Fc[:,4,None,None]*(((pupoffyx_dRC[None,:,:,:]**2).sum(axis=1))**(1.5)) +   \
            (cof_Fc[:,5:7,None,None]*pupoffyx_dRC[None,:,:,:]).sum(axis=1) +  \
            (cof_Fc[:,7:9,None,None]*pupoffyx_dRC[None,:,:,:]**3).sum(axis=1))
        PAPolfit_F = (PAPol_FRC*illum_fRC[fList]).sum(axis=(1,2))/illum_fRC[fList].sum(axis=(1,2))

        return PAPolfit_F
        
    return fit
                      
#--------------------------------------
def make_PAPol_map(illum_fRC,pupyx_dRC,pupoffyx_d,cof0_c,cfitList,dcofg_fc=[],dcal_f=[]):
    # cofnameList = ['c0','cy2','cx2','cxy','cr3','cy1','cx1','cy3','cx3']
    def fit_map(fList,*cof):
        cof_Fc = np.tile(cof0_c,(len(fList),1))
        cof_Fc[:,cfitList] = cof
        if len(dcofg_fc):
            cof_Fc = cof_Fc + dcofg_fc[fList]
        if len(dcal_f):
            cof_Fc[:,0] = cof_Fc[:,0] + dcal_f[fList]
        Rows,Cols = pupyx_dRC.shape[1:]
        ispup_RC = ((pupyx_dRC**2).sum(axis=0)<1.)
        pupoffyx_dRC = pupyx_dRC - pupoffyx_d[:,None,None]
        
      # reverse zemax X sign: zemax frame is down beam
      # don't: pupoffyx_dRC = np.array([1.,-1.])[:,None,None]*pupoffyx_dRC
        PAPol_dRC = np.zeros((5,Rows,Cols))              
        PAPol_dRC[0] =  ispup_RC[None,:,:].astype(float)* cof_Fc[0,0,None,None] 
        PAPol_dRC[1] =  ispup_RC[None,:,:].astype(float)* (cof_Fc[0,1:3,None,None]*pupoffyx_dRC**2).sum(axis=0)
        PAPol_dRC[2] =  ispup_RC[None,:,:].astype(float)* cof_Fc[0,3,None,None]*(pupoffyx_dRC.prod(axis=0))
        PAPol_dRC[3] =  ispup_RC[None,:,:].astype(float)* cof_Fc[0,4,None,None]*(((pupoffyx_dRC**2).sum(axis=0))**(1.5))
        PAPol_dRC[4] =  ispup_RC[None,:,:].astype(float)* (cof_Fc[0,5:7,None,None]*pupoffyx_dRC).sum(axis=0)

        PAPol_RC = PAPol_dRC.sum(axis=0)
        return PAPol_RC, PAPol_dRC
        
    return fit_map

#--------------------------------------------------------------------    
        
if __name__=='__main__':
    infiletxt = sys.argv[1].split(',')
    fitcofnameList = sys.argv[2].split(',')    
    kwargs = dict(x.split('=', 1) for x in sys.argv[3:])  
    spHeffcal(infiletxt, fitcofnameList, **kwargs)
    
# cd /d/pfis/khn/poleff/et21b_moon0
# python2.7 script.py spHeffcal.py 20230829/sci_blue5,20230829/sci_red3,20230830/sci_blue6,20230830/sci_red9 c0,cy1,cx1 calPA=91,90,91,90
# python2.7 script.py spHeffcal.py 20230829/sci_blue5,20230829/sci_red3,20230830/sci_blue6,20230830/sci_red9 c0,cy1,cx1 calPA=91,90,91,90 useshuttercor=True centeronly=True 
