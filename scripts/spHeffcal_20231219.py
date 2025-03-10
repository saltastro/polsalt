#! /usr/bin/env python2.7

"""
spHeffcal

Fit HeffCal Moon data  
  evaluate pupil illum dependence of HW_Calibration 

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

    fitpupoff = (kwargs.pop('fitpupoff','False') == 'True')        
    paramDict = load(open(datadir+'/skypupilparams.json'))
    primwidth = paramDict["primwidth"]
    sacpupilrad = primwidth/np.sqrt(3.)    
    calPA_DList = kwargs.pop('calPA',[])
    debug = (kwargs.pop('debug','False') == 'True')
    logfile= kwargs.pop('logfile','spHeffcal.log')    

  # first get Dir-specific data
    basedir = "/d/pfis/khn/"
    Dirs = len(DirList)
    calPA = 90.
    if len(calPA_DList)==0:
        calPA_DList = list(np.repeat([calPA],Dirs))
    else:
        calPA_DList = map(float,calPA_DList.split(','))
            
    files_D = np.zeros(Dirs,dtype=int)
    wavs_D = np.zeros(Dirs,dtype=int) 
    wav0_D = np.zeros(Dirs,dtype=int)
    maskid_D =  np.zeros(Dirs,dtype=str)
    PA_D = np.zeros(Dirs)                   # proposed PA    
    telPA_D = np.zeros(Dirs)                # actual PA, may be 180 different from PA
    grtilt_D = np.zeros(Dirs)                 
    wav_v_DList = []
                  
    for D,Dir in enumerate(DirList):    
        stokesList = sorted(glob.glob(basedir+DirList[D]+"/Moon-*rawstokes.fits"))      
        files_D[D] = len(stokesList)                    
        hdul0 = pyfits.open(stokesList[0])
        hdr0 = hdul0[0].header         
        tgtTab = Table.read(hdul0['TGT'])                    
        if (D==0):
            YX_dt = np.array([tgtTab['YCE'],tgtTab['XCE']])
            tcenter = np.argmin((YX_dt**2).sum(axis=0))          
            stokess = hdul0['SCI'].data.shape[0]
            grating =  hdr0['GRATING'].strip()                                    
            targets = len(tgtTab['CATID'])
            oktgt_Dt = np.zeros((Dirs,targets),dtype=bool)                    
        oktgt_Dt[D] = (tgtTab['CULL'] == '')              
        wav0_D[D] = hdul0['SCI'].header['CRVAL1']        
        wavs_D[D] = hdul0['SCI'].data.shape[-1]
        dwav = hdul0['SCI'].header['CDELT1']
        wav_v_DList.append(wav0_D[D] + dwav*np.arange(wavs_D[D]))
        PA_D[D] = hdr0['PA']                                   
        telPA_D[D] = hdr0['TELPA']
        maskid_D[D] =  hdr0['MASKID'].strip()
        grtilt_D[D] = hdr0['GRTILT']        # this is proposed grang                                                  
 
    t_T_DList = [np.where(oktgt_Dt[D])[0] for D in range(Dirs)]
    
  # now get file-dependent data          
    files_tv_DList = [np.zeros((targets,wavs_D[D])) for D in range(Dirs)]
    rho_f_DList = [np.zeros(files_D[D]) for D in range(Dirs)]  
    MBYX_df_DList = [np.zeros((2,files_D[D])) for D in range(Dirs)]     # SALT coords
    MByx_df_DList = [np.zeros((2,files_D[D])) for D in range(Dirs)]     # RSS coords    
    hdul_f_DList = [np.empty(files_D[D],dtype=object) for D in range(Dirs)] 
    fileList_DList = [[] for D in range(Dirs)]
           
    for D,Dir in enumerate(DirList):
        rsslog.message(" Inputting data from: %s "% Dir, logfile)
        fileList_DList[D] = sorted(glob.glob(basedir+Dir+"/Moon-*rawstokes.fits"))        
        obsDict = create_obslog(fileList_DList[D],keywordfile)        
        for f in range(files_D[D]):
            hdul_f_DList[D][f] = pyfits.open(fileList_DList[D][f])                                     
            files_tv_DList[D][oktgt_Dt[D]] += (hdul_f_DList[D][f]['BPM'].data == 0).all(axis=0)
            rho_f_DList[D][f] = obsDict['TRKRHO'][f]                             
            MBYX_df_DList[D][:,f] = np.array([obsDict['MBY'][f],obsDict['MBX'][f]]).astype(float)
            MByx_df_DList[D][:,f] = rotate2d(MBYX_df_DList[D][:,f], rho_f_DList[D][f])

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
        
    okz_w = (zcof_pzcw[0] != 0.).all(axis=(0,1))       
                   
  # add new cofs and decide what to fit
    newcofnameList = ['cy1','cx1','cy3','cx3']
    cofnameList = zcofnameList+newcofnameList
    cofs = len(cofnameList)
    
    var_V = np.array(["y","x","r"])
    pow_Vc = np.array([[0,2,0,1,0,1,0,3,0],[0,0,2,1,0,0,1,0,3],[0,0,0,0,3,0,0,0,0]])
    vars = pow_Vc.shape[0]
                
    fitcofs = len(fitcofnameList)
    cfitList = [cofnameList.index(name) for name in fitcofnameList]
    calnameList = list(set((zcofnameList + fitcofnameList)))    # zcofs plus new fit cofs
    calcofs = len(calnameList)
    ccalList = sorted([cofnameList.index(name) for name in calnameList])
    calnameList = [cofnameList[cof] for cof in ccalList]
    
    rsslog.message(('\n Fit Coefficients: '+fitcofs*'%s ') % tuple(fitcofnameList), logfile)

  # obtain illumfiles
    model0_RC = skypupillum(np.zeros(2), np.zeros(2), "", 0.)[0]   # get RC dims
    Rows,Cols = model0_RC.shape
    rcmax_d = np.array([Rows-1,Cols-1])/2.
    pupyx_dRC = (np.indices((Rows,Cols)) - rcmax_d[:,None,None])/sacpupilrad
    ispup_RC = ((pupyx_dRC**2).sum(axis=0)<1.) 
       
    illum_ftRC_DList = [np.zeros((files_D[D],targets,Rows,Cols),dtype=int) for D in range(Dirs)]

    for D in range(Dirs):       
        for f,t in np.ndindex(files_D[D],targets):                   
            model_RC, modelarea, modelareanom = skypupillum(YX_dt[:,t], MBYX_df_DList[D][:,f], "", rho_f_DList[D][f])
            illum_ftRC_DList[D][f,t] = ispup_RC.astype(float)[None,None,:,:]*model_RC

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
        cof_tgpc = np.zeros((targets,2,2,cofs))
        err_tgpc = np.zeros_like(cof_tgpc)
        vstokes_ftp_DList = [np.zeros((files_D[D],targets,2)) for D in range(Dirs)]
        for t in np.where(okcomp_t)[0]:   
            compwav = wav_v_DList[0][okcomp_tv_DList[0][t]].mean()
            zcof_pc = interp1d(wav_w,zcof_pzcw[:,t],kind='cubic')(compwav)
            for g in (0,1):
                obsDList = obs_D[[g,g+2]]            
                illum_ftRC = np.concatenate((illum_ftRC_DList[obsDList[0]],illum_ftRC_DList[obsDList[1]]))            
                vstokes_fp = np.zeros((0,2))
                MByx_fd = np.zeros((0,2))
                for a,D in enumerate(obsDList):
                    T = np.where(t_T_DList[D]==t)[0][0]            
                    for f in range(files_D[D]):
                        hdul = hdul_f_DList[D][f]                           
                        stokes_Sv = hdul['SCI'].data[:,T]
                        var_Sv = hdul['VAR'].data[:,T]                        
                        vstokes_p = viewstokes(stokes_Sv[:,okcomp_tv_DList[D][t]],    \
                            var_Sv[:,okcomp_tv_DList[D][t]])[0].mean(axis=1)
                        vstokes_fp = np.vstack((vstokes_fp,vstokes_p))
                        vstokes_ftp_DList[D][f,t] = vstokes_p
                ok_f = np.ones(vstokes_fp.shape[0],dtype=bool)                                
                for p in (0,1):
                    cof0_c = np.zeros(cofs)
                    cof0_c[:zcofs] = zcof_pc[p]                              
                    fileList = np.where(ok_f)[0]
                                                               
                    cof_C, covar_CC = curve_fit(make_PAPol(illum_ftRC[:,t],pupyx_dRC,pupoffyx_dt[:,t],cof0_c,cfitList), \
                        fileList, vstokes_fp[fileList,p],p0=tuple(cof0_c[cfitList]))
                    cof_c = np.copy(cof0_c)
                    cof_c[cfitList] = cof_C
                    err_c = np.zeros(cofs)
                    err_c[cfitList] = np.sqrt(np.diagonal(covar_CC))                        
                    cof_tgpc[t,g,p] = cof_c
                    err_tgpc[t,g,p] = err_c
                    
        calPAisoff_D = (np.array(calPA_DList) <> calPA)                        
        if calPAisoff_D.sum():
            goffpol = np.where(tiltList==grtilt_D[calPAisoff_D])[0][0]    # assume offset polaroid is for one grating setting          
            dcalPA = cof_tgpc[tcenter,goffpol,1,0] - cof_tgpc[tcenter,1-goffpol,1,0]            
            cof_tgpc[:,goffpol,1,0] -= dcalPA
            rsslog.message(("\n Cal polaroid PA offset for Dirs "+calPAisoff_D.sum()*" %2i"+":  %8.3f degs") %  \
                (tuple(np.where(calPAisoff_D)[0])+(dcalPA,)), logfile)
                                    
        dcalPA_f = np.zeros(0)
        for D in range(Dirs):         
            dcalPA_f = np.concatenate((dcalPA_f,np.repeat(calPAisoff_D[D]*dcalPA, files_D[D])))        
            np.savetxt("vstokes_ftp_"+str(D)+".txt",np.vstack((np.tile(MByx_df_DList[D],(1,2)),  \
                vstokes_ftp_DList[D].transpose((1,2,0)).reshape((targets,-1)))).T,fmt=2*" %10.5f"+targets*" %10.4f") 
                             
        dcofg_tpc = np.diff(cof_tgpc,axis=1)[:,0]/2.
        dcofg_ftpc = np.tile(-dcofg_tpc,(files_D[0],1,1,1))
        for D in range(1,Dirs):
            dcofg_ftpc = np.vstack((dcofg_ftpc,np.tile(-dcofg_tpc*(-1.)**D,(files_D[D],1,1,1))))           
        oktgt_t &= okcomp_t
        np.savetxt("cof_tgpc.txt",cof_tgpc.reshape((-1,cofs)),fmt=" %10.4f")
        np.savetxt("dcofg_tpc.txt",np.vstack((dcofg_tpc.reshape((-1,cofs)),dcofg_ftpc.reshape((-1,cofs)))),fmt=" %10.4f")        
        np.savetxt("err_tgpc.txt",err_tgpc.reshape((-1,cofs)),fmt=" %10.4f")     

    print dcalPA_f

    exit()
    
  # get calibration data (wav_v), and bin down to 50 Ang (wav_w) (same as zemax file)
    files = files_D.sum()
    vstokes_fptw = np.zeros((files,2,targets,wavs))
    ok_ftw = np.ones((files,targets,wavs),dtype=bool)
    wbin = 50.
    wedge_w = np.append(wav_w - wbin/2., wav_w.max()+wbin/2.)
    rsslog.message("\n Input files: \n   f      MBy        MBx      rho     telpa    grtilt    file", logfile)
    D_f =  np.concatenate([np.repeat(D,files_D[D]) for D in range(Dirs)])
    illum_ftRC = np.concatenate([illum_ftRC_DList[D] for D in range(Dirs)])
    MByx_df = np.hstack(MByx_df_DList)
    rho_f = np.hstack(rho_f_DList)
    telPA_f = np.concatenate([np.repeat(telPA_D[D],files_D[D]) for D in range(Dirs)])
    grtilt_f =  np.concatenate([np.repeat(grtilt_D[D],files_D[D]) for D in range(Dirs)])    
    nameList = [ "/".join(f.split('/')[-3:]) for f in sum(fileList_DList,[])] 
                
    for f,hdul in enumerate(np.hstack(hdul_f_DList)):       
        ok_Tv = (hdul['BPM'].data==0).all(axis=0)
        stokes_STv = hdul['SCI'].data*ok_Tv[None,:,:]
        var_STv = hdul['VAR'].data*ok_Tv[None,:,:]                   
        wav_v = wav_v_DList[D_f[f]]
        colArray = np.where(wav_v > 0.)[0]
        wedgeArray = np.where((wedge_w >= wav_v[colArray].min()) &    \
                              (wedge_w < wav_v[colArray].max()))[0]
        wavout = wedgeArray[:-1]   
        cbinedge_W = interp1d(wav_v[colArray],(colArray+0.5))(wedge_w[wedgeArray])
                   
        for t in np.where(oktgt_t)[0]:
            T = np.where(t_T_DList[D_f[f]]==t)[0][0]            
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
       
    rsslog.message('\n Fit summary\n target file  %P Wstd %P Wrms   PA Wstd  PA Wrms', logfile)        
  # for each target, compute fit_pw fit file
    cof_tcpw = np.zeros((targets,cofs,2,wavs))
    fit_ftpw = np.zeros((files,targets,2,wavs)) 
    
    oktgt_t = np.zeros(targets,dtype=bool)
    ok_tw = np.zeros((targets,wavs),dtype=bool)
    oktgt_t[12] = True 
            
    for t in np.where(oktgt_t)[0]:    
        wmean_pf = np.zeros((2,files))
        wstd_pf = np.zeros((2,files))
        diff_pfw = np.zeros((2,files,wavs))     
        ok_tw[t] = ok_ftw[:,t].sum(axis=0) > fitcofs
       
        for p in (0,1):
            cof0_wc = np.zeros((wavs,cofs))
            cof0_wc[:,:zcofs] = zcof_pzcw[p,z_t[t]].T                            
            for w in np.where(ok_tw[t])[0]:
                ok_f = ok_ftw[:,t,w]
                fileList = np.where(ok_f)[0]
                cof0_c = cof0_wc[w]                                          
                cof_C, covar_CC = curve_fit(make_PAPol(illum_ftRC[:,t],pupyx_dRC,pupoffyx_dt[:,t],cof0_c,   \
                    cfitList,dcofg_fc=dcofg_ftpc[:,t,p]), \
                    fileList, vstokes_fptw[fileList,p,t,w],p0=tuple(cof0_c[cfitList]))
                cof_c = np.copy(cof0_c)
                cof_c[cfitList] = cof_C
                err_c = np.zeros(cofs)
                err_c[cfitList] = np.sqrt(np.diagonal(covar_CC))
                fit_F =make_PAPol(illum_ftRC[:,t],pupyx_dRC,pupoffyx_dt[:,t],cof0_c,    \
                    cfitList,dcofg_fc=dcofg_ftpc[:,t,p])(fileList,*tuple(cof_c[cfitList]))
                cof_tcpw[t,:,p,w] = cof_c
                fit_ftpw[ok_f,t,p,w] = fit_F
            for f in range(files):
                diff_pfw[p,f] = (vstokes_fptw[f,p,t] - fit_ftpw[f,t,p])      
                wmean_pf[p,f] = diff_pfw[p,f][ok_ftw[f,t]].mean()
                wstd_pf[p,f] =  diff_pfw[p,f][ok_ftw[f,t]].std()
        fmean_pw = diff_pfw.mean(axis=1)
        fstd_pw = diff_pfw.std(axis=1)
           
        for f in range(files):
            rsslog.message(' %3i %3i %8.3f %8.3f %8.3f %8.3f' %     \
                (t, f, wmean_pf[0,f], wstd_pf[0,f],wmean_pf[1,f],wstd_pf[1,f]), logfile)
        np.savetxt("fdiffmean_"+str(t)+"_pw.txt",np.vstack((wav_w,fmean_pw,fstd_pw)).T,fmt=" %7.0f "+4*"%8.3f ")

  # save cal cof file after scrunching _w down to cal _W 
    fitstring = ','.join(fitcofnameList)        
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
        hdr += ("\n fovYX  %6.2f %6.2f" % tuple(YX_dt[:,t]))
        hdr += ("\n grang0 %6.3f" % grang0)                    
        hdr += ("\n offyx  %8.4f %8.4f" % tuple(pupoffyx_dt[:,t]))          
        hdr += ("\n calPA  %6.2f" % calPA)
                                         
        for v in range(vars): hdr += ("\n %5s  "+calcofs*"%10i ") % ((var_V[v],)+tuple(pow_Vc[v][ccalList]))
            
        targetfile = "et21b_target_"+str(t).zfill(2)+".txt"
        np.savetxt(targetfile, np.vstack((lbl_l,cof_cl[ccalList].reshape((calcofs,-1)))).T,    \
            header=hdr, fmt="%9.2f "+calcofs*"%10.3f ")
        calzip.write(targetfile)
        calzip.close()
        os.remove(targetfile)

  # save et21 fit file
    PAPolfitzip = 'ET21B_PAPol_fit_'+fitstring+'.zip' 
    fitzip = ZipFile(PAPolfitzip,mode='w')   

    for t in np.where(oktgt_t)[0]:
        hdr = '/'.join(os.getcwd().split('/')[-2:])+' et21skytest '+fitstring+' '+  \
            str(datetime.date.today()).replace('-','')                
        hdr += ("\n YX   %6.2f %6.2f" % tuple(YX_dt[:,t]))            
        targetfile = "et21b_target_"+str(t).zfill(2)+".txt"                   
        np.savetxt(targetfile, np.vstack((np.tile(wav_w,2),fit_ftpw[:,t].reshape((files,-1)))).T,    \
            header=hdr, fmt="%9.2f "+files*"%10.3f ")
        fitzip.write(targetfile)
        fitzip.close()
        os.remove(targetfile)    
                             
    return

#--------------------------------------
def make_PAPol(illum_fRC,pupyx_dRC,pupoffyx_d,cof0_c,cfitList,dcofg_fc=[]):
    # cofnameList = ['c0','cy2','cx2','cxy','cr3','cy1','cx1','cy3','cx3']
    def fit(fList,*cof):
        cof_Fc = np.tile(cof0_c,(len(fList),1))
        cof_Fc[:,cfitList] = cof
        if len(dcofg_fc):
            cof_Fc = cof_Fc + dcofg_fc[fList]            
        Rows,Cols = pupyx_dRC.shape[1:]
        ispup_RC = ((pupyx_dRC**2).sum(axis=0)<1.)
        pupoffyx_dRC = pupyx_dRC - pupoffyx_d[:,None,None]
        
      # reverse zemax X sign: zemax frame is down beam
        pupoffyx_dRC = np.array([1.,-1.])[:,None,None]*pupoffyx_dRC              
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
def make_PAPol_old(illum_fRC,pupyx_dRC,pupoffyx_d,cof0_c,cfitList):
    # cofnameList = ['c0','cy2','cx2','cxy','cr3','cy1','cx1','cy3','cx3']
    def fit(fList,*cof):
        cof_c = cof0_c
        cof_c[cfitList] = cof
        Rows,Cols = pupyx_dRC.shape[1:]
        ispup_RC = ((pupyx_dRC**2).sum(axis=0)<1.)
        pupoffyx_dRC = pupyx_dRC - pupoffyx_d[:,None,None]
        
      # reverse zemax X sign: zemax frame is down beam
        pupoffyx_dRC = np.array([1.,-1.])[:,None,None]*pupoffyx_dRC      
        
        PAPol_RC =  ispup_RC.astype(float)*(cof_c[0] +  \
            (cof_c[1:3,None,None]*pupoffyx_dRC**2).sum(axis=0) +    \
            cof_c[3,None,None]*pupoffyx_dRC.prod(axis=0) +    \
            cof_c[4,None,None]*(((pupoffyx_dRC**2).sum(axis=0))**(1.5)) +   \
            (cof_c[5:7,None,None]*pupoffyx_dRC).sum(axis=0) +  \
            (cof_c[7:,None,None]*pupoffyx_dRC**3).sum(axis=0))

        PAPolfit_f = (PAPol_RC[None,:,:]*illum_fRC[fList]).sum(axis=(1,2))/illum_fRC[fList].sum(axis=(1,2))
        return PAPolfit_f
    return fit
                      
#--------------------------------------
def netrms(pupoffyx_d,illum_fRC,pupyx_dRC,cof0_pwc,cfitList,stokes_pfw):
    wavs = cof0_pwc.shape[1]
    MBs = illum_fRC.shape[0]
    err_pfw = np.zeros((2,MBs,wavs))

    for w in range(wavs):
        for p in (0,1):
            cof0_c = cof0_pwc[p,w]            
            cof_C, covar_CC = curve_fit(make_PAPol(illum_fRC,pupyx_dRC,pupoffyx_d,cof0_c,cfitList),range(MBs), \
                stokes_pfw[p,:,w],p0=tuple(cof0_c[cfitList]))
            cof_c = np.copy(cof0_c)
            cof_c[cfitList] = cof_C
            fit_f = make_PAPol(illum_fRC,pupyx_dRC,pupoffyx_d,cof0_c,cfitList)(range(MBs),*tuple(cof_c[cfitList]))
            err_pfw[p,:,w] = (fit_f - stokes_pfw[p,:,w])

    err_f =err_pfw.mean(axis=(0,2))       # give P and PA equal wt for now
    rms = np.sqrt((err_f**2).mean())

    return rms, err_f
    
def fminrms(pupoffyx_d,illum_fRC,pupyx_dRC,cof0_pwc,cfitList,stokes_pfw):
    return netrms(pupoffyx_d,illum_fRC,pupyx_dRC,cof0_pwc,cfitList,stokes_pfw)[0]

#--------------------------------------------------------------------    
        
if __name__=='__main__':
    infiletxt = sys.argv[1].split(',')
    fitcofnameList = sys.argv[2].split(',')    
    kwargs = dict(x.split('=', 1) for x in sys.argv[3:])  
    spHeffcal(infiletxt, fitcofnameList, **kwargs)
    
# cd /d/pfis/khn/poleff/et21b_moon0
# python2.7 script.py spHeffcal.py 20230829/sci_blue5,20230829/sci_red3,20230830/sci_blue6,20230830/sci_red9 c0,cy1,cx1
