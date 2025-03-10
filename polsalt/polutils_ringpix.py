
# polarimetry utilities, including:

# datedfile(filename,date)
# datedline(filename,date)
# def rotate2d(yx_ds, rot, center=np.zeros(2))
# greff(grating,grang,artic,dateobs,wav)
# rssdtralign(datobs,trkrho)
# rssmodelwave(grating,grang,artic,trkrho,cbin,cols,datobs)
# pixellate(pupshape_RC,pupyx_dRC,**kwargs)
# skypupillum(YX_d,MByx_d,TRKyx_d,trkrho,**kwargs)
# heffcal(heffcalfile,YX_dt,MByx_d,TRKyx_d,grang,trkrho,wav_w,**kwargs)
# polzerocal(polzerocalfile,YX_dt,wav_w)
# configmap(infileList,confitemList,debug='False')
# image_number(image_name)
# list_configurations(infileList, logfile)
# configmapset(obsTab, configList=('GRATING','GR-ANGLE', 'CAMANG'))
# list_configurations_old(infileList, logfile)
# blksmooth1d(ar_x,blk,ok_x)
# angle_average(ang_d)
# fence(arr)
# legfit_cull(x_x,y_x,ok_x,deg,fence='outer')
# specpolrotate(stokes_Sw,var_Sw,covar_Sw,par_w,normalized=False)
# viewstokes(stokes_Sw,err2_Sw,ok_w=[True],tcenter=0.)
# fargmax(arr)

# this is pysalt-free

import os, sys, glob, shutil, inspect, warnings
import numpy as np
from json import load
from astropy.io import fits as pyfits, ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import interp1d
from scipy import interpolate as ip
from scipy import linalg as la
from astropy.table import Table,unique
from scipy.special import legendre as legFn
import rsslog
from obslog import create_obslog

datadir = os.path.dirname(__file__) + '/data/'
keywordfile = datadir+"obslog_config.json"

# ------------------------------------

def datedfile(filename,date):
    """ select file based on observation date and latest version

    Parameters
    ----------
    filename: text file name pattern, including "yyyymmdd_vnn" place holder for date and version
    date: yyyymmdd of observation

    Returns: file name

    """

    fileList = sorted(glob.glob(filename.replace('yyyymmdd_vnn','????????_v??')))
    if len(fileList)==0: return ""
    dateoffs = filename.find('yyyymmdd')
    dateList = [file[dateoffs:dateoffs+8] for file in fileList]
    file = fileList[0]
    for (f,fdate) in enumerate(dateList):
        if date < fdate: continue
        for (v,vdate) in enumerate(dateList[f:]):
            if vdate > fdate: continue
            file = fileList[f+v]
 
    return file     
# ------------------------------------

def datedline(filename,date):
    """ select line from file based on observation date and latest version

    Parameters
    ----------
    filename: string
        skips lines without first field [yyyymmdd]_v[nn] datever label
    date: string or int yyyymmdd of observation

    Returns: string which is selected line from file (including datever label)

    """
    line_l = [ll for ll in open(filename) if ll[8:10] == "_v"]
    datever_l = [line_l[l].split()[0] for l in range(len(line_l))]

    line = ""
    for (l,datever) in enumerate(datever_l):
        if (int(date) < int(datever[:8])): continue
        for (v,vdatever) in enumerate(datever_l[l:]):
            if (int(vdatever[:8]) > int(datever[:8])): continue
            datever = datever_l[l+v]
            line = line_l[l+v]
 
    return line
# ----------------------------------------------------------

def rotate2d(yx_ds, rot, center=np.zeros(2)):
    """rotate an array of 2d coordinates

    Parameters:
    yx_ds: 2d numarray of 2d coordinates.  Optional 1d, returns 1d
        _d: 0,1 for y,x
        _s: index of coordinates
    rot: amount to rotate (degrees)
    center: y,x coordinates of rotation center (default 0,0)

    """
    dims = len(np.shape(yx_ds))
    if (dims==1): yx_ds = yx_ds[:,None]
    c = np.cos(np.radians(rot))
    s = np.sin(np.radians(rot))
    rotate = np.transpose([[c, s],[-1.*s, c]])
    yx1_ds = yx_ds - center[:,None]
    yxout_ds = (np.dot(yx1_ds.T,rotate)).T
    yxout_ds +=  center[:,None]
    if (dims==1): return yxout_ds[:,0]
    return yxout_ds

#--------------------------------------------------------

def greff(grating,grang,artic,dateobs,wav,alftarg=0.):
    """ grating efficiency, zero outside 1st order and eff < grateffedge

    Parameters
    ----------
    grating: string.  grating name
    grang (deg)
    artic (deg)
    dateobs yyyymmdd
    wav (ang), may be 1D array
    alphtarg (deg) optional target off-axis in dispersion dirn
    
    Returns: efficiency (poss array), effp/effs (poss array)

    """
    
#   p/s added 9 July, 2017 khn
#   alphtarg added 20220821

    grname=np.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=np.loadtxt(datadir+"gratings.txt",usecols=(1,2),unpack=True)
    gr300wav,gr300eff,gr300ps=np.loadtxt(datadir+"grateff_0300.txt",usecols=(0,1,2),unpack=True)
    grng,grdn,grthick,grtrans,grbroaden=np.loadtxt(datadir+"grateff_v1.txt", \
        usecols=(1,2,3,4,5),unpack=True)
    spec_dp=np.array(datedline(datadir+"RSSspecalign.txt",dateobs).split()[1:]).astype(float)

    Grat0 = spec_dp[0]
    grateffedge = 0.04

    grnum = np.where(grname==grating)[0][0]
    lmm = grlmm[grnum]
    alpha_r = np.radians(grang+Grat0+alftarg)

    if grnum == 0:          # SR grating
        eff = interp1d(gr300wav,gr300eff,kind='cubic',bounds_error=False)(wav)
        ps = interp1d(gr300wav,gr300ps,kind='cubic',bounds_error=False)(wav) 
    else:                   # Kogelnik gratings
        ng = grng[grnum]
        dn = grdn[grnum]
        thick = grthick[grnum]
        tran = grtrans[grnum]
        broaden = grbroaden[grnum]
        beta_r = np.arcsin(wav*lmm/1.e7 - np.sin(alpha_r))
        betag_r = np.arcsin(np.sin(beta_r)/ng)
        alphag_r = np.arcsin(np.sin(alpha_r)/ng)
        KK = 2.*np.pi*lmm/1.e7
        nus = (np.pi*dn/wav)*thick/np.cos(alphag_r)
        nup = nus*np.cos(alphag_r + betag_r)
        xi = (thick/(2*np.cos(alphag_r)))*(KK*np.sin(alphag_r) - KK**2*wav/(4.*np.pi*ng))/broaden
        sins = np.sin(np.sqrt(nus**2+xi**2))
        sinp = np.sin(np.sqrt(nup**2+xi**2))
        effs = sins**2/(1.+(xi/nus)**2)
        effp = sinp**2/(1.+(xi/nup)**2)
        eff = np.where((sins>0)&(sinp>0)&((effs+effp)/2. > grateffedge),tran*(effs+effp)/2.,0.)
        ps = np.where(eff > 0., effp/effs, 0.)
    return eff,ps
    
# ------------------------------------

def rssdtralign(datobs,trkrho):
    """return predicted RSS optic axis relative to detector center (in unbinned pixels), plate scale
    Correct for flexure, and use detector alignment history.

    Parameters 
    ----------
    datobs: int
        yyyymmdd of first data of applicability  
    trkrho: float
        tracker rho in degrees

    """

  # optic axis is center of imaging mask.  In columns, same as longslit position
    rc0_pd=np.loadtxt(datadir+"RSSimgalign.txt",usecols=(1,2))
    flex_p = np.array([np.sin(np.radians(trkrho)),np.cos(np.radians(trkrho))-1.])
    rcflex_d = (rc0_pd[0:2]*flex_p[:,None]).sum(axis=0)

    row0,col0,C0 = np.array(datedline(datadir+"RSSimgalign.txt",datobs).split()[1:]).astype(float)
    row0,col0 = np.array([row0,col0]) + rcflex_d        # sign changed 20190610

    return row0, col0, C0
# ------------------------------------

def rssmodelwave(grating,grang,artic,trkrho,cbin,cols,datobs):
    """compute wavelengths from model of RSS

    Parameters 
    ----------
    datobs: int
        yyyymmdd of first data of applicability    

     TODO:  replace using PySpectrograph
  
    """

    row0,col0 = rssdtralign(datobs,trkrho)[:2]
    spec_dp=np.array(datedline(datadir+"RSSspecalign.txt",datobs).split()[1:]).astype(float)
    Grat0,Home0,ArtErr,T2Con,T3Con = spec_dp[:5]
    FCampoly=spec_dp[5:]

    grname=np.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=np.loadtxt(datadir+"gratings.txt",usecols=(1,2),unpack=True)
    grnum = np.where(grname==grating)[0][0]
    lmm = grlmm[grnum]
    alpha_r = np.radians(grang+Grat0)
    beta0_r = np.radians(artic*(1+ArtErr)+Home0) - 0.015*col0/FCampoly[0] - alpha_r
    gam0_r = np.radians(grgam0[grnum])
    lam0 = 1e7*np.cos(gam0_r)*(np.sin(alpha_r) + np.sin(beta0_r))/lmm
    modelcenter = 3162.  #   image center (unbinned pixels) for wavelength calibration model
    ww = lam0/1000. - 4.
    fcam = np.polyval(FCampoly[::-1],ww)
    disp = (1e7*np.cos(gam0_r)*np.cos(beta0_r)/lmm)/(fcam/.015)
    dfcam = (modelcenter/1000.)*disp*np.polyval([FCampoly[5-x]*(5-x) for x in range(5)],ww)

    T2 = -0.25*(1e7*np.cos(gam0_r)*np.sin(beta0_r)/lmm)/(fcam/47.43)**2 + T2Con*disp*dfcam
    T3 = (-1./24.)*modelcenter*disp/(fcam/47.43)**2 + T3Con*disp
    T0 = lam0 + T2
    T1 = modelcenter*disp + 3*T3
    X = (np.array(range(cols))-cols/2)*cbin/modelcenter
    lam_X = T0+T1*X+T2*(2*X**2-1)+T3*(4*X**3-3*X)
    return lam_X

# ---------------------------------------------------------------------------------
def pixellate(pupshape_RC,pupyx_dRC,**kwargs):
  # pixellate round shape into centered shapes.  Outer layer has equal-area bins.
  # pupshape is boolean
  # pupyx_dRC is centered and normalized to radius

    Rows,Cols = pupyx_dRC.shape[1:]
    pupdia = 2.*(Cols-1)/np.diff(pupyx_dRC[1,Rows/2][[0,-1]])[0]

    centrbindia = kwargs.pop('centrbin','default')
    if (centrbindia == 'default'):
        centrbindia = pupdia/3.

    pupr_RC = np.sqrt((pupyx_dRC**2).sum(axis=0))
    puppix_RC = np.zeros_like(pupshape_RC,dtype=int)

    pixel = 1
    ispix_RC = (pupshape_RC & (pupr_RC <= (centrbindia/pupdia)))
    puppix_RC[ispix_RC] = pixel

    angwidth = 360./8.
    pupang_RC = np.mod((np.degrees(np.arctan2(pupyx_dRC[0],pupyx_dRC[1]))+ 180. + angwidth/2.), 360.) - angwidth/2.
    for layerpix in range(8):
        pixel += 1
        ispix_RC = (pupshape_RC & (puppix_RC == 0) & (pupr_RC <= 1.) &  \
            (pupang_RC >= -angwidth/2 + layerpix*angwidth) &    \
            (pupang_RC < -angwidth/2 + (layerpix+1)*angwidth))
        puppix_RC[ispix_RC] = pixel

    return puppix_RC

# ---------------------------------------------------------------------------------
def skypupillum(YX_d, MByx_d, TRKyx_d, trkrho, **kwargs):
    """compute sky pupil illumination at FOV, MB, and rho from ET10 model
    Parameters
    ----------
    YX_d: numpy 1d array
        _d: Y,X (mm) at RSS FP
        _t: targets
    MByx_d: numpy 1d array
        _d: MBY,MBX (m) from fits header (same for all targets)
    TRKyx_d: numpy 1d array
        _d: TRKY,TRKX (m) from fits header (same for all targets).  Ignore if ""        
    trkrho: TRKRHO (deg) from fits header (same for all targets)
    
    Returns: 
    -------
    illum_v[1:]: numpy 1d array
        relative illumination of outer pup pixellations 
    modelarea: size of MB illum
    modelareanom: size of illum if MB correctly placed    

    """
    pupilparamfile = kwargs.pop('pupilparamfile','polsalt')      
    if (pupilparamfile == 'polsalt'):
        pupilparamfile = datadir+'skypupilparams.json'      
    debug = kwargs.pop('debug',False)
    useMPlat = kwargs.pop('useMPlat',False)  
    logfile= kwargs.pop('logfile','pupilmodel.log')                   

    paramDict = load(open(pupilparamfile))
    primwidth = paramDict["primwidth"]
    dprimdMB = paramDict["dprimdMB"]      
    obscdia = paramDict["obscdia"]
    obscoffrc_d = np.array(paramDict["obscoffrc_d"])
    dobscdtarg_s = np.array(paramDict["dobscdtarg_s"])
    centrbindia = paramDict["centrbindia"]        
    mbyxslope_d = np.array(paramDict["mbyxslope_d"])
    mbyxinter_d = np.array(paramDict["mbyxinter_d"])
    mbtrkyxcof_dd = np.vstack((mbyxslope_d,mbyxinter_d)).T       

    YXtarg = 5.                                 # mm target separation in ET10        
    sacpupildia = 2.*primwidth/np.sqrt(3)       # 11m SAC pupil in ET10 image pixels        
    pupPixs = int(sacpupildia/2.)*2 +1  
        
    pupshapeyx_ds = np.indices((pupPixs,pupPixs)).reshape((2,-1)) -  \
        int(pupPixs/2)*np.ones((2,pupPixs**2))
    pupshape_s = (pupshapeyx_ds**2).sum(axis=0) <= (sacpupildia/2.)**2   
    pupshape_RC = pupshape_s.reshape((pupPixs,pupPixs))
    pupyx_dRC = pupshapeyx_ds.reshape((2,pupPixs,pupPixs))/(sacpupildia/2.)
    pupbin_RC = pixellate(pupshape_RC,pupyx_dRC,centrbin=centrbindia)
    illumbins = pupbin_RC.max()
                         
    obscoff_sd = obscoffrc_d[None,:] + dobscdtarg_s[:,None]*YX_d[None,:]/YXtarg
    puprad = int(pupPixs/2)                                    
    obscshape0_RC = np.sqrt((np.arange(-puprad,puprad+1) - obscoff_sd[0,0])[:,None]**2 + 
                           (np.arange(-puprad,puprad+1) - obscoff_sd[0,1])[None,:]**2) <= obscdia/2.
    obscshape1_RC = np.sqrt((np.arange(-puprad,puprad+1) - obscoff_sd[1,0])[:,None]**2 + 
                           (np.arange(-puprad,puprad+1) - obscoff_sd[1,1])[None,:]**2) <= obscdia/2.                                                       

    primoff_d = dprimdMB*np.array([1.,-1.])*MByx_d*1000.
    primoff_d = rotate2d(primoff_d,-trkrho)              
    primshapeyx_ds = np.indices((pupPixs,pupPixs)).reshape((2,-1)) -  \
        int(pupPixs/2)*np.ones((2,pupPixs**2)) - primoff_d[:,None]
    prim_s = np.ones(pupPixs**2,dtype=bool)             
    for azi in [30,90,150]:
        y_s = rotate2d(primshapeyx_ds,-(azi-trkrho))[0] 
        prim_s &= (np.abs(y_s) <= primwidth/2.)

    MBshape_RC = (prim_s).reshape((pupPixs,pupPixs))   
    primshape_RC = np.copy(MBshape_RC)
    if len(TRKyx_d):           
        TRKMByx_d = mbtrkyxcof_dd[:,1] + mbtrkyxcof_dd[:,0]*TRKyx_d     # TRKMByx shows correct MB position
    else:
        TRKMByx_d = MByx_d                                              # ignore TRK is not provided
    MBerrmax = .001                             # if MB off by more than 1 mm, AND the MB and TRK prims       
    modelarea00 = 6771.
 
    if (np.abs(MByx_d - TRKMByx_d).max() > MBerrmax):
        TRKprimoff_d = dprimdMB*np.array([1.,-1.])*TRKMByx_d*1000.
        TRKprimoff_d = rotate2d(TRKprimoff_d,-trkrho)        
        TRKprimshapeyx_ds = np.indices((pupPixs,pupPixs)).reshape((2,-1)) -  \
            int(pupPixs/2)*np.ones((2,pupPixs**2)) - TRKprimoff_d[:,None]
        TRKprim_s = np.ones(pupPixs**2,dtype=bool)             
        for azi in [30,90,150]:
            y_s = rotate2d(TRKprimshapeyx_ds,-(azi-trkrho))[0]
            TRKprim_s &= (np.abs(y_s) <= primwidth/2.)
        primshape_RC = (TRKprim_s).reshape((pupPixs,pupPixs))              
        MBshape_RC = (prim_s & TRKprim_s).reshape((pupPixs,pupPixs))
        primoff_d = TRKprimoff_d
        primshapeyx_ds = TRKprimshapeyx_ds        
        
    if useMPlat:                                    #superpose maintenance platform if present
        MPdr_D = np.array(paramDict["MPdr_D"])      # _D=rectangle 0,1, _d=side 0,1
        MPdc_Dd = np.array(paramDict["MPdc_Dd"]).reshape((2,2))        
        MPlatyx_ds = np.indices((pupPixs,pupPixs)).reshape((2,-1)) -  \
            int(pupPixs/2)*np.ones((2,pupPixs**2)) - primoff_d[:,None]
        MPlatyx_ds = rotate2d(MPlatyx_ds, trkrho)            
        MPlat_s = np.zeros(pupPixs**2,dtype=bool)                                                                          
        for D in (0,1):
            MPlat_s |= (MPlatyx_ds[0] < MPdr_D[D]) &     \
                       (MPlatyx_ds[1] > MPdc_Dd[D,0]) & (MPlatyx_ds[1] <= MPdc_Dd[D,1])                                         
        MPlatshape_RC = (MPlat_s).reshape((pupPixs,pupPixs))
        MBshape_RC = MBshape_RC & np.logical_not(MPlatshape_RC)
        primshape_RC = primshape_RC & np.logical_not(MPlatshape_RC)                 
    
    model_RC = pupshape_RC & (MBshape_RC & np.logical_not(obscshape0_RC | obscshape1_RC))
    modelarea = model_RC.sum()/modelarea00
    modelareanom = (pupshape_RC &   \
        (primshape_RC & np.logical_not(obscshape0_RC |obscshape1_RC))).sum()/modelarea00         
            
  # illum is the difference in % illumination of the illumination pixel from expected for uniform 
    illumtot = float(model_RC[pupshape_RC].sum())              
    illum_v = np.zeros(illumbins)            
    for v in range(illumbins):
        illum_v[v] = float(model_RC[pupbin_RC==(v+1)].sum())/illumtot
        illum_v[v] -= float((pupbin_RC==(v+1)).sum())/float(pupshape_RC.sum())

    if debug:
        pixellatefile = "pixellate_debug.txt"
        if (not os.path.exists(pixellatefile)):        
            pixellateFile = open(pixellatefile,'w')    
            for R in range(pupPixs):
                print >>pixellateFile, (pupPixs*" %1i" % tuple(pupbin_RC[R].astype(int)))    
        debugfile="skypupillum_debug.txt"
        processpid = os.getpid()        
        if (not os.path.exists(debugfile)):        
            debugFile = open(debugfile,'w')    
            print >>debugFile, processpid
        else:
            filepid = int(open(debugfile,'r').readline())
            if (filepid != processpid):
                debugFile = open(debugfile,'w')    
                print >>debugFile, processpid                                    
            debugFile = open(debugfile,'a')
        if (len(TRKyx_d)==0): TRKyx_d = np.zeros(2)        
        print >>debugFile, ("\nmodel: "+6*"\n %8.5f"+"\n% 8.3f \n %8.3f\n %8.3f\n"+illumbins*" %8.5f\n") %   \
            (tuple(YX_d)+tuple(MByx_d)+tuple(TRKyx_d)+(trkrho, modelarea, modelareanom)+tuple(illum_v))
        for R in range(pupPixs):
            print >>debugFile, (pupPixs*" %1i" % tuple(model_RC[R].astype(int)))

    return illum_v[1:], modelarea, modelareanom, model_RC

# ----------------------------------------------------------
def heffcal(heffcalfile,YX_dt,MByx_d,TRKyx_d,grang,trkrho,wav_w, **kwargs):
    """get P and PA (HWPol), hw efficiency cal file    
    Parameters
    ----------
    heffcalfile: text
        name of heffcal file to use
    YX_dt: numpy 2d array
        _d: Y,X (mm) at RSS FP
        _t: targets
    MByx_d: numpy 1d array
        _d: MBY,MBX (m) from fits header (same for all targets)
    TRKyx_d: numpy 1d array
        _d: TRKY,TRKX (m) from fits header (same for all targets).  Ignore if ""        
    grang: float grating angle (deg)
    rho: TRKRHO (deg) from fits header (same for all targets)
    wav_w: numpy 1d array
        _w: wavelengths (Ang)
    illum_vt: illum= kwarg, override sky illumination model (eg for lamp) 
    
    Returns: numpy 2d arrays:
    -------
    heffcal_ptw[0,1]: 2 numpy 2d arrays
        hwcal polarization stokes for each target and wavelength.  =0 where invalid
    
    """
    """
    _p: (0,1) %P, PA
    _W: wavelengths in cal file
    _c: coefficents (terms in fit)
    _v: variables in cal file
    _V: standard variable order, (Y,X) + illumbins
    
    """
    debug = kwargs.pop('debug',False)    
    logfile= kwargs.pop('logfile','hwcal.log')       
    illum_vt= kwargs.pop('illum','')
    useMPlat = kwargs.pop('useMPlat',False)      
    fovrad = 52.  
           
  # input heffcal file data, _p=0,1 are stacked
    wav_pW = np.loadtxt(heffcalfile,usecols=0).reshape((2,-1))
    Wavs = wav_pW.shape[1]
    targets = YX_dt.shape[1]    
    wavs = wav_w.shape[0]    
    
    wav_W = wav_pW[0]
    heffcal_pcW = np.loadtxt(heffcalfile)[:,1:].reshape((2,Wavs,-1)).transpose((0,2,1))
    ok_W = (heffcal_pcW[0,0] != 0.)
    cofs = heffcal_pcW.shape[1]
    hdrs = open(heffcalfile).read().count("#")        
    hdrlineList = open(heffcalfile).readlines()[:hdrs]
    hdrtitleList = [hdrlineList[x][1:].split()[0] for x in range(hdrs)]
    
    PAline = hdrtitleList.index("calPA")     
    calPA = np.fromstring(hdrlineList[PAline][9:],dtype=float,sep=" ")[0]        
    normline =  hdrtitleList.index("YXnorm")
    YXnorm = np.fromstring(hdrlineList[normline][9:],dtype=float,sep=" ")[0]
    isgrangdata = hdrtitleList.count("Grang0")
    illumvarline = hdrtitleList.index("1")
    vars = hdrs - illumvarline        
    varList = hdrtitleList[illumvarline:hdrs]
    isintvarList = [x.decode().isnumeric() for x in varList] 
    illumbins = (np.array(varList)[isintvarList]).astype(int).max()

  # assemble variable values for each target 
    varname_V = np.array(['Y','X'] + [str(b) for b in range(1,illumbins+1)])
    varidx_V = np.array([varList.index(varname) for varname in varname_V])
    ok_t = ((np.sqrt((YX_dt**2).sum(axis=0)) < fovrad) & (np.abs(YX_dt[0]) < 0.5*fovrad))
    var_Vt = np.zeros((vars,targets))    
    var_Vt[0:2] = YX_dt/YXnorm
    varill0_V,modelarea,modelareanom =  \
        skypupillum(YX_dt[:,0], MByx_d, TRKyx_d, trkrho, useMPlat=useMPlat, debug=debug)[:3]
    
    if len(illum_vt):
        var_Vt[2:] = illum_vt
    else:
        if (varill0_V.shape[0] != illumbins):
            rsslog.message('  HW Eff cal file %s incompatible with illum model, exitting' % heffcalfile , logfile)
            exit()     
        for t in range(targets):           
            var_Vt[2:,t] = skypupillum(YX_dt[:,t], MByx_d, TRKyx_d, trkrho, useMPlat=useMPlat)[0]
   
  # assemble illum fit terms, using variable flags for each term
    term_Vct = np.zeros((vars,cofs,targets))    
    for V in range(vars):
        term_ft = np.array([np.ones(targets),var_Vt[V],var_Vt[V]**2])        
        useVar_c = np.fromstring(hdrlineList[illumvarline+varidx_V[V]][8:], \
            dtype=int,sep=" ")        
        term_Vct[V] = term_ft[useVar_c,:]

  # evaluate illum fit terms, interpolate HWPol to desired wavelengths
    a_ct = term_Vct.prod(axis=0)    
    heffcal_ptW = (a_ct[None,:,:,None]*heffcal_pcW[:,:,None,:]).sum(axis=1)            
    ok_w = (wav_w > (wav_W[ok_W]).min()) & (wav_w < (wav_W[ok_W]).max())
    ok_tw = (ok_t[:,None] & ok_w[None,:])
    heffcal_ptw = np.zeros((2,targets,wavs))
    for p in [0,1]:
        heffcal_ptw[p,ok_tw] = interp1d(wav_W[ok_W],heffcal_ptW[p,ok_t][:,ok_W], \
            kind='cubic',bounds_error=False)(wav_w[ok_w]).flatten()                               
    heffcal_ptw[1,ok_tw] = heffcal_ptw[1,ok_tw] + calPA

  # apply dPA vs FOV correction, if applicable
    if isgrangdata:
        grangline = hdrtitleList.index("Grang0")
        grang0 = np.fromstring(hdrlineList[grangline][9:],dtype=float,sep=" ")[0]   
        grangvars = 2
        dpacof_c = np.fromstring(hdrlineList[grangline+1+grangvars][8:],dtype=float,sep=" ")
        grangcofs = dpacof_c.shape[0]         
        term_vct = np.zeros((grangvars,grangcofs,targets))
            
        for v in range(grangvars):
            var_t = YX_dt[v]/YXnorm
            term_ft = np.array([np.ones(targets),var_t])        
            usevar_c = np.fromstring(hdrlineList[grangline+1+v][8:], \
                dtype=int,sep=" ")                        
            term_vct[v] = term_ft[usevar_c,:]        
        a_ct = term_vct.prod(axis=0)
        dpa_t = (grang - grang0)*(a_ct*dpacof_c[:,None]).sum(axis=0)   
        heffcal_ptw[1,ok_tw] = heffcal_ptw[1,ok_tw] + np.tile(dpa_t,(wavs,1)).T[ok_tw]  
 
    return heffcal_ptw, ok_tw, varill0_V, modelarea, modelareanom
# ----------------------------------------------------------
def polzerocal(polzerocalfile,YX_dt,wav_w):
    """ Q and U (HZero) vs wavelength from polzero cal file    
    Parameters
    ----------
    polzerocalfile: text
        name of polzerocal file to use
    YX_dt: numpy 2d array
        _d: Y,X (mm) at RSS FP
        _t: targets
    wav_w: numpy 1d array
        _w: wavelengths (Ang)

    Returns: numpy 2d arrays:
    -------
    polzerocal_ptw[0,1]: 2 numpy 2d arrays
        polzerocal polarization stokes for each target and wavelength.  =0 where invalid
    
    """
    """
    _p: (0,1) Q,U 
    """        
  # input polzerocal file data, _p=0,1 are stacked
    wav_pW = np.loadtxt(polzerocalfile,usecols=0).reshape((2,-1))
    Wavs = wav_pW.shape[1]
    targets = YX_dt.shape[1]    
    wavs = wav_w.shape[0]
    
    polzerocal_ptw = np.zeros(((2,targets,wavs)))
    ok_tw = np.ones((targets,wavs),dtype=bool)      
    
    wav_W = wav_pW[0]
    polzerocal_pcW = np.loadtxt(polzerocalfile)[:,1:].reshape((2,Wavs,-1)).transpose((0,2,1))
    ok_W = (polzerocal_pcW[0,0] != 0.)
    cofs = polzerocal_pcW.shape[1]
    
    varList = ['Y','X','Ty','Tx','r']   
    Vars = open(polzerocalfile).read().count("#") - 4        
    hdrlineList = open(polzerocalfile).readlines()[1:(Vars+4)]
    
    PAoff = np.fromstring(hdrlineList[0][9:],dtype=float,sep=" ")[0]
    YXnorm = np.fromstring(hdrlineList[1][9:],dtype=float,sep=" ")[0]    
    mbnorm = np.fromstring(hdrlineList[2][9:],dtype=float,sep=" ")[0]       
    varidx_V = np.zeros(Vars,dtype=int)
    fntype_V = np.empty(Vars,dtype=str)
    fn_Vc = np.zeros((Vars,cofs),dtype=int)
 
    for V in range(Vars): 
        varname,fntype_V[V] = hdrlineList[V+3][3:8].split()
        varidx_V[V] = varList.index(varname)
        fn_Vc[V] = np.fromstring(hdrlineList[V+3][8:],dtype=int,sep=" ")

  # polzerocal_p for each target
    ok_t = ((np.sqrt((YX_dt**2).sum(axis=0)) < 1.1*YXnorm) & (np.abs(YX_dt[0]) < 0.55*YXnorm))    
    var_vt = YX_dt/YXnorm    
    TYX_d = rotate2d(Tyx_d[:,None],-rho)[:,0]/mbnorm

    var_vt = np.vstack((var_vt,np.repeat(TYX_d,targets).reshape((2,-1)),np.repeat(rho,targets)))
    
    term_Vct = np.zeros((Vars,cofs,targets))    
    for V in range(Vars):
        var_t = var_vt[varidx_V[V]]
        if fntype_V[V]=='l':              # legendre polynomials
            term_ft = np.array([np.ones(targets),var_t,(3.*var_t**2-1.)/2.])
        elif fntype_V[V]=='s':            # sinusoidal
            term_ft = np.array([np.ones(targets),np.sin(np.radians(var_t)),np.cos(np.radians(var_t))])
        term_Vct[V] = term_ft[fn_Vc[V],:]
            
    Wnorm = 3000.
    W0 = 6500.
    W_w = (wav_w - W0)/Wnorm
    x_ds = np.empty((dims,0))
    for t in range(x_dt.shape[1]):
        x_ds = np.hstack((x_ds,np.vstack((np.repeat(x_dt[:,t][:,None],  \
            wavs,axis=1),W_w.reshape((1,-1))))))  

  # evaluate fit terms, interpolate HWPol to desired wavelengths
    a_cs = (x_ds[:,None,:]**pow_dc[:,:,None]).prod(axis=0) 
    hzerocal_ptw = (a_cs[None,:,:,None]*hzerocal_pcW[:,:,None,:]).sum(axis=1).reshape((2,targets,wavs))
    ok_tw = np.repeat(ok_t[:,None],wavs,axis=1)
 
    return polzerocal_ptw, ok_tw
 
# ----------------------------------------------------------
def configmap(infileList,confitemList,debug=False):
    """general purpose mapper of observing configurations

    Parameters
    ----------
    infileList: list
        List of filenames 
    confitemList: list
        List of header keywords which define a configuration

    Returns
    -------
    obs_i: numpy 1d array
        observation number (unique object, config) for each file
    config_i: numpy 1d array
        config number (unique config) for each file
    obstab: astropy Table
        object name and config number for each observation     
    configtab: astropy Table
        config items for each config
     
    """
    
  # create the observation log
    obsDict = create_obslog(infileList,keywordfile)
    images = len(infileList)

  # make table of unique polarimetric configurations
    confdatListi = []
    for i in range(images):
        if obsDict['BS-STATE'][i] == 'Removed': continue
        confdatList = []
        for item in confitemList:
            confdatList.append(obsDict[item][i])
        confdatListi.append(confdatList)

    dtypeList = map(type,confdatList)           
    configTab = Table(np.array(confdatListi),names=confitemList,dtype=dtypeList) 
    config_i = np.array([np.where(configTab[i]==unique(configTab))   \
                        for i in range(images)]).flatten()
    configTab = unique(configTab)
                        
  # make table of unique observations
    obsdatListi = []
    for i in range(images):
        object = obsDict['OBJECT'][i].replace(' ','')
        obsdatListi.append([object, config_i[i]])

    obsTab = Table(np.array(obsdatListi),names=['object','config'],dtype=[str,int])
    obs_i = np.array([np.where(obsTab[i]==unique(obsTab))   \
                        for i in range(images)]).flatten()
    obsTab = unique(obsTab)
                        
    return obs_i,config_i,obsTab,configTab

# ------------------------------------

def image_number(image_name):
    """Return the number for an image"""
    return int(os.path.basename(image_name).split('.')[0][-4:])
# ------------------------------------

def list_configurations(infileList, logfile):
    """Produce a list of files of similar configurations

    Parameters
    ----------
    infileList: str
        list of input files

    logfile: str
        logging file

    Returns
    -------
    iarc_a: list
        list of indices for arc images

    iarc_i:
        list of indices for images

    imageno_i: 
        list of image numbers
        
    """
    # set up the observing dictionary
    arclampList = ['Ar','CuAr','HgAr','Ne','NeAr','ThAr','Xe']
    obsDict = create_obslog(infileList,keywordfile)

    # hack to remove potentially bad data
    for i in reversed(range(len(infileList))):
        if int(obs_Dict['BS-STATE'][i][1])!=2: del infileList[i]
    obsDict = create_obslog(infileList,keywordfile)

    # inserted to take care of older observations
    old_data=False
    for date in obs_Dict['DATE-OBS']:
        if int(date[0:4]) < 2015: old_data=True

    if old_data:
        printstdlog("Configuration map for old data", logfile)
        iarc_a, iarc_i, confno_i, confdatList = list_configurations_old(infileList, log)
        arcs = len(iarc_a)
        configDict = {}
        for i in set(confno_i):
            imageDict={}
            imageDict['arc']=[infileList[iarc_a[i]]]
            iList = [infileList[x] for x in np.where(iarc_i==iarc_a[i])[0]]
            iList.remove(imageDict['arc'][0])
            imageDict['object'] = iList
            configDict[confdatList[i]] = imageDict
        return configDict

    # delete bad columns
    obsDict = create_obslog(infileList,keywordfile)
    for k in obsDict.keys():
        if len(obsDict[k])==0: del obsDict[k]
    obsTab = Table(obsDict)

    # create the configurations list
    configDict={}
    confdatList = configmapset(obsTab, configList=('GRATING', 'GR-ANGLE', 'CAMANG', 'BVISITID'))

    infileList = np.array(infileList)
    for grating, grtilt, camang, blockvisit in confdatList:
        imageDict = {}
        #things with the same configuration 
        mask = ((obsTab['GRATING']==grating) *  
                     (obsTab['GR-ANGLE']==grtilt) * 
                     (obsTab['CAMANG']==camang) *
                     (obsTab['BVISITID']==blockvisit)
               )

        objtype = obsTab['CCDTYPE']    # kn changed from OBJECT: CCDTYPE lists ARC more consistently
        lamp = obsTab['LAMPID']
        isarc = ((objtype == 'ARC') | np.in1d(lamp,arclampList))
                                        # kn added check for arc lamp when CCDTYPE incorrect        
        imageDict['arc'] = infileList[mask * isarc]

        # if no arc for this config look for a similar one with different BVISITID
        if len(imageDict['arc']) == 0:
            othermask = ((obsTab['GRATING']==grating) *  \
                     ((obsTab['GR-ANGLE'] - grtilt) < .03) * ((obsTab['GR-ANGLE'] - grtilt) > -.03) * \
                     ((obsTab['CAMANG'] - camang) < .05) * ((obsTab['CAMANG'] - camang) > -.05) *   \
                     (obsTab['BVISITID'] != blockvisit))
            imageDict['arc'] = infileList[othermask * (objtype == 'ARC')]
            if len(imageDict['arc']) > 0:
                printstdlog("Warning: using arc from different BLOCKID", logfile)                
            
        imageDict['flat'] = infileList[mask * (objtype == 'FLAT')]
        imageDict['object'] = infileList[mask * ~isarc *  (objtype != 'FLAT')]
        if len(imageDict['object']) == 0: continue
        configDict[(grating, grtilt, camang, blockvisit)] = imageDict

    return configDict
# ------------------------------------

def configmapset(obsTab, configList=('GRATING','GR-ANGLE', 'CAMANG')):
    """Determine how many different configurations are in the list

    Parameters
    ----------
    obsTab: ~astropy.table.Table
        Observing table of image headers

    Returns
    -------
    configList: list
        Set of configurations
    """
    return list(set(zip(*(obsTab[x] for x in configList))))
# ------------------------------------

def list_configurations_old(infileList, log):
    """For data observed prior 2015

    """
    obsDict = create_obslog(infileList,keywordfile)

    # Map out which arc goes with which image.  Use arc in closest wavcal block of the config.
    # wavcal block: neither spectrograph config nor track changes, and no gap in data files
    infiles = len(infileList)
    newtrk = 5.                                     # new track when rotator changes by more (deg)
    trkrho_i = np.array(map(float,obsDict['TRKRHO']))
    trkno_i = np.zeros((infiles),dtype=int)
    trkno_i[1:] = ((np.abs(trkrho_i[1:]-trkrho_i[:-1]))>newtrk).cumsum()

    infiles = len(infileList)
    grating_i = [obsDict['GRATING'][i].strip() for i in range(infiles)]
    grang_i = np.array(map(float,obsDict['GR-ANGLE']))
    artic_i = np.array(map(float,obsDict['CAMANG']))
    configdat_i = [tuple((grating_i[i],grang_i[i],artic_i[i])) for i in range(infiles)]
    confdatList = list(set(configdat_i))          # list tuples of the unique configurations _c
    confno_i = np.array([confdatList.index(configdat_i[i]) for i in range(infiles)],dtype=int)
    configs = len(confdatList)

    imageno_i = np.array([image_number(infileList[i]) for i in range(infiles)])
    filegrp_i = np.zeros((infiles),dtype=int)
    filegrp_i[1:] = ((imageno_i[1:]-imageno_i[:-1])>1).cumsum()
    isarc_i = np.array([(obsDict['OBJECT'][i].upper().strip()=='ARC') for i in range(infiles)])

    wavblk_i = np.zeros((infiles),dtype=int)
    wavblk_i[1:] = ((filegrp_i[1:] != filegrp_i[:-1]) \
                    | (trkno_i[1:] != trkno_i[:-1]) \
                    | (confno_i[1:] != confno_i[:-1])).cumsum()
    wavblks = wavblk_i.max() +1

    arcs_c = (isarc_i[:,None] & (confno_i[:,None]==range(configs))).sum(axis=0)
    np.savetxt("wavblktbl.txt",np.vstack((trkrho_i,imageno_i,filegrp_i,trkno_i, \
                confno_i,wavblk_i,isarc_i)).T,fmt="%7.2f "+6*"%3i ",header=" rho img grp trk conf wblk arc")

    for c in range(configs):                               # worst: no arc for config, remove images
        if arcs_c[c] == 0:
            lostimages = imageno_i[confno_i==c]
            log.message('No Arc for this configuration: ' \
                +("Grating %s Grang %6.2f Artic %6.2f" % confdatList[c])  \
                +("\n Images: "+lostimages.shape[0]*"%i " % tuple(lostimages)), with_header=False)
            wavblk_i[confno_i==c] = -1
            if arcs_c.sum() ==0:
                log.message("Cannot calibrate any images", with_header=False)
                exit()
    iarc_i = -np.zeros((infiles),dtype=int)

    for w in range(wavblks):
            blkimages =  imageno_i[wavblk_i==w]
            if blkimages.shape[0]==0: continue
            iarc_I = np.where((wavblk_i==w) & (isarc_i))[0]
            if iarc_I.shape[0] >0:
                iarc = iarc_I[0]                        # best: arc is in wavblk, take first
            else:
                conf = confno_i[wavblk_i==w][0]       # fallback: take closest arc of this config
                iarc_I = np.where((confno_i==conf) & (isarc_i))[0]
                blkimagepos = blkimages.mean()
                iarc = iarc_I[np.argmin(imageno_i[iarc_I] - blkimagepos)]
            iarc_i[wavblk_i==w] = iarc
            log.message(("\nFor images: "+blkimages.shape[0]*"%i " % tuple(blkimages)) \
                +("\n  Use Arc %5i" % imageno_i[iarc]), with_header=False)
    iarc_a = np.unique(iarc_i[iarc_i != -1])
    return iarc_a, iarc_i, confno_i, confdatList
#--------------------------------------

def blksmooth1d(ar_x,blk,ok_x):
# blkaverage, then spline interpolate result

    blks = ar_x.shape[0]/blk
    offset = (ar_x.shape[0] - blks*blk)/2
    ar_X = ar_x[offset:offset+blks*blk]                  # cut to integral blocks
    ok_X = ok_x[offset:offset+blks*blk]    
    ar_by = ar_X.reshape(blks,blk)
    ok_by = ok_X.reshape(blks,blk)

    okcount_b = (ok_by.sum(axis=1) > blk/2)
    ar_b = np.zeros(blks)
    for b in np.where(okcount_b)[0]: ar_b[b] = ar_by[b][ok_by[b]].mean()
    gridblk = np.arange(offset+blk/2,offset+blks*blk+blk/2,blk)
    grid = np.arange(ar_x.shape[0])
    arsm_x = ip.griddata(gridblk[okcount_b], ar_b[okcount_b], grid, method="cubic", fill_value=0.)
    oksm_x = (arsm_x != 0.)

    return arsm_x,oksm_x
# ----------------------------------------------------------

def angle_average(ang_d):
# average nparray of angles (floating deg), allowing for wrap at 360
  
    sc_d = SkyCoord(ang_d * u.deg,0. * u.deg)
    sep_d = sc_d[0].separation(sc_d)
    sep_d[sc_d[0].position_angle(sc_d).deg == 270.] *= -1.
    angmean = (sc_d[0].ra + sep_d.mean()).deg % 360.
    return angmean
# ------------------------------------

def fence(arr):
  # return lower outer, lower inner, upper inner, and upper outer quartile fence
    Q1,Q3 = np.percentile(arr,(25.,75.))
    IQ = Q3-Q1
    return Q1-3*IQ, Q1-1.5*IQ, Q3+1.5*IQ, Q3+3*IQ
# ---------------------------------------------------------------------------------

def legfit_cull(x_x,y_x,ok_x,deg,xlim_d=(-1.,1.),yerr=0.,docull=True,IQcull=3., \
    maxerr=None,usefitchi=False,debugname=''):
  # do legendre polyfit, culling weighted errors to fence (one at a time), default outer cull 
  #   if no fence cull, apply option maxerr cull
  # at most 25% of startvals
  # x_x should be on -1,1 interval
  # legfit_d is coefficients low to high (for legval)

    startvals = ok_x.sum()
    if (isinstance(yerr,float)):
        yerr_x = np.zeros_like(x_x)
        wt_x = np.ones_like(x_x)
    else:
        yerr_x = yerr
        wt_x = 1./yerr_x
    docull_x = docull*np.ones_like(x_x).astype(bool)
    x_r = np.linspace(xlim_d[0],xlim_d[1])          # _r is x for returned fiterr_r x-array
              
    newcull = True
    chi_x = np.zeros_like(x_x)
    fitchi_x = np.zeros_like(x_x)    
    cull_x = np.zeros_like(x_x,dtype=bool)    
    passes = 0
    culls = 0
    idxcull = -1
    idxcullsort = [-1,]
    leg_dx = np.array([np.polyval(legFn(d),x_x) for d in range(deg+1)])     
    leg_dr = np.array([np.polyval(legFn(d),x_r) for d in range(deg+1)])   
    cullmeanerrLog = []
    cullmaxerrLog = []
    
    while newcull:
        passes += 1 
        newfenceculls, newwavculls = (0,0)
                
        a_Xd = leg_dx[:,ok_x].T * wt_x[ok_x,None]
        b_X = (y_x * wt_x)[ok_x]
        fit_d,sumsqerr = la.lstsq(a_Xd, b_X)[0:2]
        cov_dd = la.inv((a_Xd[:,:,None]*a_Xd[:,None,:]).sum(axis=0))
        var_x = np.array([(cov_dd*np.outer(leg_dx[:,idx],leg_dx[:,idx])).sum()  \
            for idx in range(len(x_x))])        
        fiterr_x = np.sqrt(var_x*sumsqerr/ok_x.sum())
        var_r = np.array([(cov_dd*np.outer(leg_dr[:,idr],leg_dr[:,idr])).sum()  \
            for idr in range(50)]) 
        fiterr_r = (np.sqrt(var_r*sumsqerr/ok_x.sum()))
        err_x = y_x - np.polynomial.legendre.legval(x_x,fit_d)
        newcull = (ok_x & docull_x).any()
        if (not newcull): continue
        
        fitchi_x[ok_x & docull_x] = (err_x/np.sqrt(yerr_x**2 + fiterr_x**2))[ok_x & docull_x]
        fitQ1,fitQ3 = np.percentile(fitchi_x[ok_x & docull_x],(25.,75.))        
        fitfence_f = [fitQ1 - IQcull*(fitQ3-fitQ1), fitQ3 + IQcull*(fitQ3-fitQ1)]
        fitoverfence_x = np.clip(np.maximum((fitfence_f[0] - fitchi_x),(fitchi_x - fitfence_f[1])),0.,None)
                        
        chi_x[ok_x] = err_x[ok_x]*wt_x[ok_x]    
        Q1,Q3 = np.percentile(chi_x[ok_x],(25.,75.))        
        fence_f = [Q1 - IQcull*(Q3-Q1), Q3 + IQcull*(Q3-Q1)]
        overfence_x = np.clip(np.maximum((fence_f[0] - chi_x),(chi_x - fence_f[1])),0.,None)
        
        useoverfence_x = [overfence_x,fitoverfence_x][usefitchi]
        newfencecull = (useoverfence_x[ok_x & docull_x] > 0.).any()        
        if newfencecull: 
            idxcull = np.where(ok_x & docull_x)[0][np.argmax(useoverfence_x[ok_x & docull_x])]                  
            cull_x[idxcull] = True
            newfenceculls = 1                
        elif maxerr:
            idxcullsort = np.argsort((ok_x & docull_x)*np.abs(err_x))[::-1]
            newwavculls = min(startvals/4,(np.abs(err_x)[ok_x & docull_x] > maxerr).sum())      
            cull_x[idxcullsort[:newwavculls]] = True                             
        culls += newfenceculls + newwavculls       
        newcull = ((culls < startvals/4) & ((newfenceculls + newwavculls) > 0))                        
        if debugname:
            if (passes==1): debugfile = open(debugname+".txt",'w') 
            print >>debugfile, passes, newfenceculls, newwavculls, culls, startvals, usefitchi, newcull
            print >>debugfile, 4*"%10.3f " % tuple(fit_d)
            print >>debugfile, 2*"%10.3f " % tuple(fence_f)
            print >>debugfile, 2*"%10.3f " % tuple(fitfence_f)
            for x in range(len(x_x)):
                print >>debugfile, ((4*" %2i"+" %8.4f"+5*" %8.3f") %    \
                    (x,ok_x[x],docull_x[x],cull_x[x],x_x[x],y_x[x],yerr_x[x],chi_x[x],fiterr_x[x],fitchi_x[x])) 
        ok_x = ok_x & np.logical_not(cull_x)
        cullmeanerrLog.append(fiterr_r.mean())
        cullmaxerrLog.append(fiterr_r.max())
    cullLog = [cullmeanerrLog,cullmaxerrLog]  
        
    return fit_d, ok_x, err_x, fiterr_x, fiterr_r, cullLog

# ----------------------------------------------------------

def specpolrotate(stokes_Sw,var_Sw,covar_Sw,par_w,normalized=False):
    """ rotate linear polarization in stokes,variance cubes

    Parameters
    ----------
    stokes_Sw: 2d np array
        _S = I,Q,U,(optional V) unnormalized stokes (size 3, or 4)
        _w = wavelength
    var_Sw: 2d np array (size 4, or 5).  If not an array, ignore errors
        _S = I,Q,U,QU variance, (optional V) QU covariance for stokes
    covar_Sw: 2d np array (size 3, or 4)
        _S = I,Q,U covariance, (optional V) covariance in wavelength for stokes
    par_w: 1d np array (if single float, expand it) 
        PA(degrees) to rotate
    normalized: if True, there is no I

    Returns stokes, var (as copy)

    """

    Qarg = int(not normalized)
    stokes_Fw = np.copy(stokes_Sw)
    var_Fw = np.copy(var_Sw)
    covar_Fw = np.copy(covar_Sw)
    if isinstance(par_w,float): par_w = np.repeat(par_w,stokes_Sw.shape[1])
    if (par_w.shape[0]==1):     par_w = np.repeat(par_w,stokes_Sw.shape[1])
    c_w = np.cos(2.*np.radians(par_w))
    s_w = np.sin(2.*np.radians(par_w))
    stokes_Fw[Qarg:] = stokes_Fw[Qarg]*c_w - stokes_Fw[Qarg+1]*s_w ,    \
        stokes_Fw[Qarg]*s_w + stokes_Fw[Qarg+1]*c_w
    if (type(var_Sw) == type(stokes_Sw)):
        var_Fw[Qarg:Qarg+2] =  var_Fw[Qarg]*c_w**2 + var_Fw[Qarg+1]*s_w**2 ,    \
            var_Fw[Qarg]*s_w**2 + var_Fw[Qarg+1]*c_w**2
        var_Fw[Qarg+2] =  c_w*s_w*(var_Fw[Qarg] - var_Fw[Qarg+1]) + (c_w**2-s_w**2)*var_Fw[Qarg+2]
        covar_Fw[Qarg:] =  covar_Fw[Qarg]*c_w**2 + covar_Fw[Qarg+1]*s_w**2 ,    \
            covar_Fw[Qarg]*s_w**2 + covar_Fw[Qarg+1]*c_w**2
    return stokes_Fw,var_Fw,covar_Fw 
    
#---------------------------------------------------------------------------------------------
def viewstokes(stokes_Sw,err2_Sw,ok_w=[True],tcenter=0.):
    """Compute normalized stokes parameters, converts Q-U to P-T, for viewing

    Parameters
    ----------
    stokes_Sw: 2d float nparray(stokes,wavelength bin)
       unnormalized stokes parameters vs wavelength

    var_Sw: 2d float nparray(stokes,wavelength bin) 
       variance for stokes_sw

    ok_w: 1d boolean nparray(stokes,wavelength bin) 
       marking good stokes values. default all ok.

    Output: normalized stokes parameters and errors, linear stokes converted to pol %, PA
       Ignore covariance.  Assume binned first if necessary.

    """
    warnings.simplefilter("error")
    stokess,wavs = stokes_Sw.shape
    stokes_vw = np.zeros((stokess-1,wavs))
    err_vw = np.zeros((stokess-1,wavs))
    if (len(ok_w) == 1): ok_w = np.ones(wavs,dtype=bool)

    stokes_vw[:,ok_w] = 100.*stokes_Sw[1:,ok_w]/stokes_Sw[0,ok_w]               # in percent
    err_vw[:,ok_w] = 100.*np.sqrt(err2_Sw[1:stokess,ok_w])/stokes_Sw[0,ok_w]     # error bar ignores covariance

    if (stokess >2):
        stokesP_w = np.zeros((wavs))
        stokesT_w = np.zeros((wavs))
        varP_w = np.zeros((wavs))
        varT_w = np.zeros((wavs))
        varpe_dw = np.zeros((2,wavs))
        varpt_w = np.zeros((wavs))
        stokesP_w[ok_w] = np.sqrt(stokes_Sw[1,ok_w]**2 + stokes_Sw[2,ok_w]**2)      # unnormalized linear polarization
        stokesT_w[ok_w] = (0.5*np.arctan2(stokes_Sw[2,ok_w],stokes_Sw[1,ok_w]))     # PA in radians
        stokesT_w[ok_w] = (stokesT_w[ok_w]-(tcenter+np.pi/2.)+np.pi) % np.pi + (tcenter-np.pi/2.)
                                                                                    # optimal PA folding                
     # variance matrix eigenvalues, ellipse orientation
        varpe_dw[:,ok_w] = 0.5*(err2_Sw[1,ok_w]+err2_Sw[2,ok_w]                          \
            + np.array([1,-1])[:,None]*np.sqrt((err2_Sw[1,ok_w]-err2_Sw[2,ok_w])**2 + 4*err2_Sw[-1,ok_w]**2))
        varpt_w[ok_w] = 0.5*np.arctan2(2.*err2_Sw[-1,ok_w],err2_Sw[1,ok_w]-err2_Sw[2,ok_w])
     # linear polarization variance along p, PA   
        varP_w[ok_w] = varpe_dw[0,ok_w]*(np.cos(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2   \
               + varpe_dw[1,ok_w]*(np.sin(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2
        varT_w[ok_w] = varpe_dw[0,ok_w]*(np.sin(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2   \
               + varpe_dw[1,ok_w]*(np.cos(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2

        stokes_vw[0,ok_w] = 100*stokesP_w[ok_w]/stokes_Sw[0,ok_w]                  # normalized % linear polarization
        err_vw[0,ok_w] =  100*np.sqrt(err2_Sw[1,ok_w])/stokes_Sw[0,ok_w]
        stokes_vw[1,ok_w] = np.degrees(stokesT_w[ok_w])                            # PA in degrees
        err_vw[1,ok_w] =  0.5*np.degrees(np.sqrt(err2_Sw[2,ok_w])/stokesP_w[ok_w])

    return stokes_vw,err_vw       
# ----------------------------------------------------------

def fargmax(arr):
    """returns simple floating argmax using quad fit to top 3 points

    Parameters:
    arr: 1d numarray

    """
    argmax = np.clip(np.argmax(arr),1,len(arr)-2)
    fm,f0,fp = tuple(arr[argmax-1:argmax+2])
    darg = -0.5*(fp - fm)/(fp + fm - 2.*f0)     
    return (argmax + darg)

