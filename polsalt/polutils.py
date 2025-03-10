
# polarimetry utilities, including:

# datedfile(filename,date)
# datedline(filename,date)
# rotate2d(yx_ds, rot, center=np.zeros(2))
# greff(grating,grang,artic,dateobs,wav)
# rssdtralign(datobs,trkrho)
# rssmodelwave(grating,grang,artic,trkrho,cbin,cols,datobs)
# skypupillum(YX_d,MByx_d,TRKyx_d,trkrho,**kwargs)
# heffcal(heffcalfile,YX_dt,MByx_d,TRKyx_d,grang,trkrho,wav_w,**kwargs)
# heffcalcorrect(heffcalcorrectfile,wav_w,stokes_Sw,ok_w,**kwargs)
# polzerocal(polzerocalfile,YX_dt,wav_w)
# configmap(infileList,confitemList,debug='False')
# image_number(image_name)
# list_configurations(infileList, logfile)
# configmapset(obsTab, configList=('GRATING','GR-ANGLE', 'CAMANG'))
# list_configurations_old(infileList, logfile)
# blksmooth1d(ar_x,blk,ok_x)
# angle_average(ang_d)
# fpchecpy(x_I,t_J,k=3,debug=False)
# fence(arr)
# chi2sample(f_ax,var_ax,ok_x,dx,debugfile='')
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
from scipy.interpolate import interp1d, CubicSpline
from scipy import interpolate as ip
from scipy import linalg as la
from astropy.table import Table,unique
from scipy.special import legendre as legFn
from zipfile import ZipFile
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
#   PG0700 added 20240726

    grname=np.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=np.loadtxt(datadir+"gratings.txt",usecols=(1,2),unpack=True)  
    grng,grdn,grthick,grtrans,grbroaden=np.loadtxt(datadir+"grateff_v1.txt", \
        usecols=(1,2,3,4,5),unpack=True)
    spec_dp=np.array(datedline(datadir+"RSSspecalign.txt",dateobs).split()[1:]).astype(float)          
    Grat0 = spec_dp[0]
    grateffedge = 0.04

    grnum = np.where(grname==grating)[0][0]    
    lmm = grlmm[grnum]
    alpha_r = np.radians(grang+Grat0+alftarg)

    if (grating=="PG0300"):     # SR grating
        gr300wav,gr300eff,gr300ps=np.loadtxt(datadir+"grateff_0300.txt",usecols=(0,1,2),unpack=True)
        eff = interp1d(gr300wav,gr300eff,kind='cubic',bounds_error=False)(wav)
        ps = interp1d(gr300wav,gr300ps,kind='cubic',bounds_error=False)(wav)
                 
    elif (grating=="PG0700"):   # VPH tilted-groove grating: interpolate in grang space
        gratefffile = datadir+"grateff_0700.txt"
        gr700Wav = np.loadtxt(gratefffile,usecols=0)        
        Wavs = gr700Wav.shape[0]
        for line in open(gratefffile, 'r').readlines():
            if (line[:7] == "# grang"): 
                grang_g = np.fromstring(line[7:],sep=' ')
                break
        grangs = grang_g.shape[0] 
        
        if (grangs==0):
            print " Error: faulty file "+gratefffile
            exit()
            
        eff_gW = np.loadtxt(gratefffile,usecols=range(1,2*(grangs+1)-1,2),unpack=True)                           
        ps_gW = np.loadtxt(gratefffile,usecols=range(2,2*(grangs+1),2),unpack=True)       
        eff_W = np.zeros(Wavs)
        ps_W = np.zeros(Wavs)        
        for W in range(Wavs):
            eff_W[W] = interp1d(grang_g,eff_gW[:,W])(grang)
            ps_W[W] = interp1d(grang_g,ps_gW[:,W])(grang)          
        eff = interp1d(gr700Wav,eff_W,kind='cubic',bounds_error=False)(wav)
        ps = interp1d(gr700Wav,ps_W,kind='cubic',bounds_error=False)(wav)
        
    else:                       # Kogelnik gratings
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
def skypupillum(YX_d, MByx_d, TRKyx_d, trkrho, imgtime, **kwargs):
    """compute sky pupil illumination at FOV, MB, and rho from ET10 model
    Parameters
    ----------
    YX_d: numpy 1d array
        _d: Y,X (mm) at RSS FP
    MByx_d: numpy 1d array
        _d: MBY,MBX (m) from fits header (same for all targets)
    TRKyx_d: numpy 1d array
        _d: TRKY,TRKX (m) from fits header (same for all targets).  Ignore if ""        
    trkrho: TRKRHO (deg) from fits header (same for all targets)
    
    Returns: 
    -------
    model_RC: numpy 2d float array.  pupil illumination (0. - 1.) 
    modelarea: sum of illum relative to filled pupil
    modelareanom: modelarea if MB correctly placed    

    """
    pupilparamfile = kwargs.pop('pupilparamfile','polsalt')      
    if (pupilparamfile == 'polsalt'):
        pupilparamfile = datadir+'skypupilparams.json'           
    debug = kwargs.pop('debug',False)
    useshuttercor =  kwargs.pop('useshuttercor',False)     
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
    shuttertime = 0. 
    if useshuttercor: shuttertime = paramDict["shuttertime"]      

    YXtarg = 5.                                 # mm target separation in ET10        
    sacpupildia = 2.*primwidth/np.sqrt(3)       # 11m SAC pupil in ET10 image pixels        
    pupPixs = int(sacpupildia/2.)*2 +1  
        
    pupshapeyx_ds = np.indices((pupPixs,pupPixs)).reshape((2,-1)) -  \
        int(pupPixs/2)*np.ones((2,pupPixs**2))
    pupshape_s = (pupshapeyx_ds**2).sum(axis=0) <= (sacpupildia/2.)**2   
    pupshape_RC = pupshape_s.reshape((pupPixs,pupPixs))
    pupyx_dRC = pupshapeyx_ds.reshape((2,pupPixs,pupPixs))/(sacpupildia/2.)
                         
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

    shutteropentime = 100.                           # fits image exptime is short by 100 ms open time
    shuttercor = (shuttertime/1000.)/(imgtime + shutteropentime/1000.)    
    shuttercor_RC = 1. - shuttercor*np.sqrt((pupyx_dRC**2).sum(axis=0))
        
    model_RC = shuttercor_RC*   \
        (pupshape_RC & (MBshape_RC & np.logical_not(obscshape0_RC | obscshape1_RC))).astype(float)
    modelarea = model_RC.sum()/modelarea00
    modelareanom = (shuttercor_RC*  \
        (pupshape_RC & (primshape_RC & np.logical_not(obscshape0_RC |obscshape1_RC))).astype(float)).sum()/modelarea00         

    if debug:
        shuttercorFile = open("shuttercor_debug.txt",'w')    
        for R in range(pupPixs):
            print >>shuttercorFile, (pupPixs*" %6.3f" % tuple(shuttercor_RC[R]))                
        skypupFile = open("skypup_debug.txt",'w')    
        for R in range(pupPixs):
            print >>skypupFile, (pupPixs*" %6.3f" % tuple(model_RC[R]))    

    return model_RC, modelarea, modelareanom

# ----------------------------------------------------------
def heffcal(heffcalfile,YX_dt,MByx_d,TRKyx_d,grang,trkrho,imgtime,wav_w,wvplt, **kwargs):
    """get P and PA (HWPol), hw efficiency cal file    
    Parameters
    ----------
    heffcalfile: text
        name of heffcal zip file to use
    YX_dt: numpy 2d array
        _d: Y,X (mm) at RSS FP
        _t: targets
    MByx_d: numpy 1d array
        _d: MBY,MBX (m) from fits header (same for all targets)
    TRKyx_d: numpy 1d array
        _d: TRKY,TRKX (m) from fits header (same for all targets).  Ignore if ""        
    grang: float grating angle (deg)
    rho: TRKRHO (deg) from fits header (same for all targets)
    imgtime: raw image EXPTIME (sec) (same for all targets)    
    wav_w: numpy 1d array
        _w: wavelengths (Ang)
    wvplt: string ("04", "26", from rawstokes)
    illum_vt: illum= kwarg, override sky illumination model (eg for lamp) 
    
    Returns: numpy 2d arrays:
    -------
    heffcal_ptw[0,1]: 2 numpy 2d arrays
        hwcal polarization stokes for each target and wavelength.  =0 where invalid
    
    """
    """
    _p: (0,1) %P, PA
    _z: files in heffcal zip file 
    _W: wavelengths in heffcal zip file
    _t: YX targets in request    
    _T: YX targets in heffcal zip file    
    _c: coefficents (terms in fit)
    _v: variables in cal file
    _V: standard variable order, (pupil y,x,r)
    _t: targets in science data
    
    """
    debug = kwargs.pop('debug',False)    
    logfile= kwargs.pop('logfile','hwcal.log')       
    illum_vt= kwargs.pop('illum','')
    useMPlat = kwargs.pop('useMPlat',False)
    wpcalcor = kwargs.pop('wpcalcor','')           
    pupilparamfile = kwargs.pop('pupilparamfile','polsalt')      
    if (pupilparamfile == 'polsalt'):
        pupilparamfile = datadir+'skypupilparams.json'          
    fovrad = 52.
    paramDict = load(open(pupilparamfile))    
    primwidth = paramDict["primwidth"]
    sacpupilrad = primwidth/np.sqrt(3.) 
              
    targets = YX_dt.shape[1]    
    wavs = wav_w.shape[0]  
               
  # input heffcal file data. if zip, targets are stacked in zip. _p=0,1 are stacked inside each target 
    calfilezipped = (heffcalfile.split('.')[-1]  == 'zip')

    if calfilezipped:  
        calZip = ZipFile(heffcalfile,mode='r')
        zipList = calZip.namelist()
        Targets = len(zipList)    
      # get common header calibration data from first target
        callineList = list(filter(None,calZip.read(zipList[0]).split('\n')))      # remove empty lines
    else:
        Targets = 1
        callineList = list(filter(None,open(heffcalfile,'r').read().split('\n'))) # remove empty lines
                
    hdrs = [x[0] for x in callineList].count('#')
    cofs = len(callineList[hdrs].split()) - 1   
    hdrlineList = callineList[:hdrs]
    hdrtitleList = [hdrlineList[x][1:].split()[0] for x in range(hdrs)]

    PAline = hdrtitleList.index("calPA")     
    calPA = np.fromstring(hdrlineList[PAline][9:],dtype=float,sep=" ")[0]
    useshuttercor = (hdrtitleList.count("useshuttercor") > 0)
    pacorfile = ''
    if (hdrtitleList.count("pacor") > 0):        
        pacorfile = hdrlineList[hdrtitleList.index("pacor")].split()[-1]         
    isgrangdata = hdrtitleList.count("grang0")            
    varlineList = list(np.where(np.array(map(len,hdrtitleList))==1)[0])
    vars = len(varlineList)       
    varList = [hdrtitleList[x] for x in varlineList]
    varname_V = np.array(['y','x','r'])
    varidx_V = np.array([varList.index(varname) for varname in varname_V])
    pow_Vc = np.zeros((vars,cofs),dtype=int)
    for V in range(vars): 
        pow_Vc[V] = np.fromstring(hdrlineList[varlineList[varidx_V[V]]][9:],dtype=int,sep=" ")

    if isgrangdata:
        grang0line = hdrtitleList.index("grang0")     
        grang0 = np.fromstring(hdrlineList[grang0line][9:],dtype=float,sep=" ")[0]
        data_px = np.array([x.split()[0] for x in callineList[hdrs:]]).astype(float).reshape((2,-1))
        dgrang = data_px[0,0]
        wav_pW = data_px[:,1:]        
    else:     
        wav_pW = np.array([x.split()[0] for x in callineList[hdrs:]]).astype(float).reshape((2,-1))
    Wavs = wav_pW.shape[1]
    wav_W = wav_pW[0]    

  # get target-dependent calibration data                
    YX_dT = np.zeros((2,Targets))
    pupoffyx_dT = np.zeros((2,Targets))
    grangcor_Tpc = np.zeros((Targets,2,cofs))
    heffcal_TpcW = np.zeros((Targets,2,cofs,Wavs))
    ok_TW = np.zeros((Targets,Wavs),dtype=bool)
        
    for T in range(Targets):
        if calfilezipped:
            callineList = list(filter(None,calZip.read(zipList[T]).split('\n')))
        hdrlineList = callineList[:hdrs]
        hdrtitleList = [hdrlineList[x][1:].split()[0] for x in range(hdrs)]                
        YXline = hdrtitleList.index("fovYX")
        YX_dT[:,T] = np.fromstring(hdrlineList[YXline][9:],dtype=float,sep=" ")[:2]
        pupoffline = hdrtitleList.index("offyx")
        pupoffyx_dT[:,T] = np.fromstring(hdrlineList[pupoffline][9:],dtype=float,sep=" ")[:2]
        if isgrangdata:
            data_pcx = np.array([x.split() for x in callineList[hdrs:]]).astype(float)[:,1:] \
                .reshape((2,Wavs+1,-1)).transpose((0,2,1))
            grangcor_Tpc[T] = data_pcx[:,:,0] 
            heffcal_TpcW[T] = data_pcx[:,:,1:]                       
        else:                           
            heffcal_TpcW[T] = np.array([x.split() for x in callineList[hdrs:]]).astype(float)[:,1:] \
                .reshape((2,Wavs,-1)).transpose((0,2,1))
        ok_TW[T] = (heffcal_TpcW[T,0,0] != 0.)

  # FOR NOW: assume on-axis (YX_t ~ 0)  Use on-axis T=T0
  # LATER: use interp2d to interpolate polarization map coefficients in T for each t  
    T0 = np.argmin(np.sqrt((YX_dT**2).sum()))       # on-axis cal target   
    heffcal_tpcW = np.tile(heffcal_TpcW[T0][None,:,:,:],(targets,1,1,1))
    pupoffyx_dt  = np.tile(pupoffyx_dT[:,T0][None,:],(1,targets))
    grangcor_tpc = np.tile(grangcor_Tpc[T0][None,:],(targets,1,1))
    ok_W = ok_TW[T0]
    ok_t = ((np.sqrt((YX_dt**2).sum(axis=0)) < fovrad) & (np.abs(YX_dt[0]) < 0.5*fovrad))
    wpangcor = 0.
    if ("wpcor" in hdrtitleList):
        wpcorline = hdrtitleList.index("wpcor")     
        wpcalcor = np.fromstring(hdrlineList[wpcorline][9:],dtype=float,sep=" ")[0] 
        wpangcor = wpcalcor*float(wvplt[-2])    
    Rows,Cols = (skypupillum(YX_dT[:,T0], MByx_d, TRKyx_d, trkrho-wpangcor, imgtime))[0].shape
    rcmax_d = np.array([Rows-1,Cols-1])/2.
    inpup_RC = ((np.indices((Rows,Cols)) - rcmax_d[:,None,None])/sacpupilrad <= 1.)
    pupyxr_VRC = np.zeros((3,Rows,Cols))
    heffcal_ptW = np.zeros((2,targets,Wavs))
    modelarea_t = np.zeros(targets)
    modelareanom_t = np.zeros(targets)
    minillum_t = np.zeros(targets)            

  # finally, put sky illumination and polarization pupil map together for each p, W and t
        
    for t in range(targets):
    # get model sky pupil illumination for target       
        skymodel_RC, modelarea_t[t], modelareanom_t[t] =   \
            skypupillum(YX_dt[:,t], MByx_d, TRKyx_d, trkrho-wpangcor, imgtime,   \
            useshuttercor=useshuttercor, useMPlat=useMPlat, pupilparamfile=pupilparamfile)
        minillum_t[t] = skymodel_RC[skymodel_RC>0.].min()

    # compute polarization pupil map vs wavelength for target from coefficients           
      # first, compute the dependent variable part of each term (does not depend on p or W)
        pupyxr_VRC[:2] = (np.indices((Rows,Cols)) +     \
            (pupoffyx_dt[:,t] - rcmax_d)[:,None,None])/sacpupilrad
        pupyxr_VRC[2] = np.sqrt((pupyxr_VRC[:2]**2).sum(axis=0))      
        polterm_cRC = np.ones((cofs,Rows,Cols))
        for V in range(vars):
            use_c = (pow_Vc[V] > 0)
            polterm_cRC[use_c] = polterm_cRC[use_c]*pupyxr_VRC[V][None,:,:]**pow_Vc[V,use_c][:,None,None]
            
      # next, multiply by coefficient from cal file and sum over terms
        polmodelcor_pRC = ((grang-grang0)/dgrang)*  \
            (grangcor_tpc[t][:,:,None,None]*polterm_cRC[None,:,:,:]).sum(axis=1)
                                         
        for W in range(Wavs):            
            polmodel_pRC =  polmodelcor_pRC +   \
                (heffcal_tpcW[t,:,:,W][:,:,None,None]*polterm_cRC[None,:,:,:]).sum(axis=1)
            
          # final answer for each wavelength is mean over pupil of polmodel weighted by skymodel                        
            heffcal_ptW[:,t,W] = (skymodel_RC[None,:,:]*polmodel_pRC).sum(axis=(1,2))/skymodel_RC.sum()

  # interpolate heffcal to desired wavelengths                
    ok_w = (wav_w > (wav_W[ok_W]).min()) & (wav_w < (wav_W[ok_W]).max())    
    ok_tw = (ok_t[:,None] & ok_w[None,:])
    heffcal_ptw = np.zeros((2,targets,wavs))
    for p in [0,1]:
        heffcal_ptw[p,ok_tw] = interp1d(wav_W[ok_W],heffcal_ptW[p,ok_t][:,ok_W], \
            kind='cubic',bounds_error=False)(wav_w[ok_w]).flatten()                               
    heffcal_ptw[1,ok_tw] = heffcal_ptw[1,ok_tw] + calPA
 
    return heffcal_ptw, ok_tw, pacorfile, modelarea_t, modelareanom_t, minillum_t

# ----------------------------------------------------------
def heffcalcorrect(heffcorrectfile,wav_w,stokes_Sw,var_Sw,covar_Sw,ok_w,**kwargs):
    """correct stokes file for RSS PA dependence   
    Parameters
    ----------
    heffcorrectfile: text
        name of heffcalcorrect txt file to use (one target only)
   
    wav_w: numpy 1d array
        _w: wavelengths (Ang)
    stokes_Sw: numpy 2d arrays
        input stokes, RSS PA frame
    slitdwav: float slit width wavlength
        
    Returns: numpy 2d arrays:
    -------
    stokes_Sw: numpy 2d arrays
        corrected polarization stokes for each wavelength.  =0 where invalid
    
    """
    """
    _S:  I, Q, U (unnormalized) stokes
    _v: (0,1) %P, PA
    _W: wavelengths in smoothed PAin file
    _x: wavelengths in correctfile 
    
    """
    debug = kwargs.pop('debug',False)    
    logfile= kwargs.pop('logfile','hwcal.log') 
    slitdwav = kwargs.pop('slitdwav','')

  # first smooth input PA down to slitdwav, to avoid coupling in statistical noise
    wavs = len(wav_w)
    dwav = np.diff(wav_w)[0]      
    if (slitdwav is None):
        slitdwav = 4.*dwav
    Wavs = int(round(dwav*wavs/slitdwav))
    dwdW = wavs/Wavs
    slitdwav = dwav*dwdW
    wend = int(dwdW*Wavs)
    wav_W = np.arange(wav_w[0]+slitdwav/2.,wav_w[0]+slitdwav/2.+ slitdwav*Wavs,slitdwav)     
    count_W = ok_w[:wend].reshape((dwdW,Wavs)).sum(axis=0)
    ok_W = (count_W>0)
    stokes_SW = np.zeros((3,Wavs))
    stokes_SW[:,ok_W] = stokes_Sw[:,:wend].reshape((3,dwdW,Wavs)).sum(axis=1)[:,ok_W]/count_W[ok_W]
    var_SW = np.zeros((4,Wavs))
    var_SW[:,ok_W] = var_Sw[:,:wend].reshape((4,dwdW,Wavs)).sum(axis=1)[:,ok_W]/count_W[ok_W]**2    
    stokes_vW = viewstokes(stokes_SW,var_SW,ok_W)[0]
    tcenter = np.median(stokes_vW[1,ok_W])

  # get correctfile data, interpolate to _W
    correctlineList = list(filter(None,open(heffcorrectfile, 'r').read().split('\n')))    # remove empty lines
    hdrs = [x[0] for x in correctlineList].count('#')
    hdrlineList = correctlineList[:hdrs]
    hdrtitleList = [hdrlineList[x][1:].split()[0] for x in range(hdrs)]     
    PAs = len(correctlineList[hdrs].split()) - 1    
    wav_vx = np.array([x.split()[0] for x in correctlineList[hdrs:]]).astype(float).reshape((2,-1))
    correctwavs = wav_vx.shape[1]
    wav_x = wav_vx[0]       
    ok_x = np.zeros(correctwavs,dtype=bool)                     
    PAline = hdrtitleList.index("calPA")
    PA_P = np.zeros(5)
    PA_P[1:4] = np.fromstring(hdrlineList[PAline][9:],dtype=float,sep=" ")
    PA_P[[0,4]] = [-90.,90.]    
    heffcorrect_vPx = np.zeros((2,5,correctwavs))                               
    heffcorrect_vPx[:,1:4] = np.array([x.split() for x in correctlineList[hdrs:]]).astype(float)[:,1:] \
        .reshape((2,correctwavs,-1)).transpose((0,2,1))
    ok_x = (heffcorrect_vPx[0,1] != 0.)
    xmin,xmax = np.where(ok_x)[0][[0,-1]]        
    heffcorrect_vPx[0,[0,4]] = ok_x*100.*np.ones(correctwavs)
    heffcorrect_vPx[1,[0,4]] = np.zeros(correctwavs)        

    if debug:    
        np.savetxt("heffcorrect_vPx_debug.txt",np.vstack((wav_x,ok_x,heffcorrect_vPx.reshape((-1,correctwavs)))).T, 
            fmt=" %8.1f %2i"+10*" %10.4f")
        
    heffcorrect_vPW = np.zeros((2,5,Wavs))
    Wmin = np.where(wav_W > wav_x[xmin]-(wav_x[xmin+1]-wav_x[xmin])/2.)[0][0]
    Wmax = np.where(wav_W < wav_x[xmax]+(wav_x[xmax]-wav_x[xmax-1])/2.)[0][-1]
    ok_W = ((np.arange(Wavs) >=Wmin) & (np.arange(Wavs) <= Wmax))
    heffcorrect_vPW[:,:,ok_W] =    \
        CubicSpline(wav_x[ok_x],heffcorrect_vPx[:,:,ok_x],axis=2)(wav_W[ok_W])

  # compute heffcorrect_vW for PA stokes_vW[1] using periodic spline over PA
  #   abscissa of Spline is on pa = (0,1); PA = (-90,90)
  #   use first iteration of Newton-Raphson to solve for PAout
    heffcorrect_vW = np.zeros((2,Wavs))
    pa_P = (PA_P+90.)/180.
    pastokes_W = (stokes_vW[1] +90.)/180.

    Wout = -1
    if debug:
        Wavout = 7500.
        if (np.fabs(wav_W-Wavout).min()<20.):
            Wout = np.where(np.fabs(wav_W-Wavout) == np.fabs(wav_W-Wavout).min())[0][0]
    
    for W in np.where(ok_W)[0]:
        dPASpline = CubicSpline(pa_P,heffcorrect_vPW[1,:,W],bc_type='periodic')
        paout = pastokes_W[W] - dPASpline(pastokes_W[W])/(180.-dPASpline.derivative()(pastokes_W[W]))        
        heffcorrect_vW[1,W] = dPASpline(paout)
        poldegSpline = CubicSpline(pa_P,heffcorrect_vPW[0,:,W],bc_type='periodic')
        heffcorrect_vW[0,W] = poldegSpline(paout)

        if (debug & (W==Wout)):       
            hdr = ((" %i3"+3*" %8.3f"+"\n"+ 5*" %8.3f") % ((Wout, wav_W[Wout], stokes_vW[1,W], pastokes_W[W],     \
                dPASpline(pastokes_W[W]), dPASpline.derivative()(pastokes_W[W]), paout) + tuple(heffcorrect_vW[:,W])))
            p_p = np.arange(0.,1.05,.05)
            P_p = 180.*p_p - 90.
            np.savetxt("heffSpline_p_debug.txt",   \
                np.vstack((p_p,P_p,dPASpline(p_p),dPASpline.derivative()(p_p),poldegSpline(p_p))).T,    \
                header=hdr, fmt=" %8.3f")
    
  # interpolate to original wav scale, apply, and return
    heffcorrect_vw = interp1d(wav_W[ok_W],heffcorrect_vW[:,ok_W],fill_value='extrapolate')(wav_w)

    if debug:
        np.savetxt("heffcorrect_vPW_debug.txt",np.vstack((wav_W,ok_W,heffcorrect_vPW.reshape((-1,Wavs)),heffcorrect_vW)).T, \
            fmt=" %8.1f %2i"+12*" %10.4f")    
        np.savetxt("heffcorrect_vw_debug.txt",np.vstack((wav_w,ok_w,heffcorrect_vw)).T, \
            fmt=" %8.1f %2i"+2*" %10.4f")

    stokescor_Sw = np.copy(stokes_Sw) 
    stokescor_Sw[1:,ok_w] = stokes_Sw[1:,ok_w]/(heffcorrect_vw[0,ok_w]/100.)
    varcor_Sw = np.copy(var_Sw) 
    varcor_Sw[1:,ok_w] = var_Sw[1:,ok_w]/(heffcorrect_vw[0,ok_w]/100.)**2
    covarcor_Sw = np.copy(covar_Sw) 
    covarcor_Sw[1:,ok_w] = covar_Sw[1:,ok_w]/(heffcorrect_vw[0,ok_w]/100.)**2
    stokescor_Sw,varcor_Sw,covarcor_Sw = specpolrotate(stokescor_Sw,varcor_Sw,covarcor_Sw,-heffcorrect_vw[1])
    
    return stokescor_Sw,varcor_Sw,covarcor_Sw
               
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

def fpchecpy(x_I,t_J,k=3,debug=False):
  # check Schoenberg-Whitney criteria for LSQSpline
  # sttempts to duplicate scipy fortran dfitpack/fpchec
  # returns which criterion failed
  # x_i, t_j use fortran indexing 1,...
    n = t_J.shape[0]
    m = x_I.shape[0]
    x_i = np.append(0.,x_I)
    t_j = np.append(0.,t_J)
    k1 = k+1
    k2 = k1+1
    nk1 = n-k1
    nk2 = nk1+1
  #       
    if (nk1<k1): return 1, 0
    if (nk1>m): return 1, 1
  #        
    jj = np.copy(n)
    for j in range(1,k+1):
        if (t_j[j] > t_j[j+1]): return 2, 0, j
        if (t_j[jj] < t_j[jj-1]): return 2, 1, jj
        jj = jj-1
  #        
    for j in range(k2,nk2+1):
        if(t_j[j] <= t_j[j-1]): return 3, j
  #        
    if (x_i[1] < t_j[k1]): return 4, 0
    if (x_i[m] > t_j[nk2]): return 4, 1    
  #
    if(x_i[1] >= t_j[k2]): return 5, 0
    if(x_i[m] <= t_j[nk1]): return 5, 1
    i = 1
    ll = np.copy(k2)
    nk3 = nk1 - 1
    if (nk3 < 2): return 0
    for j in range(2,nk3+1):
        tj = t_j[j]
        ll = ll+1
        tll = t_j[ll]        
        if debug: print j, tj,tll
        
        while True:   
            i = i+1
            if (i > m): return 5, 2, j            
            if debug: print "  ", i, x_i[i]
            
            if (x_i[i] > tj): break            
        if (x_i[i] >= tll): return 5, 3, j
  #        
    return 0    
# ------------------------------------          

def SWcheck(x_i,t_j,k=3):
  # tabulate Schoenberg-Whitney criterion 5 for LSQSpline
  # returns count of data points t[j] < x < t[j+k+1].  Must be > 0
    knots = t_j.shape[0]
    datapts = x_i.shape[0]
    ptsSW_j = np.zeros(knots-(k+1),dtype=int)
    for j in range(knots-(k+1)):
        ptsSW_j[j] = ((t_j[j] < x_i) & (x_i < t_j[j+k+1])).sum()
    return ptsSW_j 
# ------------------------------------

def fence(arr):
  # return lower outer, lower inner, upper inner, and upper outer quartile fence
    Q1,Q3 = np.percentile(arr,(25.,75.))
    IQ = Q3-Q1
    return Q1-3*IQ, Q1-1.5*IQ, Q3+1.5*IQ, Q3+3*IQ
# ---------------------------------------------------------------------------------

def chi2sample(f_ax,var_ax,ok_x,X_x,dX,debugfile=''):
  # return chisq in X, length dX, over f_ax arrays
  # f_ax, var_ax: inputs, can be one-dim in x
  # ok_x: 
  #    if scalar (eg True), is constant
  # X_x:  
  #    x values (float)
  # dX:
  #    if scalar, dX is constant
  #    if array, is dX_X
  # _h: chi2sample index 
  # returns: chi2dof_ha,mean_ha,Xmean_h,xlen_h

    isonedim = (f_ax.ndim == 1)
    if isonedim:
        f_ax = np.expand_dims(f_ax,0)
        var_ax = np.expand_dims(var_ax,0)     
    xes = f_ax.shape[1]
    
    if np.isscalar(ok_x):
        ok_x = ok_x*np.ones(xes,dtype=bool)
             
    if np.isscalar(dX):
        dX_x = dX*np.ones(xes)    
    else:
        dX_x = dX

    if debugfile:
        debugFile = open(debugfile,'w')
        debugFile = open(debugfile,'a')
                                
    chi2dof_aList = []
    mean_aList = []
    var_aList = []       
    XmeanList = []
    xlenList = []
    
    Xmax = X_x[ok_x][-1]
    
    for x0 in np.where(ok_x)[0]:
        X0 = X_x[x0]        
        if (X0+dX_x[x0] >= Xmax): break
        xhArray = np.where(ok_x & (X_x > X0) & (X_x < X0+dX_x[x0]))[0]
        xhallArray = np.where((X_x > X0) & (X_x < X0+dX_x[x0]))[0] 
        
        if (len(debugfile) & (len(xhArray)<len(xhallArray)/2)):        
            debugFile.write(' %4i %8.2f %8.2f %4i %11.6f \n' %  \
                (x0, X0, X0+dX_x[x0], len(xhArray), f_ax[0,xhArray[-1]]))        
                             
        if (len(xhArray)<len(xhallArray)/2): continue
                         
        chi2dof_aList.append(f_ax[:,xhArray].sum(axis=1)**2/var_ax[:,xhArray].sum(axis=1))
        mean_aList.append(f_ax[:,xhArray].mean(axis=1))
        var_aList.append(var_ax[:,xhArray].mean(axis=1)/len(xhArray))        
        XmeanList.append(X_x[xhArray].mean())
        xlenList.append(len(xhArray))
        
        if debugfile:        
            debugFile.write(' %4i %8.2f %8.2f %4i %11.6f %10.1f %11.6f %12.3e %8.2f \n' %  \
                (x0, X0, X0+dX_x[x0], len(xhArray), f_ax[0,xhArray[-1]],    \
                chi2dof_aList[-1][0],mean_aList[-1][0],var_aList[-1][0],XmeanList[-1]))
       
    chi2dof_ha = np.array(chi2dof_aList)
    mean_ha = np.array(mean_aList)
    var_ha = np.array(var_aList)       
    Xmean_h = np.array(XmeanList) 
    xlen_h = np.array(xlenList)
    
    if debugfile: debugFile.close
    if isonedim:                       
        return chi2dof_ha[:,0],mean_ha[:,0],var_ha[:,0],Xmean_h,xlen_h
    else:
        return chi2dof_ha,mean_ha,var_ha,Xmean_h,xlen_h

# ---------------------------------------------------------------------------------

def legfit_cull(x_x,y_x,ok_x,deg,xlim_d=(-1.,1.),yerr=0.,docull=True,IQcull=3., \
    maxerr=None,usefitchi=False,debugname=''):
    """do legendre polyfit, culling weighted errors to fence (one at a time), default outer cull 
     if no fence cull, apply option maxerr cull
    at most 25% of startvals
    x_x: array of floats
        legendre abscissae of data points (should be on -1,1 interval)
    y_x: array of floats
        data
    ok_x: array of booleans
    deg: int degree (1 = linear)
    yerr: if array of floats, errors for data, else unwtd (default)
    xlim_d: tuple of floats.  If specified, limit fit to smaller interval
    docull: 
        if True (default), do fence culling, with IQcull fence (3 default)
        if False, use maxerr
    usefitchi: default(False), use chis from input errs
        if True: add in (in quadrature) error to fit

    returns: fit_d, ok_x, err_x, fiterr_x, fiterr_r, cullLog  
        fit_d is coefficients low to high (for legval)
   """
    startvals = ok_x.sum()
    if (isinstance(yerr,float)):
        yerr_x = np.zeros_like(x_x)
        wt_x = np.ones_like(x_x)
    else:
        yerr_x = yerr
        wt_x = np.zeros_like(x_x)
        wt_x[ok_x] = 1./yerr_x[ok_x]
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
            print >>debugfile, (deg+1)*"%10.3f " % tuple(fit_d)
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

    Returns stokes, var, covar (as copy)

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

