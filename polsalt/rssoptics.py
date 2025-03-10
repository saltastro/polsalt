
# optics-related tools, including:

# RSScoldistort(yx_ds,wav_s,coltem,distTab=[],debug=False)
# RSScamdistort(alfyx_ds,wav_s,camtem,distTab=[],debug=False)
# RSSdisperse(alfyx_ds,wav_s,grating,grang,artic,dateobs,order=1,debug=False)
# RSSpolsplit(alfyx_ds,wav_s,distTab=[],EOratTab=[],debug=False)
# RSScolpolcam(YX_ds, wav_s, coltem, camtem, yxOEoff_d=np.zeros(2))
# RSScolgratpolcam(YX_ds, wav_s, coltem, camtem, grating, grang, artic, dateobs,  
#      order=1,yxOEoff_d=np.zeros(2),dalfyx_d=np.zeros(2),debug=False)
# RSSpolgeom(hdul,wavl,yxOEoff_d=np.zeros(2))

import os, sys, glob, copy, shutil, inspect
import numpy as np
from scipy.interpolate import interp1d, griddata
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table,unique
import astropy.table as ta
from polutils import rssdtralign, datedfile, datedline
from polmaptools import Tableinterp,rotate2d,ccdcenter

datadir = os.path.dirname(__file__) + '/data/'

# ------------------------------------
def RSScoldistort(YX_ds,wav_s,coltem,distTab=[],debug=False):
    """apply collimator distortion to array of Y,X coordinates

    Parameters:
    YX_ds: 2d numarray input coordinates (mm, SALT focal plane)
        _d: 0,1 for Y,X
        _s: index of inputs
    wav_s: 1d numarray (or float, if all the same) wavelength (Ang)
    disttab: astropy Table of distortion model parameters, vs wavelength

    Returns: alfyx_ds (degrees) angles in collimated space

    """
    stars = YX_ds.shape[1]
    if np.isscalar(wav_s): wav_s = np.repeat(wav_s,stars)

    imgoptfile = datadir+'RSSimgopt.txt'
    if len(distTab)==0:
        distTab = ascii.read(imgoptfile,data_start=1,   \
            names=['Wavel','Fcoll','Acoll','Bcoll','ydcoll','xdcoll','Fcam','acam','alfydcam','alfxdcam'])
    dFcoldt,dFcamdt = ascii.read(imgoptfile,data_end=1)[0]
    sdistTab = Tableinterp(distTab,'Wavel',wav_s)
    Fcam,Fcoll = Tableinterp(distTab,'Wavel',5500.)['Fcam','Fcoll'][0]

    YXdcoll_ds = np.array([sdistTab['ydcoll'],sdistTab['xdcoll']])* Fcoll/Fcam

  # correct for collimator distortion
    Rd_s = np.sqrt(((YX_ds - YXdcoll_ds)**2).sum(axis=0))/50.
    dist_s = (1. + sdistTab['Acoll']*Rd_s**2 + sdistTab['Bcoll']*Rd_s**4)
    YXd_ds = YXdcoll_ds + dist_s*(YX_ds - YXdcoll_ds)
    alfyx_ds = np.degrees(YXd_ds/(sdistTab['Fcoll'] + dFcoldt*(coltem - 7.5)))

    if debug: np.savetxt("coldist.txt",np.vstack((wav_s,YX_ds,dist_s,YXd_ds,alfyx_ds)).T,  \
        fmt=3*"%7.2f "+"%10.4f "+2*"%7.2f "+2*"%9.4f ")

    return alfyx_ds

# ------------------------------------
def RSScamdistort(alfyx_ds,wav_s,camtem,distTab=[],debug=False):
    """apply camera distortion to array of y,x ray angles

    Parameters:
    alfyx_ds: 2d numarray input coordinates (deg)
        _d: 0,1 for y,x
        _s: index of inputs
    wav_s: 1d numarray (or float, if all the same) wavelength (Ang)
    disttab: astropy Table of distortion model parameters, vs wavelength

    Returns: yx_ds (mm) at the detector

    """
    stars = alfyx_ds.shape[1]
    if np.isscalar(wav_s): wav_s = np.repeat(wav_s,stars)

    imgoptfile = datadir+'RSSimgopt.txt'
    if len(distTab)==0:
        distTab = ascii.read(imgoptfile,data_start=1,   \
            names=['Wavel','Fcoll','Acoll','Bcoll','ydcoll','xdcoll','Fcam','acam','alfydcam','alfxdcam'])
    dFcoldt,dFcamdt = ascii.read(imgoptfile,data_end=1)[0]       
    sdistTab = Tableinterp(distTab,'Wavel',wav_s)

    alfyxdcam_ds = np.array([sdistTab['alfydcam'],sdistTab['alfxdcam']])

  # correct for camera distortion 
    alfrd_s = np.sqrt(((alfyx_ds - alfyxdcam_ds)**2).sum(axis=0))/5.
    dist_s = (1. + sdistTab['acam']*alfrd_s**2)
    alfyxd_ds = alfyxdcam_ds + dist_s*(alfyx_ds - alfyxdcam_ds)
    yx_ds = (sdistTab['Fcam'] + dFcamdt*(camtem - 7.5))*np.radians(alfyxd_ds)

    if debug: np.savetxt("camdist.txt",np.vstack((wav_s,alfyx_ds,dist_s,alfyxd_ds,yx_ds)).T,  \
        fmt="%7.2f "+5*"%10.4f "+2*"%7.2f ")

    return yx_ds

# ------------------------------------
def RSSdisperse(alfyx_ds,wav_s,grating,grang,artic,dateobs,order=1,debug=False):
    """apply grating dispersion to array of y,x ray angles

    Parameters:
    alfyx_ds: 2d numarray input coordinates (deg)
        _d: 0,1 for y,x
        _s: index of inputs
    wav_s: 1d numarray (or float, if all the same) wavelength (Ang)
    grating,grang,artic: grating configuration

    Returns: alfdispyx_ds (degrees) angles in articulated collimated space

    """
    stars = alfyx_ds.shape[1]
    if np.isscalar(wav_s): wav_s = np.repeat(wav_s,stars)
    mjdateobs = str(int(dateobs))
    
    specfile = datadir+'RSSspecalign.txt'
    Grat0,Home0,ArtErr = np.array(datedline(specfile,dateobs).split()[1:]).astype(float)[:3]    
    gratingfile = datedfile(datadir+"gratings_yyyymmdd_vnn.txt",mjdateobs)    
    grname=np.loadtxt(gratingfile,dtype=str,usecols=(0,))
    grlmm,tilt0,sinart,cosart=np.loadtxt(gratingfile,usecols=(1,2,3,4),unpack=True)
    grnum = np.where(grname==grating)[0][0]
    lmm = grlmm[grnum]
    dgamma = tilt0[grnum] + sinart[grnum]*np.sin(np.radians(artic)) +   \
         cosart[grnum]*np.cos(np.radians(artic))

  # compute dispersion, simple tilt correction for now
    alpha_s = -alfyx_ds[1] + grang + Grat0
    gamma_s = alfyx_ds[0]
    sinbeta = np.clip(order*lmm*wav_s/(1e7*np.cos(np.radians(gamma_s))) - np.sin(np.radians(alpha_s)),-1.,1.)
    beta_s = np.degrees(np.arcsin(sinbeta))
        
    alfdispyx_ds = np.zeros_like(alfyx_ds)
    alfdispyx_ds[0] = gamma_s + dgamma
    alfdispyx_ds[1] = grang + beta_s - (artic*(1+ArtErr)+Home0)

    if debug:
        np.savetxt("disperse.txt",np.vstack((wav_s,alfyx_ds,alpha_s,beta_s,gamma_s,alfdispyx_ds)).T,  \
        fmt="%7.2f "+7*"%10.4f ")

    return alfdispyx_ds
# ------------------------------------
def RSSpolsplit(alfyx_ds,wav_s,debug=False):
    """apply wollaston prism distortion and splitting for beam OE to array of y,x angles

    Parameters:
    alfyx_ds: 2d numarray input angles (deg), relative to optical axis
        _d: 0,1 for y,x
        _s: index of input targets
    wav_s: float or 1d numarray float
        wavelength (Ang) for point s.  If not an array, use for all points
    distTab: astropy Table of distortion model parameters for this beam, vs wavelength

    Returns: w-distorted and deviated alfyx_dps, (_p = 0,1 for O,E), relative to optical axis

    """

    stars = alfyx_ds.shape[1]
    if np.isscalar(wav_s): wav_s = np.repeat(wav_s,stars)
    allkey = ['y_0','y_Y','y_YY','y_YYY','y_YXX','y_XX','x_0','x_X','x_XY','x_XXY','x_XXX']
    poldistfile = datadir+'RSSpoldist.txt'
    paramTab = ascii.read(poldistfile, header_start=0, data_start=1,data_end=2)    
    paramTab = ta.hstack([paramTab,ascii.read(poldistfile, header_start=2,data_start=3, data_end=4)])
    useEOrat = ('Wavs' not in paramTab.colnames)
    if useEOrat:
        distTab = ascii.read(poldistfile, header_start=4,data_start=5)
    else:
        distTab = ascii.read(poldistfile, header_start=4,data_start=5, data_end=5+paramTab['Wavs'][0])
    tabwavs = len(distTab)
    colList = distTab.colnames
     
    for key in allkey:
        if key in colList: continue
        paramTab.add_column(ta.Column(name='EO'+key,data=(1.,)))
        distTab.add_column(ta.Column(name=key,data=np.zeros(tabwavs)))

    sdistTab = Tableinterp(distTab,'Wavel',wav_s)
    alfyxout_dps = np.zeros((2,2,stars))
    bsrot, yoff, xoff = paramTab['bsrot','yoff','xoff'][0]
    
    for p in (0,1):
      # rotate about alf=0,0 into prism ref, apply distortion, rotate back out
        alfyxin_ds = rotate2d(alfyx_ds,-bsrot)
        alfyout_s = yoff + sdistTab['y_0']  \
            + sdistTab['y_Y']*alfyxin_ds[0]     \
            + 0.001*sdistTab['y_YY']*alfyxin_ds[0]**2   \
            + 0.001*sdistTab['y_XX']*alfyxin_ds[1]**2   \
            + 0.001*sdistTab['y_YYY']*alfyxin_ds[0]**3  \
            + 0.001*sdistTab['y_YXX']*alfyxin_ds[0]*alfyxin_ds[1]**2
        alfxout_s = xoff + sdistTab['x_0']  \
            + sdistTab['x_X']*alfyxin_ds[1]     \
            + 0.001*sdistTab['x_XY']*alfyxin_ds[1]*alfyxin_ds[0]    \
            + 0.001*sdistTab['x_XXY']*alfyxin_ds[1]**2*alfyxin_ds[0]    \
            + 0.001*sdistTab['x_XXX']*alfyxin_ds[1]**3

        if debug:
            np.set_printoptions(threshold=1000000)
            fileobj = open('ydisttab.txt','w')      
            print >> fileobj, Table([sdistTab['y_0'],sdistTab['y_Y']*alfyxin_ds[0], \
                0.001*sdistTab['y_YY']*alfyxin_ds[0]**2, \
                0.001*sdistTab['y_XX']*alfyxin_ds[1]**2,    \
                0.001*sdistTab['y_YYY']*alfyxin_ds[0]**3,   \
                0.001*sdistTab['y_YXX']*alfyxin_ds[0]*alfyxin_ds[1]**2])  
            fileobj = open('xdisttab.txt','w')  
            print >> fileobj, Table([sdistTab['x_0'],sdistTab['x_X']*alfyxin_ds[1], \
                0.001*sdistTab['x_XY']*alfyxin_ds[1]*alfyxin_ds[0], \
                0.001*sdistTab['x_XXY']*alfyxin_ds[1]**3])
            np.set_printoptions(threshold=1000)
            
        alfyxout_dps[:,p] = rotate2d(np.array([alfyout_s,alfxout_s]),bsrot)

        if (p==1): continue        
        if useEOrat:                          # second pass is E, name[2:] strips off the 'EO'        
            for name in paramTab.colnames[3:]: sdistTab[name[2:]] *= paramTab[name]
        else:
            for col in colList:
                distTab[col] = ascii.read(poldistfile, header_start=5+tabwavs,data_start=6+tabwavs)[col]                        
            sdistTab = Tableinterp(distTab,'Wavel',wav_s)

    return alfyxout_dps

# ---------------------------------------------------------------------------------
def RSScolpolcam(YX_ds, wav_s, coltem, camtem, yxOEoff_d=np.zeros(2),dalfyx_d=np.zeros(2),debug=False):
  # complete RSS distortion for imaging polarimetric data

    alfyxo_ds = RSScoldistort(YX_ds,wav_s,coltem,debug=debug) + dalfyx_d[:,None]    
    alfyxow_dps = RSSpolsplit(alfyxo_ds,wav_s,debug=debug) - dalfyx_d[:,None,None]    
    if np.isscalar(wav_s):
        wav_S = wav_s
    else:
        wav_S = np.tile(wav_s,2)  
    yxowa_dps = RSScamdistort(alfyxow_dps.reshape((2,-1)),wav_S,camtem,debug=debug).reshape((2,2,-1))
    yxowa_dps += (np.array([[-.5,.5],[-.5,.5]])*yxOEoff_d[:,None])[:,:,None]

    return yxowa_dps

# ---------------------------------------------------------------------------------
def RSScolgratpolcam(YX_ds, wav_s, coltem, camtem, grating, grang, artic, dateobs,  \
      order=1,yxOEoff_d=np.zeros(2),dalfyx_d=np.zeros(2),debug=False):
  # complete RSS distortion for grating spectropolarimetric data

    alfyxo_ds = RSScoldistort(YX_ds,wav_s,coltem,debug=debug) + dalfyx_d[:,None] 
    alfyxdisp_ds = RSSdisperse(alfyxo_ds,wav_s,grating,grang,artic,dateobs,order=order,debug=debug)       
    alfyxow_dps = RSSpolsplit(alfyxdisp_ds,wav_s,debug=debug) - dalfyx_d[:,None,None]    
    if np.isscalar(wav_s):
        wav_S = wav_s
    else:
        wav_S = np.tile(wav_s,2)  
    yxowa_dps = RSScamdistort(alfyxow_dps.reshape((2,-1)),wav_S,camtem,debug=debug).reshape((2,2,-1))
    yxowa_dps += (np.array([[-.5,.5],[-.5,.5]])*yxOEoff_d[:,None])[:,:,None]

    return yxowa_dps
    
# ------------------------------------
def RSSpolgeom(hdul,wavl,yxOEoff_d=np.zeros(2)):
    """Return imaging polarimetric layout 

    Parameters 
    ----------
    hdul: HDU list for relavant image
    wavl: float, wavelength (Ang)
    yxOEoff_d: 1D float numpy array, optional y,x offset to RSScolpolcam nominal E-O

    Returns 
    ----------
    yx0_dp: 2D float numpy array, [[y0_O,x0_O],[y0_E,x0_E]].  
        mm position of O,E optic axes at this wavelength relative to imaging optic axis 
    rshift: int, row of OE FOV split point - imaging CCD center row (5000 Ang)
    yxp0_dp: 2D float numpy array, 
        mm position of center of split O,E images relative to O,E optic axes at this wavelength 
    isfov_rc: 2D boolean numpy array, (full image) true inside FOV for O and E
    """

    data_rc = hdul[1].data
    rows, cols = data_rc.shape[-2:]      
    if len(data_rc.shape)>2:                            # allow for pol split data    
        rows = 2*rows           
        data_rc = data_rc[0]                
    cbin, rbin = [int(x) for x in hdul[0].header['CCDSUM'].split(" ")]
    rcbin_d = np.array([rbin,cbin])
    rccenter_d, cgapedge_c = ccdcenter(data_rc)

    camtem = hdul[0].header['CAMTEM']
    coltem = hdul[0].header['COLTEM']
    grating = hdul[0].header['GRATING'].strip()
    grang = hdul[0].header['GR-ANGLE']
    artic = hdul[0].header['CAMANG']         
    dateobs =  hdul[0].header['DATE-OBS'].replace('-','')
    trkrho = hdul[0].header['TRKRHO']
    pixmm = 0.015

    ur0,uc0,saltfps = rssdtralign(dateobs,trkrho)       # ur, uc =unbinned pixels, saltfps =micr/arcsec
    yx0_d = -0.015*np.array([ur0,uc0])                  # mm position of center in imaging from optical axis
       
    dyx_dp = np.array([[-.5,.5],[-.5,.5]])*yxOEoff_d[:,None]

    dy = 2.01                                           # pol fov height in arcmin
    dr = 3.97                                           # pol fov radius in arcmin
    xcnr = np.sqrt((dr**2-dy**2))
    YXfov_dt = np.array([[-dy,-dy,-dy,  0.,0.,0.,   dy,dy,dy],   \
                        [-xcnr,0.,xcnr,-dr,0.,dr,-xcnr,0.,xcnr]])*60.*saltfps/1000.
    yxfov_dpt = RSScolpolcam(YXfov_dt,wavl,coltem,camtem) + dyx_dp[:,:,None]

    yx0_dp = yxfov_dpt[:,:,4]

    if (grating=="N/A"):
        yxfovmean_dpt = RSScolpolcam(YXfov_dt,5000.,coltem,camtem) + dyx_dp[:,:,None]
    else:    
        yxfovmean_dpt = RSScolgratpolcam(YXfov_dt,5000.,coltem,camtem,grating,grang,artic,dateobs) +  \
            dyx_dp[:,:,None]   
           
    yxsep_d = 0.5*(yxfovmean_dpt[:,0,7] + yxfovmean_dpt[:,1,1])     # beam fov separation point 

    rshift = int(np.trunc((yxsep_d-yx0_d)[0]/(rcbin_d[0]*pixmm)))   # split row offset, same whole obs

    yxp0_dp = yx0_d[:,None] - (yx0_dp +  \
        rcbin_d[0]*pixmm*np.array([[-rshift+rows/4,-rshift-rows/4],[0.,0.]]))

    rc0_dp = (yx0_dp-yx0_d[:,None])/(rcbin_d[:,None]*pixmm) + rccenter_d[:,None]
    cfovrad_p = (yxfov_dpt[1,:,5] - yx0_dp[1])/(rcbin_d[1]*pixmm)
    rfovup0_p = (yxfov_dpt[0,:,7] - yx0_dp[0])/(rcbin_d[0]*pixmm)
    rfovdn0_p = (yx0_dp[0] - yxfov_dpt[0,:,1])/(rcbin_d[0]*pixmm) 
    rfovupcv_p = (yxfov_dpt[0,:,8] - yxfov_dpt[0,:,7])/(rcbin_d[0]*pixmm)    
    rfovdncv_p = (yxfov_dpt[0,:,1] - yxfov_dpt[0,:,2])/(rcbin_d[0]*pixmm)
    ccnr = yxfov_dpt[1,0,8]/(rcbin_d[1]*pixmm)   
    dr_pr = np.arange(rows)[None,:] - rc0_dp[0,:][:,None]
    dc_pc = np.arange(cols)[None,:] - rc0_dp[1,:][:,None]
    isfov_rc = np.zeros((rows,cols),dtype=bool)
    for p in (0,1):    
        isfov_rc |= ((np.sqrt(dr_pr[p,:,None]**2 + dc_pc[p,None,:]**2) < cfovrad_p[p,None,None]) &      \
          (dr_pr[p,:,None] < (rfovup0_p[p,None,None] + rfovupcv_p[p,None,None]*(dc_pc[p,None,:]/ccnr)**2)) & \
          (dr_pr[p,:,None] > -(rfovdn0_p[p,None,None] + rfovdncv_p[p,None,None]*(dc_pc[p,None,:]/ccnr)**2)))
    gapcolList = range(cgapedge_c[0],cgapedge_c[1])+range(cgapedge_c[2],cgapedge_c[3])
    isfov_rc[:,gapcolList] = False

    return yx0_dp, rshift, yxp0_dp, isfov_rc

# ---------------------------------------------------------------------------------

def printstdlog(string,logfile):
    print string
    print >>open(logfile,'a'), string
    return 
