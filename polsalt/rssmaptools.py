
# mapping tools, including:

# configmap(infilelist,confitemlist,debug='False')
# sextract(fits,sigma=5.,deblend=.005,minpix=10,fwhmlo=0.6,fwhmhi=1.5,cull=False,logfile='salt.log',debug=False)
# readmaskxml(xmlfile) 
# readmaskgcode(gcodefile) 
# catid(yxcat_dt, yxcand_ds, offsettol=10., errtol=4.,debug=False,logfile='salt.log',name='')
# ccdcenter(image_rc)
# gaincor(hdu)
# YXcalc(ra_t,dec_t,RAd,DECd,PAd,fps)
# impolguide(YX_dt,yx_dpt,yxOEoff_d,wavl,coltem,camtem,debug=False,name='')
# Tableinterp(Tab,interpkey,interp_x)
# rotate2d(yx_ds, rot, center=np.zeros(2))
# boxsmooth1d(ar_x,ok_x,xbox,blklim)
# blksmooth2d(ar_rc,ok_rc,rblk,cblk,blklim,mode="mean",debug=False)
# fence(arr)
# polyfit_cull(x_x,y_x,ok_x,deg,fence='outer')
# fracmax(arr,frac=0.5)
# printstdlog(string,logfile)

import os, sys, glob, copy, shutil, inspect

import numpy as np
from xml.dom import minidom
from scipy import linalg as la
from specpolutils import rssdtralign, datedfile, datedline
from scipy.interpolate import interp1d, griddata
from astropy.io import fits as pyfits
from astropy.io import ascii
from saltobslog import obslog
from astropy.table import Table,unique
import astropy.table as ta

datadir = os.path.dirname(__file__) + '/data/'
# ----------------------------------------------------------
def configmap(infilelist,confitemlist,debug=False):
    """general purpose mapper of observing configurations
    Does proper elimination of non-polarimetric images

    Parameters
    ----------
    infilelist: list
        List of filenames 
    confitemlist: list
        List of header keywords which define a configuration

    Returns
    -------
    obs_i: numpy 1d array
        observation number (unique object, config) for each file, -1=ignore
    config_i: numpy 1d array
        config number (unique config) for each fil, -1 = ignore
    obstab: astropy Table
        object name and config number for each observation     
    configtab: astropy Table
        config items for each config
     
    """
    if debug: print "debug configmap infilelist: ",infilelist
        
  # create the observation log      
    obsdict = obslog(infilelist)
    images = len(infilelist)

  # make table of unique polarimetric configurations
    ispol_i = np.array([(obsdict['BS-STATE'][i][-8:] == 'Inserted') for i in range(images)])    
    config_i = -np.ones(images,dtype=int)
    confdatListj = []
    for i in np.where(ispol_i)[0]:
        confdatList = []
        for item in confitemlist:
            confdatList.append(obsdict[item][i])
        confdatListj.append(confdatList)

    dtypeList = map(type,confdatListj[0])
    dtypeList = ['S32' if t is str else t for t in dtypeList]   # for table comparisons               
    configTab = Table(np.array(confdatListj),names=confitemlist,dtype=dtypeList)
    configTab = unique(configTab)
    
    for j,i in enumerate(np.where(ispol_i)[0]):
        configRow = Table(np.array(confdatListj[j]),names=confitemlist,dtype=dtypeList)                    
        config_i[i] = np.where(configTab==configRow)[0][0]
                        
  # make table of unique observations
    obs_i = -np.ones(images,dtype=int)  
    obsdatListj = []
    for i in np.where(ispol_i)[0]:
        object = obsdict['OBJECT'][i].replace(' ','')
        obsdatListj.append([object, config_i[i]])

    dtypeList = map(type,obsdatListj[0])
    dtypeList = ['S32' if t is str else t for t in dtypeList]    
    obsTab = Table(np.array(obsdatListj),names=['object','config'],dtype=dtypeList)
    obsTab = unique(obsTab)
    
    for j,i in enumerate(np.where(ispol_i)[0]):
        obsRow = Table(np.array(obsdatListj[j]),names=['object','config'],dtype=dtypeList)                     
        obs_i[i] = np.where(obsTab==obsRow)[0][0]
                        
    return obs_i,config_i,obsTab,configTab
# ---------------------------------------------------------------------------------

def sextract(fits,sigma=5.,deblend=.005,minpix=10,fwhmlo=0.6,fwhmhi=1.5,cull=False,logfile='salt.log',debug=False):
# run sextractor to find objects
# sigma=DETECT_THRESH is 3 in qred.sex.  5 is more reasonable default
# Version 3, returns catalog as a astropy Table with fields named as in SeXtract

    hdulist_image = pyfits.open(fits)
    shape_d = np.asarray(hdulist_image["SCI"].data.shape)
    rcbin_d = np.array(hdulist_image[0].header["CCDSUM"].split(" ")).astype(int)[::-1]
    image_rc = np.copy(hdulist_image["SCI"].data)                   
    pix_scale=0.125
    r_ap=2.0/(pix_scale*rcbin_d.min())         
    sat=0.99*image_rc.max()
    sexparamlist = ["X_IMAGE","Y_IMAGE","MAG_APER(1)","MAGERR_APER(1)","FLUX_ISO","FLUXERR_ISO", \
            "FLUX_MAX","THETA_IMAGE","ELLIPTICITY","FWHM_IMAGE","FLAGS","CLASS_STAR","EXT_NUMBER"]
    np.savetxt("sxtr_findstars.param",sexparamlist,fmt="%s")

    cmd= ('sex %s -c '+datadir+'qred.sex -PARAMETERS_NAME %s -CATALOG_NAME %s -DETECT_THRESH %f '+  \
            '-PHOT_APERTURES %f -SATUR_LEVEL %f -DEBLEND_MINCONT %f -DETECT_MINAREA %i') \
            % (fits, "sxtr_findstars.param", "out.txt", sigma, r_ap, sat, deblend, minpix)

    if debug:
        os.system(cmd+" &> "+fits.replace(".fits","_debug.txt"))
    else:
        os.system(cmd+" &> /dev/null")
    if (not os.path.exists("out.txt")):
        printstdlog( "call to SeXtractor failed",logfile)
        exit() 
        
    sextab = ascii.read("out.txt",names=sexparamlist)
    sextab = sextab[:][sextab['EXT_NUMBER']==1]
    sextab.remove_column('EXT_NUMBER')         # only use ext=1 extension in MEF

    if cull:
        rc_ds = np.array([sextab["Y_IMAGE"],sextab["X_IMAGE"]])
        fl_s = sextab["FLUX_ISO"]
        fw_s = sextab["FWHM_IMAGE"]
        stars = fl_s.shape[0]

        if debug: printstdlog( 'SeXtract total: '+str(stars), logfile)

      # combine well-detected stars closer than 1.5 bin
        ok_s = np.ones(stars,dtype=bool)
        minsep = 1.5
        dist_ss = np.sqrt(((rc_ds[:,:,None] - rc_ds[:,None,:])**2).sum(axis=0))
        comb_ss = (dist_ss < minsep).astype(int)
        rc_ds = (comb_ss[None,:,:]*fl_s[None,None,:]*rc_ds[:,None,:]).sum(axis=2)
        fl_s = (comb_ss*fl_s[None,:]).sum(axis=1)
        rc_ds /= fl_s
        fw_s = np.sqrt(((comb_ss*fw_s[None,:]).max(axis=1))**2 + ((comb_ss*dist_ss).mean(axis=1))**2) 
        for s in range(stars):
            if ok_s[s]:
                flagoff_s = ((np.eye(stars)[s] == 0) & (comb_ss[s] > 0))
                ok_s[flagoff_s] = False

      # get rid of non-stars based on fwhm compared to median
        fwhmmed = np.median(fw_s[ok_s])
        ok_s &= ((fw_s > fwhmlo*fwhmmed) & (fw_s < fwhmhi*fwhmmed))
        Stars = ok_s.sum()
        s_S = np.where(ok_s)[0]
        sextabcull = sextab[s_S]
        if debug: 
            printstdlog( 'After duplicate and FWHM cull: '+str(Stars),logfile)
            np.savetxt(fits.replace(".fits","_sxtr_culled.txt"),sextabcull, \
                fmt=6*"%12.4f "+4*"%10.3f "+"%4i %6.2f")                                   

    if debug:
        np.savetxt(fits.replace(".fits","_sxtr.txt"),sextab, \
            fmt=6*"%12.4f "+4*"%10.3f "+"%4i %6.2f") 
    else:
        os.remove("sxtr_findstars.param")

    os.remove("out.txt")
    if cull: return sextabcull, sextab
    else: return sextab
# ------------------------------------
def readmaskxml(xmlfile):
    """Read the slit information in from an xml file
       that was used to create the slitmask
    """

    # read in the xml
    dom = minidom.parse(xmlfile)
    
    # read all the parameters into a dictionary
    parameters = dom.getElementsByTagName('parameter')
    Param = {}
    for param in parameters:
        Param[str(param.getAttribute('name'))] = str(param.getAttribute('value'))

    PA = float(Param['ROTANGLE'])
    CENTERRA = float(Param['CENTERRA'])
    CENTERDEC = float(Param['CENTERDEC'])

    # read all the slits into a table
    slitTab = Table(names=('TYPE','CATID','RACE','DECCE','WIDTH','LENGTH'),     \
        dtype=('S7','<S64', float, float, float, float))
    targets = dom.getElementsByTagName('slit')
    refs = dom.getElementsByTagName('refstar')    
    for slit in targets:
        slitTab.add_row(('target',str(slit.getAttribute('id')),
            float(slit.getAttribute('xce')),float(slit.getAttribute('yce')),
            float(slit.getAttribute('width')),float(slit.getAttribute('length'))))
    for slit in refs:
        slitTab.add_row(('refstar',str(slit.getAttribute('id')),
            float(slit.getAttribute('xce')),float(slit.getAttribute('yce')),
            float(slit.getAttribute('width')),float(slit.getAttribute('length'))))            
        
    return slitTab, PA, CENTERRA, CENTERDEC

# ------------------------------------
def readmaskgcode(gcodefile):
    """Read the slit information in from a gcode file
       that was used to create the slitmask
    """
    # read in the gcode text
    lineList = open(gcodefile,'r').readlines()

    # read all the slits into a table
    slitTab = Table(names=('CATID','XCE','YCE','XLEN','YLEN'),     \
        dtype=('<S64', float, float, float, float))

    slitstart_s = np.where(np.array(lineList)=="(Start drawing a slit)\n")[0]
    lineparamList = [line[:3] for line in lineList]  
    for s,st in enumerate(slitstart_s):
        slitTab.add_row((s,
            float(lineList[st+lineparamList[st:].index('#1=')][3:-1]),
            float(lineList[st+lineparamList[st:].index('#2=')][3:-1]),            
            float(lineList[st+lineparamList[st:].index('#3=')][3:-1]),
            float(lineList[st+lineparamList[st:].index('#4=')][3:-1])))
                                
    return slitTab
      
# ------------------------------------
def catid(yxcat_dt, yxcand_ds, offsettol=10., errtol=4.,debug=False,logfile='salt.log',name=''):
    """identify candidate positions in a catalog prediction, using 2D histogram of offset vectors

    Parameters:
    yxcat_dt: 2d numarray input coordinates for catalog
        _d: 0,1 for y,x
        _t: "target" index of catalog inputs
    yxcand_ds: 2d numarray input coordinates for catalog
        _d: 0,1 for y,x
        _s: "star" index of candidate inputs
    offsettol: float mm 
        use to set size of offset histogram
    errtol: float mm
        use to set size of histogram bins

    Returns: t_i,s_i
        arrays of catalog and found indices for id'd pairs _i
    """

  # find similar offset vectors.  
  # form 2d histogram of offset, over +/- 2*offsettol in errtol bins
    targets = yxcat_dt.shape[1]
    candidates = yxcand_ds.shape[1]  
    offyx_dts = (yxcand_ds[:,None,:] - yxcat_dt[:,:,None])
    offhist_yx,yedge,xedge = np.histogram2d(offyx_dts[0].flatten(),offyx_dts[1].flatten(),  \
        bins=np.arange(-2*offsettol,2*offsettol+errtol,errtol))

  # ids in 3x3 histogram block surrounding histogram max
    j,i = np.array(np.where(offhist_yx==np.max(offhist_yx))).T[0]   
    ok_ts = ((offyx_dts[0] > yedge[max(0,j-1)]) & (offyx_dts[0] < yedge[min(j+2,yedge.shape[0]-1)]))   & \
            ((offyx_dts[1] > xedge[max(0,i-1)]) & (offyx_dts[1] < xedge[min(i+2,xedge.shape[0]-1)]))

    offyx_d = np.array([np.median(offyx_dts[0][ok_ts]),np.median(offyx_dts[1][ok_ts])])
    roff_ts = np.sqrt(((offyx_dts - offyx_d[:,None,None])**2).sum(axis=0))
    t_i = np.where(ok_ts)[0]
    s_i = np.where(ok_ts)[1]
    roff_i = roff_ts[t_i,s_i]
    ids = ok_ts.sum()
    uniquetargetids = np.unique(t_i).shape[0]
    uniquecandids = np.unique(s_i).shape[0]

    if debug: 
        printstdlog ((("Catalog entries %3i, Candidates %3i") % (targets,candidates)),logfile)
        printstdlog ((("  Ids %3i,  unique targets %3i, unique candidates %3i") %   \
                        (ids,uniquetargetids,uniquecandids)),logfile)
        printstdlog ((("  yx offset (mm):     %8.3f %8.3f") % tuple(offyx_d)),logfile)
        np.savetxt(name+"catid1.txt", np.vstack((t_i,s_i,yxcat_dt[:,t_i],yxcand_ds[:,s_i],   \
                        offyx_dts[:,t_i,s_i])).T,fmt="%4i %4i "+6*"%8.3f ")
        np.savetxt(name+"idhist.txt", offhist_yx, fmt="%3i")
        sbest_t = np.argmin(roff_ts,axis=1)
        np.savetxt(name+"sbest_t.txt",np.vstack((np.arange(targets),sbest_t,    \
            roff_ts[range(targets),sbest_t], yxcat_dt,yxcand_ds[:,sbest_t])).T,     \
            fmt="%4i %4i %7.3f  "+4*"%8.3f ")
                        
  # get best id in case of multiple id's of same target
    roffmin_t = roff_ts.min(axis=1)
    okt_i = np.in1d(roff_i,roffmin_t[t_i])
    ids = okt_i.sum() 
    uniquecandids = np.unique(s_i[okt_i]).shape[0]     

    if ((ids < ok_ts.sum()) and debug): 
        printstdlog ((("  Ids %3i,  unique candidates %3i") % (ids,uniquecandids)),logfile)
        np.savetxt(name+"catid2.txt", np.vstack((t_i[okt_i],s_i[okt_i],yxcat_dt[:,t_i[okt_i]],   \
            yxcand_ds[:,s_i[okt_i]],offyx_dts[:,t_i[okt_i],s_i[okt_i]])).T,fmt="%4i %4i "+6*"%8.3f ")

  # get best id in case of multiple id's of same candidate
    roffmin_s = roff_ts.min(axis=0)
    okts_i = np.in1d(roff_i,roffmin_s[s_i[okt_i]])
    ids = okts_i.sum() 

    if ((ids < okt_i.sum()) and debug): 
        printstdlog ((("  Ids %3i") % ids),logfile)
        np.savetxt(name+"catid3.txt", np.vstack((t_i[okts_i],s_i[okts_i],yxcat_dt[:,t_i[okts_i]],   \
            yxcand_ds[:,s_i[okts_i]],offyx_dts[:,t_i[okts_i],s_i[okts_i]])).T,fmt="%4i %4i "+6*"%8.3f ")

    return t_i[okts_i],s_i[okts_i]   

# ------------------------------------
def ccdcenter(image_rc):
    """find gaps and center of ccd image

    Parameters:
    image_rc: 2d image

    Returns: rccenter_d (int r,c of center), cgap_c (int column of 4 gap edges)

    """
    image_c = image_rc.mean(axis=0)
    rows,cols = image_rc.shape
    cstart = 0
    cedge_id = np.zeros((3,2),dtype=int)
    for i in (0,1,2):
        cedge_id[i,0] = np.argmax((image_c[cstart: ] != 0)&(image_c[cstart: ] != -1)) + cstart
        cedge_id[i,1] = np.argmax((image_c[cedge_id[i,0]: ] == 0)|  \
                                  (image_c[cedge_id[i,0]: ] == -1)) + cedge_id[i,0] - 1                                  
        if cedge_id[i,1] == cedge_id[i,0]-1: cedge_id[i,1] = cols-1         
        cstart = cedge_id[i,1]+1
    rccenter_d = np.array([rows/2,cedge_id[1].mean()])
    return rccenter_d,cedge_id.flatten()[1:5]
# ------------------------------------
def gaincor(hdul):
    """ get gain correction data for an image
    
    Parameters:
    hdul: hduList for image

    Returns: np 1D array gaincor_c.  Divide column c in image by gaincor_c

    """ 
  # assumes data final 2 dimensions are row,col  
    rows,cols = hdul['SCI'].data.shape[-2:]
    cbin, rbin = [int(x) for x in hdul[0].header['CCDSUM'].split(" ")]
    dateobs =  hdul[0].header['DATE-OBS'].replace('-','')
    utchour = int(hdul[0].header['UTC-OBS'].split(':')[0])
    mjdateobs = str(int(dateobs) - int(utchour < 10))
    gain = hdul[0].header['GAINSET']
    speed = hdul[0].header['ROSPEED']
    rccenter_d, cgap_c = ccdcenter(hdul['SCI'].data.flat[:rows*cols].reshape(rows,cols))
    c_a = np.array([0, cgap_c[0]-1024/cbin+1, cgap_c[[0,1]].mean(),   \
        cgap_c[1]+1024/cbin, cgap_c[[2,3]].mean(), cgap_c[3]+1024/cbin, cols])  

    gaincor_c = np.ones(cols)
    GainCorrectionFile = datedfile(datadir+"RSS_Gain_Correction_yyyymmdd_vnn.txt",mjdateobs)
    if (len(GainCorrectionFile)==0): return gaincor_c
    mode_ld = np.loadtxt(GainCorrectionFile, dtype='string', usecols=(0,1))
    lmode = np.where((mode_ld[:,0]==gain) & (mode_ld[:,1] == speed))[0][0]
    gaincor_a = np.loadtxt(GainCorrectionFile, usecols=range(2,8), unpack=True).T[lmode]

    for a in range(6):
        if (gaincor_a[a] == 1.): continue
        isa_c = ((np.arange(cols) >= c_a[a]) & (np.arange(cols) < c_a[a+1]))
        gaincor_c[isa_c] = gaincor_a[a]
    return gaincor_c
                
# ------------------------------------
def YXcalc(ra_t,dec_t,RAd,DECd,PAd,fps):
    """compute YX SALT fov positions (mm from LOS)

    Parameters:
    ra_t: 1d numarray, RA (degrees) for targets _t
    dec_t: 1d numarray, Dec (degrees) for targets _t
    RAd, DECd, PAd: RA, Dec, PA of LOS (degrees)
    fps: SALT focal plane scale (micron/arcsec)

    Returns: YX_dt (mm)

    """
    PAr = np.radians(PAd)
    Y_t = 3.6*fps*(np.cos(np.radians(dec_t))*(ra_t-RAd)*np.sin(PAr) + \
            (dec_t-DECd)*np.cos(PAr))
    X_t = 3.6*fps*(-np.cos(np.radians(dec_t))*(ra_t-RAd)*np.cos(PAr) +   \
            (dec_t-DECd)*np.sin(PAr))
    return np.array([Y_t,X_t])

# ----------------------------------------------
def impolguide(YX_dt,yx_dpt,yxOEoff_d,wavl,coltem,camtem,fitOEoff=True,debug=False,name=''):
  # Least squares fit of dY,dX,drot and dyOEoff,dxOEoff of star offset from predicted target position
    targets = YX_dt.shape[1]
    dYX = 0.1                           # offset for derivative  
    dYX_t = 0.1*np.ones(targets)                 
    yxcat_dpt = RSScolpolcam(YX_dt,wavl,coltem,camtem)
    yxcatY_dpt = RSScolpolcam(YX_dt+np.array([dYX_t,np.zeros(targets)]),wavl,coltem,camtem)    
    yxcatX_dpt = RSScolpolcam(YX_dt+np.array([np.zeros(targets),dYX_t]),wavl,coltem,camtem)
    dydY_pt = (yxcatY_dpt - yxcat_dpt)[0]/dYX 
    dxdX_pt = (yxcatX_dpt - yxcat_dpt)[1]/dYX     

    off_dp = np.array([[-.5,.5],[-.5,.5]])        
    dyx_dpt = yx_dpt - (yxcat_dpt + (off_dp*yxOEoff_d[:,None])[:,:,None])
        
  # Fit: dyx_dpt = [dY*dydY_pt,dX*dxdX_pt] + drot*[X*dydY_pt,-Y*dxdX_pt] + off_dp*dyxOEoff_d
    cofs = [2,5][fitOEoff]     
    A_SC = np.zeros((4*targets,cofs))
    A_SC[:2*targets,0] = dydY_pt.flatten()
    A_SC[2*targets:,1] = dxdX_pt.flatten()

    if fitOEoff:
        A_SC[:2*targets,2] = (YX_dt[1][None,:]*dydY_pt).flatten()
        A_SC[2*targets:,2] = -(YX_dt[0][None,:]*dxdX_pt).flatten()
        A_SC[:2*targets,3] = np.repeat(off_dp[0,:,None],targets,axis=1).flatten()
        A_SC[2*targets:,4] = np.repeat(off_dp[1,:,None],targets,axis=1).flatten()

    if debug: 
        np.savetxt(name+"A_SC.txt",A_SC,fmt="%8.2f")
        np.savetxt(name+"B_S.txt",dyx_dpt.flatten(),fmt="%8.2f")

    cof_C,sumsqerr = la.lstsq(A_SC,dyx_dpt.flatten())[:2]          # here is the fit

    eps_CC = la.inv((A_SC[:,:,None]*A_SC[:,None,:]).sum(axis=0))
    std_C = np.sqrt((sumsqerr/targets)*np.diagonal(eps_CC))    
    dYX_d = cof_C[0:2]
    dYXerr_d = std_C[0:2]
    if fitOEoff:
        drot = np.degrees(cof_C[2])
        droterr = np.degrees(std_C[2])       
        dyxOEoff_d = cof_C[3:]
        dyxOEofferr_d = std_C[3:] 
    else:
        drot = 0.
        droterr = 0.
        dyxOEoff_d = np.zeros(2)
        dyxOEofferr_d = np.zeros(2) 
    return dYX_d,drot,dyxOEoff_d,dYXerr_d,droterr,dyxOEofferr_d

# ---------------------------------------------------------------------------------
def Tableinterp(Tab,interpkey,interp_x):
  # make a new table, interpolated on specified key

    if np.isscalar(interp_x): interp_x = np.array([interp_x,])    
    names = Tab.colnames
    newTab = ta.Table(np.zeros((interp_x.shape[0],len(names))),names=names)
    newTab[interpkey] = interp_x
    interp_X = Tab[interpkey]

    names.remove(interpkey)
    for name in names:
        if len(Tab) > 1:
            newTab[name] = interp1d(interp_X,Tab[name],kind='cubic')(interp_x)
        else:
            newTab[name] = Tab[name].repeat(interp_x.shape[0])

    return newTab

# ----------------------------------------------------------

def rotate2d(yx_ds, rot, center=np.zeros(2)):
    """rotate an array of 2d coordinates

    Parameters:
    yx_ds: 2d numarray of 2d coordinates
        _d: 0,1 for y,x
        _s: index of coordinates
    rot: amount to rotate (degrees)
    center: y,x coordinates of rotation center (default 0,0)

    """

    c = np.cos(np.radians(rot))
    s = np.sin(np.radians(rot))
    rotate = np.transpose([[c, s],[-1.*s, c]])
    yx1_ds = yx_ds - center[:,None]
    yxout_ds = (np.dot(yx1_ds.T,rotate)).T
    yxout_ds +=  center[:,None]
    return yxout_ds

# ---------------------------------------------------------------------------------

def boxsmooth1d(ar_x,ok_x,xbox,blklim):
# sliding boxcar average (using okmask, with ok points in > blklim fraction of the box)
# ar_x: float nparray;  ok_x = bool nparray
# xbox: int;   blklim: float

    arr_x = ar_x*ok_x
    bins = ar_x.shape[0]
    kernal = np.ones(xbox)
    if xbox/2. == 0:
        kernal[0] = 0.5
        kernal = np.append(kernal,0.5)
    nkers = kernal.shape[0]
    valmask_x = np.convolve(arr_x,kernal)[(nkers-1)/2:(nkers-1)/2+bins]
    count_x = np.convolve(ok_x,kernal)[(nkers-1)/2:(nkers-1)/2+bins]
    arr_x = np.zeros(bins)
    okbin_x = count_x > xbox*blklim
    arr_x[okbin_x] = valmask_x[okbin_x]/count_x[okbin_x]
        
    return arr_x

# ---------------------------------------------------------------------------------

def blksmooth2d(ar_rc,ok_rc,rblk,cblk,blklim,mode="mean",debug=False):
# blkaverage (using mask, with blks with > blklim fraction of the pts), then spline interpolate result
# optional: median instead of mean
    rows,cols = ar_rc.shape
    arr_rc = np.zeros_like(ar_rc)
    arr_rc[ok_rc] = ar_rc[ok_rc]
    r_rc,c_rc = np.indices((rows,cols)).astype(float)

# equalize block scaling to avoid triangularization failure    
    rblk,cblk = max(rblk,cblk), max(rblk,cblk) 
    rcenter = (np.where(ok_rc)[0][-1] + np.where(ok_rc)[0][0])/2
    ccenter = ((np.where(ok_rc)[1]).max() + (np.where(ok_rc)[1]).min())/2
    drdat = (np.where(ok_rc)[0][-1] - np.where(ok_rc)[0][0])
    dcdat = ((np.where(ok_rc)[1]).max() - (np.where(ok_rc)[1]).min())
    rblks,cblks = int(np.ceil(float(drdat)/rblk)),int(np.ceil(float(dcdat)/cblk))    
    r0 = min(max(0,rcenter-rblk*rblks/2),rows-rblk*rblks)
    c0 = min(max(0,ccenter-cblk*cblks/2),cols-cblk*cblks)

    arr_RCb = arr_rc[r0:(r0+rblk*rblks),c0:(c0+cblk*cblks)]    \
        .reshape(rblks,rblk,cblks,cblk).transpose(0,2,1,3).reshape(rblks,cblks,rblk*cblk)
    ok_RCb = ok_rc[r0:(r0+rblk*rblks),c0:(c0+cblk*cblks)]    \
        .reshape(rblks,rblk,cblks,cblk).transpose(0,2,1,3).reshape(rblks,cblks,rblk*cblk)
    r_RCb = ((ok_rc*r_rc)[r0:(r0+rblk*rblks),c0:(c0+cblk*cblks)])    \
        .reshape(rblks,rblk,cblks,cblk).transpose(0,2,1,3).reshape(rblks,cblks,rblk*cblk)
    c_RCb = ((ok_rc*c_rc)[r0:(r0+rblk*rblks),c0:(c0+cblk*cblks)])    \
        .reshape(rblks,rblk,cblks,cblk).transpose(0,2,1,3).reshape(rblks,cblks,rblk*cblk)    
    ok_RC = ok_RCb.sum(axis=-1) > rblk*cblk*blklim
    arr_RC = np.zeros((rblks,cblks))
    if mode == "mean":
        arr_RC[ok_RC] = arr_RCb[ok_RC].sum(axis=-1)/ok_RCb[ok_RC].sum(axis=-1) 
    elif mode == "median":          
        arr_RC[ok_RC] = np.median(arr_RCb[ok_RC],axis=-1)
    else: 
        print "Illegal mode "+mode+" for smoothing"
        exit()
    r_RC = np.zeros_like(arr_RC); c_RC = np.zeros_like(arr_RC)
    r_RC[ok_RC] = r_RCb[ok_RC].sum(axis=-1)/ok_RCb[ok_RC].sum(axis=-1)
    c_RC[ok_RC] = c_RCb[ok_RC].sum(axis=-1)/ok_RCb[ok_RC].sum(axis=-1)

    if debug:
        np.savetxt('arr_RC.txt',arr_RC,fmt="%14.9f")
        np.savetxt('ok_RC.txt',ok_RC,fmt="%2i")

# evaluate slopes at edge for edge extrapolation   
    dar_RC = ((arr_RC[1:,:]!=0.) & (arr_RC[:-1,:]!=0.)).astype(int)     \
        * (arr_RC[1:,:] - arr_RC[:-1,:])
    dac_RC = ((arr_RC[:,1:]!=0.) & (arr_RC[:,:-1]!=0.)).astype(int)     \
        * (arr_RC[:,1:] - arr_RC[:,:-1])
    dr_RC = r_RC[1:,:] - r_RC[:-1,:]
    dc_RC = c_RC[:,1:] - c_RC[:,:-1]

    dadr_RC = np.zeros_like(dar_RC);    dadc_RC = np.zeros_like(dac_RC)
    dadr_RC[dr_RC!=0] = dar_RC[dr_RC!=0]/dr_RC[dr_RC!=0]
    dadc_RC[dc_RC!=0] = dac_RC[dc_RC!=0]/dc_RC[dc_RC!=0]
    argR = np.where(ok_RC.sum(axis=1)>0)[0]
    argC = np.where(ok_RC.sum(axis=0)>0)[0]    
    dadr_RC[argR[0],argC]    *= (arr_RC[argR[0,],argC] > 0)
    dadr_RC[argR[-1]-1,argC] *= (arr_RC[argR[-1],argC] > 0)
    dadc_RC[argR,argC[0]]    *= (arr_RC[argR,argC[0]] > 0)
    dadc_RC[argR,argC[-1]-1] *= (arr_RC[argR,argC[-1]] > 0)    
        
# force outer block positions into a rectangle to avoid edge effects, spline interpolate

    r_RC[argR[[0,-1]][:,None],argC] = (r0+(rblk-1)/2.+rblk*argR[[0,-1]])[:,None]
    c_RC[argR[:,None],argC[[0,-1]]] = (c0+(cblk-1)/2.+cblk*argC[[0,-1]])
    ok_RC = ((r_RC > 0.) & (c_RC > 0.))
        
    arr_rc = griddata((r_RC[ok_RC],c_RC[ok_RC]),arr_RC[ok_RC],  \
        tuple(np.mgrid[:rows,:cols].astype(float)),method='cubic',fill_value=0.)

    if debug:
        pyfits.PrimaryHDU(arr_rc.astype('float32')).writeto('arr_rc_0.fits',overwrite=True)
        np.savetxt('r_RC_1.txt',r_RC,fmt="%9.2f")
        np.savetxt('c_RC_1.txt',c_RC,fmt="%9.2f")

# extrapolate to original array size, zero outside
    argR_r = ((np.arange(rows) - r0)/rblk).clip(0,rblks-1).astype(int)
    argC_c = ((np.arange(cols) - c0)/cblk).clip(0,cblks-1).astype(int)
    r0,r1 = np.where(arr_rc.sum(axis=1)>0)[0][[0,-1]]
    c0,c1 = np.where(arr_rc.sum(axis=0)>0)[0][[0,-1]]

    arr_rc[r0-rblk/2:r0,c0:c1+1]   += arr_rc[r0,c0:c1+1]   +        \
                    dadr_RC[argR[0],argC_c[c0:c1+1]]*(np.arange(-int(rblk/2),0)[:,None])
    arr_rc[r1+1:r1+rblk/2,c0:c1+1] += arr_rc[r1,c0:c1+1]   +        \
                    dadr_RC[argR[-1]-1,argC_c[c0:c1+1]]*(np.arange(1,rblk/2)[:,None])
    arr_rc[r0-rblk/2:r1+rblk/2,c0-cblk/2:c0]   += arr_rc[r0-rblk/2:r1+rblk/2,c0][:,None] + \
                    dadc_RC[argR_r[r0-rblk/2:r1+rblk/2],argC[0]][:,None]*np.arange(-int(cblk/2),0)
    arr_rc[r0-rblk/2:r1+rblk/2,c1+1:c1+cblk/2] += arr_rc[r0-rblk/2:r1+rblk/2,c1][:,None] + \
                    dadc_RC[argR_r[r0-rblk/2:r1+rblk/2],argC[-1]-1][:,None]*np.arange(1,cblk/2)

    arr_rc[((np.abs(r_rc-rcenter) > drdat/2) | (np.abs(c_rc-ccenter) > dcdat/2))] = 0.    
    if debug:
        pyfits.PrimaryHDU(arr_rc.astype('float32')).writeto('arr_rc_1.fits',overwrite=True)
    
    return arr_rc

# ----------------------------------------------------------

def fence(arr):
  # return lower outer, lower inner, upper inner, and upper outer quartile fence
    Q1,Q3 = np.percentile(arr,(25.,75.))
    IQ = Q3-Q1
    return Q1-3*IQ, Q1-1.5*IQ, Q3+1.5*IQ, Q3+3*IQ

# ---------------------------------------------------------------------------------

def legfit_cull(x_x,y_x,ok_x,deg,wt=1.,IQcull=3.,debug=False):
  # do legendre polyfit, culling weighted errors to fence, default outer
  # cull at most 25% of startvals
  # x_x should be on -1,1 interval
  # legfit_d is coefficients high to low (like np polyfit)

    startvals = ok_x.sum()

    if (type(wt) != 'numpy.ndarray'):
        wt_x = wt*np.ones_like(x_x)       
    newcull = True
    chi_x = np.zeros_like(x_x)
    passes = 0
    
    while newcull:
        passes += 1 
        fit_d = np.polynomial.legendre.legfit(x_x[ok_x],y_x[ok_x],deg,w=wt_x[ok_x])                
        chi_x[ok_x] = (y_x - np.polynomial.legendre.legval(x_x,fit_d))[ok_x]*wt_x[ok_x]    
        Q1,Q3 = np.percentile(chi_x[ok_x],(25.,75.))        
        fence_f = [Q1 - IQcull*(Q3-Q1), Q3 + IQcull*(Q3-Q1)]
        cull_x = ((chi_x < fence_f[0]) | (chi_x > fence_f[1]))
        culls = cull_x.sum()         
        newcull = ((culls < startvals/4) & (cull_x[ok_x].sum()>0))                        
        if debug:
            print passes, newcull
            print 4*"%10.3f " % tuple(fit_d)
            print 2*"%10.3f " % tuple(fence_f)
            for x in range(len(x_x)):
                print ((3*" %2i"+" %8.4f"+3*" %8.3f") % (x,ok_x[x],cull_x[x],x_x[x],y_x[x],wt_x[x],chi_x[x])) 
        ok_x = ok_x & np.logical_not(cull_x)  
                     
    err_x = y_x - np.polynomial.legendre.legval(x_x,fit_d)
    stdev = np.std(err_x[ok_x])
    
    return fit_d[::-1], ok_x, err_x, stdev, culls, passes

# ---------------------------------------------------------------------------------

def fracmax(arr_x,points=2,ok_x=[True],frac=0.5):
  # return interpolated positions of fraction of max, left and right.  -1 if not found.
  # use linear fit to points closest points
  # optional good-data mask
    xs = len(arr_x)
    if (len(ok_x)==1):
        ok_x = np.ones(xs,dtype=bool)
    fmax = arr_x[ok_x].max()
    ximax = np.where(arr_x == fmax)[0][0]
    belowfrac_x = ((arr_x < frac*fmax) & ok_x)
    xleft = -1.
    if belowfrac_x[:ximax].sum():
        xileft = ximax - np.argmax(belowfrac_x[ximax:0:-1])
        xList = range(xileft-(points-2),xileft+points)        
        xList = xList[0] + np.argsort(np.abs(arr_x[xList] - frac*fmax))[:points]
        if ok_x[xList].all():
            poly_d = np.polyfit(xList,arr_x[xList],1)
            xleft = (frac*fmax - poly_d[1])/poly_d[0]       
        
    xright = -1.
    if belowfrac_x[ximax:].sum():
        xiright = ximax + np.argmax(belowfrac_x[ximax:])
        xList = range(xiright-(points-2)-1,xiright+points-1)          
        xList = xList[0] + np.argsort(np.abs(arr_x[xList] - frac*fmax))[:points]        
        if ok_x[xList].all():
            poly_d = np.polyfit(xList,arr_x[xList],1)
            xright = (frac*fmax - poly_d[1])/poly_d[0]

    return xleft, xright

# ---------------------------------------------------------------------------------

def printstdlog(string,logfile):
    print string
    print >>open(logfile,'a'), string
    return 
