#! /usr/bin/env python

# fit polarimetric imaging mode distortion model to metal shim cartesian image

# THIS ONLY WORKS FOR UNBINNED DATA 

import os, sys, time
import numpy as np
from math import pi
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.ndimage import interpolation as nd
from scipy import interpolate as ip
from astropy.io import fits as pyfits
from astropy.io import ascii
import astropy.table as ta

np.set_printoptions(threshold=np.nan)
polsaltdir = '/d/freyr/Dropbox/software/SALT/polsaltcurrent/'
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

from pyraf import iraf
from iraf import pysalt
from specpolutils import datedline, rssdtralign, rssmodelwave, configmap
from rssmaptools import sextract,catid,RSScoldistort,RSScamdistort,RSSpolsplit,Tableinterp,rotate2d

gridcols = 21
gridrows = 11
ggall = tuple(np.mgrid[5:16,0:21].reshape((2,-1)))
gtotpts = ggall[0].shape[0]
hggall = tuple(np.mgrid[0:3,5:16,0:21].reshape((3,-1)))
Hall = np.arange(3*gtotpts)  

# ---------------------------------------------------------------------------------
def distortion_impolcartesian(imfitsfile, polfitsfile, debug=False):
  # _d = (0,1) index of row,col
  # _s star index in sextract table
  # _g grid index in grid
  # _G index in flattened grid array (for curve_fit)
  # _I grid image index (0,1,2) for direct,O,E
  # _H index of points to be fitted in all grid images  

    print str(datetime.now()),'\n'

#   get filter, date, temp for im file (assume pol file is the same)                                                      
    hduim = pyfits.open(imfitsfile)
    hdupol = pyfits.open(polfitsfile)
    imname = 'img_'+imfitsfile.split('.')[0][-4:]
    polname = 'img_'+polfitsfile.split('.')[0][-4:]
    hdr = hduim[0].header
    cbin,rbin = map(int,(hdr["CCDSUM"].split(" ")))
    rows,cols = hduim['SCI'].data.shape
    maskid =  (hdr["MASKID"]).strip() 
    filter =  (hdr["FILTER"]).strip()
    trkrho = hdr['TRKRHO']
    dateobs =  hdr['DATE-OBS'].replace('-','') 
    if "COLTEM" in hdr:
        coltem = float(hdr["COLTEM"]) 
        camtem = float(hdr["CAMTEM"])
    else:
        coltem = float(hdr["COLTEMP"]) 
        camtem = float(hdr["CAMTEMP"])     

    print "date obs:", dateobs
    print "Mask:      ", maskid
    print "Filter:    ", filter
    print "Col Temp:   %7.1f" % (coltem)
    print "Cam Temp:   %7.1f" % (camtem)

  # Get distortion models, adjust for temperature and interpolate to filter wavelength
    imgoptfile = datadir+'RSSimgopt.txt'
    print "\nimgoptfile: "+open(imgoptfile).readlines()[2][:-1]
    dr0,dc0 = rssdtralign(dateobs,trkrho)[:2]                                # offset of FOV from ccd center
    r0,c0 = ccdcenter(hduim)*np.array([rbin,cbin]) + np.array([dr0,dc0])     # FOV center coordinate, unbinned pixels

    distTab = ascii.read(imgoptfile,data_start=1,   \
        names=['Wavel','Fcoll','Acoll','Bcoll','ydcoll','xdcoll','Fcam','acam','alfydcam','alfxdcam'])
    fwav = float(filter[-4:])
    fdistTab = Tableinterp(distTab,'Wavel',fwav)

    gridsep = 5.0                                                           # nominal spacing of mask grid 5 mm       
    griddata = np.array([[gridsep,-0.825,0.269,0.076]])  
    gridTab = ta.Table(griddata, names=['Mgrid','My0','Mx0','Mrot'])
    rippleTab = ta.Table(np.zeros((11,2)),names=['rippledc','rippledcdc'])
    
    poldistfile = datadir+'RSSpoldist.txt'
    print "poldistfile: "+open(poldistfile).readlines()[2][:-1]
    EOratTab = ascii.read(poldistfile, data_end=1,  \
        names=['bsrot','EOy_0','EOy_Y','EOy_YY','EOy_XX','EOx_0','EOx_X','EOx_XY','EOx_XXX'])
    polTab = ascii.read(poldistfile, data_start=1,   \
        names = ['Wavel','y_0','y_Y','y_YY','y_XX','x_0','x_X','x_XY','x_XXX'])	
    fpolTab = Tableinterp(polTab,'Wavel',fwav)                       
    parTab = ta.hstack([ta.join(fdistTab,fpolTab, keys='Wavel'),EOratTab,gridTab,])    

  # sextract the direct image and polsplit one 
  # set DEBLEND_MINCONT higher than default .005 for pol to prevent deblending of mosaic images
    print "\nstraight image:", imname
    imsexTab = sextract(imfitsfile,deblend=.005,fwhmlo=0.4,fwhmhi=3.0,cull=True,debug=debug)
    print "\npol image:", polname
    polsexTab = sextract(polfitsfile,deblend=.12,fwhmlo=0.4,fwhmhi=3.0,cull=True,debug=debug)

  # Remove points near CCD gaps
    imgap_s =(((imsexTab["X_IMAGE"] > cols/2-1024.-110.) & (imsexTab["X_IMAGE"] < cols/2-1014.)) | \
            ((imsexTab["X_IMAGE"] > cols/2+1014.) & (imsexTab["X_IMAGE"] < cols/2+1024.+110.)))
    imsexTab = imsexTab[~imgap_s]
    imrc_ds = np.array([imsexTab["Y_IMAGE"],imsexTab["X_IMAGE"]])
    polgap_s =(((polsexTab["X_IMAGE"] > cols/2-1024.-110.) & (polsexTab["X_IMAGE"] < cols/2-1014.)) | \
            ((polsexTab["X_IMAGE"] > cols/2+1014.) & (polsexTab["X_IMAGE"] < cols/2+1024.+110.)))
    polsexTab = polsexTab[~polgap_s]
    polrc_ds = np.array([polsexTab["Y_IMAGE"],polsexTab["X_IMAGE"]])    
    if debug: 
        imsexTab.write(imname+'_imsexTab.txt',format='ascii.fixed_width',   \
                        bookend=False, delimiter=None, overwrite=True)
        polsexTab.write(polname+'_polsexTab.txt',format='ascii.fixed_width',   \
                        bookend=False, delimiter=None, overwrite=True)
      
  # identify grid positions with sextract objects based on current model
  
    rc_dH = make_pos(parTab,[],r0,c0,coltem,camtem,rippleTab,debug=False)(np.tile(Hall,2)).reshape((2,-1))
    rc_dIG = rc_dH.reshape((2,3,-1))
    pixsep = gridsep*(fdistTab['Fcam']/fdistTab['Fcoll'])/0.015         
    imS_G = np.argmin((imrc_ds[0,:,None] - rc_dIG[0,None,0])**2 + \
                    (imrc_ds[1,:,None] - rc_dIG[1,None,0])**2,axis=0)
    imfound_G = ((imrc_ds[0][imS_G] - rc_dIG[0,0])**2 +  \
                    (imrc_ds[1][imS_G] - rc_dIG[1,0])**2 < (pixsep/5.)**2)
    polOS_G = np.argmin((polrc_ds[0,:,None] - rc_dIG[0,None,1])**2 + \
                    (polrc_ds[1,:,None] - rc_dIG[1,None,1])**2,axis=0)
    polfoundO_G = ((polrc_ds[0][polOS_G] - rc_dIG[0,1])**2 + \
                    (polrc_ds[1][polOS_G] - rc_dIG[1,1])**2 < (pixsep/5.)**2)
    polES_G = np.argmin((polrc_ds[0,:,None] - rc_dIG[0,None,2])**2 + \
                    (polrc_ds[1,:,None] - rc_dIG[1,None,2])**2,axis=0)
    polfoundE_G = ((polrc_ds[0][polES_G] - rc_dIG[0,2])**2 + \
                    (polrc_ds[1][polES_G] - rc_dIG[1,2])**2 < (pixsep/5.)**2)

    use_G = (imfound_G & polfoundO_G & polfoundE_G)         # only use grid points found in all images
    gusepoints = use_G.sum()
    use_H = np.tile(use_G,3)
    HuseList = np.where(use_H)[0]

    S_IG = np.array([imS_G,polOS_G,polES_G])
    S_H = S_IG.ravel()
    foundrc_dIG = np.array([imrc_ds[:,imS_G],polrc_ds[:,polOS_G],polrc_ds[:,polES_G]]).transpose((1,0,2))
    foundrc_dH = foundrc_dIG.reshape((2,-1)) 
    imrms = np.sqrt(((foundrc_dIG[0,0][use_G] - rc_dIG[0,0][use_G])**2 +   \
                    (foundrc_dIG[1,0][use_G] - rc_dIG[1,0][use_G])**2).mean())
    polOrms = np.sqrt(((foundrc_dIG[0,1][use_G] - rc_dIG[0,1][use_G])**2 +   \
                    (foundrc_dIG[1,1][use_G] - rc_dIG[1,1][use_G])**2).mean())
    polErms = np.sqrt(((foundrc_dIG[0,2][use_G] - rc_dIG[0,2][use_G])**2 +   \
                    (foundrc_dIG[1,2][use_G] - rc_dIG[1,2][use_G])**2).mean())

  # fit straight image, E and O simultaneously

    printname = ['My0','Mx0','Mrot','Acoll','Bcoll','ydcoll','xdcoll','acam','alfydcam','alfxdcam',   \
            'bsrot','y_0','y_Y','y_YY','y_XX','x_0','x_X','x_XY','x_XXX']
    varname = ['My0','Mx0','Mrot','Acoll','Bcoll','ydcoll','xdcoll','alfydcam','alfxdcam',   \
            'bsrot','y_0','y_Y','y_YY','y_XX','x_X','x_XY','x_XXX']

    print "\nBeamsplitter Distortion Fit"
    print "points imrms  Orms   Erms    My0    Mx0    Mrot    Acoll    Bcoll   ydcoll   xdcoll    acam  alfydcam alfxdcam  ",
    print "bsrot     y_0     y_Y       y_YY     y_XX     x_0     x_X      x_XY      x_XXX" 
    print (" %4i "+3*"%6.2f "+3*"%6.3f "+2*"%8.5f "+2*"%8.4f "+"%8.5f "+3*"%8.4f "+2*"%8.5f "+2*"%8.4f "+2*"%8.5f "+2*"%8.4f ") % \
         ((gusepoints,imrms,polOrms,polErms)+tuple(parTab[printname][0]))

    dofit = True
    outlieriter = 0
    while dofit:         
        optparList, covar = curve_fit(make_pos(parTab,varname,r0,c0,coltem,camtem,rippleTab),   \
            np.tile(HuseList,2), foundrc_dIG[:,:,use_G].flatten(), p0=tuple(parTab[varname][0]))

        parTab[varname][0] = optparList
        rc_dH = make_pos(parTab,[],r0,c0,coltem,camtem,rippleTab)(np.tile(Hall,2)).reshape((2,-1))
        rc_dIG = rc_dH.reshape((2,3,-1))
        imrms = np.sqrt(((foundrc_dIG[0,0][use_G] - rc_dIG[0,0][use_G])**2 +   \
                    (foundrc_dIG[1,0][use_G] - rc_dIG[1,0][use_G])**2).mean())
        polOrms = np.sqrt(((foundrc_dIG[0,1][use_G] - rc_dIG[0,1][use_G])**2 +   \
                    (foundrc_dIG[1,1][use_G] - rc_dIG[1,1][use_G])**2).mean())
        polErms = np.sqrt(((foundrc_dIG[0,2][use_G] - rc_dIG[0,2][use_G])**2 +   \
                    (foundrc_dIG[1,2][use_G] - rc_dIG[1,2][use_G])**2).mean())

        dev_dH = foundrc_dIG.reshape((2,-1)) - rc_dH
        rlower,dum,dum,rupper = fence(dev_dH[0,use_H])
        clower,dum,dum,cupper = fence(dev_dH[1,use_H])
        odev_H = np.array([(rlower-dev_dH[0]),(dev_dH[0]-rupper),
                           (clower-dev_dH[1]),(dev_dH[1]-cupper)]).max(axis=0)            
        isoutlier = odev_H[use_H].max() > 0

        if ((outlieriter==0) | ~isoutlier):            # printout on first and last iteration
            print (" %4i "+3*"%6.2f "+3*"%6.3f "+2*"%8.5f "+2*"%8.4f "+"%8.5f "+3*"%8.4f "+2*"%8.5f "+2*"%8.4f "+2*"%8.5f "+2*"%8.4f ") % \
                ((gusepoints,imrms,polOrms,polErms)+tuple(parTab[printname][0]))
            if debug:
                np.savetxt(polname+'_poldistvpred_'+str(outlieriter)+'.txt',       \
                np.vstack((np.array(np.unravel_index(HuseList,(3,gridrows,gridcols))),   \
                S_H[use_H],foundrc_dH[:,use_H],np.array([rc_dH[0,use_H],rc_dH[1,use_H]]))).T,    \
                fmt=4*"%3i "+4*"%8.2f ")
         
        if isoutlier:                           # remove the worst outlier grid point from all images and refit
            Houtlier = np.where(odev_H == odev_H[use_H].max())[0][0]
            Goutlier = (Houtlier % gtotpts)
            use_G[Goutlier] = False
            gusepoints = use_G.sum()
            use_H = np.tile(use_G,3)
            HuseList = np.where(use_H)[0]
            outlieriter += 1 
        else: dofit=False

    errTab = parTab.copy()
    errTab[0] = np.zeros(len(errTab.colnames))
    for i,name in enumerate(varname): errTab[name][0] = np.sqrt(np.diag(covar)[i])
    print (22*" "+"+/-  "+3*"%6.3f "+2*"%8.5f "+2*"%8.4f "+"%8.5f "+3*"%8.4f "+2*"%8.5f "+2*"%8.4f "+2*"%8.5f "+2*"%8.4f ") %    \
            tuple(errTab[printname][0])
                                                                      
    return

# ---------------------------------------------------------------------------------
def make_pos(parTab,varname,r0,c0,coltem,camtem,rippleTab,debug=False):
    def pos(HHreturn, *pvar):                       # H = all points raveled. HH 2x tiled for r,c curve_fit
        if len(varname): 
            for i,name in enumerate(varname): parTab[name][0] = pvar[i]

        Hreturn = HHreturn[:(len(HHreturn)/2)]
        wavel = parTab['Wavel'][0]

      # compute raw grid shape (mm) at SALT FP, then allow for mask rotation and shift   
        yx_dgg = parTab['Mgrid'][0]*(np.indices((gridrows,gridcols)) - (np.array([gridrows,gridcols])-1)[:,None,None]/2)
        yxm0_d = np.array([parTab['My0'][0],parTab['Mx0'][0]])
        yx_dG = rotate2d(yx_dgg.reshape((2,-1)),parTab['Mrot'][0]) + yxm0_d[:,None]

      # apply collimator distortion, get positions (mm) at detector FP, pixel positions on mosaiced CCD
        alfyx_dG = RSScoldistort(yx_dG,wavel,coltem,parTab,debug=debug)
        
      # for straight image, then for split image, get positions (mm) at detector FP, pixel positions on mosaiced CCD 
        imYX_dgg = RSScamdistort(alfyx_dG,wavel,camtem,parTab,debug=debug).reshape((2,gridrows,gridcols))
        imdrc_dgg = imYX_dgg/0.015                               # pixels, centered at FOV 0,0
        imdc_gg = rippleTab['rippledc'][:,None] + rippleTab['rippledcdc'][:,None]*imdrc_dgg[1]
        imrc_dgg = imdrc_dgg + np.array([r0,c0])[:,None,None] 
        imrc_dgg[1] += imdc_gg
             
        EOratTab = parTab['bsrot','EOy_0','EOy_Y','EOy_YY','EOy_XX','EOx_0','EOx_X','EOx_XY','EOx_XXX']
        alfyx_pdG = RSSpolsplit(alfyx_dG,parTab['Wavel'][0],parTab,EOratTab,debug=debug)
        alfyx_dG = alfyx_pdG.transpose((1,0,2)).reshape((2,-1))
        polYX_dpgg = RSScamdistort(alfyx_dG,wavel,camtem,parTab,debug=debug).reshape((2,2,gridrows,gridcols))
        poldrc_dpgg = polYX_dpgg/0.015                              
        poldc_pgg = rippleTab['rippledc'][None,:,None] + rippleTab['rippledcdc'][None,:,None]*poldrc_dpgg[1]
        polrc_dpgg = poldrc_dpgg + np.array([r0,c0])[:,None,None,None] 
        polrc_dpgg[1] += poldc_pgg

      # result is stack of straight, O, and E grid coordinates for selected grid points        
        rc_dH = np.stack((imrc_dgg,polrc_dpgg[:,0],polrc_dpgg[:,1]),axis=1).reshape((2,-1))[:,Hreturn] 
            
        return rc_dH.ravel()
    return pos

# ---------------------------------------------------------------------------------
def fence(arr):
  # return lower outer, lower inner, upper inner, and upper outer quartile fence
    Q1,Q3 = np.percentile(arr,(25.,75.))
    IQ = Q3-Q1
    return Q1-3*IQ, Q1-1.5*IQ, Q3+1.5*IQ, Q3+3*IQ

# ---------------------------------------------------------------------------------
def ccdcenter(hdul):
    image_rc = hdul['SCI'].data
    rows,cols = image_rc.shape
    image_c = image_rc.sum(axis=0)
    cleft = cols/2 - np.where(image_c[cols/2:0:-1]==0)[0][0]
    cright = cols/2 + np.where(image_c[cols/2:]==0)[0][0]
    return rows/2,(cleft+cright)/2.
  
# ---------------------------------------------------------------------------------
if __name__=='__main__':
    imfitsfile = sys.argv[1]
    polfitsfile = sys.argv[2]
    kwargs = dict(x.split('=', 1) for x in sys.argv[3:] if x.count('.fits')==0)        
    distortion_impolcartesian(imfitsfile,polfitsfile,**kwargs)

# cd /d/pfis/khn/20150725/sci_V3
#  python polsalt.py distortion_impolcartesian.py m*58.fits m*17.fits debug=True
