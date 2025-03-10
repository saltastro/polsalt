
"""
impolmap

Locate targets in filter polarimetry image, save with 2D target map 
Compute 1D extraction map, 1D wavelength map for imaging spectropolarimetric data

"""

import os, sys, glob, shutil, inspect, warnings

import numpy as np
from scipy import linalg as la
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage.interpolation import shift
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import Latitude,Longitude,Angle
from astropy.table import Table

from pyraf import iraf
from iraf import pysalt

from saltobslog_kn import obslog
from saltsafelog import logging

from specpolutils import rssdtralign, rssmodelwave
from rssmaptools import sextract,catid,ccdcenter,YXcalc,RSScolpolcam,RSSpolgeom,impolguide,rotate2d,Tableinterp

datadir = os.path.dirname(__file__) + '/data/'
np.set_printoptions(threshold=np.nan)

# warnings.simplefilter("error")
debug = False

# ------------------------------------

def imfilterpolmap(infileList,mapTab,logfile='salt.log',debug=False):
    """predict E and O target positions, identify targets for filterpol image 

    Parameters 
    ----------
    infileList: list of strings
    mapTab: catalog table with nominal input mask positions and angles in collimated beam for this filter


    """
    """
    _f file index in list
    _d dimension index y,x = 0,1
    _y, _x unbinned pixel coordinate relative to optic axis
    _r, _c binned pixel cordinate
    _t catalog index
    _s sextractor index star on image
    _S culled star index
    _i id index

    YX_d image coords (mm) at SALT focal plane
    yx_d: image coords (mm) at detector (relative to imaging optic axis)
    yx0_d: position (mm) of center of CCD image (relative to imaging optic axis)  
    yx_dp image coords (relative to imaging optic axis), separated by beam, polarimetric mode
    yx0_dp position (mm) of O,E optic axes at this wavelength (relative to imaging optic axis)
    yxp_dp image coords of O,E images (relative to O,E optic axes at this wavelength)
    yxp0_dp: position (mm) of center of split O,E images (relative to O,E optic axes at this wavelength) 
    """

  # get data

    obsDictf = obslog(infileList)
    files = len(infileList)
    hdul0 = pyfits.open(infileList[0])
    rows, cols = hdul0[1].data.shape
    cbin, rbin = [int(x) for x in hdul0[0].header['CCDSUM'].split(" ")]
    rcbin_d = np.array([rbin,cbin])
    rccenter_d, cgapedge_c = ccdcenter(hdul0[1].data)
    pixmm = 0.015

    objectname = obsDictf['OBJECT'][0].replace(' ','')  # strip errant spaces from fits name
    filter = obsDictf['FILTER'][0]
    camtem = hdul0[0].header['CAMTEM']
    coltem = hdul0[0].header['COLTEM']
    dateobs =  hdul0[0].header['DATE-OBS'].replace('-','')
    trkrho = obsDictf['TRKRHO'][0]
    RAd = Longitude(obsDictf['RA'][0]+' hours').degree
    DECd = Latitude(obsDictf['DEC'][0]+' degrees').degree
    PAd = obsDictf['TELPA'][0]  
    name = objectname+"_"+filter+"_imf_"
    targets = len(mapTab)
    prows = rows/2
    wavl = float(filter[3:])
    pm1_p = np.array([1.,-1.])
    ur0,uc0,saltfps = rssdtralign(dateobs,trkrho)       # ur, uc =unbinned pixels, saltfps =micr/arcsec    
    yx0_d = -0.015*np.array([ur0,uc0])                   # optical axis in mm

  # establish image fov geometry
    yx0_dp, rshift, yxp0_dp, isfov_rc = RSSpolgeom(hdul0,wavl)

    if debug:
        open(name+'geom.txt',"w")
        geomfile = open(name+'geom.txt',"a")        
        print >>geomfile, 'RSSpolgeom:'
        print >>geomfile, ('yx0_d  : %8.4f %8.4f'% tuple(yx0_d))
        print >>geomfile, ('yx0_dp : '+4*'%8.4f ' % tuple(yx0_dp.flatten()))
        print >>geomfile, 'rshift ',rshift
        print >>geomfile, ('yxp0_dp: '+4*'%8.4f ' % tuple(yxp0_dp.flatten()))     

  # apply distortion and splitting for this wavelength to get predicted O,E target map
    YX_dt = np.array([mapTab['Y'],mapTab['X']])
    yx_dpt = RSScolpolcam(YX_dt,wavl,coltem,camtem)
    yxorig_dpt = np.copy(yx_dpt)                    # for debug

  # sum images to identify targets
    summedianvar = 0.
    image_frc = np.zeros((files,rows,cols))
    okbin_rc = np.ones((rows,cols),dtype=bool)
    for f,file in enumerate(infileList):
        hdul = pyfits.open(file)
        okbin_rc &= (hdul['BPM'].data==0)
        image_frc[f] = hdul['SCI'].data
        summedianvar += np.median(hdul['VAR'].data[okbin_rc])   
    if (files < 3):                                 # minimize CR's
        medianimage_rc = image_frc.min(axis=0)
    else:
        medianimage_rc = np.median(image_frc,axis=0)
    hdul0['SCI'].data = medianimage_rc
    medianimagefile = objectname+'_'+filter+'_median.fits'
    hdul0.writeto(medianimagefile,overwrite=True)

  # run SeXtract
    deblend = .001        # set < .005 nominal to cause detection and rejection of close doubles
    minpix = 5 
    fwhmhi = 1.5
    sexTabcull,sexTab = sextract(medianimagefile,sigma=10.,deblend=deblend,minpix=minpix, \
                    fwhmhi=fwhmhi,cull=True,logfile=logfile,debug=debug)

  # compute background map using unculled sextract:
  # positions farther than from any star that would have intensity < background error
  # using Moffat function with median fwhm and beta=3
    fwbkg_s = np.array(sexTab["FWHM_IMAGE"])
    fmaxbkg_s = np.array(sexTab["FLUX_MAX"])
    bkgstars = fwbkg_s.shape[0]
    rcbkg_ds = np.zeros((2,bkgstars))
    rcbkg_ds[0] = sexTab["Y_IMAGE"] - 1.           # using python r,c (0,0) origin 
    rcbkg_ds[1] = sexTab["X_IMAGE"] - 1.

    moffatbeta = 3.
    fwhmedian = np.median(fwbkg_s)
    bkgerr = np.sqrt(summedianvar/files)
    bkgoff_s =  0.5*fwhmedian*np.sqrt((fmaxbkg_s/bkgerr)**(1./moffatbeta)-1.)/np.sqrt(2**(1./moffatbeta)-1.)
    bkgoff_s += np.clip(fwbkg_s - fwhmedian, 0., 3.*fwhmedian)    # allow for blends up to 3* single
    rcListz = [0,0]                             # a library of index arrays of stars, size = z (starting with 2)
    for z in range(2,int(bkgoff_s.max()+1)):
        rcListz.append(np.array(np.where(np.sqrt(np.arange(-z,z+1)[:,None]**2 +   \
                                                 np.arange(-z,z+1)[None,:]**2) <= z)) - z)                                         
    
    rcTups = ()                                 # tuple of star indices for image
    for s in range(bkgstars):
        src_di = rcListz[int(bkgoff_s[s])] + np.around(rcbkg_ds[:,s]).astype(int)[:,None]
        ok_i = ((src_di >=0).all(axis=0) & (src_di[0] < rows) & (src_di[1] < cols))
        rcTups += (src_di[:,ok_i],)

    isbkg_rc = np.ones((rows,cols),dtype=bool)        
    isbkg_rc[tuple(np.hstack(rcTups))] = False

  # for science targets, use sexTabcull list, with further culls:
  # (1) cull targets with companions contributing more than Moffat fn at 1.5*fwhm (.03)
    fl_s = np.array(sexTabcull["FLUX_ISO"]) 
    fw_s = np.array(sexTabcull["FWHM_IMAGE"])
    fwhmmed = np.median(fw_s)
    stars = fl_s.shape[0]
    rc_ds = np.zeros((2,stars))
    rc_ds[0] = sexTabcull["Y_IMAGE"] - 1.           # using python r,c (0,0) origin 
    rc_ds[1] = sexTabcull["X_IMAGE"] - 1.
    
    sep_ss = np.sqrt(((rc_ds[:,:,None]-rc_ds[:,None,:])**2).sum(axis=0))
    sep_ss[np.eye(stars,dtype=bool)] = 1.e9         # get rid of zeros on diagonal
    flrat_ss = fl_s[None,:]/fl_s[:,None]
    contam_ss = (flrat_ss/(1.+((sep_ss-1.5*fwhmmed)/fwhmmed)**2)**3)/0.03
    scontam_s = np.argmax(contam_ss,axis=1)
    maxcont_s = contam_ss[np.arange(stars),scontam_s]

    cullList = list(np.where(maxcont_s > 1.)[0])
    if debug:
        sepcont_s = sep_ss[np.arange(stars),scontam_s]
        flratcont_s = flrat_ss[np.arange(stars),scontam_s]
        np.savetxt(name+"contam_s.txt",np.vstack((np.arange(stars),scontam_s,maxcont_s, \
            sepcont_s,flratcont_s)).T,fmt="%4i %4i %8.2e "+2*"%18.2f ")
        np.savetxt(name+"cull1.txt",np.vstack((np.array(cullList),scontam_s[cullList],maxcont_s[cullList],  \
            sepcont_s[cullList],flratcont_s[cullList])).T,fmt="%4i %4i %8.2e "+2*"%18.2f ")

  # sort stars (inversely) by flux, final list index _S
    s_S = np.argsort(fl_s)[::-1]
    s_S = np.array([x for x in s_S if x not in cullList])
    Stars = s_S.shape[0]
    if debug:  
        printstdlog('After crowding cull: '+str(Stars),logfile)
        np.savetxt(name+"sort.txt",np.vstack((np.arange(Stars),s_S)).T,fmt="%4i")

  # separate O,E images based on Y 
  # Slistp is to be a list of np arrays,each one containing a culled, matched S idx for the two beams
    yx_dS = pixmm*rcbin_d[:,None]*(rc_ds[:,s_S] - rccenter_d[:,None]) + yx0_d[:,None] 
    yxmid_d = yx0_dp.mean(axis=1)                            # splitter midpoint
    Slistp = [np.where(yx_dS[0] < yxmid_d[0])[0],np.where(yx_dS[0] >= yxmid_d[0])[0]]   

  # (2) cull candidates more than 3x fainter than faintest predicted target and fwhm < 1.35 median       
    fl_S = fl_s[s_S]
    fw_S = fw_s[s_S]
    fwhmedian = np.median(fw_S)
 
    if debug:
        np.savetxt(name+"yxcand_O.txt",np.vstack((Slistp[0],yx_dS[0,Slistp[0]],yx_dS[1,Slistp[0]],  \
            fl_S[Slistp[0]],fw_S[Slistp[0]])).T,fmt="%4i %10.2f %10.2f %9.0f %10.2f")
        np.savetxt(name+"yxcand_E.txt",np.vstack((Slistp[1],yx_dS[0,Slistp[1]],yx_dS[1,Slistp[1]],  \
            fl_S[Slistp[1]],fw_S[Slistp[1]])).T,fmt="%4i %10.2f %10.2f %9.0f %10.2f")

    flcand10 = (fl_S[Slistp[0][:10]] + fl_S[Slistp[1][:10]]).sum()/2.   # mean flux of 10 brtest candidates
    flcat10 = (10.**-(0.4*mapTab['MAG']))[:10].sum()
    flmin = (flcand10/flcat10)*10.**-(0.4*mapTab['MAG'][-1])/3.
    for p in (0,1):
        Slistp[p] = Slistp[p][(fl_S[Slistp[p]] > flmin) & (fw_S[Slistp[p]] < 1.35*fwhmedian)]

    if debug:
        printstdlog(('After minimum flux and fwhm cull, O,E: %4i %4i' % (len(Slistp[0]),len(Slistp[1]))),logfile)
        printstdlog('\nCull to O,E pairs' ,logfile)

  # (3) cull candidates to those with identified OE pairs, find OE offset error, correct splitting
    dyxpred_dS = np.array([(yx0_dp[0,1]-yx0_dp[0,0]) +  \
        .01235*(yx_dS[0,Slistp[0]] - yx0_dp[0,0]),np.zeros(Slistp[0].shape[0])])
    o_i, e_i = catid(yx_dS[:,Slistp[0]]+dyxpred_dS, yx_dS[:,Slistp[1]] ,       \
        offsettol=64.*pixmm,errtol=8.*pixmm,debug=debug,logfile=logfile,name=name+'OE_')

    Slistp = [Slistp[0][o_i],Slistp[1][e_i]]
    dyxpred_dS = np.array([(yx0_dp[0,1]-yx0_dp[0,0]) +  \
        .01235*(yx_dS[0,Slistp[0]] - yx0_dp[0,0]),np.zeros(Slistp[0].shape[0])])
    yxOEoff_d = np.median(yx_dS[:,Slistp[1]]-(yx_dS[:,Slistp[0]]+dyxpred_dS), axis=1)
    dyfac = yxOEoff_d[0]/(yx0_dp[0,1]-yx0_dp[0,0])        
    yx_dpt[0] += dyfac*(yx_dpt[0] - yxmid_d[0,None,None])
    yx_dpt[1] += -(pm1_p*yxOEoff_d[1]/2.)[:,None]
    drsplit = int(yxOEoff_d[0]/2.)/(rcbin_d[0]*pixmm)
    isfov_rc[:prows] = shift(isfov_rc[:prows],(-drsplit,0),order=0)
    isfov_rc[prows:] = shift(isfov_rc[prows:],(drsplit,0),order=0)

    if debug:
        np.savetxt(name+"yxcand_OE.txt",np.vstack((Slistp[0],yx_dS[:,Slistp[0]],fl_S[Slistp[0]],    \
                Slistp[1],yx_dS[:,Slistp[1]],fl_S[Slistp[1]])).T, fmt= 2*"%4i %9.3f %9.3f %9.0f ")

  # find rough offset and rotation of the O beam image, using no more than the brightest 100 candidates
  # look over rot = +/-3 deg, y,x = +/-6 mm (by 0.1)
    cmatch = min(100,len(Slistp[0]))
    rot_R = np.arange(-3.,3.,.1)
    histmax_R = np.zeros(60)
    maxidx_R = np.zeros(60,dtype=int)
    offyx_Rdts = np.zeros((60,2,targets,cmatch))
    for R,rot in enumerate(rot_R):
        offyx_Rdts[R] = rotate2d(yx_dS[:,Slistp[0][:cmatch]],rot,center=yx0_dp[:,0])[:,None,:] -  \
                    yx_dpt[:,0][:,:,None]
        offhist_yx,yedge,xedge = np.histogram2d(offyx_Rdts[R,0].flatten(),offyx_Rdts[R,1].flatten(),  \
            bins=np.arange(-6.,6.1,0.1))
        maxidx_R[R] = np.argmax(offhist_yx)
        histmax_R[R] = offhist_yx.flatten()[maxidx_R[R]]
    Rmax = np.argmax(histmax_R)
    yxoff_dR = (np.array(np.unravel_index(maxidx_R,(120,120))) - 60.)/10.
    yxoff_dp = np.array([yxoff_dR[:,Rmax],yxoff_dR[:,Rmax]]).T
    yxscloff_dp = np.zeros((2,2))
    rotoff = rot_R[Rmax]
    yxupdate_dpt = yx_dpt + yxoff_dp[:,:,None] 
    yxupdate_dpt += np.array([1.,-1.])[:,None,None]*np.radians(rotoff)*   \
                ((yxupdate_dpt - yx0_dp[:,:,None])[::-1])    

    if debug:
        np.savetxt(name+"rotsearch.txt",np.vstack((rot_R,histmax_R,yxoff_dR)).T,fmt="%6.1f %4i %6.1f %6.1f") 
        printstdlog('\nRough yoff xoff (mm), rot (deg), yxOEoff_d (mm):' ,logfile)
        printstdlog((5*'%8.2f ') % (tuple(yxoff_dp[:,0])+(rotoff,)+tuple(yxOEoff_d)),logfile)

  # (4) id in O beam at this rough rot,offset,OE offset; evaluate telescope guide and OE offset; then iterate once
  # note, the id'd targets are in the same order in Slistp[0] and [1], so S_i refers to both
    yxOEoff_d = np.zeros(2)             # resetting back to zero for full fit relative to nominal model
    dYXtot_d = np.zeros(2)   
    drottot = 0
    off_dp = np.array([[-.5,.5],[-.5,.5]])     
    for iter in (1,2):
        t_i,S_i = catid(yxupdate_dpt[:,0], yx_dS[:,Slistp[0]],   \
            offsettol=2./iter, errtol=0.5/iter,debug=debug,logfile=logfile,name=name+'id'+str(iter)+'_')

      # Fit offset of star from predicted catalogue position, to get telescope guide error, beamsplitter axis offset
        ids = len(t_i)
        yxS_dpi = np.array([yx_dS[:,Slistp[0][S_i]], yx_dS[:,Slistp[1][S_i]]]).transpose((1,0,2))

        dYX_d,drot,dyxOEoff_d,dYXerr_d,droterr,dyxOEofferr_d =  \
            impolguide(YX_dt[:,t_i],yxS_dpi,yxOEoff_d,wavl,coltem,camtem,debug=debug,name=name)
        YX_dt += dYX_d[:,None] + np.array([1.,-1.])[:,None]*np.radians(drot)*YX_dt[::-1]
        dYXtot_d += dYX_d
        drottot += drot
        yxOEoff_d += dyxOEoff_d
        yxupdate_dpt = RSScolpolcam(YX_dt,wavl,coltem,camtem,yxOEoff_d=yxOEoff_d)

        if debug:
            np.savetxt(name+"catid"+str(iter)+".txt",   \
                np.vstack((t_i,S_i,yxupdate_dpt[:,:,t_i].reshape((4,-1)),yxS_dpi.reshape((4,-1)))).T, \
                fmt="%4i %4i  "+2*(4*"%8.3f ")) 

      # (5) cull targets that are within target binradius of gap, fov edge, split image edge
        fwhmedian = np.median(fw_s[S_i])
        binrad = int(2.*fwhmedian)
        binradmm = binrad*pixmm*rcbin_d.max()
        rcpidx_dpi = np.zeros((2,2,ids))
        okid_i = np.ones(ids,dtype=bool)
        for p in (0,1): 
            okid_i &= isfov_rc[list(rc_ds[:,s_S[Slistp[p][S_i]]].astype(int))]
            okid_i &= (np.abs(rc_ds[1,s_S[Slistp[p][S_i]]][:,None] - cgapedge_c[None,:]).min(axis=1) > binrad)
          # compute positions of unculled id's in O,E images to avoid edges
            for id in range(ids):
                rcpidx_dpi[0,p,id] = np.round(rc_ds[0,s_S[Slistp[p]][S_i[id]]] - rshift-p*prows).astype(int)
                rcpidx_dpi[1,p,id] = np.round(rc_ds[1,s_S[Slistp[p]][S_i[id]]]).astype(int)          
            okid_i &= ((rcpidx_dpi[0,p] >= binrad) & (rcpidx_dpi[0,p] < (prows-binrad)))
            okid_i &= ((rcpidx_dpi[1,p] >= binrad) & (rcpidx_dpi[1,p] < (cols-binrad)))

        edgeculls = (~okid_i).sum()
        if ((edgeculls>0) & debug):
            printstdlog(('removing %2i targets too close to edges' % edgeculls),logfile)
            np.savetxt(name+"edgeculls.txt",np.vstack((rc_ds[:,s_S[Slistp[0][S_i[~okid_i]]]], \
                rc_ds[:,s_S[Slistp[1][S_i[~okid_i]]]])).T,fmt="%7.1f")
        if edgeculls:
            for p in (0,1):
                Slistp[p] = np.delete(Slistp[p],S_i[~okid_i])

        if debug:
            printstdlog (("\nTarget ID's: %3i" % ids), logfile)
            printstdlog ("    Yoff   +/-     Xoff   +/-   rotoff  +/-   dyOEoff   +/-   dxOEoff   +/-  ",logfile)
            printstdlog (5*" %7.3f %5.3f " % (dYX_d[0],dYXerr_d[0],dYX_d[1],dYXerr_d[1],   \
                drot,droterr,dyxOEoff_d[0],dyxOEofferr_d[0],dyxOEoff_d[1],dyxOEofferr_d[1]),logfile)
            if iter==1:
                printstdlog('\nAfter re-id and edgeculls' ,logfile)

    yx0_dp += -pm1_p[None,:]*yxOEoff_d[:,None]/2.
    yxp0_dp += pm1_p[None,:]*yxOEoff_d[:,None]/2.      # holding the relative centers constant   
    
    if debug:
        print >>geomfile, 'After OE update:'
        print >>geomfile, ('yxOEoff_d: '+2*'%8.4f ' % tuple(yxOEoff_d.flatten()))          
        print >>geomfile, ('yx0_dp    : '+4*'%8.4f ' % tuple(yx0_dp.flatten()))
        print >>geomfile, ('yxp0_dp   : '+4*'%8.4f ' % tuple(yxp0_dp.flatten()))   

  # compute actual RA, DEC of opt axis for header
    PAd0 = PAd - drottot
    sinpa, cospa = np.sin(np.radians(PAd0)), np.cos(np.radians(PAd0))
    DECd0 = DECd - (cospa*dYXtot_d[0] - sinpa*dYXtot_d[1])/(3.6*saltfps) 
    RAd0 = RAd + (cospa*dYXtot_d[1] + sinpa*dYXtot_d[0])/(3.6*saltfps*np.cos(np.radians(DECd0)))
    YX_di = YXcalc(mapTab['RA'],mapTab['DEC'],RAd0,DECd0,PAd0,saltfps)[:,t_i]
    printstdlog('\nPointing update:\n RA  DEC (arcsec)  PA (deg)  OEyoffset OEyoffset (mm)',logfile)
    printstdlog((' %8.2f %8.2f %8.3f %8.3f %8.3f') %    \
        ((3600.*(RAd0-RAd),3600.*(DECd0-DECd),drottot,)+tuple(yxOEoff_d)),logfile)

  # split into separate O,E images (yxp), split at O,E separation point yxsep_d by integer row bins
    yxpcat_dpi = np.copy(yxupdate_dpt[:,:,t_i])
    yxp_dpi = np.zeros_like(yxpcat_dpi)
    dyxp_dp = rcbin_d[0]*pixmm*np.array([[-rshift+rows/4,-rshift-rows/4],[0.,0.]]) + (yxp0_dp - yx0_d[:,None])
    for p in (0,1):
        yxpcat_dpi[:,p] = yxpcat_dpi[:,p] + dyxp_dp[:,p,None]
        yxp_dpi[:,p] = yx_dS[:,Slistp[p][S_i]] + dyxp_dp[:,p,None]
    if debug:
        magrat = np.sqrt((yxp_dpi**2).sum()/(yxpcat_dpi**2).sum())
        printstdlog(('Found/cat magnification: %8.5f' % magrat),logfile)
        
    prccenter_d = np.array([(prows-1.)/2.,rccenter_d[1]])

  # update catalog prediction by removing unid'd rows, updating [yo,ye]  [xo,xe] 
    poskey_dp = np.array([['YO','YE'],['XO','XE']])
    rckey_dp = np.array([['R0O','R0E'],['C0O','C0E']])
    pmapTab = mapTab[t_i]               # map of catalog positions, for debug
    pmapTab['YO'],pmapTab['YE'],pmapTab['XO'],pmapTab['XE'] = tuple(yxpcat_dpi.reshape((4,-1)))
                
    pfoundTab = mapTab[t_i]             # map of found positions, for output hdu    
    offmag_pi = np.zeros((2,ids))       
    for p in (0,1):
        offmag_pi[p] = (-2.5*np.log10(fl_S[Slistp[p][S_i]]) - mapTab['MAG'][t_i])
    magoff_p = np.median(offmag_pi,axis=1)          # updated MAG
    pfoundTab['MAG'] = 0.5*((-2.5*np.log10(fl_S[Slistp[0][S_i]]) - magoff_p[0,None]) + \
                       (-2.5*np.log10(fl_S[Slistp[1][S_i]]) - magoff_p[1,None])) 
    pfoundTab['Y'],pfoundTab['X'] = tuple(YX_di)    # updated SALT positions    
    for p in (0,1):
        pfoundTab[poskey_dp[0,p]],pfoundTab[poskey_dp[1,p]] = yxp_dpi[:,p]
        pfoundTab[['FLXO','FLXE'][p]] = fl_S[Slistp[p][S_i]]
        pfoundTab[['FWHO','FWHE'][p]] = fw_S[Slistp[p][S_i]]

  # background area masked by the distorted polarimetric FOV
    isbkg_rc &= isfov_rc
    isbkg_prc = shift(isbkg_rc,(-rshift,0),order=0).reshape((2,prows,cols))

  # input all data 
    image_fprc = np.zeros((files,2,prows,cols))
    var_fprc = np.zeros_like(image_fprc)
    okbin_fprc = np.zeros((files,2,prows,cols),dtype=bool)
    isbkgcr_fprc = np.zeros_like(okbin_fprc)
    for f,file in enumerate(infileList):
        hdul = pyfits.open(file)
        image_rc = hdul['SCI'].data
        image_fprc[f] = shift(image_rc,(-rshift,0),order=0).reshape((2,prows,cols))
        var_rc = hdul['VAR'].data
        var_fprc[f] = shift(var_rc,(-rshift,0),order=0).reshape((2,prows,cols))
        okbin_rc = (hdul['BPM'].data==0)
        okbin_fprc[f] = shift(okbin_rc,(-rshift,0),order=0).reshape((2,prows,cols)) 

  # cull background for unresolved blobs       
    medianimage_prc = np.median(image_fprc,axis=0)
    medianvar_prc = np.median(var_fprc,axis=0)
    q1_p = np.zeros(2)
    q3_p = np.zeros(2)
    for p in (0,1):
        q1_p[p],q3_p[p] = np.percentile(medianimage_prc[p][(isbkg_prc&okbin_fprc[f])[p]],(25,75))
    isbkg_prc = (isbkg_prc & (medianimage_prc < (q3_p + 3.*(q3_p-q1_p))[:,None,None]))

  # mark background CR's as BP 
    isbkgcr_fprc = (isbkg_prc & ((image_fprc - medianimage_prc) > 10.*np.sqrt(medianvar_prc)))
    bkgcrs_f = isbkgcr_fprc.sum(axis=(1,2,3))
    okbin_fprc[isbkgcr_fprc] = False  

  # form O,E targetmap extension.  
  # Found target area indicated by value id+1 (0 = no target, or overlap)
    targetmap_prc = np.zeros((2,prows,cols),dtype='uint8')
    isoverlap_prc = np.zeros((2,prows,cols),dtype=bool)
    grid_drc = np.indices((prows,cols))
    for p,id in np.ndindex(2,ids):   
        istargetpi_rc = (np.sqrt((grid_drc[0] - rcpidx_dpi[0,p,id])**2 +   \
                               (grid_drc[1] - rcpidx_dpi[1,p,id])**2) < binrad)
        isoverlap_prc[p] |= (istargetpi_rc & (targetmap_prc[p] > 0))
        targetmap_prc[p][istargetpi_rc] = id + 1

    targetmap_prc[isbkg_prc & (targetmap_prc==0)] = 255
    targetmap_prc[isoverlap_prc] = 0
    if debug:
        targetbins_tp = (targetmap_prc[None,:,:,:]==np.arange(1,ids+1)[:,None,None,None]).sum(axis=(2,3))
        np.savetxt(name+"targetbins_tp.txt",np.vstack((range(ids),targetbins_tp.T)).T,fmt="%3i")

  # rckey is the (lower left) origin of the target box
    for p in (0,1):
        pfoundTab[rckey_dp[0,p]],pfoundTab[rckey_dp[1,p]] =     \
            rcpidx_dpi[:,p] - np.array([binrad,binrad])[:,None]
    if debug:    
        pmapTab.write(objectname+"_"+filter+"_pmapTab.txt",format='ascii.fixed_width',   \
            bookend=False, delimiter=None, overwrite=True)          
        pfoundTab.write(objectname+"_"+filter+"_pfoundTab.txt",format='ascii.fixed_width',   \
            bookend=False, delimiter=None, overwrite=True)   

  # for each input fits, add targetmap and MOSTab extensions, write to tm*.fits
  # tm files produced only for multi-file polarimetric observation
    boxwidth = int(2*(binrad-1)+1)
    BOXRC = (("%3i %3i") % (boxwidth,boxwidth))
    YXAXISO = ("%7.4f %7.4f" % tuple(yxp0_dp[:,0]))
    YXAXISE = ("%7.4f %7.4f" % tuple(yxp0_dp[:,1]))
    RA0 = Angle(RAd0, u.degree).to_string(unit=u.hour, sep=":")
    DEC0 = Angle(DECd0, u.degree).to_string(unit=u.degree, sep=":")
    PA0 = PAd0
    hdrList = [YXAXISO,YXAXISE,RA0,DEC0,PA0]       
    for f,file in enumerate(infileList):
        if len(infileList) == 1: break
        hdul = pyfits.open(file)
        hdr = hdul[0].header
        hdul['SCI'].data = image_fprc[f].astype('float32')
        hdul['VAR'].data = var_fprc[f].astype('float32') 
        hdul['BPM'].data = (~okbin_fprc[f]).astype('uint8')
        hdr['YXAXISO'] = (YXAXISO,"O Optic Axis (mm)")
        hdr['YXAXISE'] = (YXAXISE,"E Optic Axis (mm)")
        hdr['BOXRC'] = (BOXRC,"target box (bins)")
        hdr['RA0'] = (RA0,"RA center")
        hdr['DEC0'] = (DEC0,"DEC center")
        hdr['PA0'] = (PA0,"PA actual")
        hdul.append(pyfits.ImageHDU(data=okbin_fprc[f]*targetmap_prc,name='TMAP'))
        hdul.append(pyfits.table_to_hdu(pfoundTab))
        hdul[-1].header['EXTNAME'] = 'TGT'
        hdul.writeto("t"+file,overwrite=True)
        printstdlog(('Output file '+'t'+file+'. Background CRs: %4i' % bkgcrs_f[f]),logfile)

  # return sextract star map for this filter, split at midpoint,
  #   to be used in imaging spectropolarimetry background
  #   and hdr items for imaging spectropolarimetry tm files
    fmaxbkg_rc = np.zeros((rows,cols))
    fwbkg_rc = np.zeros((rows,cols))
    rcidx_ds = np.round(rcbkg_ds).astype(int)
    fmaxbkg_rc[rcidx_ds[0],rcidx_ds[1]] = fmaxbkg_s
    fwbkg_rc[rcidx_ds[0],rcidx_ds[1]] = fwbkg_s
    fmaxbkg_prc = shift(fmaxbkg_rc,(-rshift,0),order=0).reshape((2,prows,cols))
    fwbkg_prc = shift(fwbkg_rc,(-rshift,0),order=0).reshape((2,prows,cols))
    
    return    

# ------------------------------------
def imspecpolmap(infileList, mapTab, calHdu_F, cutwavoverride=0., logfile='salt.log',debug=False):
  # _t targets in mapTab catalog
  # _i id'd targets
  # _F calibration filter index
  # _f infileList index

  # get configuration data from first image
    obsDictf = obslog(infileList)
    files = len(infileList)
    hdul0 = pyfits.open(infileList[0])
    dateobs =  hdul0[0].header['DATE-OBS'].replace('-','')
    trkrho = hdul0[0].header['TRKRHO']    
    rows, cols = hdul0[1].data.shape
    cbin, rbin = [int(x) for x in hdul0[0].header['CCDSUM'].split(" ")]
    rcbin_d = np.array([rbin,cbin])
    pixmm = 0.015
    binmm_d = rcbin_d*pixmm
    targets = len(mapTab)
    rccenter_d, cgapedge_c = ccdcenter(hdul0[1].data)    
    prows = rows/2
    rcpcenter_d = np.array([prows/2.,rccenter_d[1]])   
    pm1_p = np.array([1.,-1.])

    objectname = obsDictf['OBJECT'][0]
    cutfilter = obsDictf['FILTER'][0]
    camtem = hdul0[0].header['CAMTEM']
    coltem = hdul0[0].header['COLTEM']
    exptime = hdul0[0].header['EXPTIME']
    imwav = 5000.                               # use for impol: close to center of spectrum

    cutwav = max(float(cutfilter[3:]),3200.,cutwavoverride)    
    name = objectname+"_"+str(int(cutwav))+"_imf_"
                       
  # compute usable unfiltered FOV: most background is from red
    dum,rshift,dum,isfovblue_rc = RSSpolgeom(hdul0,cutwav)
    isfovred_rc = RSSpolgeom(hdul0,10200.)[3]
    isfov_rc = isfovred_rc
    isfov_prc = shift(isfov_rc,(-rshift,0),order=0).reshape((2,prows,cols))

  # Get data from filtered cals _F.  Retain id'd targets _I found in all F:
    catid_t,ra_t,dec_t = mapTab['CATID'],mapTab['RA'],mapTab['DEC'] 
    targets = catid_t.shape[0]  
    calfilters = calHdu_F.shape[0]  
    yxOEoff_Fd = np.zeros((calfilters,2))
    tel0_Fd = np.zeros((calfilters,3))
    mag_Ft = np.zeros((calfilters,targets))
    isid_Ft = np.zeros((calfilters,targets),dtype=bool)
    dyxp_Fdpt = np.zeros((calfilters,2,2,targets))
    boxwidth = 0
    
    if debug:
        open(name+'geom.txt',"w")
        geomfile = open(name+'geom.txt',"a")      
    
    for F in range(calfilters):
        hdrF = calHdu_F[F][0].header
        wavl = float(hdrF['FILTER'][2:])
        trkrhoF = hdrF['TRKRHO']       
        boxwidth = max(boxwidth, int(hdrF['BOXRC'].split()[0]))
        yxp0_dp = np.array([map(float,hdrF['YXAXISO'].split()),map(float,hdrF['YXAXISE'].split())]).T
        yxOEoff_d = -np.diff((yxp0_dp - RSSpolgeom(calHdu_F[F],wavl)[2]), axis=1)[:,0]   # OEbeam model correction                     
        ur0,uc0,saltfps = rssdtralign(dateobs,trkrhoF) 
        yx0_d = -0.015*np.array([ur0,uc0])
        yx0_dp = yx0_d[:,None] - yxp0_dp -  \
            binmm_d[:,None]*np.array([[-rshift + rows/4, -rshift - rows/4],[0.,0.]])                       
        tgtTab = Table.read(calHdu_F[F]['TGT'])
                    
        RAd0 = Longitude(hdrF['RA0']+' hours').degree
        DECd0 = Latitude(hdrF['DEC0']+' degrees').degree
        PAd0 = hdrF['PA0']

        YX_di = np.array([tgtTab['Y'],tgtTab['X']])        
        yxcat_dpi = RSScolpolcam(YX_di,wavl,coltem,camtem,yxOEoff_d=yxOEoff_d)
        yxpcat_dpi = yxcat_dpi - yx0_dp[:,:,None]
        catid_i = tgtTab['CATID']
        yxp_dpi = np.array([[tgtTab['YO'],tgtTab['YE']],[tgtTab['XO'],tgtTab['XE']]]) 
        
        if debug:
            print >>geomfile       
            print >>geomfile, 'RSSpolgeom, calfil ',wavl              
            print >>geomfile, ('yx0_d  : %8.4f %8.4f'% tuple(yx0_d))
            print >>geomfile, ('yx0_dp    : '+4*'%8.4f ' % tuple(RSSpolgeom(calHdu_F[F],wavl)[0].flatten()))            
            print >>geomfile, ('yxp0_dp : '+4*'%8.4f ' % tuple(RSSpolgeom(calHdu_F[F],wavl)[2].flatten()))
            print >>geomfile, 'After OE update:'
            print >>geomfile, ('yxOEoff_d : '+2*'%8.4f ' % tuple(yxOEoff_d.flatten()))
            print >>geomfile, ('yx0_dp    : '+4*'%8.4f ' % tuple(yx0_dp.flatten()))
            print >>geomfile, ('yxp0_dp   : '+4*'%8.4f ' % tuple(yxp0_dp.flatten()))            
                   
        isid_Ft[F] = np.in1d(catid_t,catid_i)                                      
        yxOEoff_Fd[F] = yxOEoff_d
        tel0_Fd[F] = RAd0,DECd0,PAd0
        mag_Ft[F][isid_Ft[F]] = tgtTab['MAG']        
        dyxp_Fdpt[F][:,:,isid_Ft[F]] = yxp_dpi - yxpcat_dpi         # target offset from model        
        
    isId_t = isid_Ft.all(axis=0)
    Ids = isId_t.sum()
    yxOEoff_d = yxOEoff_Fd.mean(axis=0)
    yxp0_dp = RSSpolgeom(hdul0,imwav,yxOEoff_d)[2]          # reference: geometry at 5000 Ang plus mean OEbeam correction
    tel0_d = tel0_Fd.mean(axis=0)                           # reference: mean telescope position
    mag_I = mag_Ft[:,isId_t].mean(axis=0)
    dyxp_dpI = dyxp_Fdpt[:,:,:,isId_t].mean(axis=0)         # reference: deviations from model (mean over calfilters)    
    ur0,uc0,saltfps = rssdtralign(dateobs,trkrho) 
    yx0_d = -0.015*np.array([ur0,uc0]) 
    yx0_dp = yx0_d[:,None] - yxp0_dp -  \
        binmm_d[:,None]*np.array([[-rshift + rows/4, -rshift - rows/4],[0.,0.]])

    if debug:
        print >>geomfile            
        print >>geomfile, 'RSSpolgeom, reference:'
        print >>geomfile, ('yx0_d  : %8.4f %8.4f'% tuple(yx0_d))
        print >>geomfile, ('yx0_dp    : '+4*'%8.4f ' % tuple(RSSpolgeom(hdul0,imwav)[0].flatten()))
        print >>geomfile, ('yxp0_dp : '+4*'%8.4f ' % tuple(RSSpolgeom(hdul0,imwav)[2].flatten()))        
        print >>geomfile, 'After OE update:'
        print >>geomfile, 'rshift ',rshift 
        print >>geomfile, ('yxOEoff_d : '+2*'%8.4f ' % tuple(yxOEoff_d.flatten()))                                    
        print >>geomfile, ('yx0_dp : '+4*'%8.4f ' % tuple(yx0_dp.flatten()))    
        print >>geomfile, ('yxp0_dp: '+4*'%8.4f ' % tuple(yxp0_dp.flatten()))
        np.savetxt(objectname+'_dyxp_dpt.txt',np.vstack((isId_t,dyxp_Fdpt.reshape((-1,targets)), \
                dyxp_Fdpt.mean(axis=0).reshape((-1,targets)))).T,fmt="%2i "+4*(calfilters+1)*"%8.3f ")

  # compute id'd target wavmap (wavl vs row), cmap (column vs row)
  #   to establish target map for unfiltered imaging spectropolarimetry
    RAd0,DECd0,PAd0 = tel0_d    
    YX_dI = YXcalc(mapTab['RA'],mapTab['DEC'],RAd0,DECd0,PAd0,saltfps)[:,isId_t]
    wav_W = np.arange(cutwav,10600.,100.)
    Wimref = np.where(wav_W == imwav)[0][0]
    Wavs = wav_W.shape[0]
    wav_s = np.tile(wav_W,Ids)
    YX_ds = np.repeat(YX_dI,Wavs,axis=1)                
    yx_dpIW = RSScolpolcam(YX_ds,wav_s,coltem,camtem,yxOEoff_d=yxOEoff_d).reshape((2,2,Ids,Wavs))
    yxp_dpIW = yx_dpIW + dyxp_dpI[:,:,:,None] - yx0_dp[:,:,None,None]
    rcp_dpIW  = rcpcenter_d[:,None,None,None] +     \
        (yxp_dpIW - yxp0_dp[:,:,None,None])/binmm_d[:,None,None,None]              
    r0_pI = np.clip(np.ceil(rcp_dpIW[0].min(axis=2)).astype(int),0,prows-1)
    r1_pI = np.clip(np.floor(rcp_dpIW[0].max(axis=2)).astype(int),0,prows-1)    
    wavmap_pIr = np.zeros((2,Ids,prows),dtype='float32')  
    cmap_pIr = np.zeros_like(wavmap_pIr)      
            
    for p,Id in np.ndindex(2,Ids):
        Rows = (r1_pI - r0_pI)[p,Id]+1            
        dr_W = rcp_dpIW[0,p,Id] - r0_pI[p,Id,None]                   
        wavmap_pIr[p,Id,r0_pI[p,Id]:r1_pI[p,Id]+1] =   \
            interp1d(dr_W,wav_W,kind='cubic')(np.arange(Rows))                       
        cmap_pIr[p,Id,r0_pI[p,Id]:r1_pI[p,Id]+1] =   \
            interp1d(dr_W,rcp_dpIW[1,p,Id],kind='cubic')(np.arange(Rows))

  # Form targetmap.  take wavelengths below color filter cutoff off targetmap
    ridx0_pI = np.argmax(wavmap_pIr > 0,axis=2)
    ridx1_pI = prows - np.argmax(wavmap_pIr[:,:,::-1] > 0,axis=2)-1
    Rows = (ridx1_pI - ridx0_pI).max() - 1 
    okbin_rc = (hdul0['BPM'].data==0)               # assume input bpm is same for all files
    okbin_prc = shift(okbin_rc,(-rshift,0),order=0).reshape((2,prows,cols))    

  #  add 1/2 box on NIR side to use NIR cutoff for row guiding, document in TGT table
    Cols = boxwidth                       # this is an odd number,from cal targetmap
    grid_drc = np.indices((prows,cols))
    cidx_pI = np.zeros((2,Ids),dtype=int)
    for p,Id in np.ndindex(2,Ids):
        cidx_pI[p,Id] = int(np.round(cmap_pIr[p,Id,ridx0_pI[p,Id]:ridx1_pI[p,Id]].mean()))
    ridx0_pI[1] -= Cols/2
    ridx1_pI[0] += Cols/2
    Rows += Cols/2
    targetmap_prc = np.zeros((2,prows,cols),dtype='uint8')
    isoverlap_prc = np.zeros((2,prows,cols),dtype=bool)
    gridprc_da = np.indices((prows,cols))
    isbkg_prc = np.copy(isfov_prc)
    
    for p,Id in np.ndindex(2,Ids):
        istargetpI_rc = (((gridprc_da[0] >= ridx0_pI[p,Id]) & (gridprc_da[0] <= ridx1_pI[p,Id])) &   \
                         (np.abs(gridprc_da[1] - cidx_pI[p,Id]) <= Cols/2))
        isoverlap_prc[p] |= (istargetpI_rc & (targetmap_prc[p] > 0))
        targetmap_prc[p][istargetpI_rc & isfov_prc[p]] = Id + 1 
        isbkg_prc[p] &= ~istargetpI_rc

  # gather reference data for prefTab            
    poskey_dp = np.array([['YO','YE'],['XO','XE']])
    rckey_dp = np.array([['R0O','R0E'],['C0O','C0E']])
    prefTab = mapTab[isId_t]
    prefTab['MAG'] = mag_I
    prefTab['Y'],prefTab['X'] = tuple(YX_dI)
    for p in (0,1):               
        prefTab[rckey_dp[0,p]],prefTab[rckey_dp[1,p]] = (ridx0_pI[p],cidx_pI[p]-Cols/2)     # O: too far to the left, E OK
    for p in (0,1):        
        prefTab[poskey_dp[0,p]],prefTab[poskey_dp[1,p]] = tuple(yxp_dpIW[:,p,:,Wimref])
        
  # input all data, split it into OE images 
    image_fprc = np.zeros((files,2,prows,cols))
    var_fprc = np.zeros_like(image_fprc)
    okbin_fprc = np.zeros((files,2,prows,cols),dtype=bool)
    isbkgcr_fprc = np.zeros_like(okbin_fprc)
    for f,file in enumerate(infileList):
        hdul = pyfits.open(file)
        image_rc = hdul['SCI'].data
        image_fprc[f] = shift(image_rc,(-rshift,0),order=0).reshape((2,prows,cols))
        var_rc = hdul['VAR'].data
        var_fprc[f] = shift(var_rc,(-rshift,0),order=0).reshape((2,prows,cols))
        okbin_rc = (hdul['BPM'].data==0)
        okbin_fprc[f] = shift(okbin_rc,(-rshift,0),order=0).reshape((2,prows,cols)) 

  # mark background CR's beyond extreme upper quartile fence as BP           
    q1_prc,q3_prc = np.percentile(image_fprc,(25,75),axis=0)
    isbkgcr_fprc = isbkg_prc[None,:,:,:] & (image_fprc > (q3_prc + 5.*(q3_prc-q1_prc))[None,:,:,:])
    bkgcrs_f = isbkgcr_fprc.sum(axis=(1,2,3))
    okbin_fprc[isbkgcr_fprc] = False    
    if debug:   
        prefTab.write(objectname+"_"+cutfilter+"_prefTab.txt",format='ascii.fixed_width',   \
            bookend=False, delimiter=None, overwrite=True)    
        hdul0['SCI'].data = isbkg_prc.astype('uint8')
        hdul0.writeto("isbkg_prc.fits",overwrite=True)                     

    targetmap_prc[isbkg_prc] = 255
    targetmap_prc[isoverlap_prc] = 0
    YXAXISO = ("%7.4f %7.4f" % tuple(yxp0_dp[:,0]))
    YXAXISE = ("%7.4f %7.4f" % tuple(yxp0_dp[:,1]))
    BOXRC = (("%3i %3i") % (Rows,Cols))     
    RA0 = Angle(RAd0, u.degree).to_string(unit=u.hour, sep=":")
    DEC0 = Angle(DECd0, u.degree).to_string(unit=u.degree, sep=":")
    BOXRC = (("%3i %3i") % (Rows,Cols))
    CALIMG = ' '.join([calHdu_F[F].filename().split('.')[0][-4:] for F in range(calfilters)])
    printstdlog(('Blue wavelength cutoff: %6.0f' % cutwav),logfile)    
    printstdlog(('Target box Rows,Cols: %3i %3i' % (Rows,Cols)),logfile)    

  # for each input fits, add targetmap extension, write to tm*.fits
    for f,file in enumerate(infileList):
        hdul = pyfits.open(file)
        hdr = hdul[0].header
        hdul['SCI'].data = image_fprc[f].astype('float32')
        hdul['VAR'].data = var_fprc[f].astype('float32') 
        hdul['BPM'].data = (~okbin_fprc[f]).astype('uint8')
        hdul.append(pyfits.ImageHDU(data=targetmap_prc,name='TMAP'))
        hdul.append(pyfits.table_to_hdu(prefTab))
        hdul[-1].header['EXTNAME'] = 'TGT'
        hdr['YXAXISO'] = (YXAXISO,"O Optic Axis (mm)")
        hdr['YXAXISE'] = (YXAXISE,"E Optic Axis (mm)")
        hdr['BOXRC'] = (BOXRC,"target box (bins)")
        hdr['CALIMG'] = (CALIMG,"cal image no(s)")        
        hdr['RA0'] = (RA0,"RA center")
        hdr['DEC0'] = (DEC0,"DEC center")
        hdr['PA0'] = (PAd0,"PA actual")
        hdul.writeto("t"+file,overwrite=True)
        printstdlog(('Output file '+'t'+file+'. Background CRs: %4i' % bkgcrs_f[f]),logfile)

    return
   
# ------------------------------------
def printstdlog(string,logfile):
    print string
    print >>open(logfile,'a'), string
    return 

# ------------------------------------
if __name__=='__main__':
    infilelist=sys.argv[1:]      
    poltargetmap(infilelist)


