
"""
immospolmap

Locate targets in mos arc image, save with 2D target map 
Compute 1D extraction map, 1D wavelength map for imaging mos spectropolarimetric data

imarcpolmap
imspecmospolmap

"""

import os, sys, glob, shutil, inspect, warnings

import numpy as np
from scipy.linalg import lstsq
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage.interpolation import shift
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import Latitude,Longitude,Angle
from astropy.table import Table

# this is pysalt-free

import rsslog
from obslog import create_obslog
from polutils import rssdtralign, rssmodelwave
from polmaptools import sextract,catid,ccdcenter,gaincor,YXcalc,impolguide,rotate2d,Tableinterp
from rssoptics import RSScolpolcam, RSSpolgeom
from immospolextract import moffat,moffat1dfit,fence    

datadir = os.path.dirname(__file__) + '/data/'
keywordfile = datadir+"obslog_config.json"
np.set_printoptions(threshold=np.nan)

# warnings.simplefilter("error")
debug = False

# ------------------------------------

def imarcpolmap(infile,MOSTab,objectname,FOVlim=49.,logfile='salt.log',debug=False):
    """predict E and O target positions, for arclines, update based on arc for TGTmap  

    Parameters 
    ----------
    infile: string filename of arc
    MOSTab: MOS table from xml with slit positions

    """
    """
    _f file index in list
    _d dimension index y,x = 0,1
    _y, _x unbinned pixel coordinate relative to optic axis
    _r, _c binned pixel cordinate
    _i mask slits (entries in xml)
    _t culled MOS target index for extract

    YX_d image coords (mm) at SALT focal plane
    yx_d: image coords (mm) at detector (relative to imaging optic axis)
    yx0_d: position (mm) of center of CCD image (relative to imaging optic axis)  
    yx_dp image coords (relative to imaging optic axis), separated by beam, polarimetric mode
    yx0_dp position (mm) of O,E optic axes at this wavelength (relative to imaging optic axis)
    yxp_dp image coords of O,E images (relative to O,E optic axes at this wavelength)
    yxp0_dp: position (mm) of center of split O,E images (relative to O,E optic axes at this wavelength) 
    """

  # get data

    hdul = pyfits.open(infile)
    hdr = hdul[0].header
    rows, cols = hdul[1].data.shape
    cbin, rbin = [int(x) for x in hdr['CCDSUM'].split(" ")]
    rcbin_d = np.array([rbin,cbin])
    rccenter_d, cgapedge_c = ccdcenter(hdul[1].data)
    pixmm = 0.015
    pm1_p = np.array([1.,-1.])

    filter = hdr['FILTER']
    lampid = hdr['LAMPID']
    camtem = hdr['CAMTEM']
    coltem = hdr['COLTEM']
    dateobs =  hdr['DATE-OBS'].replace('-','')
    trkrho = hdr['TRKRHO']
    RAd = Longitude(hdr['RA']+' hours').degree
    DECd = Latitude(hdr['DEC']+' degrees').degree
    PAd = hdr['TELPA']  
    name = objectname+"_"+filter+"_maparc_"
    prows = rows/2
    pm1_p = np.array([1.,-1.])
    ur0,uc0,saltfps = rssdtralign(dateobs,trkrho)           # ur, uc =unbinned pixels, saltfps =micr/arcsec    
    yx0_d = -0.015*np.array([ur0,uc0])                      # optical axis in mm
    
  # choose an arc line wavelength to use for cal
    arcwav = 5461.
    wav_w = np.array([float(filter[3:])-50.,arcwav,10500.]) # ends of spectrum
    dtrfps = saltfps*np.diff(RSScolpolcam(np.array([[0.,0.],[0.,1.]]),arcwav,coltem,camtem)[1,0])[0]    
    
  # establish nominal image fov geometry
    yx0_dp, rshift, yxp0_dp, isfov_rc = RSSpolgeom(hdul,arcwav)

    if debug:
        open(name+'geom.txt',"w")
        geomfile = open(name+'geom.txt',"a")        
        print >>geomfile, 'RSSpolgeom:'
        print >>geomfile, ('yx0_d  : %8.4f %8.4f'% tuple(yx0_d))
        print >>geomfile, ('yx0_dp : '+4*'%8.4f ' % tuple(yx0_dp.flatten()))
        print >>geomfile, 'rshift ',rshift
        print >>geomfile, ('yxp0_dp: '+4*'%8.4f ' % tuple(yxp0_dp.flatten()))     

  # apply distortion and splitting for this wavelength to get predicted O,E target map for xml entries
    YX_di = np.array([MOSTab['YCE'],MOSTab['XCE']])
    entries = YX_di.shape[1]
    yx_dpwi = RSScolpolcam(np.tile(YX_di,3),np.repeat(wav_w,entries),coltem,camtem).reshape((2,2,3,entries))
    rc_dpwi = (yx_dpwi - yx0_d[:,None,None,None])/(pixmm*rcbin_d[:,None,None,None]) + rccenter_d[:,None,None,None]

  # input all data, correct amplifier gain
    rows = 2*prows                          # allow for input with odd rows 
    image_prc = np.zeros((2,prows,cols))
    var_prc = np.zeros_like(image_prc)
    okbin_prc = np.zeros((2,prows,cols),dtype=bool)
    gaincor_c = gaincor(hdul)        
    image_rc = hdul['SCI'].data/gaincor_c[None,:]    
    image_prc = shift(image_rc[:rows],(-rshift,0),order=0).reshape((2,prows,cols))
    var_rc = hdul['VAR'].data/gaincor_c[None,:]**2
    var_prc = shift(var_rc[:rows],(-rshift,0),order=0).reshape((2,prows,cols))
    okbin_rc = (hdul['BPM'].data==0)
    okbin_prc = shift(okbin_rc[:rows],(-rshift,0),order=0).reshape((2,prows,cols))

  # cull targets outside of good FOV
    isref_i =  (MOSTab['TYPE'] == 'refstar')
    oktarg_i = ((MOSTab['TYPE'] == 'target') & (np.sqrt((YX_di**2).sum(axis=0)) < FOVlim))
    i_t = np.where(oktarg_i)[0]    
    badtarg_i = (~oktarg_i & (MOSTab['TYPE'] == 'target'))
    if badtarg_i.sum():
        rsslog.message(('Cull mask slits for FOV: '+badtarg_i.sum()*'%3i ') % tuple(np.where(badtarg_i)[0]),logfile)
    isspec_i = (isref_i | oktarg_i)
    i_s = np.where(isspec_i)[0]
    specs = isspec_i.sum()
    isref_s = isref_i[isspec_i]
    istarg_s = oktarg_i[isspec_i]
    targets = istarg_s.sum()
    s_t = np.where(istarg_s)[0]

  # now deal with overlaps, targets and refs
  # first extend reference spectra to allow for round reference holes
    drref_pi = 0.5*pm1_p[:,None]*(MOSTab['WIDTH'][isref_i][None,:] -   \
        MOSTab['WIDTH'][oktarg_i].mean())*dtrfps/(1000.*pixmm*rcbin_d[0])        
    rc_dpwi[0,:,0][:,isref_i] = rc_dpwi[0,:,0][:,isref_i] - drref_pi
    rc_dpwi[0,:,2][:,isref_i] = rc_dpwi[0,:,2][:,isref_i] + drref_pi
    
  # find nominal spec box, indicated by value spec+1 (0 = no target, or overlap)
    specmap_prc = np.zeros((2,prows,cols),dtype='uint8')
    isoverlap_prc = np.zeros((2,prows,cols),dtype=bool)
    grid_r = np.arange(prows)
    grid_c = np.arange(cols)
    rc_dpws = rc_dpwi[:,:,:,isspec_i] 
    rcp_dpws = np.copy(rc_dpws)       
    cbox2_s = 0.5*(MOSTab['LENGTH']*dtrfps/(1000.*pixmm*rcbin_d[1,None]))[isspec_i]     # half width     
    dcmax = 8.                                                                          # allowance for flexure
    
    for p,s in np.ndindex(2,specs): 
        rcp_dpws[0,p,:,s] = np.round(rc_dpws[0,p,:,s] - rshift-p*prows).astype(int)                   
        isspec_rc = (((np.sign(rcp_dpws[0,p,0,s]-grid_r) * np.sign(rcp_dpws[0,p,2,s]-grid_r) <= 0))[:,None]  \
            & ((rcp_dpws[1,p,1,s]-cbox2_s[s] < grid_c) & (rcp_dpws[1,p,1,s]+cbox2_s[s] > grid_c))[None,:])
        isoverlap_prc[p] |= (isspec_rc & (specmap_prc[p] > 0))
        specmap_prc[p][isspec_rc] = s+1                
    specmap_prc[isoverlap_prc] = 0
    isspec_psrc = np.zeros((2,specs,prows,cols),dtype=bool)    
    for p,s in np.ndindex(2,specs):      
        isspec_psrc[p,s] = (specmap_prc[p] == s+1)         
 
  # evaluate column-direction flexure for targ and ref specs by comparing edge of arc to slit prediction
    dc_ps = np.zeros((2,specs))
    onslit_psc = np.zeros((2,specs,cols),dtype=bool)

  # find the slit from flux profile
    relmin = -.03
    Cols = int((cbox2_s.max() + dcmax)*2)+1
    relerr_psC = np.zeros((2,specs,Cols))
    fprof_psC = np.zeros_like(relerr_psC)
        
    for p,s in np.ndindex(2,specs):
        c_C = np.arange(np.clip(rcp_dpws[1,p,1,s]-cbox2_s[s]-dcmax,0,cols-1),   \
                        np.clip(rcp_dpws[1,p,1,s]+cbox2_s[s]+dcmax,0,cols-1)).astype(int)
        cmean = int(c_C.mean())
        Cmean = cmean - c_C[0]
        specCols_r = isspec_psrc[p,s].sum(axis=1)
        ColsArray = np.unique(specCols_r[specCols_r>0])
        ColscountArray = np.array([(specCols_r==ColsArray[i]).sum() for i in range(len(ColsArray))])
        bestCols = ColsArray[np.argmax(ColscountArray)]
        r_R = np.where(specCols_r == bestCols)[0]
        f_C = image_prc[p,r_R][:,c_C].mean(axis=0)
        
      # do initial slit find by comparing with median (targets), max(refs)
        if istarg_s[s]:
            onslit_C = (np.abs(f_C - np.median(f_C))/np.median(f_C) < 0.1)
        else:
            onslit_C = (np.abs(f_C - f_C.max())/f_C.max() < 0.1)
            onslit_psc[p,s][c_C[onslit_C]] = True
            continue                                         # rough ref position not used for flexure calc                      
                                       
      # get rid of nearby neighbors by finding gaps between slits
        isnbr_C = np.zeros_like(onslit_C)
        CslitArray = np.where(onslit_C)[0]                        
        if ((CslitArray[-1]-CslitArray[0]+1) > len(CslitArray)):        
            CgapArray = np.where(~onslit_C[CslitArray[0]:CslitArray[-1]+1])[0] + CslitArray[0]
            CgapleftArray = CgapArray[CgapArray < Cmean]
            CgaprightArray = CgapArray[CgapArray > Cmean]
            if len(CgapleftArray):
                isnbr_C[:(CgapleftArray[0]+1)] = True              
            if len(CgaprightArray):
                isnbr_C[(CgaprightArray[-1]-1):] = True                           
        onslit_C &= ~isnbr_C
        
      # now fit profile of slit candidate to polynominal and do tighter find
        fit_C = np.polyval(np.polyfit(np.where(onslit_C)[0],f_C[onslit_C],2*(~isref_s[s])),np.arange(f_C.shape[0]))        
        relerr_C = (f_C - fit_C)/np.median(f_C)
        onslit_C = ((relerr_C > 2.*relmin) & ~isnbr_C)
        onslit_C[1:-1] |= (onslit_C[:-2] & onslit_C[2:])     # eliminate one-column dropouts

      # again, get rid of nearby neighbors by finding gaps between slits
        CslitArray = np.where(onslit_C)[0]                        
        if ((CslitArray[-1]-CslitArray[0]+1) > len(CslitArray)):        
            CgapArray = np.where(~onslit_C[CslitArray[0]:CslitArray[-1]+1])[0] + CslitArray[0]
            CgapleftArray = CgapArray[CgapArray < Cmean]
            CgaprightArray = CgapArray[CgapArray > Cmean]
            if len(CgapleftArray):
                isnbr_C[:(CgapleftArray[0]+1)] = True              
            if len(CgaprightArray):
                isnbr_C[(CgaprightArray[-1]-1):] = True                           
        onslit_C &= ~isnbr_C
        
      # finally, edge is where in relative error from fit (quadratic for target) at edge changes less than relmin
        for iter in range(3):
            CslitArray = np.where(onslit_C)[0]          
            if (-np.diff(relerr_C)[CslitArray[0]] < relmin): onslit_C[CslitArray[0]] = False
            if (np.diff(relerr_C)[CslitArray[-1]-1] < relmin): onslit_C[CslitArray[-1]] = False
                        
        onslit_psc[p,s][c_C[onslit_C]] = True
        CList = range(len(c_C))
        relerr_psC[p,s][CList] = onslit_C*relerr_C[CList]       
        CslitArray = np.where(onslit_C)[0]                        
        dc_ps[p,s] = (CslitArray[-1]+CslitArray[0])/2. - Cmean       
        fprof_psC[p,s][CList] = f_C[CList]                                      # for debug

    dc_s = np.polyval(np.polyfit(YX_di[1,oktarg_i],dc_ps[:,istarg_s].mean(axis=0),1),   \
        YX_di[1,isspec_i])                                                      # allow for scale error in x 
    rcp_dpws[1] += dc_s[None,None,:]
    yx_dpws = yx_dpwi[:,:,:,isspec_i]    
    yx_dpws[1] += dc_s[None,None,:]*pixmm*rcbin_d[1]
    yxp0_dp[1] += dc_s.mean()*pixmm*rcbin_d[1]  
        
  # evaluate row-direction flexure tor targets by centroiding 5461 arc line
    drmax = 24/rcbin_d[0]                   # allow 3 arcsec flexure
    dr_ps = np.zeros((2,specs))    
    okarc_ps = np.ones((2,specs),dtype=bool)
    okarc_ps[:,isref_s] = False
          
  # first find NIR drop-off - easy to find
    drnir = 34/rcbin_d[0]                   # rows from 5461 to NIR dropoff (1/2-point)
    nirRows = drmax + drnir
    fnir_ptR = np.zeros((2,targets,nirRows))
    
    for t in range(targets):
        s = s_t[t]    
        for p in (0,1):
            r_R = np.round(np.arange(rcp_dpws[0,p,1,s],rcp_dpws[0,p,1,s] +  \
                [1,-1][p]*nirRows,[1,-1][p])).astype(int)
            if ((r_R[0] < 0) | (r_R[-1] > prows-1)):
                okarc_ps[p,s] = False
                continue
            fnir_ptR[p,t] = image_prc[p,r_R][:,onslit_psc[p,s]].sum(axis=1)
            fnirmax = fnir_ptR[p,t].max()
            Rimax = np.argmax(fnir_ptR[p,t])
            Rinirdrop = np.argmax(fnir_ptR[p,t,Rimax:] < fnirmax/2.)
            dr_ps[p,s] = [1,-1][p]*(Rimax + Rinirdrop - drnir)

    drnir_ps = np.copy(dr_ps)
            
  # now centroid 5461 line based on this first-guess
    drline = 8/rcbin_d[0]                   # narrow range to calculate line center 
    lineRows = 2*drline + 1
    fline_ptR = np.zeros((2,targets,lineRows))    
    for t in range(targets):
        s = s_t[t]        
        for p in (0,1):
            if (not okarc_ps[p,s]): continue        
            r_R = np.round(np.arange(rcp_dpws[0,p,1,s] + dr_ps[p,s] - drline,   \
                rcp_dpws[0,p,1,s] +dr_ps[p,s] + drline + 1)).astype(int)            
            if ((r_R[0] < 0) | (r_R[-1] > prows-1)):
                okarc_ps[p,s] = False
                continue
            fline_ptR[p,t] = image_prc[p,r_R][:,onslit_psc[p,s]].sum(axis=1)
            Rimax = np.argmax(fline_ptR[p,t])
            if ((Rimax==0) | (Rimax==lineRows-1)):
                okarc_ps[p,s] = False
                continue                                                
            Y1,Y2,Y3 = fline_ptR[p,t,(Rimax-1):(Rimax+2)]
            Rmax = Rimax + 0.5*(Y1-Y3)/(Y1+Y3-2.*Y2)
            dr_ps[p,s] += Rmax-drline                    

    print okarc_ps

    drfit_ps = np.zeros((2,specs))
    dr_p = np.zeros(2)
    for p in (0,1):
        drpoly = np.polyfit(YX_di[0,i_s[okarc_ps[p]]],dr_ps[p][okarc_ps[p]],1)    
        drfit_ps[p] = np.polyval(drpoly, YX_di[0,i_s])    
        dr_p[p] = np.polyval(drpoly,0.)
       
    for p in (0,1):
        rcp_dpws[0,p] = np.round(rcp_dpws[0,p].astype(float) + drfit_ps[p]).astype(int)       
        yx_dpws[0,p] += drfit_ps[p]*pixmm*rcbin_d[0]
        yxp0_dp[0,p] += dr_p[p]*pixmm*rcbin_d[0]   
     
    if debug:
        np.savetxt(name+"fnir_ptR.txt",np.vstack((np.tile(i_t,2),np.indices((2,targets)).reshape((2,-1)), \
                fnir_ptR.reshape((-1,nirRows)).T)).T,fmt=" %3i %3i %3i"+nirRows*"%10.1f ")     
        np.savetxt(name+"fline_ptR.txt",np.vstack((np.tile(i_t,2),np.indices((2,targets)).reshape((2,-1)), \
                fline_ptR.reshape((-1,lineRows)).T)).T,fmt=" %3i %3i %3i"+lineRows*"%10.1f ")                        
        np.savetxt(name+"drc_sp.txt",np.vstack((i_s,drnir_ps,dr_ps,dc_ps,drfit_ps,dc_s)).T,fmt=" %2i "+9*"%8.2f ")
        np.savetxt(name+"fprof_psC.txt",np.vstack((np.tile(i_s,2),np.indices((2,specs)).reshape((2,-1)),    \
                fprof_psC.reshape((-1,Cols)).T)).T,fmt=" %2i %2i %2i "+Cols*"%10.2f ")        
        np.savetxt(name+"relerr_psC.txt",np.vstack((np.tile(i_s,2),np.indices((2,specs)).reshape((2,-1)),    \
                relerr_psC.reshape((-1,Cols)).T)).T,fmt=" %2i %2i %2i "+Cols*"%8.4f ")

  # Target slit box indicated by MOS slit entry +1 (0 = no target, or overlap)
    targetmap_prc = np.zeros((2,prows,cols),dtype='uint8')
    isoverlap_prc = np.zeros((2,prows,cols),dtype=bool)
    i_s = 254*np.ones(specs,dtype=int)
    i_s[istarg_s] = np.where(oktarg_i)[0]
    grid_r = np.arange(prows)
    
    for p,s in np.ndindex(2,specs):                
        isspec_rc = onslit_psc[p,s,None,:] &   \
            ((np.sign(rcp_dpws[0,p,0,s]-grid_r) * np.sign(rcp_dpws[0,p,2,s]-grid_r) <= 0))[:,None]
        isoverlap_prc[p] |= (isspec_rc & (targetmap_prc[p] > 0))
        targetmap_prc[p][isspec_rc] = i_s[s]+1        
    targetmap_prc[isoverlap_prc] = 0
    targetmap_prc[targetmap_prc==255] = 0

    istarget_ptrc = np.zeros((2,targets,prows,cols),dtype=bool)    
    for p,t in np.ndindex(2,targets):      
        istarget_ptrc[p,t] = (targetmap_prc[p] == i_t[t]+1)        

    Rows = istarget_ptrc.sum(axis=2).max()              # Rows, Cols is smallest data box containing all targetmaps
    Cols = istarget_ptrc.sum(axis=3).max()
    dRows_pt = Rows - istarget_ptrc.sum(axis=2).max(axis=2)
                
  # rckey is the (lower left) origin of the target data RowsxCols box
    poskey_dp = np.array([['YO','YE'],['XO','XE']])
    rckey_dp = np.array([['R0O','R0E'],['C0O','C0E']])  
    mapTab = MOSTab.copy()
    mapTab.rename_column('TYPE','CULL')
    mapTab['CULL'][~isref_i] = ''
    mapTab['CULL'][badtarg_i] = 'fov'
    r0_tp = np.zeros((targets,2))
    dr0_tp = np.zeros((targets,2))    
    
    for p in (0,1):
        mapTab[rckey_dp[0,p]] = 0
        mapTab[rckey_dp[1,p]] = 0        
        r0_tp[:,p] = rcp_dpws[0,p][:,istarg_s].min(axis=0).astype(int)
        dr0_tp[:,p] = np.clip((dRows_pt[p]/2).astype(int),r0_tp[:,p]+Rows-prows,r0_tp[:,p])
        mapTab[rckey_dp[0,p]][oktarg_i] = (r0_tp - dr0_tp)[:,p]            # make sure data box is inside image         
        mapTab[rckey_dp[1,p]][oktarg_i] = np.argmax(onslit_psc[p][istarg_s],axis=1) -   \
            ((Cols-onslit_psc[p][istarg_s].sum(axis=1))/2).astype(int)     # center the slit in the data box

    mapTab['YO'],mapTab['YE'],mapTab['XO'],mapTab['XE'] =   \
        tuple(yx_dpwi[:,:,1].reshape((4,-1)))        

    if debug:
        print 'Rows,Cols: ',Rows,Cols  
        np.savetxt("rtgt_spw.txt",np.vstack((i_s.repeat(2),np.indices((specs,2)).reshape((2,-1)),   \
            rcp_dpws[0].transpose((1,2,0)).reshape((3,-1)))).T,fmt=" %3i %2i %2i "+3*"%4i ")                 
        np.savetxt("dRows_tp.txt",np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)), \
            dRows_pt.T.flatten())).T, fmt=" %2i %2i %2i %4i ")  
        np.savetxt("rdr_tp.txt",np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)), \
            r0_tp.flatten(),dr0_tp.flatten())).T, fmt=" %2i %2i %2i %4i %4i ")                      
            
  # for each input fits, add targetmap and MOSTab extensions, write to tm*.fits
  # tm files produced only for multi-file polarimetric observation
    YXAXISO = ("%7.4f %7.4f" % tuple(yxp0_dp[:,0]))
    YXAXISE = ("%7.4f %7.4f" % tuple(yxp0_dp[:,1]))
    RA0 = Angle(RAd, u.degree).to_string(unit=u.hour, sep=":")
    DEC0 = Angle(DECd, u.degree).to_string(unit=u.degree, sep=":")
    PA0 = PAd
    BOXRC = (("%3i %3i") % (Rows,Cols))
    CALIMG = infile.split('.')[0][-4:]
            
    hdrList = [YXAXISO,YXAXISE,RA0,DEC0,PA0]       
    hdul['SCI'].data = image_prc.astype('float32')
    hdul['VAR'].data = var_prc.astype('float32')
    hdul['BPM'].data = (~okbin_prc).astype('uint8')     
    hdr['YXAXISO'] = (YXAXISO,"O Optic Axis (mm)")
    hdr['YXAXISE'] = (YXAXISE,"E Optic Axis (mm)")
    hdr['BOXRC'] = (BOXRC,"target box (bins)")
    hdr['CALIMG'] = (CALIMG,"cal image no(s)")      
    hdr['RA0'] = (RA0,"RA center")
    hdr['DEC0'] = (DEC0,"DEC center")
    hdr['PA0'] = (PA0,"PA actual")
    hdr['REFWAV'] = (arcwav,"TGT ref wav (Ang)")      
    hdul.append(pyfits.ImageHDU(data=targetmap_prc,name='TMAP'))
    hdul.append(pyfits.ImageHDU(data=specmap_prc,name='SMAP'))    
    hdul.append(pyfits.table_to_hdu(mapTab))
    hdul[-1].header['EXTNAME'] = 'TGT'
    hdul.writeto("t"+infile,overwrite=True)
    rsslog.message(('Output file '+'t'+infile ),logfile)    
    
    return    

# ------------------------------------
def imspecmospolmap(infileList, objectname, calHdul, cutwavoverride=0., logfile='salt.log',debug=False):
  # _i mask slits (entries in xml)
  # _t targets for extract
  # _f infileList index

  # get configuration data from first image
    obsDictf = obslog(infileList)
    files = len(infileList)
    hdul0 = pyfits.open(infileList[0])
    dateobs =  hdul0[0].header['DATE-OBS'].replace('-','')
    filter = hdul0[0].header['FILTER']
    rows, cols = hdul0[1].data.shape
    cbin, rbin = [int(x) for x in hdul0[0].header['CCDSUM'].split(" ")]
    rcbin_d = np.array([rbin,cbin])
    pixmm = 0.015
    binmm_d = rcbin_d*pixmm
    prows = rows/2
    rows = 2*prows                           # allow for odd number of rows
    calhdr = calHdul[0].header     
    arcwav = float(calhdr['REFWAV'])
    
    tgt_prc = calHdul['TMAP'].data
    tgtTab = Table.read(calHdul['TGT'])
    entries = len(tgtTab['CATID'])
    oktgt_i = (tgtTab['CULL'] == '')
    i_t = np.where(oktgt_i)[0]   
    targets = oktgt_i.sum()
    istarget_ptrc = np.zeros((2,targets,prows,cols),dtype=bool)    
    for p,t in np.ndindex(2,targets):      
        istarget_ptrc[p,t] = (tgt_prc[p] == i_t[t]+1)        
           
  # input all data, split it into OE images
    dum,rshift,dum,isfov_rc = RSSpolgeom(hdul0,arcwav)   
    image_fprc = np.zeros((files,2,prows,cols))
    var_fprc = np.zeros_like(image_fprc)
    okbin_fprc = np.zeros((files,2,prows,cols),dtype=bool)            
    for f,file in enumerate(infileList):
        hdul = pyfits.open(file)
        gaincor_c = gaincor(hdul) 
        image_rc = hdul['SCI'].data[:rows]/gaincor_c[None,:]
        image_fprc[f] = shift(image_rc,(-rshift,0),order=0).reshape((2,prows,cols))
        var_rc = hdul['VAR'].data[:rows]/gaincor_c[None,:]**2
        var_fprc[f] = shift(var_rc,(-rshift,0),order=0).reshape((2,prows,cols))
        okbin_rc = (hdul['BPM'].data[:rows]==0)
        okbin_fprc[f] = shift(okbin_rc,(-rshift,0),order=0).reshape((2,prows,cols))  

  # split into target data boxes
    rckey_pd = np.array([['R0O','C0O'],['R0E','C0E']])
    Rows,Cols = np.array(calHdul[0].header['BOXRC'].split()).astype(int)
    ri_tpR = np.zeros((targets,2,Rows),dtype=int)
    ci_tpC = np.zeros((targets,2,Cols),dtype=int)
    for p in (0,1):
        ri_tpR[:,p] = np.clip(tgtTab[rckey_pd[p,0]][oktgt_i][:,None] + np.arange(Rows)[None,:], 0,prows-1)
        ci_tpC[:,p] = np.clip(tgtTab[rckey_pd[p,1]][oktgt_i][:,None] + np.arange(Cols)[None,:], 0,cols-1)    
    image_ftpRC = np.zeros((files,targets,2,Rows,Cols))
    var_ftpRC = np.zeros_like(image_ftpRC)  
    okbin_ftpRC = np.zeros((files,targets,2,Rows,Cols),dtype=bool)
    oktgt_ftpRC = np.zeros_like(okbin_ftpRC) 
    for f,t,p in np.ndindex(files,targets,2): 
        image_ftpRC[f,t,p] = image_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
        var_ftpRC[f,t,p] = var_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]] 
        okbin_ftpRC[f,t,p] = okbin_fprc[f,p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
        oktgt_ftpRC[f,t,p] = (tgt_prc==i_t[t]+1)[p][ri_tpR[t,p],:][:,ci_tpC[t,p]] 
        
  # moffat fit using min as bkg for oktgt's in TGT Tab from summed column profile over all files
    oktgt_tpC = oktgt_ftpRC[0,:,:].any(axis=2)
    image_tpC = (image_ftpRC*oktgt_ftpRC).sum(axis=(0,3))/oktgt_ftpRC.sum(axis=(0,3))
    bkg_tp = np.zeros((targets,2))
    for t,p in np.ndindex(targets,2):     
        bkg_tp[t,p] = image_tpC[t,p,oktgt_tpC[t,p]].min()
    sigma_s, fmax_s, C0_s, fiterr_sb, okprof_s =    \
        moffat1dfit((image_tpC-bkg_tp[:,:,None]).reshape((-1,Cols)),oktgt_tpC,beta=2.5)     # _s = _tp

  # cull targets with outlier sigmas
    sigmaLower,sigmalower,sigmaupper,sigmaUpper = fence(sigma_s)  
    badsigma_s = ((sigma_s < sigmaLower) | (sigma_s > sigmaUpper))
    badsigma_t = badsigma_s.reshape((targets,2)).any(axis=1)
    if badsigma_t.sum():
        ibadList = list(i_t[badsigma_t])
        rsslog.message(('Cull mask slits for sigma: '+badsigma_t.sum()*'%3i ') % tuple(ibadList),logfile)          
        okprof_s &= np.repeat(~badsigma_t,2)
        tgtTab['CULL'][ibadList] = 'sigma'
        for t in np.where(badsigma_t)[0]:
            tgt_prc[istarget_ptrc[:,t]] = 0
                
  # flag bkg cols in TGT map    
    x_sC = np.arange(Cols)[None,:] - C0_s[:,None]    
    proffit_sC = moffat(np.ones(2*targets),sigma_s,x_sC,beta=2.5)
    fbkg = (1.+(0.375*oktgt_tpC.sum(axis=2).min()/sigma_s[okprof_s].max())**2)**(-2.5)  # > 25% columns are bkg
    isbkg_tpC = ((proffit_sC < fbkg).reshape((targets,2,-1)) & oktgt_ftpRC[0,:,:].any(axis=2))
    for t,p in np.ndindex(targets,2):
        if badsigma_t[t]: continue
        cbkgArray = np.where(isbkg_tpC[t,p])[0]+ci_tpC[t,p,0]
        rtgtArray = np.where((tgt_prc[p]==i_t[t]+1).any(axis=1))[0]
        for r,c in np.array(np.meshgrid(rtgtArray,cbkgArray)).T.reshape((-1,2)):
            tgt_prc[p,r,c] = 255

    if debug:            
        tgtTab.write(objectname+"_"+filter+"_mapTab.txt",format='ascii.fixed_width',   \
            bookend=False, delimiter=None, overwrite=True)   
        np.savetxt('image_tpC.txt',np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)),    \
                image_tpC.reshape((-1,Cols)).T)).T,fmt=" %2i %2i %2i "+Cols*"%9.2f ")
        np.savetxt('moffatfit_tp.txt',np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)),    \
                sigma_s, fmax_s, C0_s, okprof_s.astype(int))).T,fmt=" %2i %2i %2i %8.3f %9.2f %8.3f %2i")
        np.savetxt('isbkg_tpC.txt',np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)),    \
                isbkg_tpC.astype(int).reshape((-1,Cols)).T)).T,fmt=" %2i %2i %2i  "+Cols*"%2i ")                 
                
  # output with calhdr data        
    for f,file in enumerate(infileList):
        hdul = pyfits.open(file)    
        hdr = hdul[0].header        
        hdul['SCI'].data = image_fprc[f].astype('float32')
        hdul['VAR'].data = var_fprc[f].astype('float32') 
        hdul['BPM'].data = (~okbin_fprc[f]).astype('uint8')
        
        hdr['YXAXISO'] = calhdr['YXAXISO']
        hdr['YXAXISE'] = calhdr['YXAXISE']
        hdr['BOXRC'] = calhdr['BOXRC']
        hdr['CALIMG'] = calhdr['CALIMG']     
        hdr['RA0'] = calhdr['RA0']
        hdr['DEC0'] = calhdr['DEC0']
        hdr['PA0'] = calhdr['PA0']
        hdr['REFWAV'] = calhdr['REFWAV']    
        hdul.append(pyfits.ImageHDU(data=tgt_prc,name='TMAP'))
        hdul.append(pyfits.table_to_hdu(tgtTab))
        hdul[-1].header['EXTNAME'] = 'TGT'    
        hdr['object'] = objectname      # overrides with lamp name for lamp observations     
        hdul.writeto("t"+file,overwrite=True)
        rsslog.message(('Output file '+'t'+file ),logfile)

    return
   
