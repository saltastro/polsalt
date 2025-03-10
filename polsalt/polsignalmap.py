
"""
polsignalmap

Collect target file data for polarimetric extraction, apply distortion corrections,
    locate target spectra. 
    

"""

import os, sys, glob, copy, shutil, inspect

import numpy as np
import pywcs
from astropy.io import fits as pyfits

from scipy.interpolate import interp1d
from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

np.set_printoptions(threshold=np.nan)

# -------------------------------------
def polsignalmap(infilename,targetfilename='',logfile='salt.log',debug=False):
    """Collect target file data for polarimetric extraction, apply distortions

    Parameters
    ----------
    infilename: str
        Fits filename example of desired configuration
    targetfilename: str
        One of catalog, slitmask, ytarget, or wavtarget file
    logfile: str
        Name of file for logging

    """
    """
    _i: target index
    _j: target offset index

    """

    with logging(logfile, debug) as log:

        hdu = pyfits.open(infilename)
        targettype = ''
        if hdu[0].header['BS-STATE'].count("Inserted")==0:
            log.message('Not a polarimetric image', with_header=False)
            exit()
        if len(targetfilename)==0:
#            targetfile = 
            targettype = 'slitmask'
        else:
            targetfile = open(targetfilename)
            targettype = targetfile.readline().split()[1]

        if hdu[0].header['GRATING'] == 'N/A':       # imaging
            if (targettype!='catalog') & (targettype!='slitmask'):
                log.message('Incorrect target file type for imspecpol', with_header=False)
                exit()
            polsignal_ij = imspecpolmap(hdu,targetfile,targettype)
            np.savetxt('targ_i.txt',targ_i,fmt="%10s %8.2f %8.2f %6.1f")
        elif targettype != 'wavtarget':             # grating point source specpol          
            if (targettype!='ytarget') & (targettype!='slitmask'):
                log.message('Incorrect target file type for specpol', with_header=False)
                exit()
            polsignal_ij = specpolmap(hdu,targetfile,targettype)
        else:
            if (targettype!='wavtarget'):           # grating diffuse longslit                
                log.message('Incorrect target file type for linepol', with_header=False)
                exit()                           
            polsignal_ij = linepolmap(hdu,targetfile,targettype)        

    return 
#------------------------------------------------
def imspecpolmap(hdu,targetfile,targettype):

  # get catalog, convert to undistorted (TAN projection) unbinned pixel positions at detector
    if targettype == 'catalog':
        cat_i = np.loadtxt(targetfile,dtype=[('obj','a10'),('ra','f4'),('dec','f4'),('mag','f4')])
        targets = cat_i.shape[0]
        rah,ram,ras = np.array(hdu[0].header['RA'].split(':')).astype(float)
        ra = (360./24.)*(ras+60*(ram+60*rah))/3600.
        decd,decm,decs = np.array(hdu[0].header['DEC'].split(':')).astype(float)
        dec = (np.sign(decd)*decs+60*(np.sign(decd)*decm+60*decd))/3600.
        pa = float(hdu[0].header['TELPA'])

        cols = int(hdu['SCI'].header['NAXIS1'])
        rows = int(hdu['SCI'].header['NAXIS2'])
        cbin, rbin = [int(x) for x in hdu[0].header['CCDSUM'].split(" ")]
        pixscale = 0.128                                   # undistorted arcsec/unbinned pixel
        wcs = pywcs.WCS(naxis=2)
        wcs.wcs.crpix = [0.5*cols/cbin,0.5*rows/rbin]            # unbinned col,row at center
        wcs.wcs.cdelt = np.array([-pixscale, pixscale]) / 3600.  # degrees/pixel
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crota = [pa,pa] 
        wcs.wcs.equinox = 2000.

        targin_di = np.array((2,targets))       
        targin_di = wcs.wcs_sky2pix(cat_i['ra'],cat_i['dec'],0)

      # apply collimator distortion 
        targcoll_di = rsscolldist(targin_di,wcs)

      # map spectrum over filter range, split into O,E, and apply polarimetric distortion
        impolmap_odij = rsspoldist(targcoll_di,wcs,hdu)

      # apply camera focal length wavelength correction 
        catimpol_oij = np.empty((2,targets,wavs),  \
            dtype=[('obj','a10'),('wav','f4'),('col','f4'),('row','f4')])
        catimpol_oij['obj'] = cat_i['obj'][None,:,None]
        catimpol_oij['wav'] = wav_j[None,None,:]
        catimpol_oij['row'],catimpol_oij['col'] = rsspoldist(impol_dij,wcs,hdu)

    else:
  # MOS imaging polarimetry

    return catimpol_oij
#------------------------------------------------
def specpolmap(header,targetfile,targettype):
    return inputtarget_ij
#------------------------------------------------
def linepolmap(header,targetfile,targettype):
    return inputtarget_ij
#------------------------------------------------
def rsscolldist(target_di,wcs):
  # correct for collimator distortion
    imgdist=np.loadtxt(datadir+"imgdist.txt",usecols=(1,2))
    rcd_d = colldist[1,::-1]    # distortion axis relative to CCD center, unbin pix
    rd, A,B = colldist[2,0],colldist[3,0],colldist[3,1]
    rccenter_d = wcs.wcs.crpix  # CCD center, unbin pix
    rc_di = target_di - (rccenter_d + rcd_d)[:,None]    # nominal target r,c relative to distortion axis)        
    r_i = np.sqrt(np.sum(rc_di**2,axis=0))
    ba_di = (rccenter_d + rcd_d)[:,None] + rc_di*(1 + A*(r_i/rd)**2 + B*(r_i/rd)**4) 

    return ba_di

#------------------------------------------------
def rsspoldist(targcoll_di,hdu):
#   _f  filters in filter list
#   _j  wavelengths in map
#   _d  row,col
#   _i  targets
#   _o  O,E
#   _m  calibration file wavelength enumeration
#   _k  distortion calibration coefficinet
  # map spectrum over filter range

    cbin, rbin = [int(x) for x in hdu[0].header['CCDSUM'].split(" ")]
    filters_f, wf_f=np.loadtxt(datadir+"filters.txt",dtype=str,usecols=(0,1),unpack=True)
    filter = hdu[0].header["FILTER"].lstrip()
    filno = np.where(filters_f==filter)[0][0]
    wav_j = np.arange(3100.,10100.,100.*rbin)               # default whole spectral range
    if filno.size > 0: 
        if filter[0:2] == 'PI':
            wav_j = np.array([wf_f[filno],])
        elif filter[0:2] = 'PC':
            wav_j = np.arange(max(3100.,100*(int(filter[3:])/100)),10100.,100.*rbin)
    wavs = wav_j.shape[0]
    targs = targcoll_di.shape[1]

    impolmap_dij = np.vstack((np.repeat(targcoll_di[:,:,None],wavs,axis=2)),   \
                              np.repeat(wav_j[None,None,:],targs,axis=1))

  # split into O,E
    lam_m = np.loadtxt(datadir+"wollaston.txt",dtype=float,usecols=(0,))
    rpix_om = np.loadtxt(datadir+"wollaston.txt",dtype=float,unpack=True,usecols=(1,2))
    rsplit_oj = interp1d(lam_m,rpix_om,kind='cubic',bounds_error=False)(wav_j)
    impolmap_odij = impolmap_dij + rsplit_oj[:,None,None,:]

  # apply polarimetric distortion
    lam_m = np.loadtxt(datadir+"poldist.txt",dtype=float,usecols=[1:])
    distcof_km = np.loadtxt(datadir+"poldist.txt",dtype=float,unpack=True,usecols=[1:])
    coffit_kc = np.polyfit(lam_m,distcof_km,2)
    distcof_okj = np.array([np.polyval(coffit_kc[k],wav_j) for k in range(16)]).reshape((2,8,-1))

    drc_di = targcoll_di - np.array([distcof
    impolmap_odij

    return impolmap_odij

#------------------------------------------------
def rsscamdist(target_odij,wcs,hdu):
  # correct for temperature and wavelength dependence of camera focal length
    imgdist=np.loadtxt(datadir+"imgdist.txt",usecols=(1,2))
    rc0_d = imgdist[0,::-1]                 # optic axis relative to CCD center, unbin pix
    Fsclpoly=imgdist[4: ]                   # Fp scale, micr/arcsec vs (wav-5000)/1000, T-7.5C
    coltemp = float(hdu[0].header["COLTEM*"][0].value)
    camtemp = float(hdu[0].header["CAMTEM*"][0].value)
    spectemp = 0.5*(coltemp+camtemp)
    ww_j = (target_dij[0,0] - 5000.)/1000.        
    dfscl_j = (np.polyval(Fsclpoly[:,0] + (spectemp - 7.5)*Fsclpoly[:,1],ww_j))/Fsclpoly[-1,0]
    rccenter_d = wcs.wcs.crpix              # CCD center, unbin pix
    rcaxis_d = rccenter_d + rc0_d
    ba_odij = rcaxis_d[None,:,None,None] + dfscl_j*(target_odij - rcaxis_d[None,:,None,None])  

    return ba_odij
#------------------------------------------------
def catcrossref(cat1,cat2):
    """Cross-reference two catalogs

    Parameters
    ----------
    cat1: str
        Text filename of reference catalog
    cat2: str
        Text filename of catalog to be matched
    Catalog: text file, fields  "name ra  dec  mag"
        name: str (a10)
        ra, dec: float deg
        mag: float
    Output:
    -------
        dra,ddc,drastd,ddcstd: offset between catalogues, rms error (deg)
        crosscat: catalog of common objects
        combcat: combined catalog, add unmatched cat2 entries shifted to reference cat coordinates
        output catalogues: structured array, "name1 ra  dec  mag name2"

    """
    """
    _i: cat1 index
    _j: cat2 index
    _k: crosscat index
    _l: comb index

    """
  # input catalog data
    cat1_i = np.loadtxt(cat1,dtype=[('obj','a10'),('ra','f4'),('dec','f4'),('mag','f4')])
    cat2_j = np.loadtxt(cat2,dtype=[('obj','a10'),('ra','f4'),('dec','f4'),('mag','f4')])
    dra_ij = cat2_j['ra'][None,:] - cat1_i['ra'][:,None] 
    ddc_ij = cat2_j['dec'][None,:]  - cat1_i['dec'][:,None]
    entries1,entries2 = dra_ij.shape
    ok_j = np.ones(entries2,dtype=bool)
    i_j = np.zeros(entries2) 
    dra = 0.
    ddc = 0.

  # iterate while centering catalogs and culling non-matches
    for iter in range(10):
        istart_j = i_j
        err_ij = np.sqrt((dra_ij-dra)**2 + (ddc_ij-ddc)**2)
        i_j = np.argmin(err_ij,axis=0)
        dra_j = dra_ij[i_j,range(entries2)]
        ddc_j = ddc_ij[i_j,range(entries2)]
      # cull non-matches based on "inner fence" quartile criterion
        draq1,draq3 = np.percentile(dra_j[ok_j],[25.,75.])
        ddcq1,ddcq3 = np.percentile(ddc_j[ok_j],[25.,75.])
        ok_j &= ((dra_j > draq1-1.5*(draq3-draq1)) & (dra_j < draq3+1.5*(draq3-draq1)))
        ok_j &= ((ddc_j > ddcq1-1.5*(ddcq3-ddcq1)) & (ddc_j < ddcq3+1.5*(ddcq3-ddcq1)))
        dra = np.median(dra_j[ok_j])
        ddc = np.median(ddc_j[ok_j])
        drastd = np.std(dra_j[ok_j])
        ddcstd = np.std(ddc_j[ok_j])
      # we are done when matching does not change
        changes = (i_j[ok_j] != istart_j[ok_j]).sum()
        if (changes==0): break

    crosscat_k = np.empty(ok_j.sum(),   \
        dtype=[('obj1','a10'),('ra','f4'),('dec','f4'),('mag','f4'),('obj2','a10')])
    crosscat_k['obj1'] = cat1_i['obj'][i_j[ok_j]]
    crosscat_k['obj2'] = cat2_j['obj'][np.where(ok_j)]
    for entry in ('ra','dec','mag'):
        crosscat_k[entry] = cat1_i[entry][i_j[ok_j]]

    combcat_l = np.empty(entries1+(~ok_j).sum(),   \
        dtype=[('obj1','a10'),('ra','f4'),('dec','f4'),('mag','f4'),('obj2','a10')])
    combcat_l['obj2'] = '--'
    combcat_l['obj2'][i_j[ok_j]] = cat2_j['obj'][ok_j]
    combcat_l['obj2'][entries1:] = cat2_j['obj'][~ok_j]
    combcat_l['obj1'] = combcat_l['obj2']
    combcat_l['obj1'][:entries1] = cat1_i['obj']
    cat2_j['ra'] -= dra
    cat2_j['dec'] -= ddc
    for entry in ('ra','dec','mag'):
        combcat_l[entry][:entries1] = cat1_i[entry]
        combcat_l[entry][entries1:] = cat2_j[entry][~ok_j]

    return crosscat_k,combcat_l,dra,ddc,drastd,ddcstd
   
#------------------------------------------------
if __name__=='__main__':
    infilename = sys.argv[1] 
    targetfilename = sys.argv[2]     
    polsignalfile(infilename,targetfilename)
