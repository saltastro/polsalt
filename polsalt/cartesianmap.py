
"""
cartesianmap

Test distortion mapping with cartesian lamp image.  OBSOLETE (2019)

"""

import os, sys, glob, shutil, inspect

import numpy as np
from astropy.io import fits as pyfits
from scipy import linalg as la

from pyraf import iraf
from iraf import pysalt

from saltobslog import obslog
from saltsafelog import logging

import impolmap as imp
from poltargetmap import sextract, RSScoldistort, RSScamdistort
from specpolutils import datedline, rssdtralign, rssmodelwave, list_configurations
from scipy.interpolate import interp1d

datadir = os.path.dirname(__file__) + '/data/'
#np.set_printoptions(threshold=np.nan)
debug = False

def cartesianmap(infile, logfile='salt.log'):
    """derive extraction map at detector cartesian lamp image

    Parameters 
    ----------
    infile: cartesian lamp image

    """
    """
    _d dimension index y,x = 0,1
    _e entry index for cats and maps, =0,1,2,3 idx,y,x,mag
    _y, _x unbinned pixel coordinate
    _r, _c bin coordinate
    _t catalog index
    _s located star on image
    _S culled star target
    """

    with logging(logfile, debug) as log:
        log.message('Pysalt Version: '+pysalt.verno, with_header=False)
        hdulist = pyfits.open(infile)        
        cbin, rbin = [int(x) for x in hdulist[0].header["CCDSUM"].split(" ")]
        hdr = hdulist[0].header
        maskid =  hdr["MASKID"].strip() 
        if (maskid != "P000000N99") & (maskid != "P000000P04"):
            print "Not a cartesian mask image" 
            exit()
        filter =  hdr["FILTER"].strip()
        wavl = float(filter[3:])
        grating = hdr["GRATING"].strip()
        grang = hdr["GR-ANGLE"]
        artic = hdr["CAMANG"]
        trkrho = hdr["TRKRHO"]
        if "COLTEM" in hdr:                             # allow for annoying SALT fits version change
            coltem = float(hdr["COLTEM"]) 
            camtem = float(hdr["CAMTEM"])
        else:
            coltem = float(hdr["COLTEMP"]) 
            camtem = float(hdr["CAMTEMP"])    
        dateobs =  hdr['DATE-OBS'].replace('-','') 
        objectname = hdr['OBJECT']
        notpol = (hdr['BS-STATE'] !=2 )
        rows, cols = hdulist[1].data.shape

      # input catalog and apply collimator ("o") distortion 
        y0,x0,C0 = rssdtralign(dateobs,trkrho)
        yx0_d = np.array([y0,x0])                         
        maskcatfilename = maskid[-3:]+"_map.txt"
        idx_t,x_t,y_t = np.loadtxt(maskcatfilename,dtype=float,unpack=True,ndmin=2)
        yx_dt = np.array([y_t,x_t]) + yx0_d[:,None]
        targets = idx_t.shape[0]
        yxo_dt = RSScoldistort(yx_dt,wavl,yx0_d,coltem)
        maskmapo_et = np.array([idx_t,yxo_dt[0],yxo_dt[1]])
        np.savetxt("maskmapo_et.txt",maskmapo_et.T,fmt="%5i %10.5f %10.5f ")

        if notpol:
            yxoa_dt = RSScamdistort(yxo_dt,wavl,yx0_d,camtem)            
            maskmapoa_et = np.array([idx_t,yxoa_dt[0],yxoa_dt[1]])
            np.savetxt("maskmapoa_et.txt",maskmapoa_et.T,fmt="%5i %10.5f %10.5f ") 

          # run SeXtract, identify and find offset
            sexj_s = sextract(infile,sigma=10.,cull=True,debug=True)
            rc_ds = np.array([sexj_s["Y_IMAGE"],sexj_s["X_IMAGE"]])
            yx_ds = np.array([rbin,cbin])[:,None]*(rc_ds - np.array([rows,cols])[:,None]/2)
            okid_t,dyxoa_dt = idstars(yxoa_dt,yx_ds,drmax=40)           # 5 arcsec max error
            dyxoa_d,yxscloa_d,rotoa,yxfirst_dt = imgalign(okid_t,yxoa_dt,dyxoa_dt,yx0_d,debug=True)

            print 'First Identified: ',okid_t.sum()

          # iterate identification once
            okid_t,ddyxoa_dt = idstars(yxfirst_dt,yx_ds,drmax=16)       # 2 arcsec max error 
            dyxoa_d,yxscloa_d,rotoa,yxoa_dt = imgalign(okid_t,yxoa_dt,dyxoa_dt,yx0_d,debug=True)

            print 'Final Identified: ',okid_t.sum()

            okid_t,dddyxoa_dt = idstars(yxoa_dt,yx_ds,drmax=16)       
            np.savetxt(maskid[-3:]+"_oa.txt",np.vstack((np.arange(targets),okid_t,yxoa_dt,dddyxoa_dt)).T, \
                    fmt="%3i %2i "+4*"%10.4f ")
           
        else:            
          # imaging polarimetry
            if grating=='N/A':
          # process filtered images for polarimetric splitting, for object finding and wavcal                              
                filterlist_i = obsdict_i['FILTER']
                filterlist_f = sorted(list(set(filterlist_i)))
                filters = len(filterlist_f)
                PIfilterlist = [f for f in range(filters) if filterlist_f[f][:3]=='PI0']

                astromapow_fpet = np.zeros((filters,2,8,targets))
                for f in PIfilterlist:
                    objectlistf_i = [objectlist_i[i] for i in range(objectfiles)   \
                        if filterlist_i[i]==filterlist_f[f] ] 
                    np.savetxt("astromapow_fpet_"+filterlist_f[f]+".txt",astromapow_fpet[f].reshape((16,-1)).T,    \
                        fmt=2*"%5i %8.2f %8.2f %6.2f %4i %8.2f %8.2f %6.2f  ")

                exit()

            elif astrocat==object+'wavcat.txt':
                linepolmap(objectlist_i,astromapo)

            else:
               # catalog stuff
                specpolmap(objectlist_i,MOSmapo,astromapo)
                
    return

# ------------------------------------

def idstars(yx_dt,yx_ds,drmax=20):
    """
    simple star-target identifier for non-polarimetric, non-crowded fields
    _d dimension index y,x = 0,1
    _y, _x unbinned pixel coordinate relative to CCD center
    _t catalog index
    _s located star on image
    _T culled target
    """

    targets = yx_dt.shape[1]
    stars = yx_ds.shape[1]
    yxerr_dst = yx_ds[:,:,None] - yx_dt[:,None,:]
    poserr_st = np.sqrt((yxerr_dst**2).sum(axis=0))
    sid_t = poserr_st.argmin(axis=0)
    dyx_dt = yxerr_dst[:,sid_t,range(targets)]
    poserr_t = poserr_st[sid_t,range(targets)]
    okid_t = (poserr_t < drmax)

    np.savetxt("idstars.txt",np.vstack((sid_t,dyx_dt[0],dyx_dt[1],poserr_t)).T,fmt="%10.4f")

    return okid_t,dyx_dt

# ------------------------------------
def imgalign(ok_t,yx_dt,dyx_dt,yx0_d,debug=False):
    """
    Least squares fit of dyx, dyxscale, and dtheta, (5 params) of star offset from predicted target position
      for id'd targets
    Also return realigned positions targets for all targets (to allow for re-identification)
    """
    yxrot_dt = yx_dt - yx0_d[:,None]                            # rotation, scale center is the optical axis
    ids = ok_t.sum()
    A_SC = np.zeros((2*ids,5))
    A_SC[:,4] = yxrot_dt[::-1,ok_t].flatten().T                 # rotation: fit dy vs x and dx vs (-)y

    for d in range(2):
        A_SC[(ids*d):(ids*(d+1)),d] = np.ones(ids).T
        A_SC[(ids*d):(ids*(d+1)),d+2] = yxrot_dt[d,ok_t].T      # scale: fit dy vs y and dx vs x
        if d==1:
            A_SC[(ids*d):(ids*(d+1)),4] *= -1                   # rotation: dx vs -y

#    np.savetxt("A_SC.txt",A_SC,fmt="%8.2f")

    offcofs_C,sumsqerr = la.lstsq(A_SC,dyx_dt[:,ok_t].flatten())[:2]   # here is the fit

    eps_CC = la.inv((A_SC[:,:,None]*A_SC[:,None,:]).sum(axis=0))
    std_C = np.sqrt((sumsqerr/ids)*np.diagonal(eps_CC))
    yxoff_d = offcofs_C[:2]
    yxstd_d = std_C[:2]
    yxscloff_d = offcofs_C[2:4]
    yxsclstd_d = std_C[2:4]                                       
    rotoff = np.degrees(np.arcsin(offcofs_C[4]))
    rotstd = np.degrees(np.arcsin(std_C[4]/np.sqrt(2.)))        # 2x the samples for common slope

    if debug:
        print "\n    yoff   +/-     xoff   +/-    yscloff  +/-    xscloff  +/-     rotoff +/-"
        print "%8.2f %5.2f %8.2f %5.2f %8.4f %5.4f %8.4f %5.4f %8.3f %5.3f " %    \
            (yxoff_d[0],yxstd_d[0],yxoff_d[1],yxstd_d[1],   \
            yxscloff_d[0],yxsclstd_d[0],yxscloff_d[1],yxsclstd_d[1],rotoff,rotstd)

    yxnew_dt = yx_dt + yxoff_d[:,None] + yxscloff_d[:,None]*yxrot_dt +   \
                np.radians(rotoff)*np.array([1,-1])[:,None]*yxrot_dt[::-1]

    return yxoff_d,yxscloff_d,rotoff,yxnew_dt

# --------------------

if __name__=='__main__':
    infile=sys.argv[1]      
    cartesianmap(infile)

# debug:
# cd /d/pfis/khn/20150725/8005_bsout
# Pypolsalt cartesianmap.py m*.fits
