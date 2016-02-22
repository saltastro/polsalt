
"""
oksmooth

General purpose smoothing which properly uses pixel mask

"""

import os, sys, glob, shutil, inspect

import numpy as np
import pyfits
from scipy.interpolate import griddata

np.set_printoptions(threshold=np.nan)

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
    rblks,cblks = int(rows/rblk),int(cols/cblk)

# equalize block scaling to avoid triangularization failure    
    rfac,cfac = max(rblk,cblk)/rblk, max(rblk,cblk)/cblk     
    r0,c0 = (rows % rblk)/2,(cols % cblk)/2
    arr_RCb = arr_rc[r0:(r0+rblk*rblks),c0:(c0+cblk*cblks)]    \
        .reshape(rblks,rblk,cblks,cblk).transpose(0,2,1,3).reshape(rblks,cblks,rblk*cblk)
    ok_RCb = ok_rc[r0:(r0+rblk*rblks),c0:(c0+cblk*cblks)]    \
        .reshape(rblks,rblk,cblks,cblk).transpose(0,2,1,3).reshape(rblks,cblks,rblk*cblk)
    r_RCb = rfac*((ok_rc*r_rc)[r0:(r0+rblk*rblks),c0:(c0+cblk*cblks)])    \
        .reshape(rblks,rblk,cblks,cblk).transpose(0,2,1,3).reshape(rblks,cblks,rblk*cblk)
    c_RCb = cfac*((ok_rc*c_rc)[r0:(r0+rblk*rblks),c0:(c0+cblk*cblks)])    \
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
    dar_RC = arr_RC[1:,:] - arr_RC[:-1,:]
    dac_RC = arr_RC[:,1:] - arr_RC[:,:-1]
    dr_RC = r_RC[1:,:] - r_RC[:-1,:]
    dc_RC = c_RC[:,1:] - c_RC[:,:-1]

    dadr_RC = np.zeros_like(dar_RC);    dadc_RC = np.zeros_like(dac_RC)
    dadr_RC[dr_RC!=0] = rfac*dar_RC[dr_RC!=0]/dr_RC[dr_RC!=0]
    dadc_RC[dc_RC!=0] = cfac*dac_RC[dc_RC!=0]/dc_RC[dc_RC!=0]
    argR = np.where(ok_RC.sum(axis=1)>0)[0]
    argC = np.where(ok_RC.sum(axis=0)>0)[0]    
    dadr_RC[argR[0],argC]    *= (arr_RC[argR[0,],argC] > 0)
    dadr_RC[argR[-1]-1,argC] *= (arr_RC[argR[-1],argC] > 0)
    dadc_RC[argR,argC[0]]    *= (arr_RC[argR,argC[0]] > 0)
    dadc_RC[argR,argC[-1]-1] *= (arr_RC[argR,argC[-1]] > 0)    

    if debug:
        np.savetxt('dadr_RC.txt',dadr_RC,fmt="%14.9f")
        np.savetxt('dadc_RC.txt',dadc_RC,fmt="%14.9f")
        np.savetxt('r_RC_0.txt',r_RC,fmt="%9.2f")
        np.savetxt('c_RC_0.txt',c_RC,fmt="%9.2f")
        
# force outer block positions into a rectangle to avoid edge effects, spline interpolate

    r_RC[argR[[0,-1]][:,None],argC] = rfac*(r0+(rblk-1)/2.+rblk*argR[[0,-1]])[:,None]
    c_RC[argR[:,None],argC[[0,-1]]] = cfac*(c0+(cblk-1)/2.+cblk*argC[[0,-1]])
        
    arr_rc = griddata((r_RC[ok_RC],c_RC[ok_RC]),arr_RC[ok_RC],  \
        tuple(np.mgrid[:rfac*rows:rfac,:cfac*cols:cfac].astype(float)),method='cubic',fill_value=0.)

    if debug:
        pyfits.PrimaryHDU(arr_rc.astype('float32')).writeto('arr_rc_0.fits',clobber=True)

# extrapolate to original array size
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
    
    if debug:
        pyfits.PrimaryHDU(arr_rc.astype('float32')).writeto('arr_rc_1.fits',clobber=True)
    
    return arr_rc

