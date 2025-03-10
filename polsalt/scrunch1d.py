#! /usr/bin/env python

# Resample data into new bins, preserving flux
# New version 150912, much faster
# New version 170504, fixed case where output bin coverage is larger than input bin coverage
# New version 170909, again fixed case where output bin coverage is larger than input bin coverage
# ToDo: update nomenclature to be like scrunchvar1d below

import os, sys, time, glob, shutil
import numpy as np

def scrunch1d(input,binedge):
# new binedges are in coordinate system x where the left edge of the 0th input bin is at 0.0
    na = input.size
    nx = binedge.size - 1
    input_a = np.append(input,0)                         # deal with edge of array
#    okxbin = ((binedge>=0) & (binedge<=na))                      
    okxbin = ((binedge[1:]>0) & (binedge[:-1]<na))
    okxedge = np.zeros(binedge.size,dtype=bool)
    okxedge[:-1] |= okxbin
    okxedge[1:] |= okxbin                      
    output_x = np.zeros(nx)

# _s: subbins divided by both new and old bin edges
    ixmin,ixmax = np.where(okxedge)[0][[0,-1]]
    iamin = int(binedge[ixmin])
    iamax = int(binedge[ixmax])
    x_s = np.append(binedge[okxedge],range(int(np.ceil(binedge[ixmin])),iamax+1))
    x_s,argsort_s = np.unique(x_s,return_index=True)
    x_s = np.maximum(x_s,0.)                            # 20170909: deal with edge of array
    x_s = np.minimum(x_s,na)                            # 20170909: deal with edge of array
    ia_s = x_s.astype(int)
    ix_s = np.append(np.arange(ixmin,ixmax+1),-1*np.ones(iamax-iamin+1))[argsort_s].astype(int)
    while (ix_s==-1).sum():
        ix_s[ix_s==-1] = ix_s[np.where(ix_s==-1)[0] - 1]

#    np.savetxt("scrout_s.txt",np.vstack((ia_s,ix_s,x_s)).T,fmt="%5i %5i %10.4f")

# divide data into subbins, preserving flux
    ix_x = np.zeros(nx+1).astype(int)
    s_x = np.zeros(nx+1).astype(int)

    input_s = input_a[ia_s[:-1]]*(x_s[1:] - x_s[:-1])
    ix_x[ixmin:(ixmax+1)], s_x[ixmin:(ixmax+1)] = np.unique(ix_s,return_index=True)
    ns_x = s_x[1:] - s_x[:-1]

#    np.savetxt("scrout_x.txt",np.vstack((ix_x,np.append(ns_x,[0]),s_x)).T,fmt="%5i")

# sum it into the new bins
    for s in range(ns_x.max()):
        output_x[ns_x > s] += input_s[s_x[:nx][ns_x > s]+s]

    return output_x

#--------------------------------------------------------------------
def scrunchvar1d(var_a,xbinedge_B):
#   _a: input bins
#   _s: subbins divided by both new and old bin edges
#   _S: subbin edges
#   _b: output bins
#   _B: output bin edges

# new xbinedges are in coordinate system x where x=0. at he left edge of the 0th input bin 
    na = var_a.size
    nB = xbinedge_B.size
    nb = nB - 1
    vvar_a = np.append(var_a,0)                         # deal with edge of array                  
    okbin_b = ((xbinedge_B[1:]>0) & (xbinedge_B[:-1]<na))
    okedge_B = np.zeros(nB,dtype=bool)
    okedge_B[:-1] |= okbin_b
    okedge_B[1:] |= okbin_b                      
    var_b = np.zeros(nb)
    covar_b = np.zeros(nb)

    bmin,bmax = np.where(okedge_B)[0][[0,-1]]
    amin = int(xbinedge_B[bmin])
    amax = int(xbinedge_B[bmax])
    x_S = np.append(xbinedge_B[okedge_B],range(int(np.ceil(xbinedge_B[bmin])),amax+1))
    x_S,argsort_S = np.unique(x_S,return_index=True)
    x_S = np.clip(x_S,0.,na)                                                    
    a_S = x_S.astype(int)
    b_S = np.append(np.arange(bmin,bmax+1),-1*np.ones(amax-amin+1))[argsort_S].astype(int)
    while (b_S==-1).sum():
        b_S[b_S==-1] = b_S[np.where(b_S==-1)[0] - 1]

# divide data into subbins, preserving variance
    x_B = np.zeros(nB).astype(int)
    s_B = np.zeros(nB).astype(int)
    x_B[bmin:(bmax+1)], s_B[bmin:(bmax+1)] = np.unique(b_S,return_index=True)
    ns_b = s_B[1:] - s_B[:-1]                   # number of subbins in each new bin
    dxbin_s = x_S[1:] - x_S[:-1]
    var_s = vvar_a[a_S[:-1]]*dxbin_s**2
    covar_s = (a_S[:-1]==a_S[1:]).astype(int)*vvar_a[a_S[:-1]]*np.append(dxbin_s[1:]*dxbin_s[:-1],0.)

# sum subbins into the new bins
    for ds in range(ns_b.max()):
        var_b[ns_b > ds] += var_s[s_B[:nb][ns_b > ds]+ds]
        covar_b[ns_b > ds] += covar_s[s_B[:nb][ns_b > ds]+ds]

    return var_b, covar_b

# test: scrunch1d.py flx_a,var_a,xbinedge_B
if __name__=='__main__':
    flx_a=np.loadtxt(sys.argv[1])
    var_a=np.loadtxt(sys.argv[2])
    xbinedge_B=np.loadtxt(sys.argv[3])
    flx_b = scrunch1d(flx_a,xbinedge_B)
    var_b, covar_b = scrunchvar1d(var_a,xbinedge_B)
    np.savetxt('scrunchedfile.txt',np.vstack((flx_b,var_b,covar_b)).T,fmt="%14.4f")
