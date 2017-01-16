#! /usr/bin/env python

# Resample data into new bins, preserving flux
# New version 150912, much faster
# New version 160322, fixed case where output bin coverage is larger than input bin coverage

import os, sys, time, glob, shutil
import numpy as np

def scrunch1d(input,binedge):
# new binedges are in coordinate system x where the left edge of the 0th input bin is at 0.0
    na = input.size
    nx = binedge.size - 1
    input_a = np.append(input,0)                         # deal with edge of array
    okxbin = ((binedge>=0) & (binedge<=na))                      
    output_x = np.zeros(nx)

# _s: subbins divided by both new and old bin edges
    ixmin = np.where(okxbin)[0][0]
    ixmax = np.where(okxbin)[0][-1]
    iamin = int(binedge[ixmin])
    iamax = int(binedge[ixmax])
    x_s = np.append(binedge[okxbin],range(int(np.ceil(binedge[ixmin])),iamax+1))
    x_s,argsort_s = np.unique(x_s,return_index=True)
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
        output_x[ns_x > s] += input_s[s_x[ns_x > s]+s]

    return output_x

if __name__=='__main__':
    input=np.loadtxt(sys.argv[1])
    binedge=np.loadtxt(sys.argv[2])
#    for n in range(1000): scrunch1d(input,binedge)
    np.savetxt('outputfile.txt',scrunch1d(input,binedge),fmt="%14.8f")
