#! /usr/bin/env python

# Resample data into new bins, preserving flux
# New version 150912, much faster

import os, sys, time, glob, shutil
import numpy as np

def scrunch1d(input,binedge):
# new binedges are in coordinate system x where the left edge of the 0th input bin is at 0.0
    na = input.size
    nx = binedge.size - 1
    input_a = np.append(input,0)                         # deal with edge of array
    binedge_x = binedge.clip(0,na)                       # deal with edge of array
    output_x = np.zeros(nx)

# _s is list of subbins divided by both new and old bin edges
    x_s = np.append(binedge_x,range(int(binedge_x.min())+1,int(binedge_x.max()+1)))
    x_s,argsort_x = np.unique(x_s,return_index=True)
    ia_s = x_s.astype(int)
    ix_s = np.append(np.arange(nx+1),-1*np.ones(na+1))[argsort_x].astype(int)
    while (ix_s==-1).sum():
        ix_s[ix_s==-1] = ix_s[np.where(ix_s==-1)[0] - 1]

# divide data into subbins, preserving flux
    input_s = input_a[ia_s[:-1]]*(x_s[1:] - x_s[:-1])
    ix_x, s_x = np.unique(ix_s,return_index=True)
    ns_x = s_x[1:] - s_x[:-1]

# sum it into the new bins
    for s in range(ns_x.max()):
        output_x[ns_x > s] += input_s[s_x[ns_x > s]+s]

    return output_x

if __name__=='__main__':
    input=np.loadtxt(sys.argv[1])
    binedge=np.loadtxt(sys.argv[2])
#    for n in range(1000): scrunch1d(input,binedge)
    np.savetxt('outputfile.txt',scrunch1d(input,binedge),fmt="%14.8f")
