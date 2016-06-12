#! /usr/bin/env python

"""
specpolfilter

get filter values from stokes.fits data 

"""

import os, sys, glob, inspect
import numpy as np
import pyfits
from scipy.interpolate import interp1d

from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog

def specpolfilter(filter, infilelist):
    if filter in ("U","B","V"):
        filterfile = iraf.osfn("pysalt$data/scam/filters/Johnson_"+filter+".txt")
    elif filter in ("R","I"):
        filterfile = iraf.osfn("pysalt$data/scam/filters/Cousins_"+filter+".txt")         
#    else:
#        (filter file in cwd)
    wav_l,feff_l = np.loadtxt(filterfile,dtype=float,unpack=True)

    obss = len(infilelist)
    obsdict=obslog(infilelist)

    for b in range(obss):
        hdul = pyfits.open(infilelist[b])
        dwav = float(hdul['SCI'].header['CDELT1'])
        wav0 = float(hdul['SCI'].header['CRVAL1'])
        wavs = int(hdul['SCI'].header['NAXIS1'])
        ctypelist = (hdul['SCI'].header['CTYPE3']).split(',')
        stokes_sw = hdul['SCI'].data[:,0,:]
        var_sw = hdul['VAR'].data[:,0,:]
        pstokess = len(ctypelist)-1     
        ok_w = (hdul['BPM'].data[:,0,:] == 0).all(axis=0)
        wav_w = wav0 + dwav*np.arange(wavs)
        feff_w = interp1d(wav_l,feff_l,kind='linear',bounds_error=False)(wav_w)
        ok_w &= ~np.isnan(feff_w)
        feff_w[~ok_w] = 0.
        stokesfil_s = (feff_w*stokes_sw).sum(axis=1)/feff_w.sum()
        varfil_s = (feff_w**2*var_sw).sum(axis=1)/(feff_w.sum()**2)
        nstokesfil_s = 100.*stokesfil_s/stokesfil_s[0]
        nerrfil_s = 100.*np.sqrt(varfil_s[:pstokess+1])/stokesfil_s[0]
        print ("Filter "+pstokess*"%10s   Err  ") % tuple(ctypelist[1:])
        print ("%4s "+pstokess*"%8.3f %8.3f  ") % \
                (tuple(filter)+tuple(np.vstack((nstokesfil_s[1:],nerrfil_s[1:])).T.ravel()))

    return()

#--------------------------------------
 
if __name__=='__main__':
    filter=sys.argv[1]
    infilelist=sys.argv[2:]
    specpolfilter(filter, infilelist)
