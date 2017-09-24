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
np.set_printoptions(threshold=np.nan)

def specpolfilter(filterlist, infilelist):


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
        print "\n"+infilelist[b]
        print ("Filter "+pstokess*"%5s      Err     ") % tuple(ctypelist[1:])
        for filter in filterlist:
            if filter in ("U","B","V"):
                filterfile = iraf.osfn("pysalt$data/scam/filters/Johnson_"+filter+".txt")
            elif filter in ("R","I"):
                filterfile = iraf.osfn("pysalt$data/scam/filters/Cousins_"+filter+".txt")         
#            else:
#                (filter file in cwd)
            wav_l,feff_l = np.loadtxt(filterfile,dtype=float,unpack=True)
            feff_l[feff_l < .0001] = 0.
            feff_w = interp1d(wav_l,feff_l,kind='linear',bounds_error=False)(wav_w)

            okf_w = (ok_w & (feff_w > .0003))
            feff_w[~okf_w] = 0.

            if feff_w[okf_w].sum() == 0: continue
            stokesfil_s = (feff_w[okf_w]*stokes_sw[:,okf_w]).sum(axis=1)/feff_w[okf_w].sum()
            varfil_s = (feff_w[okf_w]**2*var_sw[:,okf_w]).sum(axis=1)/(feff_w[okf_w].sum()**2)
            nstokesfil_s = 100.*stokesfil_s/stokesfil_s[0]
            nerrfil_s = 100.*np.sqrt(varfil_s[:pstokess+1])/stokesfil_s[0]
            print ("%4s "+pstokess*"%9.4f %7.4f  ") % \
                (tuple(filter)+tuple(np.vstack((nstokesfil_s[1:],nerrfil_s[1:])).T.ravel()))

    return()

#--------------------------------------
 
if __name__=='__main__':
    for filters in range(len(sys.argv)): 
        if sys.argv[filters+1].count(".fits"): break
    filterlist=sys.argv[1:filters+1]     
    infilelist=sys.argv[filters+1:]
    specpolfilter(filterlist, infilelist)
