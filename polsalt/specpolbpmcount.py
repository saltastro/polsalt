
"""
specpolbpmcount

Print out BPM count in specpol fits files

"""

import os, sys, glob, shutil, inspect

import numpy as np
from astropy.io import fits as pyfits

np.set_printoptions(threshold=np.nan)

# -------------------------------------
def specpolbpmcount(infilelist):
    """Print out BPM count in specpol fits files

    """
    files = len(infilelist)
    for i in range(files):
        bpm_sw = pyfits.open(infilelist[i])['BPM'].data.reshape((2,-1))                
        print "%s %4i %4i" % ((infilelist[i],)+tuple(bpm_sw.sum(axis=1)))

    return 

# ------------------------------------

if __name__=='__main__':
    infilelist=sys.argv[1:]      
    specpolbpmcount(infilelist)
