
"""
specpolesim

Write a flat poisson noise simulation signal into e*.fits polarimetry files

"""

import os, sys

import numpy as np
from astropy.io import fits as pyfits

# ---------------------------------------------------------------------------------
def specpolesim(infilelist):
    infiles = len(infilelist)
    hdulist0 = pyfits.open(infilelist[0])

    for i in range(infiles): 
        if infilelist[i][0] != 'e':
            print 'Must be all e*.fits files'
            exit()

    wavs = hdulist0['SCI'].header['NAXIS1']

    for i in range(infiles):    
        hdul =  pyfits.open(infilelist[i]) 
        hdul['SCI'].data = np.random.poisson(10000.,2*wavs).reshape((2,1,-1))
        hdul['VAR'].data = np.repeat(10000.,2*wavs).reshape((2,1,-1))
        hdul.writeto(infilelist[i],clobber=True)
        print "O,E mean error: %8.3f %8.3f" % (np.std(hdul['SCI'].data[0]),np.std(hdul['SCI'].data[1]))       

    return
# ---------------------------------------------------------------------------------
if __name__=='__main__':
    infilelist=sys.argv[1:]
    specpolesim(infilelist)
