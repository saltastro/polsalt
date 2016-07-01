import os
import sys
import copy
import numpy as np
from astropy.io import fits

import polsalt
from polsalt.specpolwollaston import correct_wollaston, read_wollaston

datadir = os.path.dirname(polsalt.__file__) + '/data/'


def correct_files(hdu,tilt=0):
    """For a given input file, apply corrections for wavelength, 
       distortion, and bad pixels

    Parameters
    ----------
    input_file: astropy.io.fits.HDUList
    """
    
    cbin, rbin = [int(x) for x in hdu[0].header['CCDSUM'].split(" ")]
    beams, rows, cols = hdu[1].data.shape
    
    #temporary cludge
    thdu = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(hdu[1].data[0])])
    thdu[0].header = hdu[0].header
    thdu[1].name = 'SCI'
    rpix_oc = read_wollaston(thdu, wollaston_file=datadir+"wollaston.txt")
    drow_oc = (rpix_oc-rpix_oc[:,cols/2][:,None])/rbin
    drow_oc += -(tilt/cbin)*(np.arange(cols) - cols/2)/cols
 
    for i in range(1, len(hdu)):
       for o in range(beams):
          hdu[i].data[o] = correct_wollaston(hdu[i].data[o], -drow_oc[o])
        
    return hdu

if __name__=='__main__':

    import glob
    if '*' in sys.argv[-1]:
      images = glob.glob(sys.argv[-1])
    else:
      images = sys.argv[1:]
    for img in images:
        hdu=correct_files(fits.open(img))
        hdu.writeto('c' + img, clobber=True)
  
