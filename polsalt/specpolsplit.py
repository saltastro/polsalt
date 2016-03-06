
"""
specpolsplit

Split O and E beams

"""

import os, sys, glob, shutil, inspect

import numpy as np
import pyfits
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
from scipy import linalg as la

from pyraf import iraf
from iraf import pysalt

from saltobslog import obslog

from specidentify import specidentify
from saltsafelog import logging

import reddir
from specpolutils import rssmodelwave

np.set_printoptions(threshold=np.nan)
debug = True

#----------------------------------------------------------------------
def specpolsplit(hdu, splitrow=None, wollaston_file=None):
    """ Split the O and E beams  

    Parameters
    ----------
    hdu: fits.HDUList
       Polarimetric observations data

    splitrow: None or int
       Row to split the image.  If None, the value will be calculated

    wollaston_file: None or str
       File containing the central position of the split O and E beams
 
    Return
    ------
    whdu: fits.HDUList
       New header object with extensions split between the O and E beams

    splitrow: float
       Row at which to split the images


    """

    rows,cols = hdu[1].data.shape

    #determine the row to split the file from the estimated center
    if splitrow is None and wollaston_file:
        # use arc to make first-guess wavecal from model
        # locate beamsplitter split point based on the center of the chips 
        # given in the wollaston file

        #set up data
        data= hdu['SCI'].data
        grating = hdu[0].header['GRATING']
        grang = hdu[0].header['GR-ANGLE']
        artic = hdu[0].header['CAMANG']
        cbin, rbin = [int(x) for x in hdu[0].header['CCDSUM'].split(" ")]


        #load data from wollaston file
        lam_m = np.loadtxt(wollaston_file,dtype=float,usecols=(0,))
        rpix_om = np.loadtxt(wollaston_file,dtype=float,unpack=True,usecols=(1,2))
        lam_c = rssmodelwave(grating,grang,artic,cbin,cols)
        axisrow_o = ((2052 + interp1d(lam_m,rpix_om,kind='cubic',bounds_error=False) \
                     (lam_c[cols/2]))/rbin).astype(int)

        data_y = hdu[1].data.sum(axis=1)
        top = axisrow_o[1] + np.argmax(data_y[axisrow_o[1]:] <  0.5*data_y[axisrow_o[1]])
        bot = axisrow_o[0] - np.argmax(data_y[axisrow_o[0]::-1] <  0.5*data_y[axisrow_o[0]])
        splitrow = 0.5*(bot + top)
    elif splitrow is None and wollaston_file is None:
         splitrow = rows/2.0


    offset = int(splitrow - rows/2)                 # how far split is from center of detector

    # split arc into o/e images
    padbins = (np.indices((rows,cols))[0]<offset) | (np.indices((rows,cols))[0]>rows+offset)

    image_rc = np.roll(hdu['SCI'].data,-offset,axis=0)
    image_rc[padbins] = 0.
    hdu['SCI'].data = image_rc.reshape((2,rows/2,cols))
    var_rc = np.roll(hdu['VAR'].data,-offset,axis=0)
    var_rc[padbins] = 0.
    hdu['VAR'].data = var_rc.reshape((2,rows/2,cols))
    bpm_rc = np.roll(hdu['BPM'].data,-offset,axis=0)
    bpm_rc[padbins] = 1
    bpm_orc = bpm_rc.reshape((2,rows/2,cols))
    hdu['BPM'].data = bpm_orc

    return hdu, splitrow
