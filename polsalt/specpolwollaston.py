
"""
specpolsplit

Split O and E beams

"""


import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift

from specpolutils import rssmodelwave



def read_wollaston(hdu, wollaston_file):
    """ Correct the O or E beam for distortion due to the beam splitter

    Parameters
    ----------
    hdu: fits.HDUList
       Polarimetric observations data

    wollaston_file: None or str
       File containing the central position of the split O and E beams
 
    Return
    ------
    woll_pix: ~numpy.ndarray
       A two column array represnting the center pixels of the O+E beams for 
       the given configuration

    """
    #set up data
    data= hdu['SCI'].data
    rows,cols = data.shape
    grating = hdu[0].header['GRATING'].strip()
    grang = hdu[0].header['GR-ANGLE']
    artic = hdu[0].header['CAMANG']
    cbin, rbin = [int(x) for x in hdu[0].header['CCDSUM'].split(" ")]


    #load data from wollaston file
    lam_m = np.loadtxt(wollaston_file,dtype=float,usecols=(0,))
    rpix_om = np.loadtxt(wollaston_file,dtype=float,unpack=True,usecols=(1,2))
    lam_c = rssmodelwave(grating,grang,artic,cbin,cols)
    return interp1d(lam_m,rpix_om,kind='cubic',bounds_error=False)(lam_c)


def specpolwollaston(hdu, wollaston_file=None):
    """ Correct the O or E beam for distortion due to the beam splitter

    Parameters
    ----------
    hdu: fits.HDUList
       Polarimetric observations data

    beam: str
       Either the 'O' or the 'E' beam

    wollaston_file: None or str
       File containing the central position of the split O and E beams
 
    Return
    ------
    whdu: fits.HDUList
       New object with each extension corrected

    """
    rows,cols = hdu[1].data.shape
    cbin, rbin = [int(x) for x in hdu[0].header['CCDSUM'].split(" ")]

    #determine the shift
    rpix_oc = read_wollaston(data, wollaston_file)
    drow_shift = (rpix_oc-rpix_oc[:,cols/2][:,None])/rbin

    for i in range(len(hdu)):
        if hdu[i].data.any():
           for o in (0,1):
               hdu[i].data[o] = correct_wollaston(hdu[i].data[o], drow_shift[o])

    return hdu

def correct_wollaston(data, drow_shift):
    """Correct the distortion in the data by a shift

    Parameters
    ----------
    data: ~numpy.ndarray
       Data to be corrected

    drow_shift: ~numpy.ndarray
       Shift to be applied to each column

    Returns
    -------
    sdata: ~numpy.ndarray
       Corrected data
    
    """
    
    rows,cols = data.shape
    sdata = np.zeros(data.shape, dtype='float32')
    for c in range(cols):
        shift(data[:,c], drow_shift[c], sdata[:,c])
    return sdata
