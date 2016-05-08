
"""
AGNEXTRACT

extract galaxy spectra from image or images and combine them

"""

import os, sys, glob, shutil

import numpy as np
from scipy.ndimage.filters import median_filter

from astropy.io import fits



from PySpectrograph.Spectra import findobj, Spectrum


import argparse

import pylab as pl

def extract(data, error, mask, wave, y1, y2):
    """Extract the spectra

    Parameters
    ----------
    data: numpy.ndarray
        Flux data for spectra

    error: numpy.ndarray
        error data for spectra

    mask: numpy.ndarray
        mask data for spectra

    wave: numpy.ndarray
        wavelength map for spectra

   Returns
   -------
    
    """

    # estimate the wavelength range to be used
    ys, xs = data.shape
    w = wave[int(ys/2.0),:]
    wmin = w[w>0].min()
    wmax = w[w>0].max()
    wmask = ((w > wmin) * (w < wmax))
    wbin = (wmax-wmin)/wmask.sum()
    w = np.arange(wmin, wmax, wbin)

    #estimate the sky
    s = np.zeros_like(w)
    count = 0
    dy = y2-y1
    for y in range(y1-2*dy, y1-dy) + range(y2+dy, y2+2*dy):
        s += np.interp(w, wave[y, wmask], data[y, wmask])
        count += 1
    s = s / count

    
    
    
    # extract the spectra
    f = np.zeros_like(w)
    e = np.zeros_like(w)
    b = np.zeros_like(w)
    for y in range(y1, y2):
        f += np.interp(w, wave[y, wmask], data[y, wmask]) - s
        # not exactly correct but estimate
        e += np.interp(w, wave[y, wmask], error[y, wmask])**2
        b += np.interp(w, wave[y, wmask], mask[y, wmask])

    e = e**0.5
    b = 1.0 * ( b > 0 )
    b = b.astype('uint8')
    return w, f, e, b, wbin

def write_spectra(wave, sci_ow, var_ow, badbin_ow, header, wbin, outfile):
    """Write out the spectra in the correct format

    """
    header.update('VAREXT',2)
    header.update('BPMEXT',3)
    header.update('CRVAL1',wave[0]+wbin/2.)  # this needs to be fixed
    header.update('CRVAL2',0)
    header.update('CDELT1',wbin)  # this needs to be fixed
    header.update('CTYPE1','Angstroms')
    hduout = fits.PrimaryHDU(header=header)
    hduout = fits.HDUList(hduout)

    #what's wavs?  
    #what's the initial shape? 
    hduout.append(fits.ImageHDU(data=sci_ow, header=header, name='SCI'))
    header.update('SCIEXT',1,'Extension for Science Frame',before='VAREXT')
    hduout.append(fits.ImageHDU(data=var_ow, header=header, name='VAR'))
    hduout.append(fits.ImageHDU(data=badbin_ow, header=header, name='BPM'))

    hduout.writeto(outfile,clobber=True,output_verify='warn')

if __name__=='__main__':
   calfile=None

   parser = argparse.ArgumentParser(description='Extract SALT Polarimetric data')
   parser.add_argument('image', help='Image to extract the spectra', nargs='*')
   parser.add_argument('--yo', dest='yo', type=int, help='y position of o beam')
   parser.add_argument('--ye', dest='ye', type=int, help='y position of e beam')
   parser.add_argument('--dy', dest='dy', type=int, help='y width of psf')
   #parser.add_argument('-w', dest='basic_wave', default=True, action='store_false',
   #                 help='Skip wavelength calibration')
   args = parser.parse_args()
   thresh = 5
   convert = False

    
   for img in args.image:
    
       hdu = fits.open(img)
       y1 = args.yo - args.dy
       y2 = args.yo + args.dy
       o = 0
       data = hdu[1].data[o]
       error = hdu[2].data[o]
       mask = hdu[3].data[o]
       wave = hdu[4].data[o]
       wo, fo, eo, bo, wbin = extract(data, error, mask, wave, y1, y2)
       y1 = args.ye - args.dy
       y2 = args.ye + args.dy
       o = 1
       data = hdu[1].data[o]
       error = hdu[2].data[o]
       mask = hdu[3].data[o]
       wave = hdu[4].data[o]
       we, fe, ee, be, wbin = extract(data, error, mask, wave, y1, y2)
    
       sci_list = [[fo], [fe]]
       err_list = [[eo], [ee]]
       bad_list = [[bo], [be]]
       w = wo
   
       write_spectra(w, sci_list, err_list, bad_list,  hdu[0].header, wbin, 'e' + img)
   exit()

   #pl.figure()
   #pl.axes([0.1, 0.7, 0.8, 0.25])
   #pl.plot(wo, fe)
   #pl.axes([0.1, 0.4, 0.8, 0.25])
   #pl.plot(wo, ee)
   ##pl.axes([0.1, 0.1, 0.8, 0.25])
   #pl.plot(wo, be)
   #pl.show()
