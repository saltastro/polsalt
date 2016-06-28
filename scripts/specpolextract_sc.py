
"""

extract spectropolarimetry from image or images

"""

import os, sys, glob, shutil

import numpy as np
from astropy.io import fits as pyfits
from scipy.ndimage.filters import median_filter
from correct_files import correct_files

from PySpectrograph.Spectra import findobj, Spectrum

import pylab as pl

def specpolextract_sc(infilelist,yoffo,dyasec,logfile):

    log = open(logfile,'a')

    # first straighten out the spectra
    for img in infilelist:
        if (os.path.exists('c' + img)): continue
        hdu=correct_files(pyfits.open(img))
        hdu.writeto('c' + img, clobber=True)
        print 'c' + img
        print >>log,'c' + img

    infilelist = sorted(glob.glob('cw*fits'))
    for img in infilelist:
        hdu = pyfits.open(img)
        if hdu[0].header['LAMPID'].strip() != 'NONE': continue
        ybin = int(hdu[0].header['CCDSUM'].split(' ')[1])
        dy = int(dyasec*(8/ybin))
        yo_o = np.zeros(2,dtype=int)
        yo_o[0] = np.median(hdu[1].data[0].sum(axis=1).argsort()[-10:]) + yoffo
        yo_o[1] = np.median(hdu[1].data[1].sum(axis=1).argsort()[-10:]) + yoffo*1.0075  # allow for magnification
        print img,"  yo,ye,dy: ",yo_o,dy  
        print >>log,img," yo,ye,dy: ",yo_o,dy
        log.flush

    # establish the wavelength range to be used
        ys, xs = hdu[1].data.shape[1:]
        hduw = pyfits.open(img[1:])                                     # use wavmap from wm (cwm messed up)                                      
        wave_oyx = hduw[4].data                                              
        wbin = wave_oyx[0,yo_o[0],xs/2]-wave_oyx[0,yo_o[0],xs/2-1]      # bin to nearest power of 2 angstroms          
        wbin = 2.**(np.rint(np.log2(wbin)))                                 
        wmin_ox = wave_oyx.max(axis=1)                                     
        wmin = wmin_ox[wmin_ox>0].reshape((2,-1)).min(axis=1).max()     # ignore wavmap 0 values        
        wmax = wave_oyx[:,yo_o].reshape((2,-1)).max(axis=1).min()       # use wavs that are in both beams          
        wmin = (np.ceil(wmin/wbin)+0.5)*wbin                            # note: sc extract uses bin edges                                
        wmax = (np.floor(wmax/wbin)-0.5)*wbin                               

        o = 0 
        y1 = yo_o[o] - dy
        y2 = yo_o[o] + dy
        data = hdu[1].data[o]
        var = hdu[2].data[o]
        mask = hdu[3].data[o]
        wave = wave_oyx[o]
        w, fo, vo, bo = extract(data, var, mask, wave, y1, y2, wmin, wmax, wbin)

        o = 1
        y1 = yo_o[o] - dy
        y2 = yo_o[o] + dy
        data = hdu[1].data[o]
        var = hdu[2].data[o]
        mask = hdu[3].data[o]
        wave = wave_oyx[o]
        w, fe, ve, be = extract(data, var, mask, wave, y1, y2, wmin, wmax, wbin)
    
        sci_list = [[fo], [fe]]
        var_list = [[vo], [ve]]
        bad_list = [[bo], [be]]
   
        write_spectra(w, sci_list, var_list, bad_list,  hdu[0].header, wbin, 'e' + img)


def extract(data, var, mask, wave, y1, y2, wmin, wmax, wbin):
    """Extract the spectra

    Parameters
    ----------
    data: numpy.ndarray
        Flux data for spectra

    var: numpy.ndarray
        variance data for spectra

    mask: numpy.ndarray
        mask data for spectra

    wave: numpy.ndarray
        wavelength map for spectra

   Returns
   -------
    
    """
    w = np.arange(wmin, wmax, wbin)
    y0 = int((y1+y1)/2)
    wmask = ((wave[y0] > wmin) * (wave[y0] < wmax))

    #compute correction to preserve flux
    dwave = wave[y0,1:]-wave[y0,:-1]
    wavemean = wave[y0].mean()
    dwavemean = dwave.mean()
    dwavpoly = np.polyfit(wave[y0,1:]-wavemean,dwave-dwavemean,3)
    fluxcor = np.polyval(dwavpoly,w-wavemean) + dwavemean

    #estimate the sky
    s = np.zeros_like(w)
    count = 0
    dy = y2-y1
    for y in range(y1-3*dy, y1-2*dy) + range(y2+2*dy, y2+3*dy):
        s += np.interp(w, wave[y, wmask], data[y, wmask])
        count += 1
    s = s / count

    # extract the spectra
    f = np.zeros_like(w)
    v = np.zeros_like(w)
    b = np.zeros_like(w)
    for y in range(y1, y2):
        f += np.interp(w, wave[y, wmask], data[y, wmask]) - s
        # not exactly correct but estimate
        v += np.interp(w, wave[y, wmask], var[y, wmask])
        b += np.interp(w, wave[y, wmask], mask[y, wmask])

    f /= fluxcor
    v /= fluxcor**2
    b = 1.0 * ( b > 0 )
    b = b.astype('uint8')
    return w, f, v, b

def write_spectra(wave, sci_ow, var_ow, badbin_ow, header, wbin, outfile):
    """Write out the spectra in the correct format

    """
    header.update('VAREXT',2)
    header.update('BPMEXT',3)
    header.update('CRVAL1',wave[0]+wbin/2.)  # this needs to be fixed
    header.update('CRVAL2',0)
    header.update('CDELT1',wbin)  # this needs to be fixed
    header.update('CTYPE1','Angstroms')
    hduout = pyfits.PrimaryHDU(header=header)
    hduout = pyfits.HDUList(hduout)

    #what's the initial shape? 
    hduout.append(pyfits.ImageHDU(data=sci_ow, header=header, name='SCI'))
    header.update('SCIEXT',1,'Extension for Science Frame',before='VAREXT')
    hduout.append(pyfits.ImageHDU(data=var_ow, header=header, name='VAR'))
    hduout.append(pyfits.ImageHDU(data=badbin_ow, header=header, name='BPM'))

    hduout.writeto(outfile,clobber=True,output_verify='warn')

