
"""
specpolrawstokes

Form spectropolarimetry filterpair E-O raw stokes diffsums.

"""

import os
import sys
import glob
import shutil
import inspect

import numpy as np
from astropy.io import fits as pyfits
from scipy.interpolate import interp1d

from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging
from specpolutils import greff

import reddir
# from . import reddir
datadir = os.path.dirname(inspect.getfile(reddir)) + "/data/"
np.set_printoptions(threshold=np.nan)

def specpolrawstokes(infilelist, logfile='salt.log', debug=False):
    """Produces an unnormalized stokes measurement in intensity from
       a pair of WP filter positions

    Parameters
    ----------
    infilelist: list
        List of filenames that include an extracted spectra

    logfile: str
        Name of file for logging

    Notes
    -----
    The input file is a FITS file containing a 1D extracted spectrum with an e and o level  
        includes the intensity, variance, covariance, and bad pixels as extracted from the 2D spectrum.
    For each pair of stokes measurements, it produces an output FITS file now with two columns that are 
        the intensity and difference for the pair measured as a function of wavelength 
        and also includes the variance, covariance, and bad pixel maps.  
    The output file is named as the target name_configuration number_wave plate positions_number of repeats.fits
    """
    # set up some files that will be needed
    obsdate = os.path.basename(infilelist[0]).split('.')[0][-12:-4]

    patternfile = open(datadir + 'wppaterns.txt', 'r')
    with logging(logfile, debug) as log:

        log.message('specpolrawstokes version: 20171226', with_header=False)        
        # create the observation log
        obs_dict = obslog(infilelist)
        images = len(infilelist)

        hsta_i = np.array([int(round(s)) for s in obs_dict['HWP-STA']])
        qsta_i = np.array([int(round(s)) for s in obs_dict['QWP-STA']])
        img_i = np.array(
            [int(os.path.basename(s).split('.')[0][-4:]) for s in infilelist])
        wpstate_i = [['unknown', 'out', 'qbl', 'hw', 'hqw']
                     [int(s[1])] for s in obs_dict['WP-STATE']]
        wppat_i = obs_dict['WPPATERN']
        object_i = obs_dict['OBJECT']
        config_i = np.zeros(images, dtype='int')
        obs_i = -np.ones(images, dtype='int')

    # make table of observations
        configs = 0
        obss = 0
        for i in range(images):
            hdul = pyfits.open(infilelist[i])
            if (wpstate_i[i] == 'unknown'):
                log.message( 'Warning: Image %s WP-STATE UNKNOWN, assume it is 3 (HW)' % img_i[i], with_header=False)
                wpstate_i[i] = 'hw'
            elif (wpstate_i[i] == 'out'):
                log.message( 'Image %i not in a WP pattern, will skip' % img_i[i], with_header=False)
                continue
            if object_i[i].count('NONE'):
                object_i[i] = obs_dict['LAMPID'][i]
            object_i[i] = object_i[i].replace(' ', '')
            cbin, rbin = np.array(obs_dict["CCDSUM"][i].split(" ")).astype(int)
            grating = obs_dict['GRATING'][i].strip()
            grang = float(obs_dict['GR-ANGLE'][i])
            artic = float(obs_dict['CAMANG'][i])
            wav0 = hdul[0].header['CRVAL1']
            dwav = hdul[0].header['CDELT1']
            wavs = hdul['SCI'].data.shape[-1]

            confdat_d = [rbin, cbin, grating, grang, artic, wav0, dwav, wavs, wppat_i[i]]
            obsdat_d = [ object_i[i], rbin, cbin, grating, grang, artic, wppat_i[i]]
            if configs == 0:
                confdat_cd = [confdat_d]
                obsdat_od = [obsdat_d]
            configs = len(confdat_cd)
            config = 0
            while config < configs:
                if confdat_d == confdat_cd[config]:
                    break
                config += 1
            if config == configs:
                confdat_cd.append(confdat_d)
            config_i[i] = config
            obss = len(obsdat_od)
            obs = 0
            while obs < obss:
                if obsdat_d == obsdat_od[obs]:
                    break
                obs += 1
            if obs == obss:
                obsdat_od.append(obsdat_d)
            obs_i[i] = obs

        patternlist = patternfile.readlines()

        log.message(
            'Raw Stokes File     OBS CCDSUM GRATING  GR-ANGLE  CAMANG    WPPATERN',
            with_header=False)

    # Compute E-O raw stokes

        wav0_i = np.array([pyfits.getheader(infilelist[i],'SCI')['CRVAL1'] for i in range(images)]).astype(float)
        wavs_i = np.array([pyfits.getheader(infilelist[i],'SCI')['NAXIS1'] for i in range(images)]).astype(int)
        for obs in range(obss):
            idx_j = np.where(obs_i == obs)[0]
            i0 = idx_j[0]
            name_n = []
            if wppat_i[i0].count('NONE'):
                log.message((('\n     %s  WP Pattern: NONE. Calibration data, skipped') % infilelist[i0]),  \
                    with_header=False)
                continue
            if wppat_i[i0].count('UNKNOWN'):
                if (hsta_i[idx_j] % 2).max() == 0:
                    wppat = "LINEAR"
                else:
                    wppat = "LINEAR-HI"
                for i in idx_j:
                    wppat_i[i] = wppat
            if not(((wpstate_i[i0] == 'hw') & (wppat_i[i0] in ('LINEAR', 'LINEAR-HI'))
                    | (wpstate_i[i0] == 'hqw') & (wppat_i[i0] in ('CIRCULAR', 'CIRCULAR-HI', 'ALL-STOKES')))):
                print "Observation", obs, ": wpstate ", wpstate_i[i0], \
                    " and wppattern ", wppat_i[i0], "not consistent"
                continue
            for p in patternlist:
                if (p.split()[0] == wppat_i[i0]) & (p.split()[2] == 'hwp'):
                    wpat_p = np.array(p.split()[3:]).astype(int)
                if (p.split()[0] == wppat_i[i0]) & (p.split()[2] == 'qwp'):
                    wpat_dp = np.vstack(wpat_p, np.array(p.split()[3:]))
            stokes = 0
            j = -1

        # using overlapping wavelengths
            dwav = pyfits.getheader(infilelist[idx_j[0]],'SCI')['CDELT1']
            wav0 = wav0_i[idx_j].max()
            col1_j = ((wav0_i[idx_j] - wav0)/dwav).astype(int)
            wavs = (wavs_i[idx_j] - col1_j).min()

            while j < (len(idx_j) - 2):
                j += 1
                i = idx_j[j]
                if (wpstate_i[i] == 'hw'):
                    if (np.where(wpat_p[0::2] == hsta_i[i])[0].size > 0):
                        idxp = np.where(wpat_p == hsta_i[i])[0][0]
                        if hsta_i[i + 1] != wpat_p[idxp + 1]:
                            continue
                    else:
                        continue
                if (wpstate_i[i] == 'hqw'):
                    if (np.where(wpat_dp[0::2] == (hsta_i[i], qsta_i[i]))[0].size > 0):
                        idxp = np.where( wpat_dp == ( hsta_i[i], qsta_i[i]))[0][0]
                        if (hsta_i[i + 1], qsta_i[i + 1]) != wpat_dp[None, idxp + 1]:
                            continue
                    else:
                        continue

                first_pair_file = infilelist[i]
                second_pair_file = infilelist[i + 1]

                name = object_i[i] + '_c' + str(config_i[i]) + '_h' + str(hsta_i[i]) + str(hsta_i[i + 1])
                if (wpstate_i[i] == 'hqw'):
                    name += 'q' + ['m', 'p'][qsta_i[i] == 4] + ['m', 'p'][qsta_i[i + 1] == 4]

                count = " ".join(name_n).count(name)
                name += ('_%02i' % (count + 1))

                create_raw_stokes_file( first_pair_file, second_pair_file, wav0, wavs,
                    output_file=name + '.fits', wppat=wppat_i[i], debug=debug)
                log.message('%20s  %1i  %1i %1i %8s %8.2f %8.2f %12s' %
                            (name, obs, rbin, cbin, grating, grang, artic, wppat_i[i]), with_header=False)
                name_n.append(name)
                i += 1
                stokes += 1

    return


def create_raw_stokes_file(first_pair_file, second_pair_file, wav0, wavs, output_file, wppat=None, debug=False):
    """Create the raw stokes file.

    Parameters
    ----------
    first_pair_file: str
       Name of first file in pair for pattern

    second_pair_file: str
       Name of second file in pair for pattern

    wav0,wavs: float,int
       output file shape

    output_file: str
       Name of output file

    wppat: str
       Name of wave plate pattern

    Notes
    ------
    Writes out a FITS file containing the unnormalized intensity and stokes difference for the pair
    _f  filter pair index (0,1)
    _O  o/e index for unnormailzed extractions (0,1)
    _w  wavelength index
    _S  unnormalized raw stokes index (I,S = 0,1)

    """

    sci_fOw = np.zeros((2, 2, wavs))
    var_fOw = np.zeros_like(sci_fOw)
    covar_fOw = np.zeros_like(sci_fOw)
    bpm_fOw = np.zeros_like(sci_fOw)
    dwav = pyfits.getheader(first_pair_file, 'SCI', 1)['CDELT1']
    grating = pyfits.getheader(first_pair_file, 'SCI')['GRATING'].strip()
    grang = float(pyfits.getheader(first_pair_file, 'SCI')['GR-ANGLE'])
    artic = float(pyfits.getheader(first_pair_file, 'SCI')['CAMANG'])
    dateobs = pyfits.getheader(first_pair_file, 'SCI')['DATE-OBS'].replace('-','')
    for f, img in enumerate([first_pair_file, second_pair_file]):
        hdulist = pyfits.open(img)
        wav0f = pyfits.getheader(img, 'SCI', 1)['CRVAL1']
        c0 = int((wav0-wav0f)/dwav)
        sci_fOw[f] = hdulist['SCI'].data[:,:,c0:c0+wavs].reshape((2, -1))
        var_fOw[f] = hdulist['VAR'].data[:,:,c0:c0+wavs].reshape((2, -1))
        covar_fOw[f] = hdulist['COV'].data[:,:,c0:c0+wavs].reshape((2, -1))
        bpm_fOw[f] = hdulist['BPM'].data[:,:,c0:c0+wavs].reshape((2, -1))

  # Mark as bad bins with negative intensities
    bpm_fOw[sci_fOw < 0] = 1

  # compute intensity, E-O stokes spectrum, VAR, COV, BPM.
  # fits out: unnormalized (int,stokes),(length 1) spatial,wavelength
  # wavelength marked bad if it is bad in either filter or order

    bpm_w = (bpm_fOw.sum(axis=0).sum(axis=0) > 0).astype(int)
    wok = (bpm_w == 0)

    stokes_Sw = np.zeros((2, wavs), dtype='float32')
    var_Sw = np.zeros_like(stokes_Sw)
    covar_Sw = np.zeros_like(stokes_Sw)
    wav_w = np.arange(wav0,wav0+dwav*wavs,dwav)

  # compute and correct for estimated throughput change (improves pupil variation correction)
    EoverO_w = greff(grating,grang,artic,dateobs,wav_w)[1]
    
    f21 = np.median(((sci_fOw[1][0,wok]*EoverO_w[wok] + sci_fOw[1][1,wok])/      \
                     (sci_fOw[0][0,wok]*EoverO_w[wok] + sci_fOw[0][1,wok])))

    if debug:
        name = output_file.split('.')[0] 
        print name," f21 = ",f21
        np.savetxt(name+"_ps.txt",np.vstack((wav_w,wok,sci_fOw.reshape((4,-1)),EoverO_w)).T,fmt="%7.1f %2i "+5*"%8.4f ")

  # intensity: 0.5 * (O1 + E1 + (O2 + E2)/f21)
    stokes_Sw[0][wok] = 0.5 * (sci_fOw[0][:, wok].sum(axis=0) + sci_fOw[1][:, wok].sum(axis=0)/f21)
    var_Sw[0][wok] = 0.25 * (var_fOw[0][:, wok].sum(axis=0) + var_fOw[1][:, wok].sum(axis=0)/f21**2)
    covar_Sw[0][wok] = 0.25 * (covar_fOw[0][:, wok].sum(axis=0) + covar_fOw[1][:, wok].sum(axis=0)/f21**2)

  # stokes difference: defined as int * 0.5 * ( (E1-E2/f21)/(E1+E2/f21) - (O1-O2/f21)/(O1+O2/f21))
    stokes_Sw[1, wok] = 0.5*    \
        ((sci_fOw[0, 1, wok] - sci_fOw[1, 1, wok]/f21) / (sci_fOw[0, 1, wok] + sci_fOw[1, 1, wok]/f21)
       - (sci_fOw[0, 0, wok] - sci_fOw[1, 0, wok]/f21) / (sci_fOw[0, 0, wok] + sci_fOw[1, 0, wok]/f21))
    var_Sw[1, wok] = 0.25*  \
        ((var_fOw[0, 1, wok] + var_fOw[1, 1, wok]/f21**2) / (sci_fOw[0, 1, wok] + sci_fOw[1, 1, wok]/f21) ** 2
       + (var_fOw[0, 0, wok] + var_fOw[1, 0, wok]/f21**2) / (sci_fOw[0, 0, wok] + sci_fOw[1, 0, wok]/f21) ** 2)
    covar_Sw[1, wok] = 0.25*  \
        ((covar_fOw[0, 1, wok] + covar_fOw[1, 1, wok]/f21**2) / (sci_fOw[0, 1, wok] + sci_fOw[1, 1, wok]/f21) ** 2
       + (covar_fOw[0, 0, wok] + covar_fOw[1, 0, wok]/f21**2) / (sci_fOw[0, 0, wok] + sci_fOw[1, 0, wok]/f21) ** 2)

    stokes_Sw[1] *= stokes_Sw[0]
    var_Sw[1] *= stokes_Sw[0] ** 2
    covar_Sw[1] *= stokes_Sw[0] ** 2
    bpm_Sw = np.array([bpm_w, bpm_w], dtype='uint8').reshape((2, wavs))

    hduout = pyfits.PrimaryHDU(header=hdulist[0].header)
    hduout = pyfits.HDUList(hduout)
    if wppat:
        hduout[0].header['WPPATERN'] = wppat
    header = hdulist['SCI'].header.copy()
    header['VAREXT'] =  2
    header['COVEXT'] =  3
    header['BPMEXT'] =  4
    header['CRVAL1'] =  wav0
    header['CTYPE3'] =  'I,S'
    hduout.append( pyfits.ImageHDU( data=stokes_Sw.reshape( (2, 1, wavs)), header=header, name='SCI'))
    header.set('SCIEXT', 1, 'Extension for Science Frame', before='VAREXT')
    hduout.append( pyfits.ImageHDU( data=var_Sw.reshape( (2, 1, wavs)), header=header, name='VAR'))
    hduout.append( pyfits.ImageHDU( data=covar_Sw.reshape( (2, 1, wavs)), header=header, name='COV'))
    hduout.append( pyfits.ImageHDU( data=bpm_Sw.reshape( (2, 1, wavs)), header=header, name='BPM'))

    hduout.writeto(output_file, overwrite=True, output_verify='warn')

# ----------------------------------------------------------
if __name__=='__main__':
    infilelist=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)
    if len(kwargs): kwargs = {k:bool(v) for k,v in kwargs.iteritems()}        
    specpolrawstokes(infilelist,**kwargs)
