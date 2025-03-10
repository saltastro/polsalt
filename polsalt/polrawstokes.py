
"""
polrawstokes

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
from astropy.table import Table

# this is pysalt-free

import rsslog
from obslog import create_obslog
from polutils import greff, angle_average

datadir = os.path.dirname(__file__) + '/data/'
keywordfile = datadir+"obslog_config.json"
np.set_printoptions(threshold=np.nan)

import warnings
# warnings.filterwarnings("error",category=RuntimeWarning)    # trace warnings

def polrawstokes(infileList, **kwargs):
    """Produces an unnormalized stokes measurement in intensity from
       a pair of WP filter positions

    Parameters
    ----------
    infileList: list
        List of filenames with extracted spectra

    logfile: str
        Name of file for logging

    Notes
    -----
    The input file is a FITS file containing 1D extracted spectra with an e and o level  
        includes the intensity, variance, covariance, and bad pixels as extracted from the 2D spectrum.
        data dimensions are (wavelength,target,O/E=(0,1))
    For each pair of stokes measurements, it produces an output FITS file now with two columns that are 
        the intensity and difference for the pair measured as a function of wavelength 
        and also includes the variance, covariance, and bad pixel maps.  
    The output file is named as the target-name_configuration-number_wave-plate-positions_number-of-repeats.fits
    """
    """
    _c configuration index
    _d data index within confdat (0-8) or obsdat (0-6)
    _i file index in infileList order
    _j index within files of an observation

    """
    logfile= kwargs.pop('logfile','polrawstokes.log')
    with_stdout = kwargs.pop('with_stdout',True)       
    if with_stdout: rsslog.history(logfile)        

    if with_stdout: rsslog.message(str(kwargs), logfile)        # for some reason, history is not printing kwargs
    debug = kwargs.pop('debug',False)
    if isinstance(debug, str): debug=(debug=="True")
                
    obsdate = os.path.basename(infileList[0]).split('.')[0][-12:-4]
    patternfile = open(datadir + 'wppaterns.txt', 'r')
    if with_stdout: rsslog.message('polrawstokes version: 20211002', logfile) 
     
  # create the observation log
    obsDict = create_obslog(infileList,keywordfile)
    images = len(infileList)
    hsta_i = np.array([(int(round(s/11.25)) % 8) for s in np.array(obsDict['HWP-ANG']).astype(float)])
    qsta_i = np.array([((int(round(s/11.25))+8) % 16)-8 for s in np.array(obsDict['QWP-ANG']).astype(float)])
    img_i = np.array(
        [int(os.path.basename(s).split('.')[0][-4:]) for s in infileList])
    wpstate_i = [['unknown', 'out', 'qbl', 'hw', 'hqw']
        [int(s[1])] for s in np.array(obsDict['WP-STATE'])]
    wppat_i = np.array(obsDict['WPPATERN'],dtype='object')
    object_i = np.array(obsDict['OBJECT'],dtype='object')
    bvisitid_i = np.array(obsDict['BVISITID']).astype(int)
    config_i = np.zeros(images, dtype='int')
    obs_i = -np.ones(images, dtype='int')

  # make table of observations
    configs = 0
    obss = 0
    for i in range(images):
        hdul = pyfits.open(infileList[i])
        if (wpstate_i[i] == 'unknown'):
            if with_stdout: rsslog.message( 'Warning: Image %s WP-STATE UNKNOWN, assume it is 3 (HW)' % img_i[i], logfile)
            wpstate_i[i] = 'hw'
        elif (wpstate_i[i] == 'out'):
            if with_stdout: rsslog.message( 'Image %i not in a WP pattern, will skip' % img_i[i], logfile)
            continue
        if object_i[i].count('NONE'):
            object_i[i] = obsDict['LAMPID'][i]
        object_i[i] = object_i[i].replace(' ', '')
        cbin, rbin = np.array(obsDict["CCDSUM"][i].split(" ")).astype(int)
        grating = obsDict['GRATING'][i].strip()
        grang = float(obsDict['GR-ANGLE'][i])
        artic = float(obsDict['CAMANG'][i])
        filter = obsDict['FILTER'][i].strip()
        wav0 = hdul[0].header['CRVAL1']
        dwav = hdul[0].header['CDELT1']
        targets,wavs = hdul['SCI'].data.shape[-2:]

        confdat_d = [rbin, cbin, grating, grang, artic, filter, wav0, dwav, wavs, targets, wppat_i[i]]
        obsdatList = [object_i[i],bvisitid_i[i],targets,rbin,cbin,grating,grang,artic,filter,wppat_i[i]]
        if configs == 0:
            confdat_cd = [confdat_d]
            obsTabo = Table([[ele] for ele in obsdatList], \
                names = ('OBJECT','BVISITID','TARGETS','RBIN','CBIN','GRATING','GRANG','ARTIC','FILTER','WPPAT'))
        configs = len(confdat_cd)
        config = 0
        while config < configs:
            if confdat_d == confdat_cd[config]:
                break
            config += 1
        if config == configs:
            confdat_cd.append(confdat_d)
        config_i[i] = config
        obss = len(obsTabo)
        obs = 0
        while obs < obss:
            if (list(obsTabo[obs])==obsdatList):
                break
            obs += 1
        if obs == obss:
            obsTabo.add_row(obsdatList)
                
        obs_i[i] = obs

    patternlist = patternfile.readlines()
    wav0_i = np.array([pyfits.getheader(infileList[i],'SCI')['CRVAL1'] for i in range(images)]).astype(float)
    wavs_i = np.array([pyfits.getheader(infileList[i],'SCI')['NAXIS1'] for i in range(images)]).astype(int)
    name_i = np.copy(object_i)              # ensure output name unique for each observation 
               
    for object in np.unique(object_i):
        iList = list(np.where(object_i == object)[0])           
        namedups = len(np.unique(obs_i[iList]))                        
        if (namedups > 1): 
            name_i[iList] = [(object+("-%02i" % obs_i[i])) for i in iList]
        
    if debug:
        print "confdat_cd: "
        for c in range(configs):
            print confdat_cd[c]
        print "obsTabo:  \n",obsTabo, "\n"

  # Compute E-O raw stokes
    if with_stdout: rsslog.message(
    'Raw Stokes File     Obs Tgts CCDSUM GRATING  GR-ANGLE  CAMANG  FILTER     WPPATERN',
        logfile)
            
    for obs in range(obss):    
        idx_j = np.where(obs_i == obs)[0]            
        i0 = idx_j[0]
        name_n = []
        if wppat_i[i0].count('NONE'):
            if with_stdout: rsslog.message((('\n     %s  WP Pattern: NONE. Calibration data, skipped') % infileList[i0]),  \
                logfile)
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
        if (wppat_i[i0] in ('LINEAR', 'LINEAR-HI')):                
            qsta_i[idx_j] = 0                    # allow same pattern-match sw for all modes                
        for p in patternlist:
            if (p.split()[0] == wppat_i[i0]) & (p.split()[2] == 'hwp'):
                wpat_p = np.array(p.split()[3:]).astype(int)
                wpat_dp = np.vstack((wpat_p, np.zeros_like(wpat_p))).astype(int)
            if (p.split()[0] == wppat_i[i0]) & (p.split()[2] == 'qwp'):
                wpat_dp = np.vstack((wpat_p, np.array(p.split()[3:]).astype(int) % 32))
        j = -1

      # using overlapping wavelengths
        dwav = pyfits.getheader(infileList[idx_j[0]],'SCI')['CDELT1']
        wav0 = wav0_i[idx_j].max()
        col1_j = ((wav0_i[idx_j] - wav0)/dwav).astype(int)
        wavs = (wavs_i[idx_j] - col1_j).min()
        firstpairList = []
        secondpairList = []

        while (j < (len(idx_j)-1)):
            j += 1
            i = idx_j[j]                
            if np.where((wpat_dp.T == (hsta_i[i], qsta_i[i])).all(axis=1))[0].size ==0: continue                                    
            if (len(firstpairList)==0):                    
                if (len(np.where((wpat_dp[:,0::2].T == (hsta_i[i], qsta_i[i])).all(axis=1))[0]) > 0):
                    idxp = np.where((wpat_dp.T == (hsta_i[i], qsta_i[i])).all(axis=1))[0][0]
                    firstpairList = [infileList[i],]                                                                                    
                continue                  
                        
            if (len(secondpairList)==0):
                if ((hsta_i[i],qsta_i[i]) == wpat_dp[:,idxp]).all():
                    firstpairList.append(infileList[i])
                    continue
                elif ((hsta_i[i],qsta_i[i]) == wpat_dp[:,idxp+1]).all():
                    secondpairList = [infileList[i],]
                            
            elif ((hsta_i[i],qsta_i[i]) == wpat_dp[:,idxp+1]).all():
                secondpairList.append(infileList[i])                    
                        
            if (len(secondpairList) > 0):
                if (j < (len(idx_j)-1)):
                    if ((hsta_i[i+1],qsta_i[i+1]) == wpat_dp[:,idxp+1]).all():
                        continue                              
                                
            name = name_i[i] + '_c' + str(config_i[i]) + '_h' + str(wpat_p[idxp]) + str(wpat_p[idxp+1])
            if (wpstate_i[i] == 'hqw'):
                name += 'q' + ['m','0','p'][[-4,0,4].index(wpat_dp[1,idxp])] +     \
                            ['m','0','p'][[-4,0,4].index(wpat_dp[1,idxp+1])]

            count = " ".join(name_n).count(name)
            name += ('_%02i' % (count + 1))

            if debug:
                print firstpairList, secondpairList, wav0, wavs

            allpol_stokes_file(firstpairList, secondpairList, wav0, wavs,
                output_file=name + '.fits', wppat=wppat_i[i], debug=debug)                
            if with_stdout: rsslog.message((('%-20s %1i   %3i  %1i %1i %8s %8.2f %8.2f %8s %12s') %  \
                ((name,obs)+tuple(obsTabo[obs])[2:])), logfile)
            name_n.append(name)
            firstpairList = []
            secondpairList = []

    return

#----------------------------------------------------------------

def allpol_stokes_file(firstpairList, secondpairList, wav0, wavs, output_file, wppat=None, debug=False):
    """Create the raw stokes file, allpol version

    Parameters
    ----------
    firstpairList: List(str)
       List of files in first pair for pattern

    secondpairList: List(str)
       List of files in second pair for pattern

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
    _i  detector iterations within filter pair (0,...)
    _t  target index (0,...)
    _P  o/e index for unnormalized extractions (0,1)
    _w  wavelength index
    _S  unnormalized raw stokes index (I,S = 0,1)

    """
    hdul = pyfits.open(firstpairList[0])
    dwav = hdul[0].header['CDELT1']
    grating = hdul[0].header['GRATING'].strip()
    grang = float(hdul[0].header['GR-ANGLE'])
    artic = float(hdul[0].header['CAMANG'])
    dateobs = hdul[0].header['DATE-OBS'].replace('-','') 
    
    oktargs = 1     # for backwards compatibility with TGT-less ec data    
    tgthdu = None
    if 'TGT' in [hdul[x].name for x in range(len(hdul))]:    
        tgthdu = hdul['TGT']
        tgtTab = Table.read(hdul['TGT'])    
        oktgt_i = (tgtTab['CULL']=="")
        oktargs = oktgt_i.sum()        
    wcshdu = None
    if 'WCSDVARR' in [hdul[x].name for x in range(len(hdul))]:
        wcshdu = hdul['WCSDVARR']
    sci_fPTw = np.zeros((2, 2, oktargs, wavs))
    var_fPTw = np.zeros_like(sci_fPTw)
    cov_fPTw = np.zeros_like(sci_fPTw)
    bpm_fPTw = np.zeros_like(sci_fPTw)
    exptime_f = np.zeros(2)
    imgtime_f = np.zeros(2)    
    obsDT_f = np.zeros(2,dtype=object)
    telpa_f = np.zeros(2)
    mbyx_df = np.zeros((2,2))
    
    for f, imgList in enumerate([firstpairList, secondpairList]):
        hduList = pyfits.open(imgList[0])
        exptime_f[f] = hduList[0].header['EXPTIME']
        imgtime_f[f] = np.copy(exptime_f[f])
        obsDT_f[f] = np.datetime64(hduList[0].header['DATE-OBS']+'T'+hduList[0].header['UTC-OBS'])
        telpa_f[f] = hduList[0].header['TELPA']
        mbyx_df[:,f] = np.array([hduList[0].header['MBY'], hduList[0].header['MBX']])
        wav0f = pyfits.getheader(imgList[0], 'SCI', 1)['CRVAL1']
        c0 = int((wav0-wav0f)/dwav)
        obsendDT = obsDT_f[f] + np.timedelta64(int(round(exptime_f[f])),'s')    
        sci_fPTw[f] = hduList['SCI'].data[:,:,c0:c0+wavs]
        var_fPTw[f] = hduList['VAR'].data[:,:,c0:c0+wavs]
        cov_fPTw[f] = hduList['COV'].data[:,:,c0:c0+wavs]
        bpm_fPTw[f] = hduList['BPM'].data[:,:,c0:c0+wavs]
        if (f==0): werr_PTw = hdul['WERR'].data[:,:,c0:c0+wavs]           
        if len(imgList) > 1:
            telpa_i = telpa_f[f]*np.ones(len(imgList))      # fix to detector iterations problem        
            for i in range(1,len(imgList)):
                hduList = pyfits.open(imgList[i])
                telpa_i[i] = hduList[0].header['TELPA']                
                exptime_f[f] += hduList[0].header['EXPTIME']
                obsstartDT = np.datetime64(hduList[0].header['DATE-OBS']+hduList[0].header['UTC-OBS'])                
                sci_fPTw[f] += hduList['SCI'].data[:,:,c0:c0+wavs]
                var_fPTw[f] += hduList['VAR'].data[:,:,c0:c0+wavs]
                cov_fPTw[f] += hduList['COV'].data[:,:,c0:c0+wavs]
                bpm_fPTw[f] = np.maximum(bpm_fPTw[f],hduList['BPM'].data[:,:,c0:c0+wavs])
            telpa_f[f] =  angle_average(telpa_i)
            obsendDT = obsstartDT + np.timedelta64(int(round(exptime_f[f])),'s')                
        obsDT_f[f] += (obsendDT - obsDT_f[f])/2.

  # Mark as bad bins with negative O+E intensities  
    bpm_fPTw[np.repeat((sci_fPTw.sum(axis=1) < 0.)[:,None,:,:],2,axis=1)] = 1
    
  # compute intensity, E-O stokes spectrum, VAR, COV, BPM.
  # fits out: unnormalized (int,stokes),target,wavelength
  # wavelength marked bad if it is bad in either of pair

    bpm_Tw = (bpm_fPTw.sum(axis=(0,1)) > 0).astype(int)
    wok_Tw = (bpm_Tw == 0)

    stokes_STw = np.zeros((2, oktargs, wavs), dtype='float32')
    var_STw = np.zeros_like(stokes_STw)
    cov_STw = np.zeros_like(stokes_STw)
    wav_w = np.arange(wav0,wav0+dwav*wavs,dwav)

  # compute and correct for estimated throughput change (improves pupil variation correction)
    if grating == 'N/A':
        f21 = 1.
    else:
        Tok = wok_Tw.sum(axis=1) > wok_Tw.sum(axis=1).max()/2   
        wok = wok_Tw[Tok,:].all(axis=0)                                
        EoverO_w = greff(grating,grang,artic,dateobs,wav_w)[1]
        f21 = np.median((sci_fPTw[1,0][:,wok].sum(axis=0)*EoverO_w[wok] + sci_fPTw[1,1][:,wok].sum(axis=0))/      \
                        (sci_fPTw[0,0][:,wok].sum(axis=0)*EoverO_w[wok] + sci_fPTw[0,1][:,wok].sum(axis=0)))

        if debug:
            name = output_file.split('.')[0] 
            print name," Tok.sum(), wok.sum(), f21 : ", Tok.sum(), wok.sum(), f21
            np.savetxt(name+"_ps.txt",np.vstack((wav_w,wok,EoverO_w,sci_fPTw.reshape((-1,len(wok))))).T,    \
                fmt="%7.1f %2i "+(1+4*oktargs)*"%8.4f ")

  # intensity: 0.5 * (O1 + E1 + (O2 + E2)/f21)
    stokes_STw[0][wok_Tw] = 0.5 * (sci_fPTw[0][:, wok_Tw].sum(axis=0) +     \
                            sci_fPTw[1][:, wok_Tw].sum(axis=0)/f21)
    var_STw[0][wok_Tw] = 0.25 * (var_fPTw[0][:, wok_Tw].sum(axis=0) +       \
                            var_fPTw[1][:, wok_Tw].sum(axis=0)/f21**2)
    cov_STw[0][wok_Tw] = 0.25 * (cov_fPTw[0][:, wok_Tw].sum(axis=0) +       \
                            cov_fPTw[1][:, wok_Tw].sum(axis=0)/f21**2)

  # stokes difference: defined as int * 0.5 * ( (E1-E2/f21)/(E1+E2/f21) - (O1-O2/f21)/(O1+O2/f21))
    stokes_STw[1, wok_Tw] = 0.5*    \
        ((sci_fPTw[0, 1][wok_Tw] - sci_fPTw[1, 1][wok_Tw]/f21) /    \
                (sci_fPTw[0, 1][wok_Tw] + sci_fPTw[1, 1][wok_Tw]/f21) -  \
         (sci_fPTw[0, 0][wok_Tw] - sci_fPTw[1, 0][wok_Tw]/f21) /    \
                (sci_fPTw[0, 0][wok_Tw] + sci_fPTw[1, 0][wok_Tw]/f21))
    var_STw[1][wok_Tw] = 0.25*  \
        ((var_fPTw[0, 1][wok_Tw] + var_fPTw[1, 1][wok_Tw]/f21**2) /     \
                (sci_fPTw[0, 1][wok_Tw] + sci_fPTw[1, 1][wok_Tw]/f21) ** 2  +   \
         (var_fPTw[0, 0][wok_Tw] + var_fPTw[1, 0][wok_Tw]/f21**2) /     \
                (sci_fPTw[0, 0][wok_Tw] + sci_fPTw[1, 0][wok_Tw]/f21) ** 2)
    cov_STw[1][wok_Tw] = 0.25*  \
        ((cov_fPTw[0, 1][wok_Tw] + cov_fPTw[1, 1][wok_Tw]/f21**2) /     \
                (sci_fPTw[0, 1][wok_Tw] + sci_fPTw[1, 1][wok_Tw]/f21) ** 2  +   \
         (cov_fPTw[0, 0][wok_Tw] + cov_fPTw[1, 0][wok_Tw]/f21**2) /     \
                (sci_fPTw[0, 0][wok_Tw] + sci_fPTw[1, 0][wok_Tw]/f21) ** 2)


    stokes_STw[1] *= stokes_STw[0]
    var_STw[1] *= stokes_STw[0] ** 2
    cov_STw[1] *= stokes_STw[0] ** 2
    bpm_STw = np.array([bpm_Tw, bpm_Tw], dtype='uint8').reshape((2, oktargs, wavs))
    okwerr_Tw = (werr_PTw > 0.).all(axis=0)
    werr_Tw = np.zeros((oktargs,wavs))
    werr_Tw[okwerr_Tw] = np.sqrt((werr_PTw**2).mean(axis=0)[okwerr_Tw])

    hduout = pyfits.PrimaryHDU(header=hduList[0].header)
    hduout = pyfits.HDUList(hduout)
    if wppat:
        hduout[0].header['WPPATERN'] = wppat
    hduout[0].header['EXPTIME'] =  exptime_f.sum()
    hduout[0].header['IMGTIME'] =  imgtime_f.mean()    
    obsDT = obsDT_f[0] + (obsDT_f[1] - obsDT_f[0])/2.    
    hduout[0].header['DATE-OBS'] =  str(obsDT).split('T')[0]
    hduout[0].comments['DATE-OBS'] = 'Date of center of observation'    
    hduout[0].header['UTC-OBS'] =  str(obsDT).split('T')[1]
    hduout[0].comments['UTC-OBS'] = 'UTC of center of observation'
    del hduout[0].header['TIME-OBS']            # obsolete               
    hduout[0].header['TELPA'] =  telpa_f[0]     # 1st hdr is at end of 1st image, close to center
    hduout[0].header['MBY'],hduout[0].header['MBX'] =  tuple(mbyx_df[:,0])
    hduout[0].header['PAIRF21'] = ('%18.6f' % f21)
    header = hduList['SCI'].header.copy()
    header['VAREXT'] =  2
    header['COVEXT'] =  3
    header['BPMEXT'] =  4
    header['WERREXT'] =  5    
    header['CRVAL1'] =  wav0
    header['CTYPE3'] =  'I,S'
    hduout.append( pyfits.ImageHDU( data=stokes_STw, header=header, name='SCI'))
    header.set('SCIEXT', 1, 'Extension for Science Frame', before='VAREXT')
    hduout.append( pyfits.ImageHDU( data=var_STw, header=header, name='VAR'))
    hduout.append( pyfits.ImageHDU( data=cov_STw, header=header, name='COV'))
    hduout.append( pyfits.ImageHDU( data=bpm_STw, header=header, name='BPM'))
    hduout.append( pyfits.ImageHDU( data=werr_Tw, header=header, name='WERR'))    
    if tgthdu:  hduout.append( tgthdu )
    if wcshdu:  hduout.append( wcshdu )

    hduout.writeto(output_file, overwrite=True, output_verify='warn')
    return

# ----------------------------------------------------------
if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)
    if len(kwargs): kwargs = {k:bool(v) for k,v in kwargs.iteritems()}        
    polrawstokes(infileList,**kwargs)

# debug:
# M30
# cd /d/pfis/khn/20161023/sci
# python polsalt.py polrawstokes.py et*011[3-9].fits et*0120.fits

