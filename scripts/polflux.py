
"""
polflux

in current directory, save fluxcal table for flux standards in ec* files
if an appropriate fluxcal file now exists, apply to listed _stokes.fits files 

"""

import os
import sys
import glob
import copy
import shutil
import inspect

import numpy as np
from astropy.io import fits as pyfits
from astropy.table import Table
from scipy.interpolate import interp1d

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
keywordfile = datadir+"obslog_config.json"
sys.path.extend((polsaltdir+'/polsalt/',))

# this is pysalt-free

import rsslog
from obslog import create_obslog
from scrunch1d import scrunch1d
from polutils import configmap

np.set_printoptions(threshold=np.nan)
debug = True

def polflux(infilelist, logfile='salt.log', debug=False, with_stdout=True):
    """Finds/ produces fluxcal tables, and applies to listed _stokes.fits

    Parameters
    ----------
    infilelist: list
        filename or list of finalstokes filenames 

    logfile: str
        Name of file for logging

    """
    if len(glob.glob('specpol*.log')): logfile=glob.glob('specpol*.log')[0]
    fluxcal_w = np.zeros(0)
  # Info on CAL_SPST files:
    calspstname_s,calspstfile_s=np.loadtxt(datadir+"spst_filenames.txt",    \
        dtype=str,usecols=(0,1),unpack=True)
    namelen = max(map(len,calspstname_s))

    confitemList = ['DATE-OBS','GRATING','GR-ANGLE','CAMANG']

  # Find fluxdb files already in this directory
    fluxdbtab = Table(names=['no','OBJECT']+confitemList,        \
               dtype=[int,'S'+str(namelen),'S10','S6',float,float])
    fluxdblist = sorted(glob.glob('fluxdb*.txt'))
    goodfluxdblist = copy.copy(fluxdblist)

    for e,dbfile in enumerate(fluxdblist):
        if (open(dbfile).read().count("#") < 5):
            rsslog.message('\n    Invalid flux file '+dbfile+', not used', logfile, with_stdout=with_stdout)
            goodfluxdblist.remove(dbfile)
            continue       
        confdat_d = np.genfromtxt(dbfile,usecols=2,comments='?', \
                max_rows=5,dtype=str)
        fluxdbtab.add_row(np.insert(confdat_d,0,e+1))

    fluxdblist = goodfluxdblist    
    olddbentries = len(fluxdblist)
    if olddbentries: 
        rsslog.message('\n    Existing Fluxdb:\n'+str(fluxdbtab), logfile, with_stdout=with_stdout)

  # Create new fluxdb files if new data present
    eclist = sorted(glob.glob('ec*.fits'))
    obss = 0
    if len(eclist): 
        obs_i,config_i,obstab,configtab = configmap(eclist,confitemList)
        obss = len(obstab)

    for obs in range(obss):
        object,config = obstab[obs]

      # Find spst standards
        i_j = np.where(obs_i == obs)[0]
        if object not in calspstname_s: continue

        newfluxdbtab = Table( \
            names=['no','OBJECT']+confitemList,dtype=[int,'S'+str(namelen),'S10','S6',float,float])
        newfluxdbtab.add_row([len(fluxdbtab)+1,object]+list(configtab[config]))       
        if Table(newfluxdbtab.columns[1:]) in Table(fluxdbtab.columns[1:]): continue

      # It's a new flux standard, process it
        rsslog.message('\n    New Fluxdb entry:\n'+str(newfluxdbtab), logfile, with_stdout=with_stdout)
        s = np.where(object==calspstname_s)[0][0]
        spstfile=iraf.osfn("pysalt$data/standards/spectroscopic/"+calspstfile_s[s])
        wav_f,ABmag_f = np.loadtxt(spstfile,usecols=(0,1),unpack=True)
        wbinedge_f = (wav_f[1:] + wav_f[:-1])/2.
        wbinedge_f = np.insert(wbinedge_f,0,2.*wav_f[0]-wbinedge_f[0])
        wbinedge_f = np.append(wbinedge_f,2.*wav_f[-1]-wbinedge_f[-1])
        flam_f = 10.**(-0.4*(ABmag_f+2.402))/(wav_f)**2

      # for HST standards, scrunch down to 50A bins
        if (wav_f[0] < 3000.):
            wbinedge_F = np.arange(3000.,11000.,50.)
            binedge_F = interp1d(wbinedge_f,np.arange(wbinedge_f.shape[0]))(wbinedge_F)
            flam_F = scrunch1d(flam_f,binedge_F)/np.diff(binedge_F)       # mean over new flux bin
            wav_f = (wbinedge_F[:-1] + wbinedge_F[1:])/2.
            flam_f = flam_F
            wbinedge_f = wbinedge_F
      
      # average multiple samples,E,O
        hdul = pyfits.open(eclist[i_j[0]])
        dateobs,grating,grang,artic = configtab[config]
        wav0 = hdul['SCI'].header['CRVAL1']
        dwav = hdul['SCI'].header['CDELT1']
        wavs = hdul['SCI'].data.shape[-1]
        phot_w = np.zeros(wavs)
        count_w = np.zeros(wavs)
        exptime = 0.
        samples = i_j.shape[0]
        for j in range(samples):
            hdul = pyfits.open(eclist[i_j[j]])
            phot_w += hdul['SCI'].data.reshape((2,-1)).sum(axis=0)                
            count_w += (hdul['BPM'].data.reshape((2,-1))==0).sum(axis=0)
            exptime +=  hdul['SCI'].header['EXPTIME']  
        int_w = phot_w/exptime                                  # phot/sec/bin, E+O sum
        ok_w = (count_w == 2*samples)

      # check for gain corrections. BPM==2 marks internal ccd amp intersections
        aw_pA = np.array(np.where(hdul['BPM'].data.reshape((2,-1)) == 2))[1].reshape((2,-1))
        awmin_A,aw_A,awmax_A = (aw_pA.min(axis=0), aw_pA.mean(axis=0), aw_pA.max(axis=0))
        wloList = [0, awmax_A[0]+1, aw_A[[0,1]].mean(), awmax_A[1]+1, aw_A[[1,2]].mean(), awmax_A[2]+1]
        whiList = [awmin_A[0]-1, aw_A[[0,1]].mean(), awmin_A[1]-1, aw_A[[1,2]].mean(), awmin_A[2]-1,wavs]
        wList = [0,aw_A[0],aw_A[[0,1]].mean(),aw_A[1],aw_A[[1,2]].mean(),aw_A[2],wavs]
        photedge_da = np.zeros((2,6))
        for d,a in np.ndindex(2,6):
            w1 = wloList[a] + d*(whiList[a]-wloList[a])*2/3
            w2 = whiList[a] - (1-d)*(whiList[a]-wloList[a])*2/3
            use_w = (ok_w & (np.arange(wavs) >= w1) & (np.arange(wavs) <= w2))
            cof_c = np.polyfit(np.arange(wavs)[use_w],phot_w[use_w],1)
            if debug: print d,a,('%8.2f %8.0f' % tuple(cof_c))
            photedge_da[d,a] = np.polyval(cof_c,wList[a+d])
        photrat_A = photedge_da[0,1:]/photedge_da[1,:-1]
        photrat_a = np.insert(np.cumprod(photrat_A),0,1.)
        historyDict = dict([line.split(' ',1) for line in hdul['SCI'].header['HISTORY'] ])
        if historyDict.has_key('GainCorrection:'):
            rsslog.message(('\n    Gain cors : '+historyDict['GainCorrection:']), logfile, with_stdout=with_stdout)
        else:
            rsslog.message(('\n    no gain correction'), logfile, with_stdout=with_stdout)
        rsslog.message(('    Gain Ratio:          '+6*'%6.4f ' % tuple(photrat_a)), logfile, with_stdout=with_stdout)

      # scrunch onto flux star grid
        wav_w = np.arange(wav0,wav0+wavs*dwav,dwav)
        binedge_f = (wbinedge_f - (wav_w[0] - dwav/2.))/dwav
        int_f = scrunch1d(int_w,binedge_f)/np.diff(binedge_f)       # mean of int_w over flux bin
    
        ok_f = (scrunch1d((~ok_w).astype(int),binedge_f) == 0)      # good flux bins have no bad wavs
        ok_f &= ((wav_f > wav_w[0]) & (wav_f < wav_w[-1]))

      # save flux/intensity mean over flux standard bin, extrapolate to edge
        fluxcal_F = flam_f[ok_f]/int_f[ok_f]
        wav_F = wav_f[ok_f]
        fluxcalslope_F = (fluxcal_F[1:]-fluxcal_F[:-1])/(wav_F[1:]-wav_F[:-1])
        wav_F = np.insert(wav_F,0,wav_w[0])
        wav_F = np.append(wav_F,wav_w[-1])
        fluxcal_F = np.insert(fluxcal_F,0,fluxcal_F[0]-fluxcalslope_F[0]*(wav_F[1]-wav_F[0]))
        fluxcal_F = np.append(fluxcal_F,fluxcal_F[-1]+fluxcalslope_F[-1]*(wav_F[-1]-wav_F[-2]))
        fluxdbfile = 'fluxdb_'+calspstname_s[s]+'_c'+str(config)+'.txt'
        hdr = ("OBJECT: "+object+"\nDATEOBS: %s \nGRATING: %s \nARTIC: %s \nGRANG: %s" \
            % (dateobs,grating,grang,artic))
        np.savetxt(fluxdbfile, np.vstack((wav_F,fluxcal_F)).T,fmt="%8.2f %12.3e ",header=hdr)
        fluxdbtab.add_row(list(newfluxdbtab[0]))
        fluxdblist.append(fluxdbfile)
    dbentries = len(fluxdbtab)
                
  # do fluxcal on listed stokes.fits files
    if len(fluxdbtab)==0:
        rsslog.message('\n    No fluxdb data available', logfile, with_stdout=with_stdout)
        return fluxcal_w

    if (type(infilelist) is str): infilelist = [infilelist,]
    if len(infilelist)==0:
        print "No files to calibrate"
        exit()
        
    obs_i,config_i,obstab,configtab = configmap(infilelist,confitemList)
    obss = len(obstab)
    fluxdbconftab = fluxdbtab[confitemList]

    cunitfluxed = 'erg/s/cm^2/Ang'          # header keyword CUNIT3 if data is already fluxed     
    for obs in range(obss):
        object,config = obstab[obs]    
        iobs = np.where(obs_i == obs)[0][0]
        hdul = pyfits.open(infilelist[iobs])
        if 'CUNIT3' in hdul['SCI'].header:
            if hdul['SCI'].header['CUNIT3'].replace(' ','') ==cunitfluxed:
                rsslog.message(('\n    %s already flux calibrated' % infilelist[iobs]), logfile, with_stdout=with_stdout)
                continue 
        
        fluxdbentry_e = []
        for e in range(len(fluxdbconftab)):
            if ((fluxdbconftab[e]['GRATING']==configtab[config]['GRATING']) &  \
                (fluxdbconftab[e]['CAMANG']==configtab[config]['CAMANG'])  &  \
                (abs(fluxdbconftab[e]['GR-ANGLE']-configtab[config]['GR-ANGLE']) < 0.1)):
                fluxdbentry_e.append(e)
        if len(fluxdbentry_e) == 0:
            rsslog.message(('\n    No flux calibration available for  %s' % infilelist[iobs]), logfile, with_stdout=with_stdout)
            continue

        wav0 = hdul['SCI'].header['CRVAL1']
        dwav = hdul['SCI'].header['CDELT1']
        wavs = hdul['SCI'].data.shape[-1]
        exptime =  hdul['SCI'].header['EXPTIME']
        wav_w = np.arange(wav0,wav0+wavs*dwav,dwav)            
        fluxcal_w = np.zeros(wavs)

      # average all applicable fluxdb entries after interpolation onto wavelength grid
      # if necessary, block average onto ~50Ang grid, then average onto 50 Ang grid
      # interpolate onto configuration for fluxcal                                                     
        fluxcallog = ''
        for e in fluxdbentry_e:
            wav_F,fluxcal_F = np.loadtxt(fluxdblist[e],skiprows=5,comments='?', \
                dtype=float,unpack=True)
            fluxcal_w += interp1d(wav_F,fluxcal_F,bounds_error=False)(wav_w)
            hdul[0].header.add_history("FluxCal: "+fluxdbconftab[e]["DATE-OBS"]+' '+fluxdblist[e])
            fluxcallog += ('\n    '+str(e+1)+' '+fluxdbconftab[e]["DATE-OBS"]+' '+fluxdblist[e])
        fluxcal_w /= len(fluxdbentry_e)
        fluxcal_w = (np.nan_to_num(fluxcal_w))/exptime
        hdul['SCI'].data *= fluxcal_w
        hdul['SCI'].header['CUNIT3'] = cunitfluxed
        hdul['VAR'].data *= fluxcal_w**2
        hdul['VAR'].header['CUNIT3'] = cunitfluxed
        hdul['COV'].data *= fluxcal_w**2
        hdul['COV'].header['CUNIT3'] = cunitfluxed
        hdul['BPM'].data = ((hdul['BPM'].data > 0) | (fluxcal_w ==0.)).astype('uint8')
        hdul.writeto(infilelist[iobs],overwrite=True)

        rsslog.message((('\n    %s Fluxcal:'+fluxcallog) % infilelist[iobs]), logfile, with_stdout=with_stdout)

    return fluxcal_w

# ----------------------------------------------------------

if __name__=='__main__':
    infilelist=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)
    if len(kwargs): kwargs = {k:bool(v) for k,v in kwargs.iteritems()}        
    polflux(infilelist,**kwargs)
