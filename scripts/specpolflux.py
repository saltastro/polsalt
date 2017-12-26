
"""
specpolflux

in current directory, save fluxcal table for flux standards in ec* files
if an appropriate fluxcal file now exists, apply to listed _stokes.fits files 

"""

import os
import sys
import glob
import shutil
import inspect

import numpy as np
from astropy.io import fits as pyfits
from astropy.table import Table
from scipy.interpolate import interp1d

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging
from scrunch1d import scrunch1d
from specpolutils import configmap

np.set_printoptions(threshold=np.nan)
debug = True

def specpolflux(infilelist, logfile='salt.log', debug=False):
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

    confitemlist = ['GRATING','GR-ANGLE','CAMANG']

  # Find fluxdb files already in this directory
    fluxdbtab = Table(names=['no','OBJECT']+confitemlist,        \
               dtype=[int,'S'+str(namelen),'S6',float,float])
    fluxdblist = sorted(glob.glob('fluxdb*.txt'))
    olddbentries = len(fluxdblist)
    for e,dbfile in enumerate(fluxdblist):        
        confdat_d = np.genfromtxt(dbfile,usecols=2,comments='?', \
                max_rows=4,dtype=str)
        fluxdbtab.add_row(np.insert(confdat_d,0,e+1))
    if olddbentries: 
        printstdlog('\n    Existing Fluxdb:\n'+str(fluxdbtab), logfile)

  # Create new fluxdb files if new data present
    eclist = sorted(glob.glob('ec*.fits'))
    obs_i,config_i,obstab,configtab = configmap(eclist,confitemlist)
    obss = len(obstab)

    for obs in range(obss):
        object,config = obstab[obs]

      # Find spst standards
        i_j = np.where(obs_i == obs)[0]
        if object not in calspstname_s: continue

        newfluxdbtab = Table( \
            names=['no','OBJECT']+confitemlist,dtype=[int,'S'+str(namelen),'S6',float,float])
        newfluxdbtab.add_row([len(fluxdbtab)+1,object]+list(configtab[obs]))       
        if Table(newfluxdbtab.columns[1:]) in Table(fluxdbtab.columns[1:]): continue

      # It's a new flux standard, process it
        s = np.where(object==calspstname_s)[0][0]

        spstfile=iraf.osfn("pysalt$data/standards/spectroscopic/"+calspstfile_s[s])
        wav_f,ABmag_f = np.loadtxt(spstfile,usecols=(0,1),unpack=True)
        flam_f = 10.**(-0.4*(ABmag_f+2.402))/(wav_f)**2

        wbinedge_f = (wav_f[1:] + wav_f[:-1])/2.
        wbinedge_f = np.insert(wbinedge_f,0,2.*wav_f[0]-wbinedge_f[0])
        wbinedge_f = np.append(wbinedge_f,2.*wav_f[-1]-wbinedge_f[-1])
      
      # average multiple samples,E,O
        hdul = pyfits.open(eclist[i_j[0]])
        grating,grang,artic = configtab[config]
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
        int_w = phot_w/(2*samples*exptime)
        ok_w = (count_w == 2*samples)
      # scrunch onto flux star grid
        wav_w = np.arange(wav0,wav0+wavs*dwav,dwav)
        binedge_f = (wbinedge_f - (wav_w[0] - dwav/2.))/dwav
        int_f = scrunch1d(int_w,binedge_f)
        ok_f = (scrunch1d((~ok_w).astype(int),binedge_f) == 0)   # good flux bins have no bad wavs
        ok_f &= ((wav_f > wav_w[0]) & (wav_f < wav_w[-1]))

      # save flux/intensity mean, extrapolate to edge
        fluxcal_F = flam_f[ok_f]/int_f[ok_f]
        wav_F = wav_f[ok_f]
        fluxcalslope_F = (fluxcal_F[1:]-fluxcal_F[:-1])/(wav_F[1:]-wav_F[:-1])
        wav_F = np.insert(wav_F,0,wav_w[0])
        wav_F = np.append(wav_F,wav_w[-1])
        fluxcal_F = np.insert(fluxcal_F,0,fluxcal_F[0]-fluxcalslope_F[0]*(wav_F[1]-wav_F[0]))
        fluxcal_F = np.append(fluxcal_F,fluxcal_F[-1]+fluxcalslope_F[-1]*(wav_F[-1]-wav_F[-2]))
        fluxdbfile = 'fluxdb_'+calspstname_s[s]+'_c'+str(config)+'.txt'
        hdr = ("OBJECT: "+object+"\nGRATING: %s \nARTIC: %s \nGRANG: %s" \
            % (grating,grang,artic))
        np.savetxt(fluxdbfile, np.vstack((wav_F,fluxcal_F)).T,fmt="%8.2f %12.3e ",header=hdr)
        fluxdbtab.add_row(list(newfluxdbtab[0]))
        fluxdblist.append(fluxdbfile)
    dbentries = len(fluxdbtab)
    if (dbentries>olddbentries):
        printstdlog('\n    New Fluxdb entries:\n'+str(fluxdbtab[olddbentries:]), logfile)
                
  # do fluxcal on listed stokes.fits files
    if len(fluxdbtab)==0:
        printstdlog('\n    No fluxdb data available', logfile)
        return fluxcal_w

    if (type(infilelist) is str): infilelist = [infilelist,]
        
    obs_i,config_i,obstab,configtab = configmap(infilelist,confitemlist)
    obss = len(obstab)
    fluxdbconftab = fluxdbtab[confitemlist]

    cunitfluxed = 'erg/s/cm^2/Ang'          # header keyword CUNIT3 if data is already fluxed     
    for obs in range(obss):
        iobs = np.where(obs_i == obs)[0][0]
        hdul = pyfits.open(infilelist[iobs])
        if 'CUNIT3' in hdul['SCI'].header:
            if hdul['SCI'].header['CUNIT3'].replace(' ','') ==cunitfluxed:
                printstdlog(('\n    %s already flux calibrated' % infilelist[iobs]), logfile)
                continue 
        
        fluxdbentry_e = []
        for e in range(len(fluxdbconftab)):
            if ((fluxdbconftab[e]['GRATING']==configtab[obs]['GRATING']) &  \
                (fluxdbconftab[e]['CAMANG']==configtab[obs]['CAMANG'])  &  \
                (abs(fluxdbconftab[e]['GR-ANGLE']-configtab[obs]['GR-ANGLE']) < 0.1)):
                fluxdbentry_e.append(e)
        if len(fluxdbentry_e) == 0:
            printstdlog(('\n    No flux calibration available for  %s' % infilelist[iobs]), logfile)
            continue

        wav0 = hdul['SCI'].header['CRVAL1']
        dwav = hdul['SCI'].header['CDELT1']
        wavs = hdul['SCI'].data.shape[-1]
        wav_w = np.arange(wav0,wav0+wavs*dwav,dwav)            
        fluxcal_w = np.zeros(wavs)

      # average all applicable fluxdb entries after interpolation onto wavelength grid
        fluxcalhistory = "FluxCal: "                                                        
        fluxcallog = fluxcalhistory
        for e in fluxdbentry_e:
            wav_F,fluxcal_F = np.loadtxt(fluxdblist[e],skiprows=4,comments='?', \
                dtype=float,unpack=True)
            fluxcal_w += interp1d(wav_F,fluxcal_F,bounds_error=False)(wav_w)
            fluxcalhistory += " "+fluxdblist[e]
            fluxcallog += "  "+str(e+1)+" "+fluxdblist[e]
        fluxcal_w /= len(fluxdbentry_e)
        fluxcal_w = (np.nan_to_num(fluxcal_w))
        hdul['SCI'].data *= fluxcal_w
        hdul['SCI'].header['CUNIT3'] = cunitfluxed
        hdul['VAR'].data *= fluxcal_w**2
        hdul['VAR'].header['CUNIT3'] = cunitfluxed
        hdul['COV'].data *= fluxcal_w**2
        hdul['COV'].header['CUNIT3'] = cunitfluxed
        hdul[0].header.add_history(fluxcalhistory)
        hdul.writeto(infilelist[iobs],overwrite=True)

        printstdlog((('\n    %s '+fluxcallog) % infilelist[iobs]), logfile)

    return fluxcal_w
# ----------------------------------------------------------
def printstdlog(string,logfile):
    print string
    print >>open(logfile,'a'), string
    return

# ----------------------------------------------------------

if __name__=='__main__':
    infilelist=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)
    if len(kwargs): kwargs = {k:bool(v) for k,v in kwargs.iteritems()}        
    specpolflux(infilelist,**kwargs)
