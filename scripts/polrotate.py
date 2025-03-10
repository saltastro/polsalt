#! /usr/bin/env python

"""
polrotate

Rotate stokes.fits data, possibly multi-target

"""

import os, sys, glob, shutil, inspect, datetime, operator
import numpy as np

import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter("error")

from astropy.io import fits as pyfits
from astropy.io import ascii
polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

import rsslog
from obslog import create_obslog
from polutils import specpolrotate, viewstokes
keywordprifile = datadir+"obslog_config.json"
keywordsecfile = datadir+"obslogsec_config.json"
np.set_printoptions(threshold=np.nan)

#---------------------------------------------------------------------------------------------
def polrotate(dpa, fileList, **kwargs):
    """Rotate (possibly MOS) stokes.fits files

    Parameters
    ----------
    fileList: list of text filenames
       stokes fits files
    dpa: float PA change (deg) for all targets in all files 
    debug=: False (default)
            True (debug output)
    """
    """
  # _f file counter
  # _p (0,1) calibration file (P,PA, or Q,U)
  # _t target index in each file
  # _S (unnormalized) stokes: (I,Q,U)

    """
    logfile= kwargs.pop('logfile','sppolrotate.log')              
    rsslog.history(logfile)

    debug = (kwargs.pop('debug','False') == 'True')    
    obss = len(infileList)
    obsDict0 = create_obslog(infileList,keywordprifile)
    
    for b in range(obss):
        rsslog.message("\nRotating targets in file: "+infileList[b],logfile)
        hdul = pyfits.open(infileList[b],ignore_missing_end=True)
        stokes_Stw = hdul['SCI'].data
        var_Stw = hdul['VAR'].data     
        cov_Stw = hdul['COV'].data            
        targets = stokes_Stw.shape[1]
        for t in range(targets):                
            stokes_Stw[:,t],var_Stw[:,t],cov_Stw[:,t] =   \
                specpolrotate(stokes_Stw[:,t],var_Stw[:,t],cov_Stw[:,t],dpa)    
        outfile = infileList[b].rsplit("_",1)[0]+"_rot_"+infileList[b].rsplit("_",1)[1]
        rsslog.message("Output: "+outfile,logfile)
        hdul.writeto(outfile,overwrite=True)
    
    return        

#--------------------------------------
 
if __name__=='__main__':
    dpa = float(sys.argv[1])
    infileList=[x for x in sys.argv[2:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[2:] if x.count('.fits')==0)
    polrotate(dpa,infileList,**kwargs)
