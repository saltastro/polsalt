#! /usr/bin/env python

"""
polmask

Mask out wavelengths in stokes.fits data, including MOS. 

"""

import os, sys, glob, inspect
import numpy as np
from astropy.io import fits as pyfits
from astropy.table import Table

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

import rsslog
from obslog import create_obslog
keywordprifile = datadir+"obslog_config.json"
keywordsecfile = datadir+"obslogsec_config.json"

np.set_printoptions(threshold=np.nan)
#import warnings 
#warnings.filterwarnings("error") 

#---------------------------------------------------------------------------------------------
def polmask(infileList, Wlim_md, **kwargs):
    """difference (normalized) (possibly MOS) stokes files

    Parameters
    ----------
    infileList: list
       one or more _stokes.fits files
    Wlim_md: 2d ndarray: m pairs of Wmin,Wmax 

    outname: output file will be *_wmask_stokes.fits

    """
    """
    _b observations
    _w wavelengths in individual observations
    _m idx for multiple masks

    """    

    logfile= kwargs.pop('logfile','polmask.log')     
    debug = (kwargs.pop('debug','False') == 'True')
    
    rsslog.history(logfile)
               
    obss = len(infileList)
    masks = Wlim_md.shape[0]        
    obsDict0 = create_obslog(infileList,keywordprifile)    
    obsDictSCI = create_obslog(infileList,keywordsecfile,ext='SCI')
    dwav_b = np.array(obsDictSCI['CDELT1'])
    wav0_b = np.array(obsDictSCI['CRVAL1'])
    wavs_b = np.array(obsDictSCI['NAXIS1'] )       
    
    rsslog.message("\nMasking files:",logfile)
    
    for b in range(obss):
        rsslog.message(infileList[b],logfile)
        hdul = pyfits.open(infileList[b],ignore_missing_end=True)
        W_w =  np.linspace(wav0_b[b],wav0_b[b]+dwav_b[b]*wavs_b[b],num=wavs_b[b]) 
        Wmask_mw = ((W_w[None,:] >= Wlim_md[:,0][:,None]) & (W_w[None,:] <= Wlim_md[:,1][:,None]))
                 
        hdul['BPM'].data = ((hdul['BPM'].data==1) | Wmask_mw.any(axis=0)[None,None,:]).astype('uint8')

        for m in range(masks):
            outwmin = W_w[Wmask_mw[m]][0]
            outwmax = W_w[Wmask_mw[m]][-1]        
            hdul[0].header.add_history('Polmask: %7.2f %7.2f' % (outwmin,outwmax))
            
        outfile = infileList[b].rsplit("_",1)[0]+"_wmask_"+infileList[b].rsplit("_",1)[1]                
        rsslog.message("\nOutput: "+outfile,logfile)

        hdul.writeto(outfile,overwrite=True)
    
    return

#--------------------------------------
# for cl script, wlim_md is entered as comma-separated pairs 
if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    Wlim_md =np.vstack((np.array(x.split(',', 1)).astype(float) for x in sys.argv[1:] if ((x.count(',')==1))))
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if ((x.count('=')==1)))
    polmask(infileList, Wlim_md, **kwargs)

# current:
# cd /d/pfis/khn/20220822/sci
# python script.py polmask.py ET21B_c0_1_rawstokes.fits 7500,10000

