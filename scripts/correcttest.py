#! /usr/bin/env python2.7

"""
calcorrecttest.py

test polutils correcttest

"""

import os, sys, glob, shutil, inspect
import operator

import numpy as np
from astropy.io import fits as pyfits
from astropy.io import ascii

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
keywordfile = datadir+"obslog_config.json"
sys.path.extend((polsaltdir+'/polsalt/',))
import rsslog
from obslog import create_obslog
from polutils_correcttest import datedfile, datedline, heffcalcorrect

# -------------------------------------
def correcttest(infileList, **kwargs):

    debug = (str(kwargs.pop('debug','False')).lower() == 'true')
    obsDict = create_obslog(infileList,keywordfile)            
    files = len(infileList)
    dateobs = obsDict['DATE-OBS'][0].replace('-','')    
    row0,col0,C0 = np.array(datedline(datadir+"RSSimgalign.txt",dateobs).split()[1:]).astype(float) 
    imgoptfile = datadir+'RSSimgopt.txt'
    distTab = ascii.read(imgoptfile,data_start=1,   \
            names=['Wavel','Fcoll','Acoll','Bcoll','ydcoll','xdcoll','Fcam','acam','alfydcam','alfxdcam'])
    FColl6000 = distTab['Fcoll'][list(distTab['Wavel']).index(6000.)]
    FCam6000 = distTab['Fcam'][list(distTab['Wavel']).index(6000.)]
    tassess = 0
               
    for f,file in enumerate(infileList):
        hdul = pyfits.open(file)        
        stokes_Sw = hdul['SCI'].data[:,tassess,:]
        var_Sw = hdul['VAR'].data[:,tassess,:]        
        bpm_Sw = hdul['BPM'].data[:,tassess,:]
        ok_w = (bpm_Sw ==0).all(axis=0)
        wav0 = hdul['SCI'].header['CRVAL1']
        dwav = hdul['SCI'].header['CDELT1']
        wavs = hdul['SCI'].header['NAXIS1']
        wav_w = wav0 + dwav*np.arange(wavs)
        maskid = obsDict['MASKID'][f]
        maskasec = float(maskid[2:6])/100.
        slitdwav = (C0*maskasec*(FCam6000/FColl6000)/15.)*(wavs*dwav/(3*2048))            
        stokescor_Sw = heffcalcorrect("calcorrect.txt",wav_w,stokes_Sw,var_Sw,ok_w,slitdwav=slitdwav,debug=debug)

        outfile = file.rsplit('_',1)[0]+'_cor_stokes.fits'
        hdul['SCI'].data[:,tassess,:] = stokescor_Sw
        hdul.writeto(outfile,overwrite=True,output_verify='warn')

    return

# -------------------------------------
if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)    
    correcttest(infileList,**kwargs)
