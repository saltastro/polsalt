#! /usr/bin/env python2.7

"""
polhlist

Make a text table of rawstokes *_h*.fits files
Works with multiple-target files

"""

import os, sys

import numpy as np
import pyfits

# ---------------------------------------------------------------------------------
def polhlist(infilelist):
    infiles = len(infilelist)
    hdulist0 = pyfits.open(infilelist[0])

    dwav = hdulist0[0].header['CDELT1']
    label = raw_input('output file label: ')
    for i in range(infiles): 
        if ('_h' not in infilelist[i]):
            print 'Must be all rawstokes files'
            exit()
    wav0_i = np.array([pyfits.getheader(infilelist[i],0)['CRVAL1'] for i in range(infiles)]).astype(float)
    wavs_i = np.array([pyfits.getheader(infilelist[i],'SCI')['NAXIS1'] for i in range(infiles)]).astype(int)

    # listing overlapping wavelengths
    hdul0 =  pyfits.open(infilelist[0])
    dwav = hdul0[0].header['CDELT1']
    wav0 = wav0_i.max()
    col1_j = ((wav0_i - wav0)/dwav).astype(int)
    wavs = (wavs_i - col1_j).min()    
    wav_c = np.linspace(wav0,wav0+wavs*dwav,wavs,endpoint=False)
    targets = hdul0['SCI'].data.shape[1]

    imglist = ['_'.join(os.path.basename(infilelist[i]).split('.')[0].split('_')[-2:]) for i in range(infiles)]
    xtlist = [hdul0[x].name for x in range(len(hdul0))]
    sci_iPtc = np.zeros((infiles,2,targets,wavs),dtype="float32")
    var_iPtc = np.zeros_like(sci_iPtc)
    covar_iPtc = np.zeros_like(sci_iPtc)
    bpm_iPtc = np.zeros((infiles,2,targets,wavs),dtype="uint8")
    iscovar = (xtlist.count('COV') > 0)
    for i in range(infiles):    
        hdul =  pyfits.open(infilelist[i])
        c0 = int((wav0-wav0_i[i])/dwav)        
        sci_iPtc[i] = hdul['SCI'].data[:,:,c0:c0+wavs]
        var_iPtc[i] = hdul['VAR'].data[:,:,c0:c0+wavs]
        if iscovar: covar_iPtc[i] = hdul['COV'].data[:,:,c0:c0+wavs]
        bpm_iPtc[i] = hdul['BPM'].data[:,:,c0:c0+wavs]
    hdr = ('Tgt  '+'Wavl'+infiles*'%15s   ' % tuple(imglist))+('\n'+17*' '+infiles*"I        S        ")
    wav_tc = np.tile(wav_c,targets)
    tgt_tc = np.repeat(np.arange(targets),wavs)
    np.savetxt(label+'sci.txt',np.vstack((tgt_tc,wav_tc,sci_iPtc.reshape((2*infiles,-1)))).T,    \
        fmt="%4i %8.2f "+2*infiles*"%8.0f ",header=hdr)
    np.savetxt(label+'var.txt',np.vstack((tgt_tc,wav_tc,var_iPtc.reshape((2*infiles,-1)))).T,    \
        fmt="%4i %8.2f "+2*infiles*"%8.0f ",header=hdr)
    if iscovar:
        np.savetxt(label+'covar.txt',np.vstack((tgt_tc,wav_tc,covar_iPtc.reshape((2*infiles,-1)))).T,    \
            fmt="%4i %8.2f "+2*infiles*"%8.0f ",header=hdr)
    np.savetxt(label+'bpm.txt',np.vstack((tgt_tc,wav_tc,bpm_iPtc.reshape((2*infiles,-1)))).T,    \
        fmt="%4i %8.2f "+2*infiles*"%1i ",header=hdr)

    return
# ---------------------------------------------------------------------------------
if __name__=='__main__':
    infilelist=sys.argv[1:]
    polhlist(infilelist)
