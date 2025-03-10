#! /usr/bin/env python2.7

"""
polelist

Make a text table of e*.fits files
Works with multiple-target files

"""

import os, sys

import numpy as np
import pyfits

# ---------------------------------------------------------------------------------
def polelist(infilelist):
    infiles = len(infilelist)
    hdulist0 = pyfits.open(infilelist[0])

    dwav = hdulist0[0].header['CDELT1']
    label = raw_input('output file label: ')
    for i in range(infiles): 
        if infilelist[i][0] != 'e':
            print 'Must be all e*.fits files'
            exit()
    wav0_i = np.array([pyfits.getheader(infilelist[i],0)['CRVAL1'] for i in range(infiles)]).astype(float)
    wavs_i = np.array([pyfits.getheader(infilelist[i],'SCI')['NAXIS1'] for i in range(infiles)]).astype(int)

    # listing overlapping wavelengths
    hdul0 =  pyfits.open(infilelist[0])
    dwav = hdul0[0].header['CDELT1']
    wav0 = wav0_i.max()
#    col1_i = ((wav0_i - wav0)/dwav).astype(int)
    col1_i = ((wav0 - wav0_i)/dwav).astype(int)
    wavs = (wavs_i - col1_i).min()
    wav_c = np.linspace(wav0,wav0+wavs*dwav,wavs,endpoint=False)
    targets = hdul0['SCI'].data.shape[1]

    imglist = [os.path.basename(infilelist[i]).split('.')[0][-4:] for i in range(infiles)]
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
    sci_iPtc *= (bpm_iPtc ==0)
    hdr = ('Tgt  '+'Wavl'+infiles*'%15s   ' % tuple(imglist))+('\n'+17*' '+infiles*"O        E        ")
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
    polelist(infilelist)
