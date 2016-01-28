#! /usr/bin/env python

"""
diffsum

Convert specpolextract e*.fits file to intensity/diffsum to view E-O diffsum in specpolview

"""

import os, sys, glob, shutil, inspect, copy

import numpy as np
import pyfits

np.set_printoptions(threshold=np.nan)

#---------------------------------------------------------------------------------------------
def diffsum(infilelist):
    files = len(infilelist)

    for file in infilelist:
        hdulist = pyfits.open(file)
        stokes_sw = hdulist['SCI'].data[:,0,:]
        var_sw = hdulist['VAR'].data[:,0,:]
        ok_sw = (hdulist['BPM'].data[:,0,:] == 0)
        wavs = stokes_sw.shape[1]
        
        ok_w = ok_sw.all(axis=0)
        ok_ow = np.tile(ok_w,(2,1))
        sci_ow = np.zeros_like(stokes_sw)
        sci_ow[0,ok_w] = stokes_sw[:,ok_w].sum(axis=0)
        sci_ow[1,ok_w] = stokes_sw[1,ok_w]-stokes_sw[0,ok_w]

        var_ow = np.zeros_like(var_sw)
        var_ow[0,ok_w] = var_sw[:,ok_w].sum(axis=0)      
        var_ow[1,ok_w] = var_ow[0,ok_w]

    # write E+O, (E-O)/(E+O) spectrum, prefix "d". VAR, BPM for each spectrum. y dim is virtual (length 1)
    # for consistency with other modes
        hduout = copy.deepcopy(hdulist)        
        hduout['SCI'].data = sci_ow.astype('float32').reshape((2,1,-1))
        hduout['SCI'].header.update('CTYPE3','E+O,E-O')
        hduout['VAR'].data = var_ow.astype('float32').reshape((2,1,-1))
        hduout['VAR'].header.update('CTYPE3','E+O,E-O')
        hduout['BPM'].data = (~ok_ow).astype('uint8').reshape((2,1,-1))
        hduout['BPM'].header.update('CTYPE3','E+O,E-O')
            
        hduout.writeto('d'+file,clobber=True,output_verify='warn')
        print 'Output file '+'d'+file 
    return
 
if __name__=='__main__':
    infilelist=sys.argv[1:]
    diffsum(infilelist)
