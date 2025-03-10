
"""
bpmdiff

Make an image showing difference betwwen two bpms for same spectrum

"""

import os, sys

import numpy as np
import pyfits

# ---------------------------------------------------------------------------------
def bpmdiff(infile1,infile2):

    hdulist1 =  pyfits.open(infile1)
    hdulist2 =  pyfits.open(infile2)

    sci_rc = hdulist1['SCI'].data
    bpm1_rc = hdulist1['BPM'].data
    bpm2_rc = hdulist2['BPM'].data

    bpm1m2_rc = ((bpm1_rc==1) & (bpm2_rc==0)).astype('uint8')
    bpm2m1_rc = ((bpm2_rc==1) & (bpm1_rc==0)).astype('uint8')

    hduout = pyfits.PrimaryHDU(header=hdulist1[0].header)
    hduout = pyfits.HDUList(hduout)
    header = hdulist1['SCI'].header.copy()
    hduout.append( pyfits.ImageHDU(data=sci_rc, header=header, name='SCI'))
    hduout.append( pyfits.ImageHDU(data=bpm1m2_rc, header=header, name='BPM1'))
    hduout.append( pyfits.ImageHDU(data=bpm2m1_rc, header=header, name='BPM2'))

    outfile = infile1.split(".fits")[0]+"bpmdiff.fits"
    hduout.writeto(outfile, clobber=True, output_verify='warn')
    print outfile,' BPM difference'

    return
# ---------------------------------------------------------------------------------
if __name__=='__main__':
    infile1=sys.argv[1]
    infile2=sys.argv[2]
    bpmdiff(infile1,infile2)
