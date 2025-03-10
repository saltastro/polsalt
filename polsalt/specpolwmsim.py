
"""
specpolwmsim

Overwrite a normal noise simulation signal into wm*.fits polarimetry files

"""

import os, sys

import numpy as np
from astropy.io import fits as pyfits

# ---------------------------------------------------------------------------------
def specpolwmsim(filelist):
    files = len(filelist)
    hdul0 = pyfits.open(filelist[0])
    rows,cols = hdul0['SCI'].data.shape[1:]

    # first create a master wm file by averaging them all
    f_prc = np.zeros((2,rows,cols),dtype='float32')
    v_prc = np.zeros_like(f_prc)
    std_prc = np.zeros_like(f_prc)
    ok_prc = np.ones_like(f_prc,dtype=bool)

    for file in filelist: 
        if file[:2] != 'wm':
            print 'Must be all wm*.fits files'
            exit()
        hdul = pyfits.open(file)
        f_prc += hdul['SCI'].data/files
        v_prc += hdul['VAR'].data/files
        ok_prc &= (hdul['BPM'].data == 0)
    std_prc[ok_prc] = np.sqrt(v_prc[ok_prc])

    np.random.seed(42)        
    for file in filelist:    
        hdul = pyfits.open(file)
        hdul['SCI'].data = np.random.normal(f_prc,std_prc).astype('float32')
        hdul['VAR'].data = v_prc
        hdul['BPM'].data = (ok_prc == False).astype('uint8')
        hdul.writeto(file,clobber=True)      

    return
# ---------------------------------------------------------------------------------
if __name__=='__main__':
    filelist=sys.argv[1:]
    specpolwmsim(filelist)
