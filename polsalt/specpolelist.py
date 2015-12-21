
"""
specpolelist

Make a text table of e*.fits files

"""

import os, sys

import numpy as np
import pyfits

# ---------------------------------------------------------------------------------
def specpolelist(infilelist):
    infiles = len(infilelist)
    hdulist0 = pyfits.open(infilelist[0])
    cols = hdulist0['SCI'].data.shape[-1]
    wav0 = hdulist0['SCI'].header['CRVAL1']
    dwav = hdulist0['SCI'].header['CDELT1']
    label = raw_input('output file label: ')
    for i in range(infiles): 
        if infilelist[i][0] != 'e':
            print 'Must be all e*.fits files'
            exit()
        if pyfits.open(infilelist[i])['SCI'].data.shape[-1] != cols:
            print 'Files must be all same shape'
            exit()
    wav_c = np.linspace(wav0,wav0+cols*dwav,cols,endpoint=False)
    imglist = [os.path.basename(infilelist[i]).split('.')[0][-4:] for i in range(infiles)]
    sci_ioc = np.zeros((infiles,2,cols),dtype="float32")
    var_ioc = np.zeros_like(sci_ioc)
    bpm_ioc = np.zeros((infiles,2,cols),dtype="uint8")
    for i in range(infiles):    
        hdulist =  pyfits.open(infilelist[i])       
        sci_ioc[i] = hdulist['SCI'].data.reshape((2,cols))
        var_ioc[i] = hdulist['VAR'].data.reshape((2,cols))
        bpm_ioc[i] = hdulist['BPM'].data.reshape((2,cols))
    hdr = ('Wavl'+infiles*"%15s   " % tuple(imglist))+('\n            '+infiles*"O        E        ")

    np.savetxt(label+'sci.txt',np.vstack((wav_c,sci_ioc.reshape((2*infiles,-1)))).T,    \
        fmt="%8.2f "+2*infiles*"%8.0f ",header=hdr)
    np.savetxt(label+'var.txt',np.vstack((wav_c,var_ioc.reshape((2*infiles,-1)))).T,    \
        fmt="%8.2f "+2*infiles*"%8.0f ",header=hdr)
    np.savetxt(label+'bpm.txt',np.vstack((wav_c,bpm_ioc.reshape((2*infiles,-1)))).T,    \
        fmt="%8.2f "+2*infiles*"%1i ",header=hdr)

    return
# ---------------------------------------------------------------------------------
if __name__=='__main__':
    infilelist=sys.argv[1:]
    specpolelist(infilelist)
