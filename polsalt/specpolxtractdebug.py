
"""
specpolelist

Make a text table of extract data from c*.fits files

"""

import os, sys

import numpy as np
import pyfits

# ---------------------------------------------------------------------------------
def specpolxtractdebug(ro,re,infilelist):

    rowlist_a = np.empty(3,dtype=object)
    bins_a = np.zeros(3)
    crs_a = np.zeros(3)

    filelen = len(infilelist[0])
    print "Extraction details: "
    print (('%'+str(filelen)+'s') % "File    "),'O/E   b1   s    b2    b1rate     srate    b2rate'

    for file in infilelist:
        if file[0] != 'c':
            print 'file is not a c*.fits file'
            continue
        image = os.path.basename(file).split('.')[0][-4:]
        hdulist =  pyfits.open(file)
        cbin, rbin = [int(x) for x in hdulist[0].header['CCDSUM'].split(" ")]
        exptime = float(hdulist[0].header['EXPTIME'])
        row_p = np.array([ro,re])        
        row_px = np.add.outer(row_p,(8/rbin)*np.array([-35,-25,-5,5,25,35]))
        rows_a = (8/rbin)*np.array([10,10,10])

        wav_prc = hdulist['WAV'].data
        sci_prc = hdulist['SCI'].data
        var_prc = hdulist['VAR'].data
        bpm_prc = hdulist['BPM'].data

        for p in (0,1):
            rows,cols = sci_prc[p].shape
            okcol_c = (bpm_prc[p,(rows/4):(3*rows/4)]==0).any(axis=0)
            badcol_c = (bpm_prc[p,(rows/4):(3*rows/4)]==1).all(axis=0)

            for a in range(3): 
                rowlist_a[a] = range(row_px[p,2*a],row_px[p,2*a+1])
                bins_a[a] = okcol_c.sum() * rows_a[a]
                crs_a[a] = (okcol_c & (bpm_prc[p][rowlist_a[a]]==1)).sum()
            crrate_a = crs_a/(bins_a*exptime)

            print file,([" O: "," E: "][p]+3*"%4i "+3*"%9.2e ") % (tuple(crs_a)+tuple(crrate_a)) 
            fileout = 'xtract_'+image+'_'+str(p)+'.txt'
            rowlist = rowlist_a.sum()

            hdr = 'Wav(Ang)'+((rows_a[0]*'%7s ') % tuple(rowlist_a[0]))+'| '
            hdr += ((rows_a[1]*'%7s ') % tuple(rowlist_a[1]))+'| '
            hdr += ((rows_a[2]*'%7s ') % tuple(rowlist_a[2]))
            wav_c = wav_prc[p][row_p[p]]
            np.savetxt('sci_'+fileout,np.vstack((wav_c,sci_prc[p][rowlist])).T, header=hdr,   \
                fmt="%9.2f "+rows_a[0]*"%7.0f "+"| "+rows_a[1]*"%7.0f "+"| "+rows_a[2]*"%7.0f ")
            np.savetxt('var_'+fileout,np.vstack((wav_c,var_prc[p][rowlist])).T, header=hdr,   \
                fmt="%9.2f "+rows_a[0]*"%7.0f "+"| "+rows_a[1]*"%7.0f "+"| "+rows_a[2]*"%7.0f ")
            np.savetxt('bpm_'+fileout,np.vstack((wav_c,bpm_prc[p][rowlist])).T,   \
                fmt="%9.2f "+rows_a[0]*"%2i "+"| "+rows_a[1]*"%2i "+"| "+rows_a[2]*"%2i ")

    return
# ---------------------------------------------------------------------------------
if __name__=='__main__':
    ro=int(sys.argv[1])
    re=int(sys.argv[2])
    infilelist=sys.argv[3:]
    specpolxtractdebug(ro,re,infilelist)
