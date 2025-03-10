#! /usr/bin/env python

"""
RSSpol_HW_Calibration

Combine efficiency stokes.fits data and output a standard HW efficiency/axis calibration file

"""

import os, sys, glob
import numpy as np
import pyfits

from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from scrunch1d import scrunch1d
from specpolfinalstokes import specpolrotate
from specpolview import viewstokes
np.set_printoptions(threshold=np.nan)
import warnings 
warnings.filterwarnings("error")
#---------------------------------------------------------------------------------------------
def RSSpol_HW_Calibration(date,infilelist,debug_output=False):
    """Create standard HW efficiency calibration file

    Parameters
    ----------
    infile_list: list
       one or more _stokes.fits files

    """
    """
    _b observations
    _B observations, in order of central wavelength
    _s I,Q,U
    _S 1,p,t
    _w wavelengths in individual observations
    _W wavelengths in combined grid

    """

    polaroiddict = {'P05':"HN7",'P06':"HNP'B",'P08':"HNP'B",'P09':"APIR29-020"}
    polwavdict = {"HNP'B":(3000.,7000.),"HN7":(6000.,8600.),"APIR29-020":(4200.,10000.)}
  # Get data
    obss = len(infilelist)
    obsdict=obslog(infilelist)
    dwav_b = np.zeros(obss)
    wav0_b = np.zeros(obss)
    wavs_b = np.zeros(obss,dtype=int)
    stokeslist_sw = []
    varlist_sw = []      
    oklist_sw = []
    polaroidlist = obss*["NA"]
    filenamelen = max(len(f) for f in infilelist)

    for b in range(obss):
        hdul = pyfits.open(infilelist[b])
      # extract wavelength scale
        dwav_b[b] = hdul['SCI'].header['CDELT1']
        wav0_b[b] = hdul['SCI'].header['CRVAL1']
        wavs_b[b] = int(hdul['SCI'].header['NAXIS1'])
        wav_w = np.arange(wav0_b[b],wav0_b[b]+wavs_b[b]*dwav_b[b],dwav_b[b])
        stokeslist_sw.append(hdul['SCI'].data[:,0,:])
        varlist_sw.append(hdul['VAR'].data[:,0,:])     
        oklist_sw.append(hdul['BPM'].data[:,0,:] == 0)

      # deduce which polaroid was used
        dateobs = obsdict['DATE-OBS'][b].replace('-','')

        if int(dateobs) > 20150101:                     # we started specific barcoding the slitmask
            masktype = obsdict["MASKID"][b][-3:]
            if polaroiddict.has_key(masktype): polaroidlist[b] = polaroiddict[masktype]
        else:                                           # based on historical use before then
            if (wav0_b[b] < 4000.): polaroidlist[b] = "HNP'B"
            elif (wav0_b[b] + dwav_b[b]*wavs_b[b] > 7000.): polaroidlist[b] = "HN7"

      # strip out bad polaroid wavelengths
        oklist_sw[b] &= ((wav_w >= polwavdict[polaroidlist[b]][0])&(wav_w <= polwavdict[polaroidlist[b]][1]))
        
 #  construct common wavelength grid _W  
    dWav = dwav_b.max()
    Wav0 = dWav*(wav0_b.min()//dWav) 
    Wavs = int((dWav*((wav0_b + dwav_b*wavs_b).max()//dWav) - Wav0)/dWav)
    wav_W = np.arange(Wav0,Wav0+dWav*Wavs,dWav)
    stokes_bsW = np.zeros((obss,3,Wavs))
    var_bsW = np.zeros((obss,4,Wavs))       
    ok_bsW = np.zeros((obss,3,Wavs)).astype(bool)
    nstokes_bSW = np.zeros((obss,3,Wavs))
    nerr_bSW = np.zeros((obss,3,Wavs))       

 # get data and put on common grid, combining bins if necessary
    doflip_b = np.zeros(obss,dtype=bool)
    for b in range(obss):
        if dwav_b[b] == dWav:
            W0 = int((wav0_b[b] - Wav0)/dWav)
            stokes_bsW[b,:,W0:W0+wavs_b[b]] = stokeslist_sw[b]
            var_bsW[b,:,W0:W0+wavs_b[b]] = varlist_sw[b]
            ok_bsW[b,:,W0:W0+wavs_b[b]] = oklist_sw[b]
        else:
            wbinedge_W = (wav_W - dWav/2. - (wav0_b[b] - dwav_b[b]/2.))/dwav_b[b]
            wbinedge_W = np.append(wbinedge_W,wbinedge_W[-1]+dWav)
            for s in range(3): 
                stokes_bsW[b,s] = scrunch1d(stokeslist_sw[b][s],wbinedge_W)
                var_bsW[b,s] = scrunch1d(varlist_sw[b][s],wbinedge_W) 
                ok_bsW[b,s] = (scrunch1d((oklist_sw[b][s]).astype(int),wbinedge_W) > 0)
            var_bsW[b,3] = scrunch1d(varlist_sw[b][3],wbinedge_W) 
        nstokes_bSW[b],nerr_bSW[b] = viewstokes(stokes_bsW[b],var_bsW[b],ok_bsW[b])
        doflip_b[b] = (np.median(nstokes_bSW[b,2,ok_bsW[b,0]])>0.)                # standardize polaroid PA
    if debug_output:    
        np.savetxt("stokes_bsW.txt",np.vstack((wav_W,stokes_bsW.reshape((3*obss,Wavs)))).T,fmt="%10.2f")
        np.savetxt("nstokes_bSW.txt",np.vstack((wav_W,nstokes_bSW.reshape((3*obss,Wavs)))).T,fmt="%10.3f")
        np.savetxt("nerr_bSW.txt",np.vstack((wav_W,nerr_bSW.reshape((3*obss,Wavs)))).T,fmt="%10.3f")

 # find PA (S=2) offset between observations, in order of central wavelength
    dPA_b = np.zeros(obss)
    b_B = np.argsort(wav0_b + 0.5*dwav_b*wavs_b)
    ok_bW = ok_bsW.all(axis=1)
    for B in range(obss):
        b = b_B[B]
        ok_W = (ok_bW[b] & ok_bW[b_B[0]])
        if ok_W.sum(): 
            dPA_b[b] = (nstokes_bSW[b,2,ok_W] - 90.*int(doflip_b[b]) - nstokes_bSW[b_B[0],2,ok_W]).sum()/ok_W.sum()
    dPA_b -= (dPA_b.max()+dPA_b.min())/2. -90.*doflip_b.astype(int)

    print ("File"+(filenamelen-4)*" "+"Grating  Artic Polaroid  PA Offset")
    for b in range(obss):
        grating = obsdict['GRATING'][b].strip()
        artic = obsdict['CAMANG'][b]
        print ("%"+str(filenamelen)+"s %6s %6.2f %10s %5.2f") % (infilelist[b], grating, artic, polaroidlist[b],dPA_b[b])

 # Remove PA offsets and scrunch observations into final calibration binning _c
    wav_c = np.hstack((np.arange(3100.,3500.,100.),np.arange(3500.,5000.,50.), \
                       np.arange(5000.,7000.,100.), np.arange(7000.,10200.,200.)))
    dcwav_c = (wav_c[:-1] + wav_c[1:])/2. - wav_c[:-1] 
    dcwav_c = np.hstack((dcwav_c[0],dcwav_c,dcwav_c[-1]))
    cwavs = wav_c.shape[0]

    qu_bsc = np.zeros((obss,2,cwavs))
    varqu_bsc = np.zeros((obss,2,cwavs))
    ok_bc = np.zeros((obss,cwavs),dtype=bool)
    clim_db = np.zeros((2,obss),dtype=int)

    for b in range(obss):
        stokes_sc = np.zeros((3,cwavs))
        var_sc = np.zeros((3,cwavs))
        bins_sc = np.zeros((3,cwavs))
        stokeslist_sw[b],varlist_sw[b] = \
            specpolrotate(stokeslist_sw[b],varlist_sw[b],np.repeat(dPA_b[b],wavs_b[b]))
        wbinedge_c = (wav_c - dcwav_c[:-1] - (wav0_b[b] - dwav_b[b]/2.))/dwav_b[b] 
        wbinedge_c = np.append(wbinedge_c,wbinedge_c[-1]+dcwav_c[-1])

      # allow getting the very edge points
        wbinedge_c[np.where((wbinedge_c[:-1]<0) & (wbinedge_c[1:]>0))] = 0
        wbinedge_c[np.where((wbinedge_c[:-1]<wavs_b[b]) & (wbinedge_c[1:]>wavs_b[b]))[0]+1] = wavs_b[b]

        for s in range(3):
            stokes_sc[s] = scrunch1d(oklist_sw[b][s]*stokeslist_sw[b][s],wbinedge_c)
            var_sc[s] = scrunch1d(oklist_sw[b][s]*varlist_sw[b][s],wbinedge_c)
            bins_sc[s] = scrunch1d(oklist_sw[b][s].astype(int),wbinedge_c)
        if debug_output:
            np.savetxt("stokes_"+str(b)+"_sc.txt",stokes_sc.T,fmt="%10.0f")
            np.savetxt("var_"+str(b)+"_sc.txt",var_sc.T,fmt="%10.0f")
            np.savetxt("bins_"+str(b)+"_sc.txt",bins_sc.astype(int).T,fmt="%8.3f")

        ok_c = (bins_sc > 0).all(axis=0)
        qu_bsc[b][:,ok_c] = stokes_sc[1:,ok_c]/stokes_sc[0,ok_c]
        varqu_bsc[b][:,ok_c] = var_sc[1:,ok_c]/stokes_sc[0,ok_c]**2
        ok_bc[b] = ok_c
        clim_db[:,b] = np.where(ok_bc[b])[0][[0,-1]]

    cedge_e = np.unique(clim_db)
    cmin,cmax = np.where(ok_bc.any(axis=0))[0][[0,-1]]
    ok_c = (np.arange(cwavs)==np.arange(cmin,cmax+1)) 

  # Combine observations, using weighting ramp to avoid edges
    wt_c = np.zeros(cwavs)
    qu_sc = np.zeros((2,cwavs))
    varqu_sc = np.zeros((2,cwavs))
    for b in range(obss):
        wtb_c = (ok_bc[b]).astype(float)
        cmin,cmax = np.where(ok_bc[b])[0][[0,-1]]
        ok_c &= np.logical_not(((np.arange(cwavs)>=cmin) & (np.arange(cwavs)<=cmax) & np.logical_not(ok_bc[b])))
        if (cmin > cedge_e[0]):
            cramp = cedge_e[np.where(cedge_e > cmin)[0][0]]
            wtb_c *= np.minimum((wav_c - wav_c[cmin])/(wav_c[cramp] - wav_c[cmin]),1.)
        if (cmax < cedge_e[-1]):
            cramp = cedge_e[np.where(cedge_e < cmax)[0][-1]]
            wtb_c *= np.minimum((wav_c[cmax] - wav_c)/(wav_c[cmax] - wav_c[cramp]),1.)
        qu_sc[:,ok_c] += wtb_c[ok_c]*qu_bsc[b][:,ok_c]
        varqu_sc[:,ok_c] += wtb_c[ok_c]*varqu_bsc[b][:,ok_c]
        wt_c += wtb_c
    
    qu_sc[:,ok_c] /= wt_c[ok_c]
    varqu_sc[:,ok_c] /= wt_c[ok_c]**2

    errmax = 0.004
    ok_c &= (varqu_sc < errmax**2).all(axis=0)
    err_sC = np.sqrt(varqu_sc[:,ok_c])
    qu_sC = qu_sc[:,ok_c]
    p_C = np.sqrt((qu_sC**2).sum(axis=0))
    t_C = np.mod(np.degrees(np.arctan2(qu_sC[1],qu_sC[0]))/2. +180.,180.)   # calfile PA 0-180
    err_C = np.sqrt((err_sC**2).mean(axis=0))

 # Save calibration file
    calfile = "RSSpol_HW_Calibration_"+date+".txt"
    np.savetxt(calfile,np.vstack((wav_c[ok_c],p_C,t_C,err_C)).T, fmt="%6.0f %9.6f %8.3f %9.6f", \
      header="RSS halfwave plate calibration data\n Ang    heff    instPA     err ")
    
    return

#--------------------------------------
 
if __name__=='__main__':
    date=sys.argv[1]
    infilelist=sys.argv[2:]
    debug_output = False
    if infilelist[-1][-5:].count(".fits")==0:
        debug_output = (len(infilelist.pop()) > 0)
    RSSpol_HW_Calibration(date,infilelist,debug_output)
