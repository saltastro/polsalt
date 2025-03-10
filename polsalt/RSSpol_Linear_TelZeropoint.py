#! /usr/bin/env python

"""
RSSpol_Linear_TelPolZeropoint

Combine instrumental polarization observations, and evaluate telescope and instrument
calibration files

"""

import os, sys, glob, inspect
import numpy as np
import pyfits

from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from scipy import linalg as la
from specpolserkow import serkowski
from scrunch1d import scrunch1d

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

# np.set_printoptions(threshold=np.nan)

#---------------------------------------------------------------------------------------------
def RSSpol_Linear_TelZeropoint(datever,infilelist,debug_output=False):
    """Create standard linear zeropoint polarization calibration file, telescope contribution

    Parameters
    ----------
    infile_list: list of one or more 300 l/mm and blue 0900 l/mm unpol standard 
        _stokes.fits files 

    """
    """
    _b observations
    _f q,u (SCI); q,u,qu (VAR)
    _w wavelengths in individual observations
    _W wavelengths in coarse grid

    """

    print 

 #  read in unpol standard catalog
    unpolstdcat = open(datadir+"unpol_serk.txt",'r').readlines()
    unpolstddict = {}
    for line in unpolstdcat:
        if line[0]=="#": continue
        unpolstddict[line.split()[0]] = np.array([float(x) for x in line.split()[1:]])

 #  collect data, subtract unpol standard catalogue polarization (serkowski fit)
    uselistb = []
    objectlistb = []

    for file in infilelist:
        print file,
        object = pyfits.getheader(file)['OBJECT'].replace(" ","")  
        if not(object in unpolstddict):
            print "not in unpol cat: skip"
            continue
        print 
        uselistb.append(file)
        objectlistb.append(object)

    obss = len(uselistb)
    dwav_b = np.array([pyfits.getheader(file,'SCI')['CDELT1'] for file in uselistb])
    wav0_b = np.array([pyfits.getheader(file,'SCI')['CRVAL1'] for file in uselistb])
    wavs_b = np.array([pyfits.getheader(file,'SCI')['NAXIS1'] for file in uselistb])
    grating_b = np.array([pyfits.getheader(file)['GRATING'] for file in uselistb])
    artic_b = np.array([pyfits.getheader(file)['AR-ANGLE'] for file in uselistb])
    telpa_b = np.array([pyfits.getheader(file)['TELPA'] for file in uselistb])
    trkrho_b = np.array([pyfits.getheader(file)['TRKRHO'] for file in uselistb])  # should really use finalstokes track mean    

    stokeslistb_Fw = []
    varlistb_Fw = []
    oklistb_Fw = []
    wavlistb_w = []
    for b,file in enumerate(uselistb):
        hdul = pyfits.open(file)  
        stokeslistb_Fw.append(hdul['SCI'].data[:,0,:])
        varlistb_Fw.append(hdul['VAR'].data[:,0,:])
        oklistb_Fw.append(hdul['BPM'].data[:,0,:]==0)
        wavlistb_w.append(wav0_b[b] + dwav_b[b]*np.arange(wavs_b[b]))
        stokeslistb_Fw[-1][1:] -=   \
            serkowski(unpolstddict[objectlistb[b]][[0,2]]/100.,5500.,0.,wavlistb_w[b])*stokeslistb_Fw[-1][0]

 # bin into coarse wavelength grid _W
    dWav = 500.                         # need to bridge gap, < 175 Ang
    Wav0 = 3000. 
    Wavs = 14
    wav_W = np.arange(Wav0,Wav0+dWav*Wavs,dWav)
    stokes_bFW = np.zeros((obss,3,Wavs))
    var_bFW = np.zeros((obss,4,Wavs))
    nstokes_bfW = np.zeros((obss,2,Wavs))
    nvar_bfW = np.zeros((obss,3,Wavs))
    for b in range(obss):
        ok_w = oklistb_Fw[b].all(axis=0)
        wbinedge_W = (wav_W - wav0_b[b])/dwav_b[b]
        binwidth = wbinedge_W[1] - wbinedge_W[0]
        wbinedge_W = np.hstack((wbinedge_W,wbinedge_W[-1]+binwidth))
        wbinedge_W = np.maximum(wbinedge_W,0)
        wbinedge_W = np.minimum(wbinedge_W,wavs_b[b])

        bins_W = scrunch1d(ok_w,wbinedge_W)
        ok_W = (bins_W > 0)

        for F in (0,1,2):
            stokes_bFW[b,F,ok_W] = scrunch1d(stokeslistb_Fw[b][F],wbinedge_W)[ok_W]
        for F in (0,1,2,3):
            var_bFW[b,F,ok_W] = scrunch1d(ok_w*varlistb_Fw[b][F],wbinedge_W)[ok_W]

    nstokes_bfW[:,:,ok_W] = stokes_bFW[:,1:,ok_W]/stokes_bFW[:,0,ok_W][:,None,:]
    nvar_bfW[:,:,ok_W] = var_bFW[:,1:,ok_W]/(stokes_bFW[:,0,ok_W]**2)[:,None,:]

    np.savetxt("nstokes_bfW.txt",nstokes_bfW.reshape((-1,Wavs)).T,fmt="%12.6f")
    np.savetxt("nvar_bfW.txt",nvar_bfW.reshape((-1,Wavs)).T,fmt="%14.10f")  

 # extract telescope and instrument polarization at each wavelength by fitting stellar 
 #   inst pol data vs PA and Rho
 # for each W, (differently) weighted fit A_DC X_C = B_D. X = (q,u(inst), q,u(tel))
 # A_DC: coefficient matrix, _D=_bf (flattened)
 # B_D = data sample 

    radparho_b = np.radians(telpa_b - trkrho_b)
    radpa_b = np.radians(telpa_b)
    A_fCb = np.array([np.cos(2.*radparho_b),    -np.sin(2.*radparho_b),          \
                      np.cos(2.*radpa_b),       -np.sin(2.*radpa_b),            \
                      np.sin(2.*radparho_b),     np.cos(2.*radparho_b),          \
                      np.sin(2.*radpa_b),        np.cos(2.*radpa_b)]).reshape((2,4,obss)) 
    A_DC = A_fCb.transpose((2,0,1)).reshape((2*obss,4))

    nzpolstokes_CW = np.zeros((4,Wavs))
    nzpolerr_CW = np.zeros((4,Wavs))
    obss_W = np.zeros(Wavs)
    Bresid_DW = np.zeros((2*obss,Wavs))

    for W in range(Wavs):
        B_D = nstokes_bfW[:,:,W].flatten()
        var_D = nvar_bfW[:,:2,W].flatten()          # ignore covariance for now
        obss_W[W] = nvar_bfW[:,:,W].all(axis=1).sum()
        if obss_W[W] < obss-1: continue
        wt_D = np.zeros_like(var_D)
        wt_D[var_D>0] = 1./np.sqrt(var_D[var_D>0])
        Awtd_DC = A_DC*wt_D[:,None]
        nzpolstokes_CW[:,W],sumsqerr = la.lstsq(Awtd_DC,B_D*wt_D)[0:2]
        alpha_CC = (Awtd_DC[:,:,None]*Awtd_DC[:,None,:]).sum(axis=0)
        eps_CC = la.inv(alpha_CC)
        nzpolerr_CW[:,W] = np.sqrt((sumsqerr/(2*obss_W[W]).sum())*np.diagonal(eps_CC))
        Bresid_DW[:,W] = B_D - (A_DC*nzpolstokes_CW[:,W]).sum(axis=1)

    precision = np.std(np.median(Bresid_DW[:,ok_W],axis=1))

    np.savetxt("qresid_Wb.txt",Bresid_DW.reshape((obss,2,Wavs))[:,0].T,fmt="%10.6f")
    np.savetxt("uresid_Wb.txt",Bresid_DW.reshape((obss,2,Wavs))[:,1].T,fmt="%10.6f")

 # extrapolate the ends to 3000,10000 for successful interpolation from bin centers
    wav_W += dWav/2.
    dnzpolstokes_CW = nzpolstokes_CW[:,1:] - nzpolstokes_CW[:,:-1]
    if ok_W[-1]:
        nzpolstokesright_C = nzpolstokes_CW[:,-1]+dnzpolstokes_CW[:,-1]*(10000.-wav_W[-1])/dWav
        nzpolstokes_CW = np.hstack((nzpolstokes_CW,nzpolstokesright_C[:,None]))
        nzpolerr_CW = np.hstack((nzpolerr_CW,nzpolerr_CW[:,-1][:,None]))
        ok_W = np.append(ok_W,True)
        wav_W = np.append(wav_W,10000.)
    if ok_W[0]:
        nzpolstokesleft_C = nzpolstokes_CW[:,0]-dnzpolstokes_CW[:,0]*(wav_W[0]-3000.)/dWav
        nzpolstokes_CW = np.hstack((nzpolstokesleft_C[:,None],nzpolstokes_CW))
        nzpolerr_CW = np.hstack((nzpolerr_CW[:,0][:,None],nzpolerr_CW))
        ok_W = np.insert(ok_W,0,True)
        wav_W = np.insert(wav_W,0,3000.)

    np.savetxt("nzpolstokes_CW.txt",nzpolstokes_CW.T,fmt="%10.6f")
    np.savetxt("nzpolerr_CW.txt",nzpolerr_CW.T,fmt="%10.6f")

 # Save results to TelZeropoint calibration file 
    calfile = "RSSpol_Linear_TelZeropoint_"+datever+".txt"

    print "\n Rms deviation %8.3f%%" % (100.*precision)
    print " Saving %s" % calfile

    np.savetxt(calfile,np.vstack((wav_W[ok_W],100*nzpolstokes_CW[0:2,ok_W], \
        100*nzpolerr_CW[0,ok_W])).T, \
        header="Telescope linear polarization zero point (primary mirror coordinates) \n"+                    \
        " Ang    q (%)    u(%)   err(%)", fmt="%6.0f "+3*"%8.3f ")

    return

#--------------------------------------
 
if __name__=='__main__':
    datever=sys.argv[1]
    infilelist=sys.argv[2:]
    debug_output = False
    if infilelist[-1][-5:].count(".fits")==0:
        debug_output = (len(infilelist.pop()) > 0)
    RSSpol_Linear_TelZeropoint(datever,infilelist,debug_output)

# debug
# /d/pfis/khn/instpol
# Pypolsalt RSSpol_Linear_TelZeropoint.py 20061030_v01 `cat unpols_300.txt`
# /d/pfis/khn/instpol/v01_900
# Pypolsalt RSSpol_Linear_TelZeropoint.py 20061030_v02 `cat pg0900_blue.txt`

