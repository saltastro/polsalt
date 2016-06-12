#! /usr/bin/env python

"""
specpolcombine

Combine stokes.fits data 

"""

import os, sys, glob, inspect
import numpy as np
import pyfits

from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from scrunch1d import scrunch1d
from specpolview import viewstokes
np.set_printoptions(threshold=np.nan)

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

#--------------------------------------------------------

def greff(grating,grang,artic,wav):
#   grating efficiency, zero outside 1st order and eff < grateffedge
#   wav may be 1D array

    grname=np.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=np.loadtxt(datadir+"gratings.txt",usecols=(1,2),unpack=True)
    gr300wav,gr300eff=np.loadtxt(datadir+"grateff_0300.txt",usecols=(0,1),unpack=True)
    grng,grdn,grthick,grtrans,grbroaden=np.loadtxt(datadir+"grateff_v1.txt", \
        usecols=(1,2,3,4,5),unpack=True)
    spec=np.loadtxt(datadir+"spec.txt",usecols=(1,))
    Grat0=spec[0]
    grateffedge = 0.04

    grnum = np.where(grname==grating)[0][0]
    lmm = grlmm[grnum]
    alpha_r = np.radians(grang+Grat0)

    if grnum == 0:          # SR grating
        greff = NaturalCubicSpline(gr300wav,gr300eff)(wav)
    else:                   # Kogelnik gratings
        ng = grng[grnum]
        dn = grdn[grnum]
        thick = grthick[grnum]
        tran = grtrans[grnum]
        broaden = grbroaden[grnum]
        beta_r = np.arcsin(wav*lmm/1.e7 - np.sin(alpha_r))
        betag_r = np.arcsin(np.sin(beta_r)/ng)
        alphag_r = np.arcsin(np.sin(alpha_r)/ng)
        KK = 2.*np.pi*lmm/1.e7
        nus = (np.pi*dn/wav)*thick/np.cos(alphag_r)
        nup = nus*np.cos(alphag_r + betag_r)
        xi = (thick/(2*np.cos(alphag_r)))*(KK*np.sin(alphag_r) - KK**2*wav/(4.*np.pi*ng))/broaden
        sins = np.sin(np.sqrt(nus**2+xi**2))
        sinp = np.sin(np.sqrt(nup**2+xi**2))
        effs = sins**2/(1.+(xi/nus)**2)
        effp = sinp**2/(1.+(xi/nup)**2)
        greff = np.where((sins>0)&(sinp>0)&((effs+effp)/2. > grateffedge),tran*(effs+effp)/2.,0.)
    return greff 

#---------------------------------------------------------------------------------------------
def specpolcombine(infilelist,debug_output=False):
    """combine stokes files

    Parameters
    ----------
    infile_list: list
       one or more _stokes.fits files

    """
    """
    _b observations
    _w wavelengths in individual observations
    _W wavelengths in combined grid

    """
    obss = len(infilelist)
    obsdict=obslog(infilelist)

 #  construct common wavelength grid _W
    grating_b = obsdict['GRATING']
    grang_b = obsdict['GR-ANGLE']
    artic_b = obsdict['CAMANG']
    dwav_b = np.empty(obss)
    wav0_b = np.empty(obss)
    wavs_b = np.empty(obss)
    stokeslist_sw = []
    varlist_sw = []      
    oklist_sw = []
    for b in range(obss):
        hdul = pyfits.open(infilelist[b])
        dwav_b[b] = float(hdul['SCI'].header['CDELT1'])
        wav0_b[b] = float(hdul['SCI'].header['CRVAL1'])
        wavs_b[b] = int(hdul['SCI'].header['NAXIS1'])
        stokeslist_sw.append(hdul['SCI'].data[:,0,:])
        varlist_sw.append(hdul['VAR'].data[:,0,:])     
        oklist_sw.append(hdul['BPM'].data[:,0,:] == 0)

    dWav = dwav_b.max()
    Wav0 = dWav*(wav0_b.min()//dWav) 
    Wavs = (dWav*((wav0_b + dwav_b*wavs_b).max()//dWav) - Wav0)/dWav
    wav_W = np.arange(Wav0,Wav0+dWav*Wavs,dWav)
    stokess = stokeslist_sw[0].shape[0]
    vars = varlist_sw[0].shape[0]

    stokes_bsW = np.zeros((obss,stokess,Wavs))
    var_bsW = np.zeros((obss,vars,Wavs))       
    ok_bsW = np.zeros((obss,stokess,Wavs)).astype(bool)

 # get data and put on common grid, combining bins if necessary
    for b in range(obss):
        if dwav_b[b] == dWav:
            W0 = (wav0_b[b] - Wav0)/dWav
            stokes_bsW[b,:,W0:W0+wavs_b[b]] = stokeslist_sw[b]
            var_bsW[b,:,W0:W0+wavs_b[b]] = varlist_sw[b]
            ok_bsW[b,:,W0:W0+wavs_b[b]] = oklist_sw[b]
        else:
            wbinedge_W = (wav_W - dWav/2. - (wav0_b[b] - dwav_b[b]/2.))/dwav_b
            for s in range(stokess): 
                stokes_bsW[b,s] = scrunch1d(stokeslist_sw[b][s],wbinedge_W)
                var_bsW[b,s] = scrunch1d(varlist_sw[b][s],wbinedge_W) 
                ok_bsW[b,s] = (scrunch1d((oklist_sw[b][s]).astype(int),wbinedge_W) > 0)
            if (vars>stokess): var_bsW[b,vars] = scrunch1d(var_sw[vars],wbinedge_W) 

    if debug_output:    
        np.savetxt("stokes_bsW.txt",np.vstack((wav_W,stokes_bsW.reshape((6,Wavs)))).T,fmt="%10.3f")

 # correct (unfluxed) intensity for grating efficiency to match observations together
    for b in range(obss):
        greff_W = greff(grating_b[b],grang_b[b],artic_b[b],wav_W) 
        ok_W = (ok_bsW[b].all(axis=0) & (greff_W > 0.))
        stokes_bsW[b][:,ok_W] /= greff_W[ok_W]
        var_bsW[b][:,ok_W] /= greff_W[ok_W]**2

 # normalize at matching wavelengths _w
 # compute ratios at each wavelength, then error-weighted mean of ratio
    ismatch_W = ok_bsW.all(axis=0).all(axis=0)
    normint_bw = stokes_bsW[:,0,ismatch_W]/stokes_bsW[:,0,ismatch_W].mean(axis=0)
    varnorm_bw = var_bsW[:,0,ismatch_W]/stokes_bsW[:,0,ismatch_W].mean(axis=0)**2
    normint_b = (normint_bw/varnorm_bw).sum(axis=1)/(1./varnorm_bw).sum(axis=1)
    print normint_b
    stokes_bsW /= normint_b[:,None,None]
    var_bsW /= normint_b[:,None,None]**2

 # Do error weighted combine of observations
    stokes_sW = np.zeros((stokess,Wavs))
    var_sW = np.zeros((vars,Wavs))

    for b in range(obss):
        ok_W = ok_bsW[b].any(axis=0)
        stokes_sW[:,ok_W] += stokes_bsW[b][:,ok_W]/var_bsW[b][:stokess,ok_W]
        var_sW[:,ok_W] += 1./var_bsW[b][:,ok_W]
    ok_W = (var_sW != 0).all(axis=0)
    ok_sW = np.tile(ok_W,(3,1))
    var_sW[ok_sW] = 1./var_sW[ok_sW]
    stokes_sW[:,ok_W] *= var_sW[:stokess,ok_W]

 # Save result, name formed from unique elements of '_'-separated parts of names
    namepartlist = []
    parts = 100
    for file in infilelist:
        partlist = os.path.basename(file).split('.')[0].split('_')
        parts = min(parts,len(partlist)) 
        namepartlist.append(partlist)
    outfile = ''     
    for part in range(parts):
        outfile+='-'.join(sorted(set(zip(*namepartlist)[part])))+'_'   

    outfile = outfile[:-1]+'.fits'
    print "\n",outfile,"\n"

    hduout = hdul
    for ext in ('SCI','VAR','BPM'):
        hduout[ext].header.update('CDELT1',dWav)
        hduout[ext].header.update('CRVAL1',Wav0)
    hduout['SCI'].data = stokes_sW.astype('float32').reshape((stokess,1,-1))
    hduout['VAR'].data = var_sW.astype('float32').reshape((vars,1,-1))
    hduout['BPM'].data = (~ok_sW).astype('uint8').reshape((stokess,1,-1))
    hduout[0].header.add_history('POLCOMBINE: '+' '.join(infilelist))

    hduout.writeto(outfile,clobber=True,output_verify='warn')
    
    return

#--------------------------------------
 
if __name__=='__main__':
    infilelist=sys.argv[1:]
    debug_output = False
    if infilelist[-1][-5:].count(".fits")==0:
        debug_output = (len(infilelist.pop()) > 0)
    specpolcombine(infilelist,debug_output)
