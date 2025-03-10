#! /usr/bin/env python

"""
specpolserkow

fit serkowski law to stokes.fits data 

"""

import os, sys, glob, inspect
import numpy as np
import pyfits
from scipy.interpolate import interp1d
from scipy.optimize import fmin

np.set_printoptions(threshold=np.nan)
chi2_i = np.empty(0)
x_iV = np.empty((0,0))

#--------------------------------------
def specpolserkow(infilelist,lmax='5500',tslp='0'):
    """For each stokes fits file in infilelist, fit Serkowski law to normalized q and u.

    Parameters
    ----------
    infilelist: list
        List of _stokes filenames 
    lmax: str
        if 'free', allow lambda-max to vary, otherwise use specified fixed value (Ang)
    tslp: str
        if 'free', allow dPA/dlambda to vary, otherwise use specified fixed value (deg/Ang)

    """
    """
    _b: index in file list
    _S: unnormalized stokes : (0-3) = (I,Q,U,V)
    _s: normalized linear stokes: (0-1) = (q,u)
    _w: wavelength
    _v: Serkowski parameter: qmax,umax,lmax,pslp
    _V: parameter allow to vary
    _i: function evaluation
    """

    global chi2_i,x_iV
    obss = len(infilelist)
    maxlen = np.array([len(infilelist[b].split('.')[0]) for b in range(obss)]).max()

    print ("\nFile"+(maxlen)*" "+"%qmax    +/-     %umax    +/-   lmax(A)  +/-  deg/1000A   +/-     chi2dof") 
    for b in range(obss):

        x0_v = np.array([0.,0.,5500.,0.])
        v_V = np.array([0,1])
        if lmax=='free':    v_V = np.append(v_V,2)
        else:               x0_v[2] = float(lmax)
        if tslp=='free':    v_V = np.append(v_V,3)
        else:               x0_v[3] = float(tslp)
        x0_V = x0_v[v_V]
        vars = x0_V.shape[0]

        hdul = pyfits.open(infilelist[b])
        dwav = float(hdul['SCI'].header['CDELT1'])
        wav0 = float(hdul['SCI'].header['CRVAL1'])
        wavs = int(hdul['SCI'].header['NAXIS1'])
        wav_w = wav0 + dwav*np.arange(wavs)

        ctypelist = (hdul['SCI'].header['CTYPE3']).split(',')
        if (len(ctypelist)<3):
            print "Q,U not in ",infilelist[b]
            continue
        stokes_Sw = hdul['SCI'].data[:3,0,:]
        var_Sw = hdul['VAR'].data[:3,0,:] 
        ok_w = (hdul['BPM'].data[:3,0,:] == 0).all(axis=0)
        samples = ok_w.sum()

        nstokes_sw = np.zeros((2,wavs))
        nvar_sw = np.zeros((2,wavs))
        nstokes_sw[:,ok_w] = stokes_Sw[1:3,ok_w]/stokes_Sw[0,ok_w]
        nvar_sw[:,ok_w] = var_Sw[1:3,ok_w]/stokes_Sw[0,ok_w]**2
        x_v = x0_v

        chi2_i = np.empty(0)
        x_iV = np.empty((0,vars))

        xopt_V,chi2dof,iters,funcalls,warnflag = fmin(chi2_serk,x0_v[v_V],  \
          args=(x_v,v_V,nstokes_sw,nvar_sw,ok_w,wav_w),full_output=True,disp=False)

        x_v[v_V] = xopt_V
        x_v *= np.array([100.,100.,1.,1000.])               # q,u in %, tslp in deg/1000Ang 
        xerr_v = np.zeros(4)
        xerr_v[:2] = 100.*np.sqrt(chi2dof/(1./nvar_sw[:,ok_w]).sum(axis=1))

# xerr_v[2:] not yet being set

        print ("%-"+str(maxlen)+"s ") % infilelist[b].split('.')[0],
        print (2*"%8.4f %7.4f "+"%7.1f %5.1f "+2*"%8.3f ") % \
                tuple(np.vstack((x_v,xerr_v)).T.ravel()),
        print "%8.2f" % chi2dof

        np.savetxt("varsample.txt",np.vstack((chi2_i,x_iV.T)).T,fmt="%8.3f "+vars*"%12.5f ")

    return

#--------------------------------------
def serkowski(nstokesmax_f,lmax,tslp,wav_w):
    serk_w = np.exp(-(-0.1+1.86*lmax/10000)*(np.log(wav_w/lmax))**2)
    pmax = np.sqrt((nstokesmax_f**2).sum(axis=0))
    tmax = np.degrees(np.arctan2(*nstokesmax_f[::-1])/2.)

    return pmax*serk_w*np.array([np.cos(2.*np.radians(tmax+(wav_w-lmax)*tslp)), \
                                np.sin(2.*np.radians(tmax+(wav_w-lmax)*tslp))])        

#--------------------------------------
def chi2_serk(x_V,x_v,v_V,nstokes_fw,nvar_fw,ok_w,wav_w):
    global chi2_i,x_iV
    wavs = wav_w.shape[0]
    samples = 2.*ok_w.sum()
    x_v[v_V] = x_V
    nstokesmax_f = x_v[0:2]
    lmax,tslp = x_v[2:]
    if lmax <= 0.: return 1.E8
    nstokesserk_fw = serkowski(nstokesmax_f,lmax,tslp,wav_w)
    chi2dof = (((nstokes_fw[:,ok_w]-nstokesserk_fw[:,ok_w]))**2/nvar_fw[:,ok_w]).sum()/samples

    chi2_i = np.append(chi2_i,chi2dof)
    x_iV = np.vstack((x_iV,x_V))

    return chi2dof

#--------------------------------------
 
if __name__=='__main__':
    infilelist=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)        
    specpolserkow(infilelist,**kwargs)
