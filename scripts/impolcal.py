#! /usr/bin/env python

"""
impolcal

Sky data: Fit unpol standard, evaluating FOV dependence of Linear_TelZeropoint
Lamp data: Fit HWCal data, evaluating FOV dependence of HW_Calibration 

"""

import os, sys, glob, shutil, inspect, datetime
import numpy as np

import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter("error")

from astropy.io import fits as pyfits
from astropy.io import ascii
import astropy.table as ta
from scipy.optimize import curve_fit
from scipy import linalg as la
from scipy.interpolate import LSQUnivariateSpline
from pyraf import iraf
from iraf import pysalt

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

from specpolutils import datedfile, datedline, hwcal 
from specpolview import viewstokes 
np.set_printoptions(threshold=np.nan) 

#---------------------------------------------------------------------------------------------
def impolcal(fileList, **kwargs):
    """Compute FOV and track dependence of calibration

    Parameters
    ----------
    fileList: list of text filenames
       stokes fits files

    debug=: False (default)
            True (debug output)
    """
  # _f file counter
  # _w wavelength index of input files
  # _W wavelength output grid index
  # _p (0,1) calibration file (P,PA, or Q,U)
  # _t target index in each file
  # _T combined targets over files
  # _S (unnormalized) stokes: (I,Q,U)
    files = len(fileList)   
    debug = (kwargs.pop('debug','False') == 'True')
    outList = kwargs.pop('out',[])
    if len(outList):
        if outList == "all": outList = range(len(fileList))
        else: outList = [int(outList)]

    hdul = pyfits.open(fileList[0])
    name = os.path.basename(fileList[0]).split('.')[0]
    dateobs = hdul[0].header['DATE-OBS'].replace('-','')    
    targets,wavs = hdul['SCI'].data.shape[1:3]
    wedge_W = hdul['WCSDVARR'].data
    dwav = hdul['SCI'].header['CDELT1']     # number of unbinned wavelength bins in bin
    wedge_w = wedge_W[:(wavs+1)*dwav:dwav]  # shape wavs+1     
    wav_w = (wedge_w[:-1] + wedge_w[1:])/2.    
    lampid = hdul[0].header['LAMPID']
    issky = (lampid == "NONE")
  
  # halfwave plate cal vs wav,FOV (1 file), +Track (files>1)
  # FOV: YX. normalized to 50 mm  
  # Track MB mbyx. normalized to .04 m
    YXnorm = 50.
    mbnorm = 0.04      
    mbyx_df = np.zeros((2,files))
    targets_f = np.zeros(files,dtype=int)
    filelen = len(fileList[0])

    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)         
        mbyx_df[:,f] = np.array([hdul[0].header['MBY'], hdul[0].header['MBX']])/mbnorm
        targets_f[f] = len(pyfits.open(file)['TGT'].data)             

    Targets = targets_f.max()
    if issky: 
        print "\nOn-sky zeropoint calibration"
        Wnorm = 3000.
        W0 = 6500.
        wav_W = W0*np.ones(1)
        Wavs = 1
        stokes_fpTw = np.zeros((files,2,Targets,wavs))

    ispolcal = (not issky)
    if ispolcal: 
        print "\nPolaroid efficiency calibration"
        wav_W = np.array(range(4000,4800,50)+range(4800,6000,100)+  \
            range(6000,8000,200)+range(8000,10400,400)).astype(float)
        Wavs = len(wav_W)              
        stokes_fpTW = np.zeros((files,2,Targets,Wavs))                                  
        Wavidx = np.where(wav_W == 7000.)[0]    # for debug
          
    print "\n f  file "+(filelen-5)*" "+"targets mby      mbx"
 
    YX_dfT = np.zeros((2,files,Targets))
    stokes_fSTw = np.zeros((files,3,Targets,wavs))
    ok_fTw = np.zeros((files,Targets,wavs),dtype=bool)
    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)
        print ("%2i %"+str(filelen)+"s %3i %8.3f %8.3f") % ((f, file, targets_f[f]) + tuple(mbyx_df[:,f]))                
        stokes_Stw = hdul['SCI'].data
        var_Stw = hdul['VAR'].data  
        bpm_Stw = hdul['BPM'].data
        ok_tw = (bpm_Stw==0).all(axis=0)
                    
        tgtTab = hdul['TGT'].data
      # normalize FOV dims to edge
        if issky:
            YX_dfT[:,f,:targets_f[f]] = np.array([tgtTab['Y'],tgtTab['X']])/YXnorm
        else:                         
            YX_dfT[:,f,:targets_f[f]] = np.array([np.array(s.split('_')).astype(int)    \
                for s in tgtTab['CATID']]).T/10.
         
        PApos_t = np.degrees(np.arctan2(YX_dfT[0,f,:targets_f[f]],YX_dfT[1,f,:targets_f[f]]))
        R_t =  np.sqrt((YX_dfT[:,f,:targets_f[f]]**2).sum(axis=1))               
                
        if issky:
            stokes_ptw = np.zeros((2,targets_f[f],wavs))
            err_ptw = np.zeros_like(stokes_ptw)
            stokes_fpTw[f][:,ok_tw] = 100.*stokes_Stw[1:3,ok_tw]/stokes_Stw[0,ok_tw][None,:]    # %Q, %U
            YXpow_dc = np.array([[0,1,2,0,0,1,1,2],[0,0,0,1,2,1,2,1]])            
        if ispolcal:
            PA0 = 90.                                           # PA of polaroid         
            stokes_ptw = np.zeros((2,targets_f[f],wavs))
            err_ptw = np.zeros((2,targets_f[f],wavs))
            for t in range(targets_f[f]):                        # note: view returns %P and deg
                stokes_ptw[:,t],err_ptw[:,t] =  \
                    viewstokes(stokes_Stw[:,t],var_Stw[:,t],ok_w=ok_tw[t],tcenter=np.pi/2.)
            ok_tw = (stokes_ptw[0] != 0.)
            stokes_ptw[1,ok_tw] -= PA0
            stokes_ptw[1,ok_tw] = np.mod(stokes_ptw[1,ok_tw] + 180.,180.)
            YXpow_dc = np.array([[0,1,2,0,0,1,1,2],[0,0,0,1,2,1,2,1]])

        stokes_fSTw[f,:,:targets_f[f]] = stokes_Stw
        ok_fTw[f,:targets_f[f]] = ok_tw
        
      # for polcal, wavelength spline smooth onto grid for cal table
        if ispolcal:
            wmin,wmax = (min(wav_w[ok_tw.any(axis=0)]),max(wav_w[ok_tw.any(axis=0)]))        
            for t in range(targets_f[f]):
                ok_W = ((wav_W > wav_w[ok_tw[t]][0]) & (wav_W < wav_w[ok_tw[t]][-1]))      # grid is inside points                     
                W_w = np.where((wav_W[:-1,None] < wav_w[ok_tw[t]][None,:]) &    \
                    (wav_W[1:,None] > wav_w[ok_tw[t]][None,:]))[0]
                W_k, pts_k =  np.unique(W_w,return_counts=True)
                ok_W &= np.in1d(np.arange(Wavs),W_k)            # only use grid knots enclosing >= 1 data point
                if ((ok_tw[t].sum() < 5) | (ok_W.sum() < 4)): continue            
                k0 = np.argmax(ok_W[W_k])
                k1 = len(pts_k) - np.argmax(ok_W[W_k]) -1
                if (pts_k[k0]==1): ok_W[W_k[k0+1]] = False      # >1 data points at end, avoid spline end problems
                if (pts_k[k1]==1): ok_W[W_k[k1-1]] = False

                for p in (0,1):
                    pSpline = LSQUnivariateSpline(wav_w[ok_tw[t]],stokes_ptw[p,t,ok_tw[t]],wav_W[ok_W],k=1,     \
                        w=1./err_ptw[p,t,ok_tw[t]]**2)
                    stokes_fpTW[f,p,t,ok_W] = pSpline(wav_W[ok_W])   

  # do fit of calibration function
    YXcofs = YXpow_dc.shape[1]
    pow_dc = np.copy(YXpow_dc)
    dname_d = ['y','x']    
    if (files>1):
        mbpow_dc = [np.array([[0,1,2,0,0],[0,0,0,1,2]]),np.array([[0,1],[0,0]])][issky]
        mbcofs = mbpow_dc.shape[1] 
        pow_dc = np.vstack((np.repeat(pow_dc,mbcofs,axis=1), np.tile(mbpow_dc,YXcofs)))
        dname_d += ['by','bx']         
    if issky:
        Wpow_c = np.arange(2)     
        Wcofs = Wpow_c.shape[0]
        cofs = pow_dc.shape[1]        
        pow_dc = np.vstack((np.repeat(pow_dc,Wcofs,axis=1), np.tile(Wpow_c,cofs)))
        dname_d += ['W']                                 
    dims,cofs = pow_dc.shape                 
    stokes_pcW = np.zeros((2,cofs,Wavs))
    stokeserr_pcW = np.zeros_like(stokes_pcW)
    stokeserr_pW = np.zeros((2,Wavs))

    for W in range(Wavs):
        if (Wavs==1):
            ok_fT = (ok_fTw.any(axis=2) != 0)        
        else:
            ok_fT = (stokes_fpTW[:,0,:,W] != 0)       
            if (ok_fT.sum() < files*Targets/3): continue                        
        x_ds = YX_dfT[:,ok_fT].reshape((2,-1))
      
        if (files>1):                
            mbyx_ds = np.tile(mbyx_df[:,:,None],(1,1,Targets))[:,ok_fT].reshape((2,-1))
            x_ds = np.vstack((x_ds,mbyx_ds))
        if (Wavs == 1):
            W_w = (wav_w - W0)/Wnorm
            ok_sw = ok_fTw[ok_fT].reshape((-1,wavs))
            ok_w = ok_sw.any(axis=0)
            x_Ds = np.copy(x_ds)            
            x_ds = np.empty((dims,0))

            for s in range(x_Ds.shape[1]):
                wavsamples = ok_sw[s].sum()                          
                x_ds = np.hstack((x_ds,np.vstack((np.repeat(x_Ds[:,s][:,None],  \
                    wavsamples,axis=1),W_w[ok_sw[s]].reshape((1,-1))))))            
               
        dims,samples = x_ds.shape            
        a_cs = (x_ds[:,None,:]**pow_dc[:,:,None]).prod(axis=0)                                              
        a_sc = a_cs.T                                    
        alpha_cc = (a_sc[:,:,None]*a_sc[:,None,:]).sum(axis=0)
        eps_cc = la.inv(alpha_cc)

        for p in (0,1):
            if (Wavs == 1):
                stokes_pcW[p,:,W],sumsqerr = la.lstsq(a_sc,stokes_fpTw[:,p][ok_fTw])[0:2]                            
            else:        
                stokes_pcW[p,:,W],sumsqerr = la.lstsq(a_sc,stokes_fpTW[:,p,:,W][ok_fT])[0:2]
            stokeserr_pW[p,W] = np.sqrt(sumsqerr/samples)      # rms fit deviation
            stokeserr_pcW[p,:,W] = stokeserr_pW[p,W]*np.sqrt(np.diagonal(eps_cc))         

    ok_W = (stokes_pcW[0,0] != 0.)

    stokesrms_p = np.sqrt((stokeserr_pW[:,ok_W]**2).sum(axis=1)/ok_W.sum())
    stokesmaxchi_pc = np.abs(stokes_pcW[:,:,ok_W]/stokeserr_pcW[:,:,ok_W]).max(axis=2)
    if issky:
        stokessky_s = stokes_pcW[:,0,0]
        print "\nCentral %% Q, %% U (presumably sky), removed: %8.3f %8.3f" % tuple(stokessky_s)
        stokes_pcW[:,0,0] -= stokessky_s
        
    hdr = '/'.join(os.getcwd().split('/')[-2:])+' '+str(datetime.date.today()).replace('-','')
    for d in range(dims): hdr += ("\n %2s    "+cofs*"%9i ") % ((dname_d[d],)+tuple(pow_dc[d]))
    wav_l = np.tile(wav_W,2)    
    np.savetxt(name+"_stokeserr_pcW.txt",np.vstack((wav_l,stokeserr_pW.flatten(),    \
        stokeserr_pcW.transpose((1,0,2)).reshape((cofs,-1)))).T,fmt="%8.2f %8.3f "+cofs*"%9.4f ")              
    HWcalname = ['HWPol','HWZero'][issky]
    hwcalfile = "RSSpol_"+HWcalname+"_Im_v00.txt"
    np.savetxt(hwcalfile, np.vstack((wav_l,stokes_pcW.transpose((1,0,2)).reshape((cofs,-1)))).T,    \
        header=hdr, fmt="%8.2f "+cofs*"%9.4f ")
        
    pname_p = [['% P','PA'],['% Q','% U']][issky]        
    print ("\n rms:  %7.2f %7.2f" % tuple(stokesrms_p))    
    print ("\n maxchi vs coeffpow \n"+dims*"%4s "+" %7s %7s") % (tuple(dname_d)+tuple(pname_p))
    for c in range(cofs):
        print (dims*"%4i "+" %7.1f %7.1f") % (tuple(pow_dc[:,c])+tuple(stokesmaxchi_pc[:,c]))

    if len(outList)==0: return
        
  # save smoothed data, residual rawstokes
    for f in outList:   
        np.savetxt(name+"_stokes_"+str(f)+"_tw.txt", np.vstack((np.tile(wav_w,2), \
            stokes_fpTw[f].transpose((1,0,2)).reshape((Targets,-1)))).T, fmt="%8.2f "+Targets*"%9.4f ")
        if ispolcal:
            np.savetxt(name+"_stokes_"+str(f)+"_TW.txt",np.vstack((np.tile(wav_W,2),  \
                stokes_fpTW[f].transpose((1,0,2)).reshape((Targets,-1)))).T,    \
                fmt="%8.2f "+targets*"%9.4f ")

        stokesfit_ptw, okcal_tw = hwcal(hwcalfile,YXnorm*YX_dfT[:,f,:targets_f[f]],  \
            mbnorm*mbyx_df[:,f], wav_w)       
        ok_tw = (okcal_tw & ok_fTw[f,:targets_f[f]])
                    
        if issky:
            stokesfit_stw = stokesfit_ptw/100.
        else:
            PArad_tw = np.radians(stokesfit_ptw[1] + PA0)
            stokesfit_stw = stokesfit_ptw[None,0]*np.array([np.cos(2.*PArad_tw),np.sin(2.*PArad_tw)])/100.

        stokes_Stw = ok_tw[None,:,:].astype(float)*stokes_fSTw[f,:,:targets_f[f]]
        stokesfit_Stw = np.copy(stokes_Stw)        
        stokesfit_Stw[1:3] = stokes_Stw[0][None,:,:]*stokesfit_stw     
        
        stokesres_Stw = np.copy(stokes_Stw)   
        stokesres_Stw[1:3] = stokes_Stw[1:3] - stokesfit_Stw[1:3]
                   
        bpm_Stw = np.tile(np.logical_not(ok_tw).astype(int),(3,1,1))
        hduout = pyfits.open(fileList[f])        
        hduout['SCI'].data = stokesres_Stw
        hduout['BPM'].data = bpm_Stw    
        outfile = fileList[f].replace('stokes','resstokes')
        hduout.writeto(outfile,overwrite=True,output_verify='warn')
        print '\n    '+outfile+' Stokes I,Q,U' 

    return

#---------------------------------------------------------------------------------------------
 
if __name__=='__main__':
    files = [x.count("fits") for x in sys.argv[1:]].count(1)
    fileList = sys.argv[1:(files+1)]
    kwargs = dict(x.split('=', 1) for x in sys.argv[(files+1):])  
    impolcal(fileList, **kwargs)
