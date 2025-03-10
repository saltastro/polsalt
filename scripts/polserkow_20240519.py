#! /usr/bin/env python

"""
polserkow

fit serkowski law to stokes.fits data, both specpol and imsppol data, multiple targets 

"""

import os, sys, glob, inspect
import numpy as np
from astropy.io import fits as pyfits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from sppolview import viewstokes
from polutils import specpolrotate

np.set_printoptions(threshold=np.nan)

#--------------------------------------
def polserkow(infileList, **kwargs):
    """For stokes fits files in infileList (same object), fit (together) or remove Serkowski law.

    Parameters
    ----------
    infileList: List of str (comma-sep in linux call)
       stokes filenames
    targetList: list of ints, [0] default
    pmax: float,%
    pamax: float,deg 
    lmax: str
        if 'free', allow lambda-max to vary, otherwise use specified fixed value, 5500 default (Ang)
    tslp: str
        if 'free', allow dPA/dlambda to vary, otherwise use specified fixed value, 0 default (deg/Ang)
    K: str
        if 'free', allow serk K parameter to vary freely, otherwise use WLR dependence on lmax

    """
    """
    _S: unnormalized stokes : (0-3) = (I,Q,U,V)
    _s: normalized linear stokes: (0-1) = (q,u)
    _o: observation index
    _w: wavelength
    _W: wavelength index for stacked observations
    _v: Serkowski parameter: qmax,umax,lmax,pslp
    _V: parameter allow to vary
    _i: function evaluation
    """

    targetList = kwargs.pop('targetList',[0,])
    pmax = float(kwargs.pop('pmax',0.))/100.         # input pmax is in %
    pamax = float(kwargs.pop('pamax',0.))
    lmax = kwargs.pop('lmax',5500.)
    if (lmax<>'free'): lmax = float(lmax)
    tslp = kwargs.pop('tslp',0.)
    if (tslp<>'free'): tslp = float(tslp)
    K = kwargs.pop('K','WLR')
    if (K<>'free'): K = 0.                          # 0 will be flagged as WLR  
    wavmin = float(kwargs.pop('wavmin', 0.))           
    remove = (kwargs.pop('remove','False') == 'True')    
    debug = (kwargs.pop('debug','False') == 'True')
    
    obss = len(infileList)

    hdul0 = pyfits.open(infileList[0])       
    ctypelist = (hdul0['SCI'].header['CTYPE3']).split(',')
    if (len(ctypelist)<3):
        print "Q,U not in ",infilelist
        exit()

    if 'WCSDVARR' in [hdul0[x].name for x in range(len(hdul0))]:
        wedge_W = hdul0['WCSDVARR'].data
        dwav = hdul0['SCI'].header['CDELT1']
        wav_w = (wedge_W[:dwav*wavs:dwav] + wedge_W[dwav:(wavs+1)*dwav:dwav])/2.

    x0_v = np.array([pmax*np.cos(2.*np.radians(pamax)),pmax*np.sin(2.*np.radians(pamax)),5500.,0.,WLR(5500.)])
    bounds_lv = np.array([[-1.,-1.,4000.,-005.,0.4],[1.,1.,8500.,.005,2.0]])
    v_V = np.array([0,1],dtype=int)
    if lmax=='free':    
        v_V = np.append(v_V,2)
    else:               
        x0_v[2] = float(lmax)
    if tslp=='free':    
        v_V = np.append(v_V,3)
    else:               
        x0_v[3] = float(tslp)
    if K=='free':    
        v_V = np.append(v_V,4)
    else:               
        x0_v[4] = 0.        
    bounds_lV = bounds_lv[:,v_V]

    if (not remove):
        print ("tgt  %qmax    +/-     %umax    +/-    lmax(A)  +/-   deg/1000A  +/-     K   +/-    %pmax    +/-    PA (deg)   +/-  chi2dof") 

    stokess,targets = pyfits.open(infileList[0])['SCI'].data.shape[0:2]
    hdul_o = np.zeros(obss,dtype=object)   
    stokes_StW = np.zeros((stokess,targets,0))
    var_StW = np.zeros((stokess+1,targets,0))
    covar_StW = np.zeros((stokess,targets,0))    
    ok_StW = np.zeros_like(stokes_StW,dtype=bool)
    wav_W = np.zeros(0)    

    for o in range(obss):
        hdul_o[o] = pyfits.open(infileList[o])    
        stokes_StW = np.concatenate((stokes_StW,hdul_o[o]['SCI'].data),axis=2)
        var_StW = np.concatenate((var_StW,hdul_o[o]['VAR'].data),axis=2)
        covar_StW = np.concatenate((covar_StW,hdul_o[o]['COV'].data),axis=2)        
        wavs = hdul_o[o]['SCI'].data.shape[2]
        dwav = float(hdul_o[o]['SCI'].header['CDELT1'])
        wav0 = float(hdul_o[o]['SCI'].header['CRVAL1'])
        wav_w = wav0 + dwav*np.arange(wavs)
        ok_Stw = ((hdul_o[o]['BPM'].data == 0) & (wav_w >= wavmin)[None,None,:])                
        ok_StW = np.concatenate((ok_StW,ok_Stw),axis=2)
        wav_W = np.concatenate((wav_W,wav_w))
               
    for t in targetList: 
        ok_W = ok_StW[0,t]        
        nstokes_sW = (stokes_StW[1:3,t][:,ok_W]/stokes_StW[0,t][ok_W]).astype('float64')
        nvar_sW = (var_StW[1:,t][:,ok_W]/stokes_StW[0,t][ok_W]**2).astype('float64')
        ncovar_sW = (covar_StW[1:,t][:,ok_W]/stokes_StW[0,t][ok_W]**2).astype('float64')        
        wav_m = np.tile(wav_W[ok_W].astype('float64'),2)

        x0_v[:2] = nstokes_sW.mean(axis=1)
        x0_V = np.copy(x0_v[v_V])
        vars = x0_V.shape[0]

        xopt_V, covar_VV = curve_fit(makeserk(x0_v,v_V,nstokes_sW,nvar_sW,debug=debug), wav_m,   \
            nstokes_sW.flatten(), sigma=np.sqrt(nvar_sW[:2]).flatten(), p0=tuple(x0_V),bounds=bounds_lV,    \
            absolute_sigma=True)

        x_v = np.copy(x0_v)
        x_v[v_V] = xopt_V
        if x_v[4]==0.: x_v[4] = WLR(x_v[2])
        
      # compute revised chi2dof using sliding shorter windows
      # chi_sW rotated to put q in P dirn
      # _n: window lengths
      # _h: sliding step for each _n
        vstokes_vW, verr_vW = viewstokes(stokes_StW[:,t,ok_W],var_StW[:,t,ok_W])
        chi_sW = (nstokes_sW - serkowski(wav_W[ok_W],*x_v))

        chi_sW = specpolrotate(chi_sW,nvar_sW,ncovar_sW,-vstokes_vW[1],normalized=True)[0] 

        Wmin = np.where(wav_W==wav_W[ok_W][0])[0][0]
        Wmax = np.where(wav_W==wav_W[ok_W][-1])[0][0]
        dwav_n = np.arange(10,min(1000.,wav_W[Wmax]-wav_W[Wmin]),10)
        chilengths = len(dwav_n)
        chi2dofmax_ns = np.zeros((chilengths,2))
        chi2dofrms_ns = np.zeros((chilengths,2))        
        dwavmax_ns = np.zeros((chilengths,2))
        for n,dwavn in enumerate(dwav_n):
            Whsamples = dwavn/dwav        
            W0_H = np.arange(Wmin,np.where(wav_W==wav_W[Wmax]-dwavn)[0][0])
            ok_H = ok_W[W0_H]            
            hWList = list(W0_H[ok_H])
            chisamples = len(hWList)
            chi2dof_hs = np.zeros((chisamples,2))
            Wlen_h = np.zeros(chisamples)
            for h,W0 in enumerate(hWList):
                WList = np.where((wav_W[ok_W] > wav_W[W0]) & (wav_W[ok_W] < wav_W[W0]+dwavn))[0]
                if (len(WList)<dwavn/2): continue                 
                chi2dof_hs[h] = chi_sW[:,WList].sum(axis=1)**2/(nvar_sW[:2,WList].sum(axis=1))
                Wlen_h[h] = len(WList)
            chi2dofrms_ns[n] = chi2dof_hs.mean(axis=0)
            chi2dofmax_ns[n] = chi2dof_hs.max(axis=0)            

            if (dwavn==20.):
                np.savetxt("chi2dof_20_h.txt",np.vstack((wav_W[hWList],Wlen_h,chi2dof_hs.T)).T,fmt=" %8.3f")
            
            dwavmax_ns[n] = wav_W[np.array([np.where(chi2dof_hs[:,s]==chi2dofmax_ns[n,s])[0][0] for s in (0,1)])]
        np.savetxt("chi2dof_n.txt",np.vstack((dwav_n,dwavmax_ns.T,chi2dofrms_ns.T)).T,fmt=" %8.3f")
        exit()
        
        chi2dof = ((nstokes_sW - serkowski(wav_W[ok_W],*x_v))**2/nvar_sW).sum()/len(wav_m)        
        xerr_v = np.zeros(5)
        xerr_v[v_V] = np.sqrt(np.diag(covar_VV))
        xpr_v = x_v*np.array([100.,100.,1.,1000.,1.])               # print q,u in %, tslp in deg/1000Ang         
        xerrpr_v = xerr_v*np.array([100.,100.,1.,1000.,1.])
        pmaxpr = np.sqrt((xpr_v[:2]**2).sum())
        PA = 0.5*np.arctan2(xpr_v[1],xpr_v[0])
        PApr = (180. + np.degrees(PA)) % 180. 
        pmaxerrpr = np.sqrt((xerrpr_v[0]*np.cos(2.*PA))**2 + (xerrpr_v[1]*np.sin(2.*PA))**2)     
        PAerrpr = np.degrees(0.5)*np.sqrt((xerrpr_v[0]*np.sin(2.*PA))**2 + (xerrpr_v[1]*np.cos(2.*PA))**2)/pmaxpr       

        print ("%2i" % t),
        print (2*"%8.4f %7.4f "+"%7.1f %6.1f %8.3f %7.3f %6.3f %5.3f") % tuple(np.vstack((xpr_v,xerrpr_v)).T.ravel()),
        print ("%7.4f %7.4f %8.3f %7.3f ") % (pmaxpr,pmaxerrpr,PApr,PAerrpr),               
        print "%6.2f" % chi2dof
        
    if remove:
        for o in range(obss):
            stokes_Stw = hdul_o[o]['SCI'].data
            nstokes_sw = serkowski(wav_w,*x0_v)
            stokescor_Stw[1:3,t,ok_w] = stokes_Stw[1:3,t,ok_w] - (nstokes_sw[:,ok_w]*stokes_Stw[0,t,ok_w])  
            outfile = raw_input("output file name: ")
            if (len(outfile)==0): outfile=infile.split('stokes')[0]+'iscor_stokes.fits'
            hdul['SCI'].data = stokescor_Stw                        
            hdul[0].header.add_history("ISpol: %6.2f %8.2f %5.0f %6.2f" % (100.*pmax,pamax,lmax,tslp))            
            hdul.writeto(outfile,overwrite=True,output_verify='warn')
            print (outfile+' ISpol corrected Stokes I,Q,U')                         

    return xpr_v, xerrpr_v, chi2dof

#--------------------------------------
def WLR(lmax):
    K = -0.1+1.86*lmax/10000
    return K
    
#--------------------------------------
def serkowski(wav_w,qmax,umax,lmax,tslp,K):
    if (K==0.):
        K = WLR(lmax)
    serk_w = np.exp(-K*(np.log(wav_w/lmax))**2)
    pmax = np.sqrt(qmax**2 + umax**2)
    tmax = np.degrees(np.arctan2(umax,qmax)/2.)
    serk_sw = pmax*serk_w*np.array([np.cos(2.*np.radians(tmax+(wav_w-lmax)*tslp)), \
                                np.sin(2.*np.radians(tmax+(wav_w-lmax)*tslp))])        
    return serk_sw

#--------------------------------------
def makeserk(x0_v,v_V,nstokes_sW,nvar_sW,debug=False):
    def serk(wav_m, *par):  
        wav_W = wav_m.reshape((2,-1))[0]
        x_v = np.copy(x0_v)
        x_v[v_V] = np.array(par)                
        serkfit_sW = serkowski(wav_W,*x_v)
        if debug:
            chi2dof = ((nstokes_sW-serkfit_sW)**2/nvar_sW).sum()/len(wav_m)
            print ((len(v_V)+1)*"%16.12f ") % (tuple(x_v[v_V])+(chi2dof,))                    
        return serkfit_sW.flatten()
    
    return serk
  
#--------------------------------------------------------------------------------------------- 
if __name__=='__main__':
    infileList=sys.argv[1].split(',')
    kwargs = dict(x.split('=', 1) for x in sys.argv[2:] )        
    polserkow(infileList,**kwargs)
    
# cd /d/pfis/khn/20210224/sci_x2EtaCar
# python script.py polserkow.py EtaCarina*1234_stokes.fits remove=True pmax=3.117 pamax=87.766
# cd /d/pfis/khn/20230210/sci
# cd /d/pfis/khn/20230221/sci
# python2.7 script.py polserkow.py HD298383_1_c0_oldcal_stokes.fits,HD298383_2_c1_oldcal_stokes.fits pmax=5.28 pamax=148.404 lmax=free tslp=free K=free
# python2.7 script.py polserkow.py HD298383_1_c0_calxy_stokes.fits,HD298383_2_c1_calxy_stokes.fits pmax=5.28 pamax=148.404 lmax=free tslp=free K=free

