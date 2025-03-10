#! /usr/bin/env python2.7

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
from polutils import specpolrotate, chi2sample, fence

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
        if 'WLR' (default), use WLR dependence on lmax
        if 'free', allow serk K parameter to vary freely 
        otherwise use specified fixed value

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
        K = WLR(5500.)
    elif K=='WLR':
        x0_v[4] = 0.
    else:               
        x0_v[4] = float(K)        
    bounds_lV = bounds_lv[:,v_V]

    if (not remove):
        print ("tgt  %qmax    +/-     %umax    +/-    lmax(A)  +/-   deg/1000A  +/-     K   +/-    %pmax    +/-    PA (deg)   +/-   chi2P  chi2PA") 

    stokess,targets = pyfits.open(infileList[0])['SCI'].data.shape[0:2]
    hdul_o = np.zeros(obss,dtype=object)   
    stokes_StW = np.zeros((stokess,targets,0))
    var_StW = np.zeros((stokess+1,targets,0))
    covar_StW = np.zeros((stokess,targets,0))    
    ok_StW = np.zeros_like(stokes_StW,dtype=bool)
    wav_W = np.zeros(0)
    wav0_o = np.zeros(obss)    

    for o in range(obss):
        hdul_o[o] = pyfits.open(infileList[o])
        wav0_o[o] = float(hdul_o[o]['SCI'].header['CRVAL1'])        
    o_O = np.argsort(wav0_o)
            
    for o in o_O:                   # stack observations in order of wavelength            
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
    Wavs = wav_W.shape[0]
               
    for t in targetList: 
        ok_W = ok_StW[0,t]
        W_U = np.where(ok_W)[0]       
        nstokes_sU = (stokes_StW[1:3,t][:,ok_W]/stokes_StW[0,t][ok_W]).astype('float64')
        nvar_sU = (var_StW[1:,t][:,ok_W]/stokes_StW[0,t][ok_W]**2).astype('float64')
        ncovar_sU = (covar_StW[1:,t][:,ok_W]/stokes_StW[0,t][ok_W]**2).astype('float64')        
        wav_m = np.tile(wav_W[ok_W].astype('float64'),2)

        x0_v[:2] = nstokes_sU.mean(axis=1)
        x0_V = np.copy(x0_v[v_V])
        vars = x0_V.shape[0]

        xopt_V, covar_VV = curve_fit(makeserk(x0_v,v_V,nstokes_sU,nvar_sU,debug=debug), wav_m,   \
            nstokes_sU.flatten(), sigma=np.sqrt(nvar_sU[:2]).flatten(), p0=tuple(x0_V),bounds=bounds_lV,    \
            absolute_sigma=True)

        x_v = np.copy(x0_v)
        x_v[v_V] = xopt_V
        if x_v[4]==0.: x_v[4] = WLR(x_v[2])
        
      # compute revised chi2dof using Serkowski-appropriate 800 Ang sliding window on fit deviation
      # dev_sU rotated to put q in P dirn
      # _h: sliding step      

        PA = 0.5*np.arctan2(x_v[1],x_v[0])
        PApr = (180. + np.degrees(PA)) % 180. 
        PAserk_W = PApr + (wav_W - x_v[2])*x_v[3]        
        dev_sU = nstokes_sU - serkowski(wav_W[ok_W],*x_v)
        dev_sU = specpolrotate(dev_sU,nvar_sU,ncovar_sU,-PAserk_W[ok_W],normalized=True)[0] 

        chi2dof_hs,mean_hs,var_hs,wav_h,len_h = chi2sample(dev_sU,nvar_sU[:2],True,wav_W[ok_W],800.)              
        chi2dof_s = chi2dof_hs.mean(axis=0)

        if debug:
            np.savetxt("polserkow_debug1.txt",np.vstack((wav_W[ok_W],PAserk_W[ok_W],dev_sU)).T, \
                fmt=(" %7.0f "+3*" %10.4f")) 
            np.savetxt("polserkow_debug2.txt",np.vstack((wav_h,len_h,chi2dof_hs.T,  \
                100.*mean_hs.T,100.*np.sqrt(var_hs).T)).T,   \
                fmt=(" %7.0f %6i "+6*" %10.4f"))
               
        xerr_v = np.zeros(5)
        xerr_v[v_V] = np.sqrt(np.diag(covar_VV))
        xerr_v[[0,1,2,4]] *= np.sqrt(chi2dof_s[0])                  # pol mag errors
        xerr_v[3] *= np.sqrt(chi2dof_s[1])                          # PA related errors
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
        print "%6.2f %6.2f" % tuple(chi2dof_s)
        
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

    return xpr_v, xerrpr_v, chi2dof_s, chi2dof_hs, mean_hs, var_hs, wav_h, len_h

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
            chi2dof = ((nstokes_sW-serkfit_sW)**2/nvar_sW[:2]).sum()/len(wav_m)
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
# cd /d/pfis/khn/20230214/sci
# cd /d/pfis/khn/20230221/sci
# polserkow.py HD298383_1_c0_oldcal_stokes.fits,HD298383_2_c1_oldcal_stokes.fits pmax=5.28 pamax=148.404 lmax=free tslp=free K=free
# polserkow.py HD298383_1_c0_calxy_stokes.fits,HD298383_2_c1_calxy_stokes.fits pmax=5.28 pamax=148.404 lmax=free tslp=free K=free

# cd /d/pfis/khn/20240724/newcal, 721, 822, 823
# polserkow.py OmiSco_1_c0_12345_calxy9_stokes.fits,OmiSco_2_c0_12345_calxy9_stokes.fits pmax=4.3 pamax=33.2 lmax=free tslp=free K=free
# polserkow.py OmiSco_1_c0_12345_calxy10_stokes.fits,OmiSco_2_c0_12345_calxy10_stokes.fits pmax=4.3 pamax=33.2 lmax=free tslp=free K=free

# cd /d/pfis/khn/20240706/sci
# polserkow.py CoalSackD-1_c0_calxy_stokes.fits pmax=3.9 pamax=70 lmax=free tslp=free
# polserkow.py CoalSackD-1_c0_calxy_stokes.fits pmax=3.9 pamax=70 lmax=free tslp=free K=1.15
# cd /usr/users/khn/salt/polarimetry/Standards/Hiltner652
# polserkow.py 20241001_Hiltner652_1_calxy9_stokes.fits,20241001_Hiltner652_2_calxy9_stokes.fits pmax=6.5 pamax=179.5 lmax=free tslp=free K=free

# cd /d/pfis/khn/20250203/newcal
# polserkow.py Vela1-95_1_c0_12345678_calxy9_stokes.fits,Vela1-95_2_c0_1234_calxy9_stokes.fits pmax=8.3 pamax=172.3 lmax=free tslp=free K=free
