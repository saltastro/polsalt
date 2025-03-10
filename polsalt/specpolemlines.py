#! /usr/bin/env python

"""
specpolemlines

polarimetric emission line analysis 

"""

import os, sys, glob
import numpy as np
from astropy.io import fits as pyfits
from scipy import linalg as la

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/scripts/',))

from specpolfinalstokes import specpolrotate
from specpolview import viewstokes

from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
np.set_printoptions(threshold=np.nan)
    
#--------------------------------------
def profstat(Stokes_W, varStokes_W, lam_W, lam0, dlam, isabs='find', debug=False):
    Wavs = len(lam_W)
    Stokestot = Stokes_W.sum() 
    varStokestot = varStokes_W.sum()
    if (isabs=='find'):
        isabs = (Stokestot < 0.)
    if isabs:            
        argmax = np.clip(np.argmin(Stokes_W),2,len(Stokes_W)-3)
    else:
        argmax = np.clip(np.argmax(Stokes_W),2,len(Stokes_W)-3)
            
    fm,f0,fp = tuple(Stokes_W[argmax-1:argmax+2])
    darg = -0.5*(fp - fm)/(fp + fm - 2.*f0)    
    maxline = f0 + (f0 - (fm + fp)/2.)*darg**2
    width = 3.e5*(dlam*Stokestot/maxline)/lam0
    widerr = 3.e5*dlam*(Stokestot/Stokes_W[argmax])* \
        np.sqrt(varStokestot/Stokestot**2 + varStokes_W[argmax]/Stokes_W[argmax]**2)/lam0
                
    usevel_W = (np.abs(np.arange(Wavs) - argmax) <  0.5*Stokestot/maxline)
    if ((argmax-2 < 0) | (argmax+3 > Wavs - 1)):
        print "%8.2f line peak too close to edge of window" % lam0
        return 0., 0., width, widerr, isabs
    usevel_W[argmax-2:argmax+3] = True          # ensure at least 6 points surrounding max 
    usevel_W[argmax+3*np.sign(argmax)] = True               
    cof_d,V_dd = np.polyfit((np.arange(Wavs) - argmax)[usevel_W],Stokes_W[usevel_W],2,  \
        w=1./np.sqrt(varStokes_W)[usevel_W],cov=True)
    err_d = np.sqrt(V_dd.diagonal())
    darg = -0.5*cof_d[1]/cof_d[0]
    dargvar = 0.25*((cof_d[1]/cof_d[0])**2)*((err_d[1]/cof_d[1])**2 + (err_d[0]/cof_d[0])**2)          
    vel = 3.e5*(lam_W[argmax] + dlam*darg - lam0)/lam0
    velerr = 3.e5*np.sqrt(dargvar)*dlam/lam0

    return vel, velerr, width, widerr, isabs
            
#--------------------------------------
def specpolemlines(linefile,infilelist,debug=False):
    """For each stokes fits file in infilelist, analyse lines in lines table.

    Parameters
    ----------
    linefile: text file name
        Line: line-name cont1lam1 cont1lam2 linelam1 linelam2 cont2lam1 cont2lam2
    infilelist: list
        List of _stokes filenames 


    """
    """
    _b: index in file list
    _S: unnormalized stokes : (0-3) = (I,Q,U,V)
    _s: normalized linear stokes: (0-1) = (q,u)
    _l: line
    _d: line apertures.  0,1 cont1, 2,3 line, 4,5 cont2
    _w: wavelength idx in current file
    _W: wavelength idx in current line
    """

    line_l = np.loadtxt(linefile,dtype='string',usecols=0,ndmin=1)   
    lam0_l = np.loadtxt(linefile,usecols=1,ndmin=1)
    lam_ld = np.loadtxt(linefile,usecols=(2,3,4,5,6,7),ndmin=2)
    lines = line_l.shape[0]    
    obss = len(infilelist)

    for b in range(obss):
        print "\n",infilelist[b]
        print ("        line     Wav     EW     vel     wid       p   +/-       PA  +/-    dQPvel  +/-   dQPwid  +/-  dPA/dwid  +/- dPA/wid^2 +/-")
        print ("                 Ang    Ang   km/sec  km/sec      %            deg          %wid          %wid        deg/wid       deg/wid^2")                   
    
        hdul = pyfits.open(infilelist[b])
        dlam = float(hdul['SCI'].header['CDELT1'])
        lam0 = float(hdul['SCI'].header['CRVAL1'])
        wavs = int(hdul['SCI'].header['NAXIS1'])
        lam_w = lam0 + dlam*np.arange(wavs)
        ctypelist = (hdul['SCI'].header['CTYPE3']).split(',')
        if (len(ctypelist)<3):
            print "Q,U not in ",infilelist[b]
            continue
        stokes_Sw = hdul['SCI'].data[:,0,:]
        var_Sw = hdul['VAR'].data[:3,0,:]
        covar_Sw = hdul['COV'].data[:,0,:]  
        ok_Sw = hdul['BPM'].data[:,0,:] == 0
        ok_w = ok_Sw.all(axis=0)
        samples = ok_w.sum()

        stokes_sw = np.zeros((2,wavs))
        var_sw = np.zeros((2,wavs))
        covar_sw = np.zeros((2,wavs))
        stokes_sw[:,ok_w] = stokes_Sw[1:][:,ok_w]/stokes_Sw[0][ok_w]
        var_sw[:,ok_w] = var_Sw[1:][:,ok_w]/stokes_Sw[0][ok_w]**2
        covar_sw[:,ok_w] = covar_Sw[1:][:,ok_w]/stokes_Sw[0][ok_w]**2
        pline_l = np.zeros(lines)
        plineerr_l = np.zeros(lines)
        paline_l = np.zeros(lines)
        palineerr_l = np.zeros(lines)        

        for l in range(lines):        
            lam_d = lam_ld[l]
            w_d = ((lam_d - lam0)/dlam).round().astype(int)
            okcont1_w = (ok_w & (np.arange(wavs) >= w_d[0]) & (np.arange(wavs) <= w_d[1]))
            okcont2_w = (ok_w & (np.arange(wavs) >= w_d[4]) & (np.arange(wavs) <= w_d[5]))
            okcont_w = (okcont1_w | okcont2_w)
            wcont1 = np.where(okcont1_w)[0].mean()
            wcont2 = np.where(okcont2_w)[0].mean()            
            stokescont_Sw = stokes_Sw[:,okcont1_w].mean(axis=1)[:,None] +   \
                    ((np.arange(wavs) - wcont1)/(wcont2-wcont1))[None,:]*        \
                    (stokes_Sw[:,okcont2_w].mean(axis=1)-stokes_Sw[:,okcont1_w].mean(axis=1))[:,None]
            stokescontdevvar_S = (np.std((stokes_Sw - stokescont_Sw)[:,okcont_w],axis=1))**2                    
            stokescontphotvar_S = (var_Sw + 2.*covar_Sw)[:3,okcont_w].mean(axis=1)
            stokescontvar_S = np.maximum(stokescontphotvar_S,stokescontdevvar_S)/(okcont_w.sum())**2
                    
            okline_w = (ok_w & (np.arange(wavs) >= w_d[2]) & (np.arange(wavs) <= w_d[3]))
            stokesline_SW = (stokes_Sw - stokescont_Sw)[:,okline_w]
            var_SW = var_Sw[:,okline_w] + stokescontvar_S[:,None]
            covar_SW = covar_Sw[:,okline_w]
            lam_W = lam_w[okline_w]
            Wavs = lam_W.shape[0]
            
            stokesline_S = stokesline_SW.sum(axis=1)
            varline_S = (var_SW[:3] + 2.*covar_SW[:3]).sum(axis=1)
            velint, velinterr, widint, widinterr, isabs  =     \
                profstat(stokesline_SW[0],var_SW[0],lam_W,lam0_l[l],dlam)
            EWint = dlam*stokesline_S[0]/(stokescont_Sw[0][okline_w].mean())
            if (abs(EWint) < 0.5*dlam): continue                      # do not analyse very weak lines

            pline_l[l] = np.sqrt((stokesline_S[1:3]**2).sum())/stokesline_S[0]
            plineerr_l[l] = np.sqrt(varline_S[1:3].sum())/stokesline_S[0]
            paline_l[l] = np.mod(np.degrees(np.arctan2(stokesline_S[2],stokesline_S[1])/2.)+180,180.)
            palineerr_l[l] = 0.5*np.degrees(plineerr_l[l]/pline_l[l])   

            if (pline_l[l]/plineerr_l[l] > 10.):
                QProt_s = np.array([np.cos(np.radians(2.*paline_l[l])),np.sin(np.radians(2.*paline_l[l]))]) 
                QPline_W = (stokesline_SW[1:3]*QProt_s[:,None]).sum(axis=0)
                QPlinevar_W = (var_SW[1:3]*(QProt_s[:,None])**2).sum(axis=0)                
                dvel,velerr,dwid,widerr,dum =     \
                        profstat(QPline_W,QPlinevar_W,lam_W,lam0_l[l],dlam,isabs=isabs,debug=debug)
                lamQP0 = lam0_l[l]*(1. + dvel/3.e5)                        
                dvel = (dvel - velint)/widint
                velerr = velerr/widint
                dwid = (dwid - widint)/widint
                widerr = widerr/widint 

                PA_W = 0.5*np.arctan2(stokesline_SW[2],stokesline_SW[1])        # radians
                dPA_W = np.mod(PA_W - np.radians(paline_l[l]) + np.pi/2., np.pi) - np.pi/2. 
                varPA_W = (var_SW/stokesline_SW**2)[1:3].sum(axis=0)*   \
                    ((stokesline_SW[1]/stokesline_SW[2])/(1. +  (stokesline_SW[1]/stokesline_SW[2])**2))**2   
                cof_d,V_dd = np.polyfit(lam_W-lamQP0,dPA_W,2,cov=True)
                cof_d = np.degrees(cof_d)
                err_d = np.degrees(np.sqrt(V_dd.diagonal()))
                dPAslp,slperr = np.array([cof_d[1],err_d[1]])*widint*(lam0_l[l]/3.e5)
                dPAquad,quaderr = np.array([cof_d[0],err_d[0]])*2.*(0.5*widint*(lam0_l[l]/3.e5))**2
                                               
                print (("%12s %8.2f %6.2f "+2*"%7.1f "+"%7.3f %6.3f "+5*"%7.1f %5.1f ") % \
                    (line_l[l],lam0_l[l],EWint,velint,widint,100.*pline_l[l],100.*plineerr_l[l],paline_l[l], \
                    palineerr_l[l],100.*dvel,100.*velerr,100.*dwid,100.*widerr,dPAslp,slperr,dPAquad,quaderr))
                
                if debug:
                    dPAfit_W = np.polyval(cof_d,lam_W-lamQP0)
                    np.savetxt("line_"+str(l)+"_polprof.txt", np.vstack((lam_W-lamQP0,QPline_W,    \
                    np.sqrt(QPlinevar_W),np.degrees(dPA_W),np.degrees(np.sqrt(varPA_W)),dPAfit_W)).T,   \
                    fmt="%8.2f %10.2e %10.2e %8.3f %8.3f %8.3f ")
                    
            else:
                print ("%12s %8.2f %6.2f "+2*"%7.1f "+"%7.3f %6.3f %7.1f %5.1f ") % \
                    (line_l[l],lam0_l[l],EWint,velint,widint,100.*pline_l[l],100.*plineerr_l[l],    \
                    paline_l[l],palineerr_l[l])           

    return pline_l,plineerr_l,paline_l,palineerr_l  


#--------------------------------------           
# current use
# cd /d/pfis/khn/20210224/sci_x2EtaCar
# python polsalt.py specpolemlines.py EtaCar_lines.txt EtaCarina*_iscor3_*stokes.fits debug=True
 
if __name__=='__main__':
    linefile = sys.argv[1]
    infileList = [x for x in sys.argv[2:] if (x.count('.fits')>0) ]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.')==0)          
    specpolemlines(linefile,infileList,**kwargs)
  
