#! /usr/bin/env python

"""
specpolbinary

do BME, interstellar polarization analysis on specpol of binary star

"""

import os, sys, glob, warnings
import numpy as np
import pyfits
from scipy import linalg as la
from scipy.optimize import fmin
from scipy import interpolate as ip
from specpolserkow import serkowski

warnings.filterwarnings('ignore', message='Overwriting existing file')
np.set_printoptions(threshold=np.nan)

#--------------------------------------
def specpolbinary(period,epoch,linefile,infilelist,ISPA='free',lmax='5500',tslp='0'):
    """For each stokes fits file in infilelist, do BME analysis on continuum outside of lines in linefile
    Parameters
    ----------
    period: float
        period of binary (days)
    epoch: float
        HJD of phase 0
    lines: text file name
        Line: line-name cont1lam1 cont1lam2 linelam1 linelam2 cont2lam1 cont2lam2
    infilelist: list
        List of _stokes filenames
    ISPA: str
        if 'free', allow Interstellar PA to vary, otherwise use specified fixed value (deg)
    lmax: str
        if 'free', allow lambda-max to vary, otherwise use specified fixed value (Ang)
    tslp: str
        if 'free', allow dPA/dlambda to vary, otherwise use specified fixed value (deg/Ang) 

    """
    """
    _b: index in file list
    _S: unnormalized stokes : (0-3) = (I,Q,U,V)
    _s: normalized linear stokes: (0-1) = (q,u)
    _l: line
    _d: line apertures.  0,1 cont1, 2,3 line, 4,5 cont2
    _m: BME coefficient : (0-4) = (const,sin_b,cos_b,sin_2b,cos_2b)
    _w: wavelength in first file. Others are registered to it
    _W: wavelength in current file
    """
    global chi2_i,x_iV

    line_l = np.loadtxt(linefile,dtype='string',usecols=0)
    lines = line_l.shape[0]
    lam_ld = np.array((lines,6))
    lam_ld = np.loadtxt(linefile,usecols=(1,2,3,4,5,6))

    obss = len(infilelist)
    maxlen = np.array([len(infilelist[b].split('.')[0]) for b in range(obss)]).max()

    hdul = pyfits.open(infilelist[0])
    dlam = float(hdul['SCI'].header['CDELT1'])
    lam0 = float(hdul['SCI'].header['CRVAL1'])
    wavs = int(hdul['SCI'].header['NAXIS1'])
    lam_w = lam0 + dlam*np.arange(wavs)
    ok_w = (hdul['BPM'].data[:,0,:] == 0).all(axis=0)

    stokes_bSw = np.zeros((obss,3,wavs))
    var_bSw = np.zeros((obss,3,wavs))
    phase_b = np.zeros(obss)

    for b in range(obss):
        hdul = pyfits.open(infilelist[b])
        lam0b = float(hdul['SCI'].header['CRVAL1'])
        Wavs = int(hdul['SCI'].header['NAXIS1'])
        lam_W = lam0b + dlam*np.arange(Wavs)
        ok_W = (hdul['BPM'].data[:,0,:] == 0).all(axis=0)
        stokes_SW = hdul['SCI'].data[:,0,:]
        var_SW = hdul['VAR'].data[:3,0,:]
        jd = hdul['SCI'].header['JD']
        phase_b[b] = ((jd-2400000. - epoch)/period) % 1.

        Wshift0 = int((lam_W[0] - lam_w[0])/dlam)
        Wshift1 = int((lam_W[-1] - lam_w[-1])/dlam)
        Wlist = range(max(0,-Wshift0),min(Wavs,Wavs-Wshift1))
        wlist = range(max(0,Wshift0),min(wavs,wavs+Wshift1))

        ok_w &= (lam_w >= lam0b)
        ok_w &= (lam_w <= lam0b + Wavs*dlam)
        ok_w[wlist] &= ok_W[Wlist]
        stokes_bSw[b][:,wlist] = stokes_SW[:,Wlist]
        var_bSw[b][:,wlist] = var_SW[:,Wlist]

    stokes_bsw = np.zeros((obss,2,wavs))
    var_bsw = np.zeros((obss,2,wavs))
    stokes_bsw[:,:,ok_w] = stokes_bSw[:,1:][:,:,ok_w]/stokes_bSw[:,0][:,None,ok_w]
    var_bsw[:,:,ok_w] = var_bSw[:,1:][:,:,ok_w]/stokes_bSw[:,0][:,None,ok_w]**2

    okcont_w = np.copy(ok_w)
    for l in range(lines):
        okcont_w &= np.logical_not((lam_w >= lam_ld[l,2]) & (lam_w <= lam_ld[l,3]))

  # do BME Fourier fit vs phase, save results
    bmenamelist = ["const","cosphi","sinphi","cos2phi","sin2phi","cos2phi"]
    a_bm = np.vstack((np.ones(obss),np.cos(2.*np.pi*phase_b),np.sin(2.*np.pi*phase_b),   \
                   np.cos(2.*np.pi*2.*phase_b),np.sin(2.*np.pi*2.*phase_b))).T

    cof_msw,sumsqerr_sw = la.lstsq(a_bm,stokes_bsw.reshape((obss,-1)))[0:2]
    cof_msw = cof_msw.reshape((5,2,-1))

    sumsqerr_sw = sumsqerr_sw.reshape((2,-1))
    sigma_sw = np.sqrt(sumsqerr_sw/obss)    
    alpha_mm = (a_bm[:,:,None]*a_bm[:,None,:]).sum(axis=0)
    eps_mm = la.inv(alpha_mm)                                        # errors from Bevington, p121
    err_msw = sigma_sw[None,:,:]*np.sqrt(np.diagonal(eps_mm))[:,None,None]

    hdul = pyfits.open(infilelist[0])
    hdul['BPM'].data = np.tile(np.logical_not(ok_w).astype('uint8'),(3,1)).reshape((3,1,-1))

    for m in range(5):
        hdul['SCI'].data[1:3] = (cof_msw[m]*stokes_bSw[0,0]).astype('float32').reshape((2,1,-1))
        hdul['VAR'].data[1:3] = ((err_msw[m]*stokes_bSw[0,0])**2).astype('float32').reshape((2,1,-1))
        outfile = os.path.basename(infilelist[0]).replace('stokes',bmenamelist[m]+'_stokes')
        hdul.writeto(outfile,clobber=True,output_verify='ignore')
        print ('    %s Stokes I,Q,U') % outfile           

  # compute system parameters from BME coefficient means (BME p424)
    stokesBME_ms = cof_msw[:,:,okcont_w].mean(axis=2)
    errBME_ms = np.sqrt((err_msw[:,:,okcont_w]**2).mean(axis=2)/wavs)
    BMEPAph1 = (0.5*np.degrees(  \
        np.arctan((stokesBME_ms[1,1]+stokesBME_ms[2,0])/(stokesBME_ms[2,1]-stokesBME_ms[1,0])) -   \
        np.arctan((stokesBME_ms[1,1]-stokesBME_ms[2,0])/(stokesBME_ms[2,1]+stokesBME_ms[1,0]))) +180.) % 180.
    BMEPAph2 = (0.5*np.degrees(  \
        np.arctan((stokesBME_ms[3,1]+stokesBME_ms[4,0])/(stokesBME_ms[4,1]-stokesBME_ms[3,0])) -   \
        np.arctan((stokesBME_ms[3,1]-stokesBME_ms[4,0])/(stokesBME_ms[4,1]+stokesBME_ms[3,0]))) +180.) % 180.
    x1 = ((stokesBME_ms[1,1]+stokesBME_ms[2,0])**2 + (stokesBME_ms[2,1]-stokesBME_ms[1,0])**2) /   \
         ((stokesBME_ms[2,1]+stokesBME_ms[1,0])**2 + (stokesBME_ms[1,1]-stokesBME_ms[2,0])**2)
    BMEinclph1 = (np.degrees(np.arccos((np.sqrt(x1) - 1.)/(np.sqrt(x1) + 1.))) +180.) % 180.
    x2 = ((stokesBME_ms[3,1]+stokesBME_ms[4,0])**2 + (stokesBME_ms[4,1]-stokesBME_ms[3,0])**2) /   \
         ((stokesBME_ms[4,1]+stokesBME_ms[3,0])**2 + (stokesBME_ms[3,1]-stokesBME_ms[4,0])**2)
    BMEinclph2 = (np.degrees(np.arccos((x2**(0.25) - 1.)/(x2**(0.25) + 1.))) +180.) % 180.
    BMEph1 = (0.5*np.degrees(  \
        np.arctan((stokesBME_ms[1,1]+stokesBME_ms[2,0])/(stokesBME_ms[2,1]-stokesBME_ms[1,0])) +   \
        np.arctan((stokesBME_ms[1,1]-stokesBME_ms[2,0])/(stokesBME_ms[2,1]+stokesBME_ms[1,0]))) +180.) % 180.
    BMEph2 = (0.25*np.degrees(  \
        np.arctan((stokesBME_ms[3,1]+stokesBME_ms[4,0])/(stokesBME_ms[4,1]-stokesBME_ms[3,0])) +   \
        np.arctan((stokesBME_ms[3,1]-stokesBME_ms[4,0])/(stokesBME_ms[4,1]+stokesBME_ms[3,0]))) +180.) % 180.
    print ("\n      PA1      PA2     incl1    incl2     ph1     ph2 \n  "+6*"%8.2f ") %     \
            (BMEPAph1,BMEPAph2,BMEinclph1,BMEinclph2,BMEph1,BMEph2)

  # calculate smoothed continuum polarization wavelength dependence from BME phase dependent coefficients
    PABME_m = (0.5*np.degrees(np.arctan2(stokesBME_ms[:,1],stokesBME_ms[:,0])) + 180.) % 180.
    qintrBME_mw = np.cos(2.*np.radians(PABME_m))[:,None]*cof_msw[:,0] +    \
                np.sin(2.*np.radians(PABME_m))[:,None]*cof_msw[:,1]
    qintrvarBME_mw = (np.cos(2.*np.radians(PABME_m))[:,None]*cof_msw[:,0])**2 +    \
                (np.sin(2.*np.radians(PABME_m))[:,None]*cof_msw[:,1])**2
    wt_m = np.sqrt((stokesBME_ms**2).sum(axis=1))/(errBME_ms**2).sum(axis=1)
   
    qintrBME_w = (wt_m[1:,None]*qintrBME_mw[1:,:]).sum(axis=0)/wt_m[1:].sum()
    qintrvarBME_w = (wt_m[1:,None]**2*qintrvarBME_mw[1:,:]).sum(axis=0)/wt_m[1:].sum()**2

    for iter in range(2):                   # cull for excursions due to line profiles, etc
        a_Wd = np.vstack((np.ones(okcont_w.sum()),lam_w[okcont_w]-lam_w.mean())).T
        b_W = qintrBME_w[okcont_w]
        wt_W = 1./np.sqrt(qintrvarBME_w[okcont_w])
        cof_d,sumsqerr = la.lstsq(a_Wd*wt_W[:,None],b_W*wt_W)[0:2]
        chisq_W = ((qintrBME_w[okcont_w] - (cof_d[0] + cof_d[1]*(lam_w[okcont_w]-lam_w.mean())))*wt_W)**2
        if (iter==0): okcont_w[okcont_w] &= (chisq_W < 2.)
           
    qintrslp = cof_d[1]/cof_d[0]
    
    np.savetxt("qintrBME_w.txt",np.vstack((lam_w[okcont_w],100.*qintrBME_w[okcont_w],  \
       100.*np.sqrt(qintrvarBME_w[okcont_w]))).T,fmt="%8.1f %8.4f %8.4f ")

  # fit continuum of phase-constant wavelength dependence with Serkowski + 
  #   intrinsic(linear with BME relative slope) at BME PA
    stokesconst_sw = cof_msw[0]
    varconst_sw = err_msw[0]**2
    x0_v = np.array([0.,stokesBME_ms[0,0],stokesBME_ms[0,1],5500.,0.])
    v_V = np.arange(2)
    if ISPA=='free':    v_V = np.append(v_V,2)
    else:               x0_v[2] = float(ISPA)
    if lmax=='free':    v_V = np.append(v_V,3)
    else:               x0_v[3] = float(lmax)
    if tslp=='free':    v_V = np.append(v_V,4)
    else:               x0_v[4] = float(tslp)
    x0_V = x0_v[v_V]
    vars = x0_V.shape[0]
    x_v = x0_v
    chi2_i = np.empty(0)
    x_iV = np.empty((0,vars))

    xopt_V,chi2dof,iters,funcalls,warnflag = fmin(chi2_bmeserk,x0_V,  \
          args=(x_v,v_V,stokesconst_sw,varconst_sw,okcont_w,lam_w,BMEPAph2,qintrslp),full_output=True,disp=False)

    x_v[v_V] = xopt_V
    if not(2 in v_V):              # if PA is not free, x_v is Pmax,PA
        x_v[1:3] = x_v[1]*np.array([np.cos(2.*radians(x_v[2])),np.sin(2.*radians(x_v[2]))])
    stokesBMEconst_s = x_v[0]*np.array([np.cos(2.*np.radians(BMEPAph2)),np.sin(2.*np.radians(BMEPAph2))])

    errISM_s = 100.*np.sqrt(chi2dof/(1./varconst_sw[:,ok_w]).sum(axis=1))   # WRONG
#    np.savetxt(stokesconst_sw fit)

    print ("\nISM lambdamax  %8.2f Ang  tslope  %6.2f deg/1000 Ang \n") % (x_v[3],1000.*x_v[4])
    print ("                     %q     +/-       %u      +/-       %p       +/-      PA     +/-")
    print ("ISM pmax        "+6*"%8.4f "+2*"%8.2f ") % stokesprint(x_v[1:3],errISM_s)
    print ("BME constant    "+6*"%8.4f "+2*"%8.2f ") % stokesprint(stokesBMEconst_s,errBME_ms[0])
    print ("BME cos phase   "+6*"%8.4f "+2*"%8.2f ") % stokesprint(stokesBME_ms[1],errBME_ms[1])
    print ("BME sin phase   "+6*"%8.4f "+2*"%8.2f ") % stokesprint(stokesBME_ms[2],errBME_ms[2])
    print ("BME cos 2*phase "+6*"%8.4f "+2*"%8.2f ") % stokesprint(stokesBME_ms[3],errBME_ms[3])
    print ("BME sin 2*phase "+6*"%8.4f "+2*"%8.2f ") % stokesprint(stokesBME_ms[4],errBME_ms[4])

    print "\n %8.4f %i %i %i" % (chi2dof,iters,funcalls,warnflag)
    np.savetxt("BMEfit.txt",np.vstack((x_iV.T,chi2_i)).T,fmt=vars*"%8.4f "+"%10.2f ")

    return 
  
#--------------------------------------
def stokesprint(stokes_s,err_s):
    pp = 100.*np.sqrt((stokes_s**2).sum())
    pa = np.degrees(0.5*np.arctan2(stokes_s[1],stokes_s[0])) 
    pperr = 100.*np.sqrt((err_s**2).sum()/2.)
    paerr = np.degrees(0.5*pperr/pp)
    return 100.*stokes_s[0],100.*err_s[0],100.*stokes_s[1],100.*err_s[1],pp,pperr,pa,paerr

#--------------------------------------
def chi2_bmeserk(x_V,x_v,v_V,stokes_sw,var_sw,ok_w,lam_w,BMEPAph2,qintrslp):
    global chi2_i,x_iV
    wavs = lam_w.shape[0]
    samples = 2.*ok_w.sum()
    x_v[v_V] = x_V

    stokesintr_s = x_v[0]*np.array([np.cos(2.*np.radians(BMEPAph2)),np.sin(2.*np.radians(BMEPAph2))])
    stokesintr_sw = stokesintr_s[:,None]*(1.+qintrslp*(lam_w - lam_w.mean()))[None,:]
    if (2 in v_V):                  # if PA is free, x_v is qmax,umax, else it is Pmax,PA
        stokesmax_s = x_v[1:3]      
    else:
        stokesmax_s = x_v[1]*np.array([np.cos(2.*np.radians(x_v[2])),np.sin(2.*np.radians(x_v[2]))])

    lmax,tslp = x_v[3:]
    if ((lmax < 3000.) | (lmax > 10000.)): return 1.E8
    stokesserk_sw = serkowski(stokesmax_s,lmax,tslp,lam_w)
    chi2dof = ((stokes_sw[:,ok_w] - (stokesintr_sw[:,ok_w] +     \
                stokesserk_sw[:,ok_w]))**2/var_sw[:,ok_w]).sum()/samples
    chi2_i = np.append(chi2_i,chi2dof)
    x_iV = np.vstack((x_iV,x_V))

    return chi2dof

# ---------------------------------------------------------------------------------
# debug
# cd /d/pfis/khn/WRs/WR47/BMEallphases
# Pypolsalt specpolbinary.py 6.239 43912.48 WR47_lines.txt `cat WR47_obs.txt` lmax=free
 
if __name__=='__main__':
    period = float(sys.argv[1])
    epoch = float(sys.argv[2])
    linefile = sys.argv[3]
    infilelist=[x for x in sys.argv[4:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[4:] if x.count('.fits')==0) 
    specpolbinary(period,epoch,linefile,infilelist,**kwargs)
