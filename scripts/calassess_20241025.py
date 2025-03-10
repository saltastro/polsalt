#! /usr/bin/env python2.7

import os, sys, glob
import numpy as np
from astropy.io import fits as pyfits
from astropy.table import Table

polsaltdir = '/usr/users/khn/src/salt/polsaltcurrent/'
datadir = polsaltdir+'polsalt/data/'
sys.path.extend((polsaltdir+'/scripts/',))
from sppolview import viewstokes
from polutils import specpolrotate, chi2sample, fence
import rsslog
from obslog import create_obslog
keywordfile = datadir+"obslog_config.json"

"""
calibration assessment script for standalone accuracy, polfinalstokes printout
    paripple width vs wavelength fit for chisq
    infileList:     name_stokes.fits
    reduce all targets, do assessment only for specified catid (default t00, catid 12)
"""

ripplewidth_c = np.array([4.25E-05, -0.326, 904.967])           # from ripplewaves.xlsx
    
def calaccuracy_assess(infile, **kwargs):

    doprint = (str(kwargs.pop('doprint','False')).lower() == 'true')
    if doprint:
        logfile= 'accassess.log'
        rsslog.history(logfile)
        rsslog.message(str(kwargs), logfile)
    dofiles = (str(kwargs.pop('dofiles','False')).lower() == 'true')
    okmask_w = kwargs.pop('okmask','')
    minwav = float(kwargs.pop('minwav','0.'))     
        
    debug = (str(kwargs.pop('debug','False')).lower() == 'true')
    catid = int(kwargs.pop('catid',12))

    name = infile.rsplit("_",1)[0]             
    hdul = pyfits.open(infile)       
    wavs = hdul['SCI'].data.shape[-1]
    wav0 = hdul['SCI'].header['CRVAL1']
    dwav = hdul['SCI'].header['CDELT1']
    wav_w = wav0 + dwav*np.arange(wavs)
                                              
    rawstokess = hdul['SCI'].shape[0] - 1          
                
    stokes_Sw = np.zeros((rawstokess+1,wavs))
    var_Sw = np.zeros((rawstokess+2,wavs))
    covar_Sw = np.zeros_like(stokes_Sw)
    stokes_sw = np.zeros((rawstokess,wavs))
    var_sw = np.zeros((rawstokess+1,wavs))
    covar_sw = np.zeros_like(stokes_sw)                 
    ok_w = np.zeros(wavs,dtype=bool)      
    targets = hdul['SCI'].data.shape[1]
    tassess = 0
    if (targets > 1):
        tgtTab = Table.read(hdul['TGT'])           
        entries = len(tgtTab['CATID'])
        oktgt_i = (tgtTab['CULL'] == '')
        i_t = np.where(oktgt_i)[0]                 
        tassess = np.where(i_t==catid)[0][0]

    stokes_Sw = hdul['SCI'].data[:,tassess,:]
    var_Sw = hdul['VAR'].data[:,tassess,:]
    if ('COV' in hdul):                     # use zero if COV not computed
        covar_Sw = hdul['COV'].data[:,tassess,:] 
    bpm_Sw = hdul['BPM'].data[:,tassess,:]                        
    ok_w = (bpm_Sw ==0).all(axis=0)
    if (minwav > 0.):
        ok_w &= (wav_w > minwav)
    if len(okmask_w):
        ok_w &= okmask_w      
               
    stokes_sw[:,ok_w] = stokes_Sw[1:,ok_w]/stokes_Sw[0,ok_w]
    var_sw[:,ok_w] = var_Sw[1:,ok_w]/stokes_Sw[0,ok_w]**2
    covar_sw[:,ok_w] = covar_Sw[1:,ok_w]/stokes_Sw[0,ok_w]**2 

  # cull if intensity > 1.25* 200 Ang running median
    cullint = 1.25
    cullwav = 200.
    runmedint_w = np.zeros(wavs)
    cullint_w = np.zeros(wavs,dtype=bool)    
    for w in range(wavs):
        runwav_w = (ok_w & (wav_w > (wav_w[w] - cullwav/2.)) & (wav_w <= (wav_w[w] + cullwav/2.)))
        if (runwav_w.sum() > 8):         
            runmedint_w[w] = np.median(stokes_Sw[0,runwav_w])                    
            cullint_w[w] = stokes_Sw[0,w] > cullint*runmedint_w[w]
    okfit_w = ok_w & np.logical_not(cullint_w) 
    
    vstokes_vw, verr_vw = viewstokes(stokes_Sw,var_Sw,ok_w=ok_w)
    PAmean = np.median(vstokes_vw[1,okfit_w])        
    vstokes_vw[1] = (vstokes_vw[1] - PAmean + 90.) % 180. + PAmean - 90.              
    pawt_w = np.zeros(wavs)
    pawt_w[ok_w] = 1./verr_vw[1,ok_w]
    pafit_d = np.polyfit(wav_w[okfit_w],vstokes_vw[1,okfit_w],2,w=pawt_w[okfit_w]) 

  # now do inner fence cull on PA
    pachi_w = pawt_w*(vstokes_vw[1] - np.polyval(pafit_d,wav_w))      
    dum,pamin,pamax,dum = fence(pachi_w[okfit_w])
    cullpa_w = okfit_w & ((pachi_w < pamin) | (pachi_w > pamax))
    okfit_w = okfit_w & np.logical_not(cullpa_w)         
    pafit_d = np.polyfit(wav_w[okfit_w],vstokes_vw[1,okfit_w],2,w=pawt_w[okfit_w]) 
    pafit_w = np.polyval(pafit_d,wav_w)   

  # compute calibration mean PA ripple accuracy from stokes cycle mean, rotated to pq,pu 
    stokesrot_Sw,varrot_Sw,dum = specpolrotate(stokes_Sw,var_Sw,covar_Sw,-pafit_w)
    stokesrot_sw = np.zeros((2,wavs))
    varrot_sw = np.zeros((3,wavs))
    stokesrot_sw[:,ok_w] = stokesrot_Sw[1:,ok_w]/stokesrot_Sw[0,ok_w]
    varrot_sw[:,ok_w] = varrot_Sw[1:,ok_w]/stokesrot_Sw[0,ok_w]**2                    
    dwav_w = np.polyval(ripplewidth_c,wav_w)*(1. + np.polyval(np.polyder(ripplewidth_c),wav_w)/2.)      

    if debug:
        name = '_'.join(infile.split('.')[0].split('_')[:-1])
        np.savetxt(name+"_assessdebug.txt", \
            np.vstack((ok_w,cullint_w,cullpa_w,okfit_w,wav_w,dwav_w,pafit_w,stokes_Sw[0],runmedint_w,vstokes_vw,verr_vw)).T,  \
            fmt=" %2i %2i %2i %2i %6.1f %6.1f %8.3f  %10.3e %6i %8.3f %8.3f  %8.3f %8.3f")

    chi2dofacc_hs,meanacc_hs,varacc_hs,wavmean_h,wlen_h = chi2sample(stokesrot_sw,varrot_sw[:2],okfit_w,wav_w,dwav_w)

    accPA_h = np.degrees(meanacc_hs[:,1]/meanacc_hs[:,0])/2.
    accPAerr_h = np.degrees(np.sqrt(varacc_hs[:,1])/meanacc_hs[:,0])/2.                
    accmeanPA = (np.abs(accPA_h)*chi2dofacc_hs[:,1]).sum()/chi2dofacc_hs[:,1].sum()
    accmeanPAerr = (accPAerr_h*chi2dofacc_hs[:,1]).sum()/chi2dofacc_hs[:,1].sum()

    if doprint:    
        rsslog.message("\n Estimated calibration errors for obs : %s \n       relP(x100)  Err  PA (deg)   Err" % name, logfile)
        rsslog.message(" Accuracy              %7.3f %7.3f " % (accmeanPA, accmeanPAerr), logfile)
    if dofiles:
        np.savetxt(name+"_pafit.txt",np.vstack((wav_w,vstokes_vw[1],pafit_w,cullint_w,cullpa_w,okfit_w)).T,    \
            fmt=" %7.1f %9.4f %9.4f %2i %2i %2i")     
        np.savetxt(name+"_chi2dofacc_hs.txt",  \
            np.vstack((wavmean_h,wlen_h,chi2dofacc_hs[:,1],100.*meanacc_hs.T,100.*np.sqrt(varacc_hs[:,1]))).T,   \
            fmt=" %7.1f %4i %8.3f %8.3f %8.4f %8.4f ")
    return accmeanPA, accmeanPAerr, meanacc_hs, wavmean_h, pafit_w, okfit_w

def calprecision_assess(infileList, **kwargs):
    logfile= 'precassess.log'
    rsslog.history(logfile)
    rsslog.message(str(kwargs), logfile)
    debug = (str(kwargs.pop('debug','False')).lower() == 'true')
    catid = int(kwargs.pop('catid',12))
    
    nameList = [x.rsplit("_",1)[0] for x in infileList] 
    obsList = sorted(list(set(nameList)))
    obss = len(obsList)
       
    for o,name in enumerate(obsList):    
      # compute deviation, to remove science features
        stokesdev_csw = np.zeros((cycles,rawstokess,wavs))
        vardev_csw = np.zeros((cycles,rawstokess+1,wavs))
        covardev_csw = np.zeros_like(stokesdev_csw)             
        stokesdev_csw[:,:,ok_w] = (stokes_csw - stokes_sw[None,:])[:,:,ok_w]        
        vardev_csw[:,:,ok_w] = (var_csw + var_sw[None,:,:])[:,:,ok_w]
        covardev_csw[:,:,ok_w] = (covar_csw + covar_sw[None,:,:])[:,:,ok_w]    
                      
      # compute calibration noise precision from stokes cycle deviations, rotated to qp,up
        devmeanrelP_c = np.zeros(cycles)
        devmeanrelPerr_c = np.zeros(cycles)
        devmeanPA_c = np.zeros(cycles)
        devmeanPAerr_c = np.zeros(cycles)                
        for c in range(cycles):
            stokesrot_sw,varrot_sw,dum = specpolrotate(stokesdev_csw[c],vardev_csw[c],covardev_csw[c],-pafit_w,normalized=True)
            chi2dofprec_hs,meanprec_hs,varprec_hs,wavmean_h,wlen_h = chi2sample(stokesrot_sw,varrot_sw[:2],okfit_w,wav_w,dwav_w)
            devrelP_h = meanprec_hs[:,0]/meanacc_hs[:,0]
            devrelPerr_h = np.sqrt(varprec_hs[:,0])/meanacc_hs[:,0]    
            devPA_h = np.degrees(meanprec_hs[:,1]/meanacc_hs[:,0])/2.
            devPAerr_h = np.degrees(np.sqrt(varprec_hs[:,1])/meanacc_hs[:,0])/2.
        
            devmeanrelP_c[c] = (np.abs(devrelP_h)*chi2dofprec_hs[:,0]).sum()/chi2dofprec_hs[:,0].sum()
            devmeanrelPerr_c[c] = (devrelPerr_h*chi2dofprec_hs[:,0]).sum()/chi2dofprec_hs[:,0].sum()                            
            devmeanPA_c[c] = (np.abs(devPA_h)*chi2dofprec_hs[:,1]).sum()/chi2dofprec_hs[:,1].sum()
            devmeanPAerr_c[c] = (devPAerr_h*chi2dofprec_hs[:,1]).sum()/chi2dofprec_hs[:,1].sum() 
                       
            np.savetxt("chi2dofprec_hs_"+str(c+1)+"_"+str(o)+"_"+nametag+".txt",  \
                np.vstack((wavmean_h,wlen_h,chi2dofprec_hs.T,100.*meanprec_hs.T,100.*np.sqrt(varprec_hs).T)).T,   \
                fmt=" %7.1f %4i %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f")

            rsslog.message("   %2i  %7.2f %7.2f %7.3f %7.3f" %     \
                (c+1, 100.*devmeanrelP_c[c], 100.*devmeanrelPerr_c[c], devmeanPA_c[c], devmeanPAerr_c[c]), logfile) 

        rmsdevmeanrelP = np.sqrt((devmeanrelP_c**2).mean())
        rmsdevmeanrelPerr = np.sqrt((devmeanrelPerr_c**2).mean())
        rmsdevmeanPA = np.sqrt((devmeanPA_c**2).mean())
        rmsdevmeanPAerr = np.sqrt((devmeanPAerr_c**2).mean())                        
        rsslog.message("  rms  %7.2f %7.2f %7.3f %7.3f" %     \
            (100.*rmsdevmeanrelP, 100.*rmsdevmeanrelPerr,rmsdevmeanPA, rmsdevmeanPAerr), logfile)
                           
    return
    
#--------------------------------------------------------------------            
if __name__=='__main__':
    infile=sys.argv[1]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)
    calaccuracy_assess(infile, doprint=True, **kwargs)

# HD298383
# cd /d/pfis/khn/20230210/sci   (1.25 arcsec, 8,6)  DONE
# cd /d/pfis/khn/20230214/sci   (1.25 arcsec, 8,6)  DONE
# cd /d/pfis/khn/20230221/sci   (1.25 arcsec, 8,6)  DONE
# cd /d/pfis/khn/20230526/sci   (3 arcsec, 3)     
# cd /d/pfis/khn/20230610/sci   (1.25 arcsec, 3)
# cd /d/pfis/khn/20240321/sci   (3 arcsec, PG0700, 7)  
# cd /d/pfis/khn/20240402/sci   (3 arcsec, 7,6)       DONE  
# cd /d/pfis/khn/20240418/sci   (3 arcsec, 7)       DONE  
# cd /d/pfis/khn/20240515/sci   (3 arcsec, 8, red 1st)  

# Moon
# cd /d/pfis/khn/20230829/sci_blue5
# cd /d/pfis/khn/20230829/sci_red3
# cd /d/pfis/khn/20230830/sci_blue6
# cd /d/pfis/khn/20230830/sci_red9

# WR's
# cd /d/pfis/khn/20201202/sci
# cd /d/pfis/khn/20191127/sci
# cd /d/pfis/khn/20180429/sci
# cd /d/pfis/khn/20191231/sci
# cd /d/pfis/khn/20171202/sci
# cd /d/pfis/khn/20180223/sci
# cd /d/pfis/khn/20220627/sci
# cd /d/pfis/khn/20191224/sci
# cd /d/pfis/khn/20180224/sci
# cd /d/pfis/khn/20201222/sci
# cd /d/pfis/khn/20191225/sci   
# cd /d/pfis/khn/20200111/sci
# cd /d/pfis/khn/20201223/sci
# cd /d/pfis/khn/20200112/sci
# cd /d/pfis/khn/20180116/sci
# cd /d/pfis/khn/20180202/sci
# cd /d/pfis/khn/20180227/sci   TODO

# python2.7 script.py calassess.py WR097_c?_*calxy8_stokes.fits

