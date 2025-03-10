#! /usr/bin/env python

"""
polcombine

Combine stokes.fits data, including MOS. 

"""

import os, sys, glob, inspect
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits as pyfits
from astropy.table import Table
import rsslog
from obslog import create_obslog
from polutils import greff, specpolrotate

np.set_printoptions(threshold=np.nan)

datadir = os.path.dirname(__file__) + '/data/'
keywordprifile = datadir+"obslog_config.json"
keywordsecfile = datadir+"obslogsec_config.json"

#import warnings 
#warnings.filterwarnings("error") 

#---------------------------------------------------------------------------------------------
def polcombine(infileList, outname="default", PAmatch=False, logfile="salt.log", debug=False):
    """combine (possibly MOS) stokes files

    Parameters
    ----------
    infile_list: list
       one or more _stokes.fits files

    """
    """
    _b observations
    _i targetTab entries
    _t target index in fits (skips entries with CULL set)
    _T targets in combination
    _w wavelengths in individual observations
    _W wavelengths in combined grid

    """    
    rsslog.history(logfile)
    
    obss = len(infileList)
    obsDict0 = create_obslog(infileList,keywordprifile)
    obsDictSCI = create_obslog(infileList,keywordsecfile,ext='SCI')

 #  construct common wavelength grid _W
    grating_b = obsDict0['GRATING']
    grang_b = obsDict0['GR-ANGLE']
    artic_b = obsDict0['CAMANG']    
    dateobs_b = np.array([int(x.replace('-','')) for x in obsDict0['DATE-OBS']])
    dwav_b = np.array(obsDictSCI['CDELT1'])
    wav0_b = np.array(obsDictSCI['CRVAL1'])
    wavs_b = np.array(obsDictSCI['NAXIS1'] )       

    stokesList_Stw = []
    varList_Stw = [] 
    covList_Stw = []           
    okList_Stw = []
    werrList_tw = []    
    tgtTabList = []
    
    rsslog.message("\nCombining files:",logfile)
    for b in range(obss):
        rsslog.message(infileList[b],logfile)
        hdul = pyfits.open(infileList[b],ignore_missing_end=True)
        stokesList_Stw.append(hdul['SCI'].data)
        varList_Stw.append(hdul['VAR'].data)     
        covList_Stw.append(hdul['COV'].data)            
        okList_Stw.append(hdul['BPM'].data == 0)
        werrList_tw.append(hdul['WERR'].data)        
        if 'TGT' in [hdul[x].name for x in range(len(hdul))]: 
            tgtTabList.append(Table.read(hdul['TGT']))
    
    entries = 1
    if (len(tgtTabList) > 0):
        sametgts = (len(tgtTabList) == obss)
        if sametgts:
            tgt_i = tgtTabList[0]['CATID']
            for b in range(1,obss):
                sametgts &= (tgtTabList[b]['CATID'] == tgt_i).all()       
        if (not sametgts):
            rsslog.message('All input files to combine must have same targets' , logfile)
            exit()
        entries = tgt_i.shape[0]
        oktgt_bi = np.array([(tgtTabList[b]['CULL'] == '') for b in range(obss)]) 
        
  # do combination only for targets in common across all observations
    oktgt_i = oktgt_bi.all(axis=0)
    Targets = oktgt_i.sum()
    i_T = np.where(oktgt_i)[0]    

    dWav = dwav_b.max()
    Wav0 = dWav*(wav0_b.min()//dWav) 
    Wavs = int((dWav*((wav0_b + dwav_b*wavs_b).max()//dWav) - Wav0)/dWav)
    wav_W = np.arange(Wav0,Wav0+dWav*Wavs,dWav)
    stokess = stokesList_Stw[0].shape[0]
    vars = varList_Stw[0].shape[0]

    stokes_bSTW = np.zeros((obss,stokess,Targets,Wavs))
    var_bSTW = np.zeros((obss,vars,Targets,Wavs)) 
    cov_bSTW = np.zeros((obss,stokess,Targets,Wavs))           
    ok_bSTW = np.zeros((obss,stokess,Targets,Wavs)).astype(bool)
    werr_bTW = np.zeros((obss,Targets,Wavs))  

 # get data and put on common wavelength/target grid, combining (commensurate) bins if necessary
    for b in range(obss):
        t_T = np.where(oktgt_bi[b])[0]
        W0 = int((wav0_b[b] - Wav0)/dWav)        
        if dwav_b[b] == dWav:
            stokes_bSTW[b,:,:,W0:W0+wavs_b[b]] = stokesList_Stw[b][:,t_T]
            var_bSTW[b,:,:,W0:W0+wavs_b[b]] = varList_Stw[b][:,t_T]
            cov_bSTW[b,:,:,W0:W0+wavs_b[b]] = covList_Stw[b][:,t_T]            
            ok_bSTW[b,:,:,W0:W0+wavs_b[b]] = okList_Stw[b][:,t_T]
            werr_bTW[b,:,W0:W0+wavs_b[b]] = werrList_tw[b][t_T]            
        else:
            subbins = int(dWav/dwav_b[b])
            wav_w = np.arange(wav0_b[b],wav0_b[b]+dwav_b[b]*wavs_b[b],dwav_b[b])
            for T in range(Targets):
                w0 = np.where((wav_w & subbins)==0)[0]
                swavs = np.where((wav_w[-1] & subbins)==0)[0] - w0
                sWav0 = dWav*(wav_w[w0]//dWav)
                sWavs = int((dWav*((wav_w[w0] + dwav_b[b]*wavs_b).max()//dWav) - sWav0)/dWav)
                stokes_bSTW[b,:,:,W0:W0+sWavs] =        \
                    stokesList_Stw[b][:,t_T[T],w0:w0+swavs].reshape((-1,sWavs,subbins)).sum(axis=-1)
                var_bSTW[b,:,:,W0:W0+sWavs] =           \
                    varList_Stw[b][:,t_T[T],w0:w0+swavs].reshape((-1,sWavs,subbins)).sum(axis=-1)
                cov_bSTW[b,:,:,W0:W0+sWavs] =           \
                    covList_Stw[b][:,t_T[T],w0:w0+swavs].reshape((-1,sWavs,subbins)).sum(axis=-1)                                        
                ok_bSTW[b,:,:,W0:W0+sWavs] =            \
                    okList_Stw[b][:,t_T[T],w0:w0+swavs].reshape((-1,sWavs,subbins)).all(axis=-1)
                werr_bTW[b,:,W0:W0+sWavs] =            \
                    werrList_tw[b][t_T[T],w0:w0+swavs].reshape((-1,sWavs,subbins)).all(axis=-1)                    

 # correct (unfluxed) intensity for grating efficiency to match observations together
    stokescor_bSTW = np.zeros((obss,stokess,Targets,Wavs))
    varcor_bSTW = np.zeros((obss,vars,Targets,Wavs)) 
    for b in range(obss):
        greff_W = greff(grating_b[b],grang_b[b],artic_b[b],dateobs_b[b],wav_W)[0]
        for T in range(Targets):
            ok_W = (ok_bSTW[b,:,T].all(axis=0) & (greff_W > 0.))
            stokescor_bSTW[b,:,T][:,ok_W] = stokes_bSTW[b,:,T][:,ok_W]/greff_W[None,ok_W]
            varcor_bSTW[b,:,T][:,ok_W] = var_bSTW[b,:,T][:,ok_W]/greff_W[None,ok_W]**2

 # normalize at matching wavelengths _w, separately for each target
 # compute ratios at each wavelength, then error-weighted mean of ratio
    badtargetList = []
    ismatch_TW = np.zeros((Targets,Wavs),dtype=bool)
    for T in range(Targets):
        ismatch_TW[T] = ok_bSTW[:,:,T].all(axis=0).all(axis=0)
        if ismatch_TW[T].sum():
            normint_bw = stokescor_bSTW[:,0,T][:,ismatch_W]/     \
                stokescor_bSTW[:,0,T][:,ismatch_W].mean(axis=0)
            varnorm_bw = varcor_bSTW[:,0,T][:,ismatch_W]/         \
                stokescor_bSTW[:,0,T][:,ismatch_W].mean(axis=0)**2
            normint_b = (normint_bw/varnorm_bw).sum(axis=1)/(1./varnorm_bw).sum(axis=1)
            
      # if no mutual overlap, use overlap of each with common obs
        else:           
            minmatchs_b = np.zeros(obss,dtype=int)
            ismatch_TW[T] = np.zeros(Wavs,dtype=bool)
            for b in range(obss):
                notb_B = np.where(np.arange(obss) != b)[0]
                Wmatchs_B = (ok_bSTW[b,0,T][None,:] & ok_bSTW[notb_B,0,T]).sum(axis=1)                
                minmatchs_b[b] = Wmatchs_B.min()
            if (minmatchs_b.max()>0):
                bmatch = np.argmax(minmatchs_b)
                notb_B = np.where(np.arange(obss) != bmatch)[0]
                normint_b = np.ones(obss)
                for b in (notb_B):
                    ismatch_TW[T] |= (ok_bSTW[bmatch,:,T] & ok_bSTW[b,:,T]).all(axis=0)
                    ismatchbT_W = (ismatch_TW[T] & ok_bSTW[b,:,T])
                    normint_w = stokescor_bSTW[b,0,T][ismatchbT_W]/     \
                        stokescor_bSTW[bmatch,0,T][ismatchbT_W].mean()
                    varnorm_w = varcor_bSTW[b,0,T][ismatchbT_W]/         \
                        stokescor_bSTW[bmatch,0,T][ismatchbT_W].mean()**2
                    normint_b[b] = (normint_w/varnorm_w).sum()/(1./varnorm_w).sum()
                    
                print T, bmatch, normint_b
                                                            
            else:
                ok_bSTW[:,:,T] = False
                badtargetList.append(T)                 

        if len(badtargetList):
            rsslog.message("\nNot processing targets with no overlap: "+badtargetList,logfile)            
                
        ok_bW = ok_bSTW[:,:,T].all(axis=1)
        stokes_bSTW[:,:,T] = ok_bW[:,None,:]*stokes_bSTW[:,:,T]/normint_b[:,None,None]
        var_bSTW[:,:,T] = ok_bW[:,None,:]*var_bSTW[:,:,T]/normint_b[:,None,None]**2
        cov_bSTW[:,:,T] = ok_bW[:,None,:]*cov_bSTW[:,:,T]/normint_b[:,None,None]**2

 # if required (eg for polaroid calibrations) allow for PA change between observations
    if PAmatch:
        ok_T = ok_bSTW.any(axis=(0,1,3))
        dPA_bT = np.zeros((obss,Targets))
        for T in range(Targets):
            ismatch_W = ok_bSTW[:,:,T].all(axis=0).all(axis=0)
            PAcenter_b = (0.5*np.arctan2(stokes_bSTW[:,2,T][:,ismatch_W].mean(axis=1),   \
                    stokes_bSTW[:,1,T][:,ismatch_W].mean(axis=1)) + np.pi) % np.pi            
            PA_bw = 0.5*np.arctan2(stokes_bSTW[:,2,T][:,ismatch_W],   \
                    stokes_bSTW[:,1,T][:,ismatch_W])                # PA in radians
            PA_bw = (PA_bw-(PAcenter_b[:,None]+np.pi/2.)+np.pi) % np.pi +   \
                    (PAcenter_b[:,None]-np.pi/2.)                                                                    # optimal PA folding            
            dPA_bT[:,T] = np.degrees((PA_bw - PA_bw.mean(axis=0)[None,:]).mean(axis=1))
        dPA_b = dPA_bT.mean(axis=1)
        
        for b,T in np.ndindex(obss,Targets):
            stokes_bSTW[b,:,T],var_bSTW[b,:,T],cov_bSTW[b,:,T] =    \
                specpolrotate(stokes_bSTW[b,:,T],var_bSTW[b,:,T],cov_bSTW[b,:,T],-dPA_b[b])
            
 # Do error weighted combine of observations
    stokes_STW = np.zeros((stokess,Targets,Wavs))
    var_STW = np.zeros((vars,Targets,Wavs))
    cov_STW = np.zeros((stokess,Targets,Wavs))
    werr_TW = np.zeros((Targets,Wavs))    

    for b in range(obss):
        ok_TW = ok_bSTW[b].all(axis=0)        
        stokes_STW[:,ok_TW] += stokes_bSTW[b][:,ok_TW]/var_bSTW[b][:stokess,ok_TW]
        var_STW[:,ok_TW] += 1./var_bSTW[b][:,ok_TW]
        cov_STW[:,ok_TW] += cov_bSTW[b][:,ok_TW]/var_bSTW[b][:stokess,ok_TW]
        werr_TW[ok_TW] += werr_bTW[b][ok_TW]/var_bSTW[b][0,ok_TW]                  
        
    ok_TW = (var_STW != 0).all(axis=0)
    ok_STW = np.tile(ok_TW,(stokess,1,1))
    okvar_STW = np.tile(ok_TW,(vars,1,1))
    var_STW[okvar_STW] = 1./var_STW[okvar_STW]
    stokes_STW[ok_STW] = stokes_STW[ok_STW]*var_STW[:stokess,ok_TW].flatten()
    cov_STW[ok_STW] = cov_STW[ok_STW]*var_STW[:stokess,ok_TW].flatten()
    werr_TW[ok_TW] = werr_TW[ok_TW]*var_STW[0,ok_TW].flatten()         

 # Save result, default name formed from unique elements of '_'-separated parts of names
    if (outname=="default"):
        outfile = ''  
        namepartlist = []
        parts = 100
        for file in infileList:
            partlist = os.path.basename(file).split('.')[0].split('_')
            parts = min(parts,len(partlist)) 
            namepartlist.append(partlist)
        for part in range(parts):
            outfile+='-'.join(sorted(set(zip(*namepartlist)[part])))+'_'   
        outfile = outfile[:-1]+'.fits'
    else:
        outfile = outname+".fits"
    
    rsslog.message("\nOutput: "+outfile,logfile)

    hduout = hdul
    for ext in ('SCI','VAR','BPM'):
        hduout[ext].header['CDELT1'] = dWav
        hduout[ext].header['CRVAL1'] = Wav0
    hduout['SCI'].data = stokes_STW.astype('float32')
    hduout['VAR'].data = var_STW.astype('float32')
    hduout['COV'].data = cov_STW.astype('float32')    
    hduout['BPM'].data = (~ok_STW).astype('uint8')
    hduout['WERR'].data = werr_TW.astype('float32')         
    if (len(tgtTabList) > 0): 
        tgtTab = tgtTabList[0]
        combculled_i = (oktgt_i[None,:] != oktgt_bi).any(axis=0)
        for i in np.where(combculled_i)[0]:
            tgtTab['CULL'][i] = 'COMB'                   
        hdul['TGT'] = pyfits.table_to_hdu(tgtTab)     
    hduout[0].header.add_history('POLCOMBINE: '+' '.join(infileList))

    hduout.writeto(outfile,overwrite=True)
    
    return

#--------------------------------------
 
if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)
    polcombine(infileList,**kwargs)

# use:
# ET21Ab,r
# cd /d/pfis/khn/poleff/et21a_lamp
# python polsalt.py polcombine.py ../../20210808/sci/ET21Ab-00*rawstokes.fits ../../20211127/sci/ET21Ar-00*rawstokes.fits PAmatch=True
# ET21Bu,g,r
# cd /d/pfis/khn/poleff/et21b_lamp
# problem with this one:
# python polsalt.py polcombine.py ../../20220405/sci/ET21Bu-01*rawstokes.fits ../../20210304/sci/ET21Bg-01*rawstokes.fits ../../20220401/sci/ET21Br-01*rawstokes.fits PAmatch=True
