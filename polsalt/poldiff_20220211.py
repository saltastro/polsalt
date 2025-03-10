#! /usr/bin/env python

"""
poldiff

Difference stokes.fits data, including MOS. 

"""

import os, sys, glob, inspect
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits as pyfits
from astropy.table import Table
import rsslog
from obslog import create_obslog
from polutils import greff, angle_average

np.set_printoptions(threshold=np.nan)

datadir = os.path.dirname(__file__) + '/data/'
keywordprifile = datadir+"obslog_config.json"
keywordsecfile = datadir+"obslogsec_config.json"

#import warnings 
#warnings.filterwarnings("error") 

#---------------------------------------------------------------------------------------------
def poldiff(infileList, outname="default", logfile="salt.log", debug=False):
    """difference (normalized) (possibly MOS) stokes files

    Parameters
    ----------
    infileList: list
       one or more _stokes.fits files
    outname: output file will be outname.fits

    """
    """
    _b observations
    _i targetTab entries
    _t target index in fits (skips entries with CULL set)
    _T targets in combination
    _w wavelengths in individual observations
    _W wavelengths in combined grid

    """    
    
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

    stokesList_stw = []
    varList_stw = [] 
    covList_stw = []           
    okList_stw = []
    werrList_tw = []    
    tgtTabList = []
    
    rsslog.message("\nDifferencing files:",logfile)
    
    for b in range(obss):
        rsslog.message(infileList[b],logfile)
        hdul = pyfits.open(infileList[b],ignore_missing_end=True)
        stokesList_stw.append(hdul['SCI'].data)
        varList_stw.append(hdul['VAR'].data)     
        covList_stw.append(hdul['COV'].data)            
        okList_stw.append(hdul['BPM'].data == 0)
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
    Wav0 = dWav*np.ceil(wav0_b.max()/dWav) 
    Wav1 = dWav*np.floor((wav0_b + dwav_b*wavs_b).min()/dWav)
    Wavs = int((Wav1 - Wav0)/dWav)
    wav_W = np.arange(Wav0,Wav1,dWav)    
    stokess = stokesList_stw[0].shape[0]
    vars = varList_stw[0].shape[0]

    stokes_bsTW = np.zeros((obss,stokess,Targets,Wavs))
    var_bsTW = np.zeros((obss,vars,Targets,Wavs)) 
    cov_bsTW = np.zeros((obss,stokess,Targets,Wavs))           
    ok_bsTW = np.zeros((obss,stokess,Targets,Wavs)).astype(bool)
    werr_bTW = np.zeros((obss,Targets,Wavs))  

 # get data and put on common wavelength/target grid, combining (commensurate) bins if necessary
    for b in range(obss):
        t_T = np.where(oktgt_bi[b])[0]
        W0 = max(int(np.ceil((wav0_b[b] - Wav0)/dWav)),0)
        W1 = min(int(np.floor((wav0_b + dwav_b*wavs_b).min()/dWav)), len(wav_W)) 
               
        if dwav_b[b] == dWav:                
            print b, wav0_b[b],dwav_b[b],wavs_b[b], W0, W1, wav_W[[W0,W1-1]]        
        
            stokes_bsTW[b,:,:,W0:W1] = stokesList_stw[b][:,t_T,W0:W1]
            var_bsTW[b,:,:,W0:W1] = varList_stw[b][:,t_T,W0:W1]
            cov_bsTW[b,:,:,W0:W1] = covList_stw[b][:,t_T,W0:W1]            
            ok_bsTW[b,:,:,W0:W1] = okList_stw[b][:,t_T,W0:W1]
            werr_bTW[b,:,W0:W1] = werrList_tw[b][t_T,W0:W1]            
        else:
            subbins = int(dWav/dwav_b[b])
            wav_w = np.arange(wav0_b[b],wav0_b[b]+dwav_b[b]*wavs_b[b],dwav_b[b])
            w_W = np.array(np.where((wav_w % subbins)==0)[0],dtype=int)
            w0 = w_W[W0]
            w1 = w_W[W1]            
            wavs = (w1 - w0)/dwav_b[b]

            print b, wav0_b[b],dwav_b[b],wavs_b[b], w0, w1                   
            print stokes_bsTW[b,:,0,W0:W1].shape
            print stokesList_stw[b][:,t_T[0],w0:w1].shape
                       
            for T in range(Targets):                
                stokes_bsTW[b,:,T,W0:W1] =        \
                    stokesList_stw[b][:,t_T[T],w0:w1].reshape((-1,Wavs,subbins)).sum(axis=-1)
                var_bsTW[b,:,T,W0:W1] =           \
                    varList_stw[b][:,t_T[T],w0:w1].reshape((-1,Wavs,subbins)).sum(axis=-1)
                cov_bsTW[b,:,T,W0:W1] =           \
                    covList_stw[b][:,t_T[T],w0:w1].reshape((-1,Wavs,subbins)).sum(axis=-1)                                        
                ok_bsTW[b,:,T,W0:W1] =            \
                    okList_stw[b][:,t_T[T],w0:w1].reshape((-1,Wavs,subbins)).all(axis=-1)
                werr_bTW[b,T,W0:W1] =            \
                    werrList_tw[b][t_T[T],w0:w1].reshape((-1,Wavs,subbins)).all(axis=-1)                    

 # correct (unfluxed) intensity for grating efficiency to match observations together
    stokescor_bsTW = np.zeros((obss,stokess,Targets,Wavs))
    varcor_bsTW = np.zeros((obss,vars,Targets,Wavs)) 
    for b in range(obss):
        greff_W = greff(grating_b[b],grang_b[b],artic_b[b],dateobs_b[b],wav_W)[0]
        for T in range(Targets):
            ok_W = (ok_bsTW[b,:,T].all(axis=0) & (greff_W > 0.))
            stokescor_bsTW[b,:,T][:,ok_W] = stokes_bsTW[b,:,T][:,ok_W]/greff_W[None,ok_W]
            varcor_bsTW[b,:,T][:,ok_W] = var_bsTW[b,:,T][:,ok_W]/greff_W[None,ok_W]**2

 # normalize intensity at matching wavelengths _w
 # compute ratios at each wavelength, then error-weighted mean of ratio
    for T in range(Targets):
        ismatch_W = ok_bsTW[:,:,T].all(axis=0).all(axis=0)
        normint_bw = stokescor_bsTW[:,0,T][:,ismatch_W]/     \
            stokescor_bsTW[:,0,T][:,ismatch_W].mean(axis=0)
        varnorm_bw = varcor_bsTW[:,0,T][:,ismatch_W]/         \
            stokescor_bsTW[:,0,T][:,ismatch_W].mean(axis=0)**2
        normint_b = (normint_bw/varnorm_bw).sum(axis=1)/(1./varnorm_bw).sum(axis=1)
        ok_bW = ok_bsTW[:,:,T].all(axis=1)
        stokes_bsTW[:,:,T] = ok_bW[:,None,:]*stokes_bsTW[:,:,T]/normint_b[:,None,None]
        var_bsTW[:,:,T] = ok_bW[:,None,:]*var_bsTW[:,:,T]/normint_b[:,None,None]**2
        cov_bsTW[:,:,T] = ok_bW[:,None,:]*cov_bsTW[:,:,T]/normint_b[:,None,None]**2
    
 # Combine intensity
    stokes_sTW = np.zeros((stokess,Targets,Wavs))
    var_sTW = np.zeros((vars,Targets,Wavs))
    cov_sTW = np.zeros((stokess,Targets,Wavs))
    werr_TW = np.zeros((Targets,Wavs))    

    for b in range(obss):
        ok_TW = ok_bsTW[b].all(axis=0)        
        stokes_sTW[0,ok_TW] += stokes_bsTW[b][0,ok_TW]/var_bsTW[b][0,ok_TW]
        var_sTW[0,ok_TW] += 1./var_bsTW[b][0,ok_TW]
        cov_sTW[0,ok_TW] += cov_bsTW[b][0,ok_TW]/var_bsTW[b][0,ok_TW]
        werr_TW[ok_TW] += werr_bTW[b][ok_TW]/var_bsTW[b][0,ok_TW]                  

    ok_TW = ok_bsTW.all(axis=(0,1))
    ok_sTW = np.tile(ok_TW,(stokess,1,1))     
    var_sTW[0][ok_TW] = 1./var_sTW[0][ok_TW]
    stokes_sTW[0][ok_TW] = stokes_sTW[0][ok_TW]*var_sTW[0][ok_TW].flatten()
    cov_sTW[0][ok_TW] = cov_sTW[0][ok_TW]*var_sTW[0][ok_TW].flatten()
    werr_TW[ok_TW] = werr_TW[ok_TW]*var_sTW[0][ok_TW].flatten() 
    
 # Difference normalized stokes       
    stokes_sTW[1:][:,ok_TW] = (stokes_bsTW[1,1:][:,ok_TW]/stokes_bsTW[1,0][ok_TW] -   \
                     stokes_bsTW[0,1:][:,ok_TW]/stokes_bsTW[0,0][ok_TW])*stokes_sTW[0][ok_TW]
    var_sTW[1:][:,ok_TW] = var_bsTW[0,1:][:,ok_TW] + var_bsTW[1,1:][:,ok_TW]
    cov_sTW[1:][:,ok_TW] = cov_bsTW[0,1:][:,ok_TW] + cov_bsTW[1,1:][:,ok_TW]    

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
    hduout['SCI'].data = stokes_sTW.astype('float32')
    hduout['VAR'].data = var_sTW.astype('float32')
    hduout['COV'].data = cov_sTW.astype('float32')    
    hduout['BPM'].data = (~ok_sTW).astype('uint8')
    hduout['WERR'].data = werr_TW.astype('float32')         
    if (len(tgtTabList) > 0): 
        tgtTab = tgtTabList[0]
        combculled_i = (oktgt_i[None,:] != oktgt_bi).any(axis=0)
        for i in np.where(combculled_i)[0]:
            tgtTab['CULL'][i] = 'DIFF'                   
        hdul['TGT'] = pyfits.table_to_hdu(tgtTab)     
    hduout[0].header.add_history('POLDIFF: '+' '.join(infileList))

    hduout.writeto(outfile,overwrite=True)
    
    return

#--------------------------------------
 
if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)    
    poldiff(infileList,**kwargs)

# debug:
# ET21Ab,r
# cd /d/pfis/khn/poleff/et21a_lamp
# python polsalt.py poldiff.py ET21A_r90_C.fits ET21A_r0_C.fits outname=ET21A_Diff_r90_0

