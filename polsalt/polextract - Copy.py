
"""
polextract

Optimal extraction for polarimetric data of all modes
Write out extracted data fits (etm*) dimensions wavelength,target #

"""

import os, sys, glob, shutil, inspect

import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.signal import convolve2d
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table

# this is pysalt-free

import rsslog
from obslog import create_obslog
from polutils import datedline, rssdtralign, rssmodelwave, configmap
from immospolcatextract import immospolcatextract
from immospolextract import findpair, imfilpolextract, immospolextract
from slitlesspolextract import slitlesspolextract
from sppolextract import sppolextract, splinemapextract
from polmaptools import blksmooth2d

datadir = os.path.dirname(__file__) + '/data/'
keywordfile = datadir+"obslog_config.json"
np.set_printoptions(threshold=np.nan)

def polextract(infileList, logfile='salt.log', debug=False):
    """derive extracted target data vs wavelength and target for each configuration

    Parameters 
    ----------
    infileList: list of strings

    """
    """
    _c configuration index
    _d dimension index y,x = 0,1
    _i file index in infileList
    _j index within files of an observation
    _f filter index 
    _y, _x mm coordinate
    _r, _c bin coordinate
    _t catalog index
    _s located star on image
    _S culled star target
    """
    if debug=='True': debug=True
    rsslog.history(logfile)
        
    dfile,dtarget,dbeam = 0,0,0
      
  # group the files together
    confitemList = ['MASKID','FILTER','GRATING','GR-ANGLE','CAMANG','BVISITID']

    if (len(infileList) == 0):
        rsslog.message("\ninfileList is empty, exitting ", logfile)
        exit()            
    obs_i,config_i,obstab,configtab = configmap(infileList, confitemList, debug=debug)
    
    obss = len(obstab)
    configs = len(configtab)
    infiles = len(infileList)        

    for o in range(obss):
        maskid, filter, grating, grangle, camang, bvisitid = configtab[obstab[o]['config']]
        objectname = obstab[o]['object']
        if objectname=='ARC': continue
        name = objectname+"_"+filter[-4:]
        fileListj = [infileList[i] for i in range(infiles) if obs_i[i]==o]
        obsDictj = create_obslog(infileList,keywordfile)
        files = len(fileListj)
        hdul0 = pyfits.open(fileListj[0])
        hdr0 = hdul0[0].header
        rows, cols = hdul0[1].data.shape[1:]
        rccenter_d = np.array([rows-1.,cols-1.])/2.
        rcbin_d = np.array([int(x) for x in hdr0['CCDSUM'].split(" ")])[::-1]
        exptime = obsDictj['EXPTIME'][0]
        trkrho = obsDictj['TRKRHO'][0]
        dateobs =  obsDictj['DATE-OBS'][0].replace('-','')
        wppat = obsDictj['WPPATERN'][0]
        lampid = obsDictj['LAMPID'][0].replace(' ','')                       
        mapTab = Table.read(hdul0['TGT'])
        targets = len(mapTab)
        tgt_prc = hdul0['TMAP'].data    # this should be the same for all files           
             
      # Process with mode-specific extraction
        if grating=='N/A':              
          # split data into targets
            rckey_pd = np.array([['R0O','C0O'],['R0E','C0E']])
            Rows,Cols = np.array(hdr0['BOXRC'].split()).astype(int)
            ri_tpR = np.zeros((targets,2,Rows),dtype=int)
            ci_tpC = np.zeros((targets,2,Cols),dtype=int)          
                        
            for p in (0,1):
                ri_tpR[:,p] = np.clip(mapTab[rckey_pd[p,0]][:,None]   \
                    + np.arange(Rows)[None,:], 0,rows-1)
                ci_tpC[:,p] = np.clip(mapTab[rckey_pd[p,1]][:,None]   \
                    + np.arange(Cols)[None,:], 0,cols-1)
                                       
          # narrowband filter imaging polarimetry.  extract if stellar in wppat config.
          # evaluate, remove background with block smooth over full beam              
            if (filter[:2]=='PI'):               
                if wppat.count('NONE'): continue 
                if (lampid != 'NONE'): continue
                rsslog.message("\n"+name+" filter imaging polarimetry", logfile)
                                                        
                image_ftpRC = np.zeros((files,targets,2,Rows,Cols))
                var_ftpRC = np.zeros_like(image_ftpRC) 
                okbin_ftpRC =  np.zeros_like(image_ftpRC).astype(bool)
                oktgt_ftpRC =  np.zeros_like(okbin_ftpRC)
                bkg_ftpRC = np.zeros_like(image_ftpRC) 
                isbkg_prc = (tgt_prc == 255)
                drbkg_p = np.zeros(2)
                for p in (0,1):
                    drbkg_p[p] = np.where(isbkg_prc[p])[0][-1]-np.where(isbkg_prc[p])[0][0]
                drbkg = drbkg_p.max()
                blk = int(np.ceil(drbkg/16.))
                    
                for f,file in enumerate(fileListj):
                    hdul = pyfits.open(file)
                    image_prc = hdul['SCI'].data
                    var_prc = hdul['VAR'].data
                    okbin_prc = (hdul['BPM'].data==0) 
                    bkg_prc = np.zeros((2,rows,cols))
                    for p in (0,1):
                        bkg_prc[p] = blksmooth2d(image_prc[p],isbkg_prc[p], \
                            blk,blk,0.2,mode="mean",debug=False)                                                   
                        for t in range(targets):
                            bkg_ftpRC[f,t,p] = bkg_prc[p][ri_tpR[t,p],:][:,ci_tpC[t,p]] 
                    image_prc -= bkg_prc
                    
                    for t,p in np.ndindex(targets,2):
                        image_ftpRC[f,t,p] = image_prc[p][ri_tpR[t,p],:][:,ci_tpC[t,p]] 
                        var_ftpRC[f,t,p] = var_prc[p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
                        okbin_ftpRC[f,t,p] = okbin_prc[p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
                        oktgt_ftpRC[f,t,p] = (tgt_prc==t+1)[p][ri_tpR[t,p],:][:,ci_tpC[t,p]]
                                          
                hi_df = findpair(fileListj,logfile=logfile)                      
                imfilpolextract(fileListj,hi_df,name,  \
                    image_ftpRC,var_ftpRC,okbin_ftpRC,oktgt_ftpRC,bkg_ftpRC,logfile=logfile,debug=debug)
                         
          # unfiltered slitless imaging spectropolarimetry       
            elif (maskid=='P000000P00')&(filter[:2]=='PC'):              
                rsslog.message("\n"+name+" slitless imaging spectropolarimetry", logfile)                
                slitlesspolextract(fileListj,name,logfile=logfile,debug=debug)
                    
          # unfiltered MOS imaging spectropolarimetry
            elif (maskid[:2] != 'PL'):
                name = name+"_"+"immospol"
                MOSxmlList = glob.glob(maskid+'.xml')
                if (len(MOSxmlList) & ('REFWAV' in hdr0)):
                    rsslog.message("\n"+name+" MOS imaging spectropolarimetry from MOS xml", logfile)                
                    immospolextract(fileListj,name,logfile=logfile,debug=debug)
                else:
                    rsslog.message("\n"+name+" MOS imaging spectropolarimetry from catalog", logfile)                
                    immospolcatextract(fileListj,name,logfile=logfile,debug=debug)                    
        else:
          # longslit grating spectropolarimetry line map                     
            if ('WAVL' in mapTab.colnames):
                name = name+"_linemap"
                rsslog.message("\n"+name+" spectropolarimetry line map", logfile)
                splinemapextract(fileListj,name,logfile=logfile,debug=debug)
            else:
                name = name+"_specpol"
                rsslog.message("\n"+name+" spectropolarimetry", logfile)
                sppolextract(fileListj,name,logfile=logfile,debug=debug)
                
    return

# ------------------------------------

if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)   
    polextract(infileList,**kwargs)
    print findpair(infileList)

# debug:
# M30
# cd /d/pfis/khn/20161023/sci
# python polsalt.py polextract.py t*010[5-9].fits t*011[0-2].fits debug=True
# python polsalt.py polextract.py t*011[3-9].fits t*0120.fits debug=True
# python polsalt.py polextract.py mx*06[5-9].fits mx*07?.fits mx*08[0-8].fits
