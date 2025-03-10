#! /usr/bin/env python2.7

"""
spHeffcalcorrect

Combine, smooth StarPol data for correction file 
    correction assumed to be independent of FOV, track pos'n, for now

"""

import os, sys, glob, shutil, inspect, datetime, operator
import numpy as np

import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter("error")

from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy import linalg as la
from scipy.interpolate import LSQUnivariateSpline, interp1d
from scipy.ndimage.interpolation import shift
from zipfile import ZipFile
from json import load
polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

import rsslog
from obslog import create_obslog
from scrunch1d import scrunch1d, scrunchvar1d
from polutils import skypupillum, viewstokes, rotate2d
keywordfile = datadir+"obslog_config.json"
np.set_printoptions(threshold=np.nan) 

#---------------------------------------------------------------------------------------------
def spHeffcalcorrect(PAfilesList, **kwargs):
    """Compute smoothed calibration correction for each PA

    Parameters
    ----------
    PAfilesList [PA1,PA2,..] text files listing Dir, file, minwav for each PA
        data is finalstokes, but PA is in RSS frame, not sky
        files are in newcal, with cal=calname
    debug=: False (default)
            True (debug output)
    _P PA's
    _f file index for each PA
    _o obs index
    _F file index for an obs 
    _w wavelength index of input files
    _W wavelength calibration output grid index
    cal_w full output grid for cal file
    _v (0,1) output calibration file (P,PA, or Q,U)
    _S (unnormalized) stokes: input stokes (I,Q,U)
    _s (normalized) stokes: input stokes (q,u)

    """
    
    print (sys. version) 
    curdir = os.getcwd()     
    logfile= kwargs.pop('logfile',curdir+'/spHeffcalcorrect.log')            
    rsslog.history(logfile)
    rsslog.message(str(kwargs), logfile)

    debug = (kwargs.pop('debug','False') == 'True')
    calname= kwargs.pop('calname','calxy9')        
 
    basedir = "/d/pfis/khn/"
    PAs = len(PAfilesList)         
    polPA_P = np.zeros(PAs)                         # polaroid PA                 
    basePAlowwav, basePAhighwav = [6200.,7800.]     # wavelength range for perfect 1/2 wave

    calwav_w = np.array(range(3200,3500,100)+range(3500,5000,50)+  \
        range(5000,7000,100)+range(7000,10000,200)).astype(float)
    calwavs = len(calwav_w)
    calwedge_w = (calwav_w[:-1] + calwav_w[1:])/2.
    calwedge_w = np.concatenate(([calwav_w[0]-(calwav_w[1]-calwav_w[0])/2.],    \
        calwedge_w,[calwav_w[-1]+(calwav_w[-1]-calwav_w[-2])/2.]))
    calwbin_w = np.diff(calwedge_w)  
    calwav_w = (calwedge_w[:-1] + calwedge_w[1:])/2.    
    lbl_l = np.tile(calwav_w[:calwavs],2)
        
    vcalstokes_Pvw = np.zeros((PAs,2,calwavs))
    
    for P in range(PAs):
        rsslog.message("\n Processing data for PAfile: %s "% PAfilesList[P], logfile)     
        PAFile = PAfilesList[P]
        dir_f, scifile_f = np.loadtxt(PAFile,dtype='S80',usecols=(0,1),unpack=True)
        minwav_f = np.loadtxt(PAFile,dtype=float,usecols=2)        

        files = len(dir_f)
        dir_o, files_o = np.unique(dir_f, return_counts=True)
        obss = len(dir_o)
        scifilelen = max(map(len,scifile_f))
        rsslog.message("  Date     File "+(scifilelen-5)*" "+" Polaroid PA   Wavlo    Wavhi", logfile) 
        polPA_o = np.zeros(obss)
        vstokesListo_vw = []
        okListo_w = []
        wavListo_w = []                       
        for o in range(obss):
            if (dir_o[o].count('newcal')):
                os.chdir(basedir+dir_o[o])            
            else:
                os.chdir(basedir+dir_o[o]+'/newcal')
            f_F = np.where(dir_f==dir_o[o])[0]                      
            for F,f in enumerate(f_F):          # currently supporting only one block/obs
                if (dir_o[o].count('newcal')):
                    caldfile = scifile_f[f]
                else:                                    
                    filepartList = scifile_f[f].split('_')
                    confpartno = [ n for n in range(len(filepartList))  \
                        if ((filepartList[n][0]=='c') and (len(filepartList[n])==2)) ][0]
                    filepartList[confpartno] = 'c0'
                    caldfile = '_'.join(filepartList).replace('stokes',calname+'_stokes')            
                hdul = pyfits.open(caldfile)   
                wav0 = hdul['SCI'].header['CRVAL1']        
                wavs = hdul['SCI'].data.shape[-1]
                dwav = hdul['SCI'].header['CDELT1']
                wav_w = wav0 + dwav*np.arange(wavs)
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
                bpm_Sw = hdul['BPM'].data[:,tassess,:]                        
                ok_w = (bpm_Sw ==0).all(axis=0)                                
                ok_w &= (wav_w > minwav_f[f])
                vstokes_vw, verr_vw = viewstokes(stokes_Sw,var_Sw,ok_w=ok_w)                 
                                
                isbase_w = ((wav_w >= basePAlowwav) & (wav_w <= basePAhighwav))
                wavlo, wavhi = wav_w[np.where(ok_w)[0][0]], wav_w[np.where(ok_w)[0][-1]]              
                polPA = vstokes_vw[1,(ok_w & isbase_w)].mean()                  
                rsslog.message((" %9s %"+str(scifilelen)+"s %8.2f   %8.1f %8.1f") %   \
                    (dir_o[o],scifile_f[f],polPA,wavlo,wavhi), logfile)

                isbase_w = ((wav_w >= basePAlowwav) & (wav_w <= basePAhighwav))
                wavlo, wavhi = wav_w[np.where(ok_w)[0][0]], wav_w[np.where(ok_w)[0][-1]]                                   
                polPA_o[o] = vstokes_vw[1,(ok_w & isbase_w)].mean()                                   
       
            np.savetxt(dir_o[o].split('/')[0]+'_vstokes0_'+str(o)+'_'+calname+'_'+str(P)+'.txt',np.vstack((wav_w,ok_w,vstokes_vw)).T,    \
                fmt=' %8.2f %2i %8.4f %8.4f')

            isbase_w = ((wav_w >= basePAlowwav) & (wav_w <= basePAhighwav))
            wavlo, wavhi = wav_w[np.where(ok_w)[0][0]], wav_w[np.where(ok_w)[0][-1]]                                   
            polPA_o[o] = vstokes_vw[1,(ok_w & isbase_w)].mean() 
                            
          # Now combine observations (different polaroids)
            if (obss==1): continue
            vstokesListo_vw.append(vstokes_vw)
            okListo_w.append(ok_w)
            wavListo_w.append(wav_w)  
                               
          # if last obs, combine them, ramping blue to red at overlap:           
            if (o==(obss-1)):
              # first adjust PAs to mean of obs
                polPA_P[P] = polPA_o.mean()
                for o in range(obss):
                    vstokesListo_vw[o][1] += okListo_w[o]*(polPA_P[P] - polPA_o[o])
                                           
                wav_C = np.array(list(set(list(np.concatenate(wavListo_w)))))
                wavs = len(wav_C)
                isok_oC = np.zeros((obss,wavs),dtype=bool)
                for o in range(obss):              
                    isok_oC[o] = np.isin(wav_C,wavListo_w[o][okListo_w[o]])                  
                isokover_C = isok_oC.all(axis=0)
                Covermin = np.where(isokover_C)[0][0] 
                Covermax = np.where(isokover_C)[0][-1]                
                isover_C = ((np.arange(wavs) >= Covermin) & (np.arange(wavs) <= Covermax))

                isred_o = np.array([wavListo_w[O].mean() > wav_C[Covermax] for O in range(obss)])                                
                CListo_w = [np.where(np.isin(wav_C,wavListo_w[O]))[0] for O in range(obss)]                                                                           
                ramp_C = np.clip((np.arange(wavs,dtype=float)-Covermin)/(Covermax-Covermin),0.,1.)
                                                           
                vstokesblue_vC = np.zeros((2,wavs))
                vstokesred_vC = np.zeros((2,wavs))
                bluecount_C = np.zeros(wavs)
                redcount_C = np.zeros(wavs)                              
                for o in range(obss): 
                    vstokesblue_vC[:,CListo_w[o]] += (1.-isred_o[o])*okListo_w[o]*vstokesListo_vw[o]                
                    vstokesred_vC[:,CListo_w[o]] += isred_o[o]*okListo_w[o]*vstokesListo_vw[o]  
                    bluecount_C[CListo_w[o]] += (1.-isred_o[o])*okListo_w[o]
                    redcount_C[CListo_w[o]] += isred_o[o]*okListo_w[o]
                vstokesblue_vC[:,bluecount_C>0] = vstokesblue_vC[:,bluecount_C>0]/bluecount_C[bluecount_C>0] 
                vstokesred_vC[:,redcount_C>0] = vstokesred_vC[:,redcount_C>0]/redcount_C[redcount_C>0]                  

                vstokes_vw = vstokesblue_vC*(1.-ramp_C[None,:]) + vstokesred_vC*ramp_C[None,:]                                                                     
                ok_w = isok_oC.any(axis=0)               
                ok_w[isover_C] = (isokover_C)[isover_C]                                            
                wav_w = wav_C
                wavlo, wavhi = wav_w[np.where(ok_w)[0][0]], wav_w[np.where(ok_w)[0][-1]]                                   

                rsslog.message(("  Combined "+scifilelen*" "+" %8.2f   %8.1f %8.1f") %  \
                    (polPA_P[P], wavlo, wavhi), logfile)                             

        os.chdir(curdir)        
        np.savetxt('vstokes_'+calname+'_'+str(P)+'.txt',np.vstack((wav_w,isok_oC,ok_w,vstokes_vw)).T,    \
            fmt=' %8.2f'+(obss+1)*' %2i'+' %8.4f %8.4f')

      # scrunch _w down to cal. _W are cal grid idxs within data range            
        colArray = np.where(wav_w > 0.)[0]
        calwedgewidxArray = np.where((calwedge_w >= wav_w[colArray].min()) &    \
                                 (calwedge_w < wav_w[colArray].max()))[0]
        calwidx_W = calwedgewidxArray[:-1]                        
        cbinedgeArray = interp1d(wav_w[colArray],(colArray+0.5))(calwedge_w[calwedgewidxArray])      
        calbins_W = scrunch1d(ok_w.astype(int),cbinedgeArray)                                
        okcal_W = (calbins_W  > (cbinedgeArray[1:] - cbinedgeArray[:-1])/2.)  
                  
        for v in (0,1):
            vcalstokes_Pvw[P,v,calwidx_W[okcal_W]] =    \
                scrunch1d(ok_w*vstokes_vw[v],cbinedgeArray)[okcal_W]/calbins_W[okcal_W]

        np.savetxt('vstokes_calwav_'+calname+'_'+str(P)+'.txt', \
            np.vstack((calwav_w[calwidx_W],vcalstokes_Pvw[P,:,calwidx_W].T)).T,    \
            fmt=' %8.2f %8.4f %8.4f')

        isbase_w = ((wav_w >= basePAlowwav) & (wav_w <= basePAhighwav)) 
        polPA = vstokes_vw[1,(ok_w & isbase_w)].mean()
        vcalstokes_Pvw[P,1,calwidx_W[okcal_W]] -= polPA     # calibration is dPA from basewavs        
                
    vcalstokes_Pl = vcalstokes_Pvw.reshape((PAs,-1))
              
    hdr = '/'.join(os.getcwd().split('/')[-2:])+' cal correction file '+  \
        str(datetime.date.today()).replace('-','')        
    hdr += ("\n calPA "+PAs*"%10.2f " % tuple(polPA_P))
            
    calcorrectfile = datetime.date.today().strftime("%Y%m%d")+'_pacalcorrect.txt'
    np.savetxt(calcorrectfile, np.vstack((lbl_l,vcalstokes_Pl)).T,    \
        header=hdr, fmt="%9.2f "+PAs*"%10.3f ")

    rsslog.message('\n Saving calibration file:  %s' % calcorrectfile, logfile)   
                                    
    return

#--------------------------------------------------------------------    
        
if __name__=='__main__':
    PAfilesList = sys.argv[1:]
    spHeffcalcorrect(PAfilesList)
    
# cd ~/salt/polarimetry/Standards/Star+Pol
# spHeffcalcorrect.py StarPolfiles_135.txt StarPolfiles_0.txt StarPolfiles_45.txt
