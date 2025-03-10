#! /usr/bin/env python2.7

import os, sys, glob
import numpy as np
from astropy.io import fits as pyfits
from astropy.table import Table

polsaltdir = '/usr/users/khn/src/salt/polsaltcurrent/'
datadir = polsaltdir+'polsalt/data/'
sys.path.extend((polsaltdir+'/scripts/',))
from zipfile import ZipFile

from polfinalstokes import polfinalstokes
from sppolview import viewstokes
from polserkow import polserkow
from polutils import specpolrotate, chi2sample, fence
from calassess import calaccuracy_assess 
import rsslog
from obslog import create_obslog
keywordfile = datadir+"obslog_config.json"

"""
do finalstokes individually for raw cycles, assess calibration precision
    for PA, use paripple crossover wavelength table
    infileList:     name_c?_h??_??.fits
    finalstokes cycles will be named name_nametag
    reduce all targets, do assessment only for specified catid (default t00, catid 12)
    if serkfit file given, do serkfit on each cycle
"""

def polcycles_assess(infileList, nametag, **kwargs):
    logfile= 'polcycles_assess.log'
    rsslog.history(logfile)
    rsslog.message(str(kwargs), logfile)
    debug = (str(kwargs.pop('debug','False')).lower() == 'true')    # gives accpa, serkfit output
    detailout = (str(kwargs.pop('detailout','False')).lower() == 'true')    # gives cull,rippledev out zips
    minwav = float(kwargs.pop('minwav','0.')) 
    catid = int(kwargs.pop('catid',12))
    serkfile = kwargs.pop('serkfit','') 

    ripplewidth_c = np.array([4.25E-05, -0.326, 904.967])           # from ripplewaves.xlsx
    
    nameList = [x.rsplit("_",3)[0] for x in infileList] 
    obsList = sorted(list(set(nameList)))    
    obss = len(obsList)
    if serkfile:
        lmax,pmax,Kpar,pamax,paslp = np.loadtxt(serkfile,usecols=1)
    outfileListo= []       
       
    for o,name in enumerate(obsList):
        fidxList = np.where(np.array(nameList)==name)[0]               
        fileList = list(np.array(infileList)[fidxList])        
        cycleList = map(int,[x.rsplit("_")[-1].split('.')[0] for x in fileList])
                
        hdul0 = pyfits.open(fileList[0])       
        wavs = hdul0['SCI'].data.shape[-1]
        wav0 = hdul0['SCI'].header['CRVAL1']
        dwav = hdul0['SCI'].header['CDELT1']
        wav_w = wav0 + dwav*np.arange(wavs)
        dwav_w = np.polyval(ripplewidth_c,wav_w)*(1. + np.polyval(np.polyder(ripplewidth_c),wav_w)/2.)      
        obsdate = hdul0[0].header['DATE-OBS'].replace('-','')
        
        goodcycleList = list(set(cycleList))
        cyclepairsList = []
        for goodcycle in goodcycleList:
            cyclepairsList.append(cycleList.count(goodcycle))
        for c in range(len(goodcycleList)):
            if cyclepairsList[c] < max(cyclepairsList):
                del goodcycleList[c]
                                              
        cycles = len(goodcycleList)
        rawstokess = hdul0['SCI'].shape[0]          

      # Reduce individual cycles                
        stokes_cSw = np.zeros((cycles,rawstokess+1,wavs))
        var_cSw = np.zeros((cycles,rawstokess+2,wavs))
        covar_cSw = np.zeros_like(stokes_cSw)
        stokes_csw = np.zeros((cycles,rawstokess,wavs))
        var_csw = np.zeros((cycles,rawstokess+1,wavs))
        covar_csw = np.zeros_like(stokes_csw)                 
        ok_cw = np.zeros((cycles,wavs),dtype=bool)
        outfileListc = []
        
        for c,cycle in enumerate(goodcycleList):        
            fidxList = np.where(np.array(cycleList)==cycle)[0]
            rawList = list(np.array(fileList)[fidxList])
            outfilepartList = rawList[0].split('.')[0].split('_')
            if (len(str(int(outfilepartList[-1]))) == 1):
                outfilepartList[-1] = str(int(outfilepartList[-1]))     # get rid of leading zero
            else:
                outfilepartList[-1] = "d"+str(int(outfilepartList[-1])) # prepended d indicates cycle>9
            outfilepartList.pop(-2)                                 # get rid of hw part
            outfile = '_'.join(outfilepartList)+'_stokes.fits'
            outfileListc.append(outfile.replace('stokes',nametag+'_stokes'))
            
            if (len(rawList)<2): continue
            polfinalstokes(rawList, **kwargs)
            
            os.rename(outfile,outfileListc[c])
            hdul = pyfits.open(outfileListc[c])
            
            targets = hdul['SCI'].data.shape[1]
            tassess = 0
            if (targets > 1):
                tgtTab = Table.read(hdul0['TGT'])           
                entries = len(tgtTab['CATID'])
                oktgt_i = (tgtTab['CULL'] == '')
                i_t = np.where(oktgt_i)[0]                 
                tassess = np.where(i_t==catid)[0][0]

            stokes_cSw[c] = hdul['SCI'].data[:,tassess,:]
            var_cSw[c] = hdul['VAR'].data[:,tassess,:]
            covar_cSw[c] = hdul['COV'].data[:,tassess,:] 
            bpm_Sw = hdul['BPM'].data[:,tassess,:]                        
            ok_cw[c] = ((bpm_Sw ==0).all(axis=0) & (wav_w >= minwav))
            stokes_csw[c,:,ok_cw[c]] = stokes_cSw[c,1:,ok_cw[c]]/stokes_cSw[c,0,ok_cw[c]][:,None]
            var_csw[c,:,ok_cw[c]] = var_cSw[c,1:,ok_cw[c]]/stokes_cSw[c,0,ok_cw[c]][:,None]**2
            covar_csw[c,:,ok_cw[c]] = covar_cSw[c,1:,ok_cw[c]]/stokes_cSw[c,0,ok_cw[c]][:,None]**2 

      # compute mean and save
        stokes_Sw = stokes_cSw.mean(axis=0)
        var_Sw = var_cSw.mean(axis=0)/cycles 
        covar_Sw = covar_cSw.mean(axis=0)/cycles
        okall_w = ok_cw.all(axis=0)
        
        outfilepartList.pop(-1)                                 # get rid of cycle part
        outfile = '_'.join(outfilepartList)+'_stokes.fits'       
        outfileListo.append(outfile.replace('stokes',nametag+'_stokes'))            
        hdul['SCI'].data[:,0,:] = stokes_Sw.astype("float32")
        hdul['VAR'].data[:,0,:] = var_Sw.astype("float32")
        hdul['COV'].data[:,0,:] = covar_Sw.astype("float32")
        hdul['BPM'].data[:,0,:] = np.logical_not(okall_w).astype('uint8')        
        hdul.writeto(outfileListo[o],overwrite=True)
        rsslog.message("\n Saving mean stokes file %s\n" % outfileListo[o], logfile)
        accmeanPA, accmeanPAerr, meanacc_hs, wavmean_h, pafit_w, okfit_w =  \
            calaccuracy_assess(outfileListo[o],okmask=okall_w,minwav=minwav,debug=debug)
             
        okall_w &= okfit_w                                  # include pafit culling    
        if (wav_w[0] < minwav):
            rsslog.message(" Wavelengths restricted to >= %6.0f" % minwav, logfile)
                    
      # compute deviation
        stokes_sw = np.zeros((rawstokess,wavs))
        var_sw = np.zeros((rawstokess+1,wavs))
        covar_sw = np.zeros_like(stokes_sw)
        stokes_sw[:,okall_w] = stokes_Sw[1:,okall_w]/stokes_Sw[0,okall_w][None,:]
        var_sw[:,okall_w] = var_Sw[1:,okall_w]/stokes_Sw[0,okall_w][None,:]**2
        covar_sw[:,okall_w] = covar_Sw[1:,okall_w]/stokes_Sw[0,okall_w][None,:]**2
              
        stokesdev_csw = np.zeros((cycles,rawstokess,wavs))
        vardev_csw = np.zeros((cycles,rawstokess+1,wavs))
        covardev_csw = np.zeros_like(stokesdev_csw)             
        stokesdev_csw[:,:,okall_w] = (stokes_csw - stokes_sw[None,:])[:,:,okall_w]        
        vardev_csw[:,:,okall_w] = (var_csw + var_sw[None,:,:])[:,:,okall_w]
        covardev_csw[:,:,okall_w] = (covar_csw + covar_sw[None,:,:])[:,:,okall_w]    

        stokesrot_csw = np.zeros_like(stokesdev_csw)
        varrot_csw = np.zeros_like(vardev_csw)
                      
      # first cull narrowband outliers (extraction artifacts, etc). 20 Ang for now. cull if >3% > 6 upperfence 
        cullzipfile = obsdate+'_'+name+'_'+nametag+'_cull.zip'             
        cullzip = ZipFile(cullzipfile,mode='w')

        outlier_msg = ""
        hasout_c = np.zeros(cycles,dtype=bool)                       
        for c in range(cycles):
            stokesrot_csw[c],varrot_csw[c],dum = specpolrotate(stokesdev_csw[c],vardev_csw[c],covardev_csw[c],-pafit_w,normalized=True)

            narrowband = 20.
            pcentlim = 3.
            hifence = 6.
            chi2dofprecc_hs,meanprecc_hs,varprecc_hs,wavmeanc_h,wlenc_h =    \
                chi2sample(stokesrot_csw[c],varrot_csw[c,:2],okall_w,wav_w,narrowband)
            cyclefile = obsdate+'_'+name+'_'+str(c+1)+'_'+nametag+'_cull.txt'                         
            np.savetxt(cyclefile,  \
                np.vstack((wavmeanc_h,wlenc_h,chi2dofprecc_hs.T,100.*meanprecc_hs.T,100.*np.sqrt(varprecc_hs).T)).T,   \
                fmt=" %7.1f %4i %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f")                
            cullzip.write(cyclefile)
            os.remove(cyclefile)
            isout_hs = np.zeros((len(wavmeanc_h),2),dtype=bool)            
            iscull_h = np.zeros(len(wavmeanc_h),dtype=bool)           
            for s in (0,1):
                Q1,Q3 = np.percentile(chi2dofprecc_hs[:,s],(25.,75.))
                isout_hs[:,s] = chi2dofprecc_hs[:,s] > (Q3+hifence*(Q3-Q1))
            isout_h = isout_hs.any(axis=1)
            cullcount =  isout_h.sum()                              
            pcentcull = 100.*cullcount/len(wavmean_h)
          # find area to cull by searching inward from median +/- 1.5 cullcount
            if (pcentcull > pcentlim):
                hcullmedian = int(np.median(np.where(isout_h)[0]))
                hcull1 = int(max(0,hcullmedian - 1.5*cullcount))              
                hcull1 = hcull1 + np.where(isout_h[hcull1:])[0][0]
                hcull2 = int(min(len(wavmeanc_h),hcullmedian + 1.5*cullcount))                 
                hcull2 = hcull2 - np.where(isout_h[hcull2::-1])[0][0]          
                cullwav1 = wavmeanc_h[hcull1] 
                cullwav2 = wavmeanc_h[hcull2]
                cullwav = (cullwav1 + cullwav2)/2.
                ok_cw[c] &= ((wav_w < cullwav1) | (wav_w > cullwav2))
                if (not hasout_c.all()):
                    rsslog.message(" Outliers found:\n Cycle Percent  wav1 to wav2 Ang", logfile)
                hasout_c[c] = True                     
                rsslog.message(" %2i  %6.2f%%  %7.1f  %7.1f" % (c+1, pcentcull, cullwav1, cullwav2), logfile)

        if detailout:
            cullzip.close()
            rsslog.message('\n Saving cull file:  %s' % cullzipfile, logfile)
        else:
            os.remove(cullzipfile)

        if hasout_c.any():
            okall_w = ok_cw.all(axis=0)
            accmeanPA, accmeanPAerr, meanacc_hs, wavmean_h, pafit_w, okfit_w =  \
                calaccuracy_assess(outfileListo[o],minwav=minwav,okmask=okall_w)
#                 debugfile=obsdate+'_'+name+'_chi2dofacc_debug.txt')
            okall_w &= okfit_w  
            
      # PA accuracy vs cycle done with culled data
        accmeanPA_c = np.zeros(cycles)
        accmeanPAerr_c = np.zeros(cycles)
        meanacc_hsListc = [] 
        wavmean_hListc = []
        for c in range(cycles):        
            accmeanPA_c[c], accmeanPAerr_c[c], meanaccc_hs, wavmeanc_h =     \
                calaccuracy_assess(outfileListc[c],minwav=minwav,okmask=ok_cw[c])[:4]  

            meanacc_hsListc.append(meanaccc_hs)
            wavmean_hListc.append(wavmeanc_h)

        rsslog.message("\n Estimated calibration errors for obs : %s \n       relP(x100)  Err  PA (deg)   Err" % name, logfile)
        rsslog.message(" Cycle Accuracy ",logfile)
        for c in range(cycles):
            rsslog.message("   %2i                  %7.3f %7.3f" % (c+1, accmeanPA_c[c], accmeanPAerr_c[c]), logfile)
        rsslog.message(" mean                  %7.3f %7.3f" % (accmeanPA, accmeanPAerr), logfile)                         
            
      # put PA accuracy results into one output file, mean on the end          
        meanacc_hsListc.append(meanacc_hs)
        wavmean_hListc.append(wavmean_h)      
        hwavmin = np.array([wavmean_hListc[c].min() for c in range(cycles+1)]).min()
        hwavmax = np.array([wavmean_hListc[c].max() for c in range(cycles+1)]).max()
        Hwavs = int((hwavmax-hwavmin)/dwav)+1
        wavmean_H = hwavmin + dwav*np.arange(Hwavs)
        dpamean_HC = np.zeros((Hwavs,cycles+1))
        for c in range(cycles+1):
            wavmeanc_h = hwavmin + ((wavmean_hListc[c] - hwavmin)/dwav).astype(int)*dwav     # line up with mean wavs
            nodup_h = np.append((wavmeanc_h[:-1] != wavmeanc_h[1:]),True)
            wavmeanc_h = wavmeanc_h[nodup_h]
            dpameanc_h = np.degrees(meanacc_hsListc[c][nodup_h,1]/meanacc_hsListc[c][nodup_h,0])/2.
            ish_H = np.isin(wavmean_H,wavmeanc_h)                       
            dpamean_HC[ish_H,c] = dpameanc_h

        partList = outfileListo[o].split('_')
        configpart = [i for i in range(len(partList)) if (partList[i][0]=='c' and len(partList[i])==2)][0]
        name = '_'.join(partList[:configpart])
         
        hdr = (" wavl "+(cycles)*"%8i "+"  mean") % tuple(np.arange(cycles)+1)
        if debug:
            np.savetxt(name+"_"+nametag+"_accdpa.txt",np.vstack((wavmean_H,dpamean_HC.T)).T, \
                header=hdr,fmt=" %7.1f "+(cycles+1)*"%8.3f ")  
                      
      # compute calibration noise precision from stokes cycle deviations, rotated to qp,up
        devzipfile = obsdate+'_'+name+'_'+nametag+'_chi2ripple.zip'      
        devzip = ZipFile(devzipfile,mode='w') 
                      
        devmeanrelP_c = np.zeros(cycles)
        devmeanrelPerr_c = np.zeros(cycles)
        devmeanPA_c = np.zeros(cycles)
        devmeanPAerr_c = np.zeros(cycles)
        rsslog.message(" Cycle Precision",logfile)

        for c in range(cycles):
            chi2dofprec_hs,meanprec_hs,varprec_hs,wavmean_h,wlen_h =    \
                chi2sample(stokesrot_csw[c],varrot_csw[c,:2],okall_w,wav_w,dwav_w)
#                debugfile=obsdate+'_'+name+'_chi2dofprec_debug.txt')
            cyclefile = obsdate+'_'+name+'_'+str(c+1)+'_'+nametag+'_chi2ripple.txt'             
            np.savetxt(cyclefile,  \
                np.vstack((wavmean_h,wlen_h,chi2dofprec_hs.T,100.*meanprec_hs.T,100.*np.sqrt(varprec_hs).T)).T,   \
                fmt=" %7.1f %4i %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f")                
            devzip.write(cyclefile)
            os.remove(cyclefile)       
            
            devrelP_h = meanprec_hs[:,0]/meanacc_hs[:,0]
            devrelPerr_h = np.sqrt(varprec_hs[:,0])/meanacc_hs[:,0]    
            devPA_h = np.degrees(meanprec_hs[:,1]/meanacc_hs[:,0])/2.
            devPAerr_h = np.degrees(np.sqrt(varprec_hs[:,1])/meanacc_hs[:,0])/2.
        
            devmeanrelP_c[c] = (np.abs(devrelP_h)*chi2dofprec_hs[:,0]).sum()/chi2dofprec_hs[:,0].sum()
            devmeanrelPerr_c[c] = (devrelPerr_h*chi2dofprec_hs[:,0]).sum()/chi2dofprec_hs[:,0].sum()                            
            devmeanPA_c[c] = (np.abs(devPA_h)*chi2dofprec_hs[:,1]).sum()/chi2dofprec_hs[:,1].sum()
            devmeanPAerr_c[c] = (devPAerr_h*chi2dofprec_hs[:,1]).sum()/chi2dofprec_hs[:,1].sum() 

            rsslog.message("   %2i  %7.2f %7.2f %7.3f %7.3f   %s" %     \
                (c+1, 100.*devmeanrelP_c[c], 100.*devmeanrelPerr_c[c], devmeanPA_c[c], devmeanPAerr_c[c], \
                outlier_msg), logfile) 

        rmsdevmeanrelP = np.sqrt((devmeanrelP_c**2).mean())
        rmsdevmeanrelPerr = np.sqrt((devmeanrelPerr_c**2).mean())
        rmsdevmeanPA = np.sqrt((devmeanPA_c**2).mean())
        rmsdevmeanPAerr = np.sqrt((devmeanPAerr_c**2).mean())                        
        rsslog.message("  rms  %7.2f %7.2f %7.3f %7.3f" %     \
            (100.*rmsdevmeanrelP, 100.*rmsdevmeanrelPerr,rmsdevmeanPA, rmsdevmeanPAerr), logfile)

        if detailout:
            devzip.close()
            rsslog.message(' Saving ripple chi2 file:  %s' % devzipfile, logfile)
        else:
            os.remove(devzipfile)
        
  # do Serkowski fit on stacked observations, if required
    if serkfile:
        rsslog.message(("\n Serkowski fit for files: "+obss*" %s") % tuple(outfileListo), logfile)
        xpr_v,xerrpr_v,chi2dof_s,chi2dof_hs,mean_hs,var_hs,wav_h,len_h =     \
            polserkow(outfileListo,lmax='free',pmax=pmax,K='free',pamax=pamax,tslp='free')   
        pmaxpr = np.sqrt((xpr_v[:2]**2).sum())
        PA = 0.5*np.arctan2(xpr_v[1],xpr_v[0])
        PApr = (180. + np.degrees(PA)) % 180. 
        pmaxerrpr = np.sqrt((xerrpr_v[0]*np.cos(2.*PA))**2 + (xerrpr_v[1]*np.sin(2.*PA))**2)     
        PAerrpr = np.degrees(0.5)*np.sqrt((xerrpr_v[0]*np.sin(2.*PA))**2 + (xerrpr_v[1]*np.cos(2.*PA))**2)/pmaxpr       
        xpr_V = np.hstack((xpr_v[2:],pmaxpr,PApr))
        xerrpr_V = np.hstack((xerrpr_v[2:],pmaxerrpr,PAerrpr))     
  
    if serkfile: 
        rsslog.message("\n lmax(A)  +/-   deg/1000A  +/-     K   +/-    %pmax    +/-    PA (deg)   +/-  chi2P  chi2PA", logfile) 
        rsslog.message("%7.1f %6.1f %8.3f %7.3f %6.3f %5.3f %7.4f %7.4f %8.3f %7.3f %6.2f %6.2f" %    \
            (tuple(np.vstack((xpr_V,xerrpr_V)).T.ravel())+tuple(chi2dof_s)), logfile)
        if debug:
            np.savetxt("chisq_"+nametag+"_.txt",np.vstack((wav_h,len_h,chi2dof_hs.T)).T,fmt=" %8.0f %4i %8.3f %8.3f")                                 
    return

#--------------------------------------------------------------------
def binstokes(stokes_Sw, var_Sw, covar_Sw, wav_w, ok_w, bin_w, bin_W):
    Bins = bin_W.shape[0]
    stokess = stokes_Sw.shape[0]    
    isbin_Ww = (bin_W[:,None] == bin_w[None,:])
    stokes_SW = (stokes_Sw[:,None,:]*isbin_Ww).sum(axis=2)
    var_SW = ((var_Sw[:stokess,None,:] + 2.*covar_Sw[:,None,:])*isbin_Ww).sum(axis=2)              
    ok_W = isbin_Ww.sum(axis=1) > 0
            
    wav_W = np.zeros(Bins) 
    wav_W[ok_W] = (wav_w[None,:]*isbin_Ww).sum(axis=1)[ok_W]/isbin_Ww.sum(axis=1)[ok_W] 
    
    return stokes_SW, var_SW, wav_W, ok_W
            
#--------------------------------------------------------------------
def bindata(data_w, var_w, wav_w, ok_w, bin_w, bin_W):
    Bins = bin_W.shape[0]  
    isbin_Ww = (bin_W[:,None] == bin_w[None,:])
    ok_W = isbin_Ww.sum(axis=1) > 0
    data_W = np.zeros(Bins)    
    data_W[ok_W] = (data_w[None,:]*isbin_Ww).sum(axis=1)[ok_W]/isbin_Ww[ok_W].sum(axis=1)
    var_W = np.zeros(Bins)    
    var_W[ok_W] = (var_w[None,:]*isbin_Ww).sum(axis=1)[ok_W]/(isbin_Ww[ok_W].sum(axis=1))**2                         
    wav_W = np.zeros(Bins) 
    wav_W[ok_W] = (wav_w[None,:]*isbin_Ww).sum(axis=1)[ok_W]/isbin_Ww[ok_W] .sum(axis=1)
    
    return data_W, var_W, wav_W, ok_W
    
#--------------------------------------------------------------------            
if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)
    polcycles_assess(infileList, **kwargs)

# HD298383
# cd /d/pfis/khn/20230210/sci   (1.25 arcsec, 8,6)  DONE
# cd /d/pfis/khn/20230214/sci   (1.25 arcsec, 8,6)  DONE
# cd /d/pfis/khn/20230221/sci   (1.25 arcsec, 8,6)  DONE
# cd /d/pfis/khn/20230526/sci   (3 arcsec, 3)     
# cd /d/pfis/khn/20230610/sci   (1.25 arcsec, 3)
# cd /d/pfis/khn/20240321/sci   (3 arcsec, PG0700, 7)   DONE.  there is a grating-dependent PA ripple  
# cd /d/pfis/khn/20240402/sci   (3 arcsec, 7,6)       DONE  
# cd /d/pfis/khn/20240418/sci   (3 arcsec, 7)       DONE  
# cd /d/pfis/khn/20240515/sci   (3 arcsec, 8, red 1st)  
# polcycles_assess.py HD298383_?_c?_h??_??.fits nametag=oldcal useoldheffcal=True minwav=4000.
# cd ../newcal
# polcycles_assess.py HD298383_?_c?_h??_??.fits nametag=calxy8 usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip minwav=4000.
# polcycles_assess.py HD298383_?_c?_h??_??.fits nametag=calxy9 usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip minwav=4000

# OmiSco
# cd /d/pfis/khn/20240721/sci
# cd /d/pfis/khn/20240724/sci
# cd /d/pfis/khn/20240822/sci
# cd /d/pfis/khn/20240823/sci
# polcycles_assess.py OmiSco_?_c?_h??_??.fits nametag=oldcal useoldheffcal=True minwav=4000.
# cd ../newcal
# polcycles_assess.py OmiSco_?_c?_h??_??.fits nametag=calxy8 usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip minwav=4000.
# polcycles_assess.py OmiSco_?_c?_h??_??.fits nametag=calxy9 usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip minwav=4000

# Hiltner652
# cd /d/pfis/khn/20241001/sci
# polcycles_assess.py Hiltner652_?_c?_h??_??.fits nametag=oldcal useoldheffcal=True minwav=4000.
# cd ../newcal
# polcycles_assess.py Hiltner652_?_c?_h??_??.fits nametag=calxy8 usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip minwav=4000
# polcycles_assess.py Hiltner652_?_c?_h??_??.fits nametag=calxy9 usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip minwav=4000

# NGC2024-1
# cd /d/pfis/khn/20241107/sci
# polcycles_assess.py NGC2024-1_?_c?_h??_??.fits nametag=oldcal useoldheffcal=True minwav=4000.
# cd ../newcal
# polcycles_assess.py NGC2024-1_?_c?_h??_??.fits nametag=calxy8 usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip minwav=4000
# polcycles_assess.py NGC2024-1_?_c?_h??_??.fits nametag=calxy9 usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip minwav=4000

# Coalsack
# cd /d/pfis/khn/20240706/sci
# polcycles_assess.py CoalSackD-1_c0_h??_??.fits nametag=oldcal useoldheffcal=True
# polcycles_assess.py CoalSackD-1_c0_h??_??.fits nametag=calxy usrHeffcalfile=RSSpol_Heff_Moon0_7_c0,cy1,cx1.zip serkfit=CoalSackD-1_serkfit.txt
# specpolview.py CoalSackD-1_?_??_oldcal_stokes.fits OmiSco_?_??_calxy_stokes.fits bin=40A save=text

# Star+Pol
# cd /d/pfis/khn/20241214/newcal
# polcycles_assess.py HD8648_c0_h??_??.fits nametag=calxy9 usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip
# specpolview.py HD074000_c0_?_calxy9_stokes.fits bin=20A save=plottext

# Moon
# cd /d/pfis/khn/20230829/sci_blue5
# polcycles_assess.py Moon-00_c?_h??_??.fits Moon-20_c?_h??_??.fits Moon-39_c?_h??_??.fits nametag=oldcal useoldheffcal=True
# polcycles_assess.py Moon_c?_h??_??.fits nametag=oldcal useoldheffcal=True
# cd /d/pfis/khn/20230829/sci_red3
# cd /d/pfis/khn/20230830/sci_blue6
# cd /d/pfis/khn/20230830/sci_red9
# polcycles_assess.py Moon_c?_h??_??.fits nametag=oldcal useoldheffcal=True
# polcycles_assess.py Moon_c?_h??_??.fits nametag=calxy usrHeffcalfile=RSSpol_Heff_Moon0_c0,cy1,cx1.zip

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
# cd /d/pfis/khn/20180227/sci
# cd /d/pfis/khn/20191228/sci
# cd /d/pfis/khn/20210404/sci
# cd /d/pfis/khn/20191204/sci
# cd /d/pfis/khn/20191221/sci
# polcycles_assess.py WR*21_c?_h??_??.fits nametag=oldcal useoldheffcal=True 
# polcycles_assess.py WR*21_c?_h??_??.fits nametag=calxy usrHeffcalfile=RSSpol_Heff_Moon0_7_c0,cy1,cx1.zip 
# specpolview.py WR*21_c?_oldcal_stokes.fits WR*21_c?_calxy_stokes.fits bin=40A save=text
# cd /d/pfis/khn/20220227/newcal
# polcycles_assess.py WR*48_c?_h??_??.fits nametag=calxy10 
# specpolview.py WR*48_c?_?_calxy10_stokes.fits bin=20A connect=hist save=plottext
