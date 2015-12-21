#! /usr/bin/env python

"""
specpolview

Plot and text output of stokes data, optionally binned

"""

import os, sys, glob, shutil, inspect

import numpy as np
import pyfits
import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter  
plt.ioff()
np.set_printoptions(threshold=np.nan)

def specpolview(infile_list, bincode='', saveoption = ''):
    """View output results
    Parameters
    ----------
    infile_list: list
       one or more _stokes.fits files
    bincode  
       unbin (= ''), nnA (nn Angstroms), nn% (binned to %)
   saveoption  
       '' (text to terminal), text (text to file), plot (terminal plot and pdf file), textplot (both)
    """
    obss = len(infile_list)
    bintype = 'unbin'
    if len(bincode):
        if bincode[-1]=='%': 
            bintype = 'percent'
            errbin = float(bincode[ :-1])
        elif bincode[-1]=='A': 
            bintype = 'wavl'
            blk = int(bincode[ :-1])
        elif bincode != 'unbin': 
            print "unrecognized binning option, set to unbinned"
            bintype = 'unbin'

    savetext = saveoption.count('text')>0
    saveplot = saveoption.count('plot')>0 
    plotcolor_o = ['b','g','r','c','m','y','k'] 

    for obs in range(obss):
        hdul = pyfits.open(infile_list[obs])
        name = os.path.basename(infile_list[obs]).split('.')[0]
        obsdate = hdul[0].header['DATE-OBS']
        stokes_sw = hdul['SCI'].data[:,0,:]
        var_sw = hdul['VAR'].data[:,0,:]
        bpm_sw = hdul['BPM'].data[:,0,:]
        wavs = stokes_sw.shape[1]
        wav0 = hdul['SCI'].header['CRVAL1']
        dwav = hdul['SCI'].header['CDELT1']
        wav_w = np.mgrid[wav0:(wav0+wavs*dwav):dwav]
        stokes_s = hdul['SCI'].header['CTYPE3'].split(',')
        plots = len(stokes_s)
        if plots > 2: pa_type = hdul[0].header['POLCAL'].split(" ")[0]
        wok_s = (bpm_sw==0)

    # set up multiplot

        if obs==0: 
            fig,plot_s = plt.subplots(plots,1,sharex=True)
            plt.xlabel('Wavelength (Ang)')
            plot_s[0].set_ylabel('Intensity')
            for s in range(1,plots): plot_s[s].set_ylabel(stokes_s[s]+' Polarization (%)')
            if plots > 2:
                plot_s[1].set_ylabel('Linear Polarization (%)')
                plot_s[2].set_ylabel(pa_type+' PA (deg)')            
            fig.set_size_inches((8.5,11))
            fig.subplots_adjust(left=0.175)

        plotcolor = plotcolor_o[obs % len(plotcolor_o)]
    # plot intensity
        label = name
        ww = -1; 
        while (bpm_sw[0,ww+1:]==0).sum() > 0:
            w = ww+1+np.where(bpm_sw[0,ww+1:]==0)[0][0]
            ww = wavs
            dw = np.where(bpm_sw[0,w:]>0)[0]  
            if dw.size: ww = w + dw[0] - 1 
            plot_s[0].plot(wav_w[w:ww],stokes_sw[0,w:ww],color=plotcolor,label=label)
            label = '_'+name    

    #   for I,Q,U, and I,Q,U,V, compute unbinned linear polarization, variance
        if plots > 2:
            stokes_s[1:3] = '    P', (pa_type[:3]+' T')
            stokesp_w = np.zeros((wavs));   stokest_w = np.zeros((wavs))
            varp_w = np.zeros((wavs));   vart_w = np.zeros((wavs))
            varpe_dw = np.zeros((2,wavs));  varpt_w = np.zeros((wavs))
            wok = wok_s[0] & wok_s[1] & wok_s[2]
            stokesp_w[wok] = np.sqrt(stokes_sw[1,wok]**2 + stokes_sw[2,wok]**2)     # unnormalized linear polarization
            stokest_w[wok] = (0.5*np.arctan2(stokes_sw[2,wok],stokes_sw[1,wok]))    # PA in radians
            stokestmean = 0.5*np.arctan2(stokes_sw[2,wok].mean(),stokes_sw[1,wok].mean())
            pafold = np.pi*stokestmean/abs(stokestmean)
            stokest_w[np.abs(stokest_w-stokestmean)>np.pi/2.] += pafold
            stokest_w -= (int((stokestmean + np.pi)/np.pi) -1)*np.pi      # 0 < mean < pi
        # variance matrix eigenvalues, ellipse orientation
            varpe_dw[:,wok] = 0.5*(var_sw[1,wok]+var_sw[2,wok]                          \
                    + np.array([1,-1])[:,None]*np.sqrt((var_sw[1,wok]-var_sw[2,wok])**2 + 4*var_sw[-1,wok]**2))
            varpt_w[wok] = 0.5*np.arctan2(2.*var_sw[-1,wok],var_sw[1,wok]-var_sw[2,wok])
        # linear polarization variance along p, PA   
            varp_w[wok] = varpe_dw[0,wok]*(np.cos(2.*stokest_w[wok]-varpt_w[wok]))**2   \
                       + varpe_dw[1,wok]*(np.sin(2.*stokest_w[wok]-varpt_w[wok]))**2
            vart_w[wok] = varpe_dw[0,wok]*(np.sin(2.*stokest_w[wok]-varpt_w[wok]))**2   \
                       + varpe_dw[1,wok]*(np.cos(2.*stokest_w[wok]-varpt_w[wok]))**2

        if bintype != 'unbin':
        # Set up bins, blocked, or binned to error based on stokes 1 or on linear stokes p
            if bintype == 'wavl':
                bin_w = (wav_w / blk -0.5).astype(int) - int((wav_w / blk -0.5).min())
                bins = bin_w.max()
                bin_w[~wok_s[1]] = -1
            else:
                if plots<=2: binvar_w = var_sw[1]
                else:        binvar_w = varp_w
                ww = -1; b = 0;  bin_w = -1*np.ones((wavs))
                while (bpm_sw[0,ww+1:]==0).sum() > 0:
                    w = ww+1+np.where(bpm_sw[0,ww+1:]==0)[0][0]
                    cumsvar_w = np.cumsum(binvar_w[w:]*(bpm_sw[0,w:]==0))    \
                                /np.cumsum(stokes_sw[0,w:]*(bpm_sw[0,w:]==0))**2
                    err_w = np.sqrt(cumsvar_w)
                    ww = wavs
                    dw = np.where(bpm_sw[0,w:]>0)[0]  
                    if dw.size: ww = w + dw[0] - 1      # stopping point override: end or before bad pixel
                    dw = np.where(err_w[:ww-w] < errbin/100.)[0]
                    if dw.size: ww = w + dw[0]          # err goal is reached first
                    bin_w[w:ww+1] = b
                    b += 1
                bins  = b

        # calculate binned data
            bin_b = np.arange(bins)                    
            bin_bw = (bin_b[:,None] == bin_w[None,:])
            stokes_sb = (stokes_sw[:,None,:]*bin_bw).sum(axis=2)
            var_sb = (var_sw[:,None,:]*bin_bw).sum(axis=2) 
            bpm_sv = ((bpm_sw[:,None,:]*bin_bw).sum(axis=2)==bin_bw.sum(axis=1)).astype(int)
            bok = (bpm_sv[1:].sum(axis=0) == 0) 
            wav_v = (wav_w[None,:]*bin_bw).sum(axis=1)[bok]/bin_bw.sum(axis=1)[bok]
            dwavleft_v = wav_v - wav_w[(np.argmax((wav_w[None,:]*bin_bw)>0,axis=1))[bok]] + dwav/2.
            dwavright_v = wav_w[wavs-1-(np.argmax((wav_w[None,::-1]*bin_bw[:,::-1])>0,axis=1))[bok]] - wav_v - dwav/2.
            stokes_sv = np.zeros((plots,bok.sum()));   errstokes_sv = np.zeros((plots,bok.sum()))
            stokes_sv[1:] = 100*stokes_sb[1:,bok]/stokes_sb[0,bok]
            errstokes_sv[1:] =  100*np.sqrt(var_sb[1:plots,bok])/stokes_sb[0,bok]
            if plots > 2:
                stokesp_b = np.zeros((bins));   stokest_b = np.zeros((bins))
                varp_b = np.zeros((bins));   vart_b = np.zeros((bins))
                varpe_db = np.zeros((2,bins));  varpt_b = np.zeros((bins))
                stokesp_b[bok] = np.sqrt(stokes_sb[1,bok]**2 + stokes_sb[2,bok]**2)     # unnormalized linear polarization
                stokest_b[bok] = (0.5*np.arctan2(stokes_sb[2,bok],stokes_sb[1,bok]))    # PA in radians
                stokest_b[bok] = ((stokest_b[bok]+np.pi/2.+np.median(stokest_b[bok])) % np.pi) \
                                - np.median(stokest_b[bok])                             # get rid up 180 wraps
            # variance matrix eigenvalues, ellipse orientation
                varpe_db[:,bok] = 0.5*(var_sb[1,bok]+var_sb[2,bok]                          \
                    + np.array([1,-1])[:,None]*np.sqrt((var_sb[1,bok]-var_sb[2,bok])**2 + 4*var_sb[-1,bok]**2))
                varpt_b[bok] = 0.5*np.arctan2(2.*var_sb[-1,bok],var_sb[1,bok]-var_sb[2,bok])
            # linear polarization variance along p, PA   
                varp_b[bok] = varpe_db[0,bok]*(np.cos(2.*stokest_b[bok]-varpt_b[bok]))**2   \
                       + varpe_db[1,bok]*(np.sin(2.*stokest_b[bok]-varpt_b[bok]))**2
                vart_b[bok] = varpe_db[0,bok]*(np.sin(2.*stokest_b[bok]-varpt_b[bok]))**2   \
                       + varpe_db[1,bok]*(np.cos(2.*stokest_b[bok]-varpt_b[bok]))**2
                stokes_sv[1] = 100*stokesp_b[bok]/stokes_sb[0,bok]
                errstokes_sv[1] =  100*np.sqrt(var_sb[1,bok])/stokes_sb[0,bok]
                stokes_sv[2] = np.degrees(stokest_b[bok])
                errstokes_sv[2] =  0.5*np.degrees(np.sqrt(var_sb[2,bok])/stokesp_b[bok])             
            for s in range(1,plots):
                plot_s[s].errorbar(wav_v,stokes_sv[s],color=plotcolor,fmt='.',    \
                    yerr=errstokes_sv[s],xerr=(dwavleft_v,dwavright_v),capsize=0)

        # unbinned
        else:
            stokes_sv = np.full((plots,wavs),0.);   errstokes_sv = np.full((plots,wavs),0.);    wav_v = wav_w
            stokes_sv[1:plots,wok_s[0]] = 100*stokes_sw[1:plots,wok_s[0]]/stokes_sw[0,wok_s[0]]
            errstokes_sv[1:plots,wok_s[0]] =  100*np.sqrt(var_sw[1:plots,wok_s[0]])/stokes_sw[0,wok_s[0]]
            if plots > 2:
                stokes_sv[1,wok_s[0]] = 100*stokesp_w[wok_s[0]]/stokes_sw[0,wok_s[0]]
                errstokes_sv[1,wok_s[0]] =  100*np.sqrt(var_sw[1,wok_s[0]])/stokes_sw[0,wok_s[0]]
                stokes_sv[2,wok_s[0]] = np.degrees(stokest_w[wok_s[0]])
                errstokes_sv[2,wok_s[0]] =  0.5*np.degrees(np.sqrt(var_sw[2,wok_s[0]])/stokesp_w[wok_s[0]]) 
        # show gaps in plot, remove them from text
            for s in range(1,plots):
                ww = -1; 
                while (bpm_sw[s,ww+1:]==0).sum() > 0:
                    w = ww+1+np.where(bpm_sw[s,ww+1:]==0)[0][0]
                    ww = wavs
                    dw = np.where(bpm_sw[s,w:]>0)[0]  
                    if dw.size: ww = w + dw[0] - 1             
                    plot_s[s].plot(wav_v[w:ww],stokes_sv[s,w:ww],color=plotcolor,label=label)
            stokes_sv = stokes_sv[:,wok_s[0]]
            errstokes_sv = errstokes_sv[:,wok_s[0]]
            wav_v = wav_v[wok_s[0]]

        textfile = sys.stdout
        if savetext: textfile = name+'_'+bincode+'.txt'
        fmt_s = ['%8.2f ','%8.4f ','%8.3f ','%8.4f ']
        fmt = fmt_s[0]+2*(' '+"".join(fmt_s[1:plots]))
        hdr = name+'   '+obsdate+'\n\n'+'Wavelen  '+(5*" ").join(stokes_s[1:plots])+"  "+   \
                " Err  ".join(stokes_s[1:plots])+' Err '           
        np.savetxt(textfile,np.vstack((wav_v,stokes_sv[1:],errstokes_sv[1:])).T, fmt=fmt, header=hdr) 

    plot_s[0].set_ylim(bottom=0)                            # intensity plot baseline 0
    if plots >2: plot_s[1].set_ylim(bottom=0)               # linear polarization % plot baseline 0
    if obss>1: plot_s[0].legend(fontsize='x-small',loc='lower center')
    else: plot_s[0].set_title(name+"   "+obsdate) 
    if saveplot:
        plotfile = ('_').join(infile_list[0].split('_')[:2])+'_'+bincode+'.pdf'
        plt.savefig(plotfile,orientation='portrait')
        if os.popen('ps -C evince -f').read().count(plotfile)==0:
            os.system('evince '+plotfile+' &')
    else: plt.show(block=True)
    return
 
if __name__=='__main__':
    infile_list=sys.argv[1:]
    saveoption = ''; bincode = ''
    if infile_list[-1].count('text') | infile_list[-1].count('plot'):
        saveoption = infile_list.pop()
    if infile_list[-1][-5:] != '.fits':
        bincode = infile_list.pop()
    specpolview(infile_list, bincode, saveoption)

    


