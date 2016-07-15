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

#---------------------------------------------------------------------------------------------
def specpolview(infile_list, bincode='unbin', saveoption = '', debug_out=False):
    """View output results

    Parameters
    ----------
    infile_list: list
       one or more _stokes.fits files

    bincode  
       unbin, nnA (nn Angstroms), nn% (binned to %)

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

    debug = saveoption.count('debug')>0
    savetext = (saveoption.count('text')>0) | debug
    saveplot = (saveoption.count('plot')>0) | debug
    plotcolor_o = ['b','g','r','c','m','y','k']

    for obs in range(obss):
        hdul = pyfits.open(infile_list[obs])
        name = os.path.basename(infile_list[obs]).split('.')[0]
        obsdate = hdul[0].header['DATE-OBS']
        stokes_sw = hdul['SCI'].data[:,0,:]
        var_sw = hdul['VAR'].data[:,0,:]
        bpm_sw = hdul['BPM'].data[:,0,:]
        stokess,wavs = stokes_sw.shape
        wav0 = hdul['SCI'].header['CRVAL1']
        dwav = hdul['SCI'].header['CDELT1']
        wav_w = wav0 + dwav*np.arange(wavs)
        ok_sw = (bpm_sw==0)
        ok_w = ok_sw.all(axis=0)

    # set up multiplot

        if obs==0:
            stokeslist = hdul['SCI'].header['CTYPE3'].split(',')
            fig,plot_s = plt.subplots(stokess,1,sharex=True)
            plt.xlabel('Wavelength (Ang)')
            plot_s[0].set_ylabel('Intensity')
            for s in range(1,stokess): plot_s[s].set_ylabel(stokeslist[s]+' Polarization (%)')
            if stokeslist[1]=="S": plotname = name.split("_")[-2]
            else: plotname = 'stokes'
            stokeslist[1:] = ('  % '+stokeslist[s] for s in range(1,stokess))
            if stokess > 2:
                if 'PATYPE' in hdul[0].header:
                    pa_type = hdul[0].header['PATYPE']
                else:
                    pa_type = hdul[0].header['POLCAL'].split(" ")[0]    # old style
                stokeslist[1:3] = '  % P', (pa_type[:3]+' T')   
                plot_s[1].set_ylabel('Linear Polarization (%)')
                plot_s[2].set_ylabel(pa_type+' PA (deg)')         
            fig.set_size_inches((8.5,11))
            fig.subplots_adjust(left=0.175)

            namelist=[]

            fmt_s = ['%8.2f ','%8.4f ','%8.3f ','%8.4f ']
            fmt = 2*(' '+"".join(fmt_s[1:stokess]))
            hdr = 'Obs'+(len(name)-3)*' '+'Mean '+(5*" ").join(stokeslist[1:stokess])+"  "+   \
                " Err  ".join(stokeslist[1:stokess])+' Err'+'   Syserr'
            if savetext: print "\n",hdr    

    # calculate, print means (stokes wtd in norm space by 1/mean var)
        hassyserr = hdul[0].header.has_key('SYSERR')
        nstokes_sw = np.zeros_like(stokes_sw)
        nvar_sw = np.zeros_like(var_sw)
        wt_w = np.zeros(wavs)
        nstokes_sw[:,ok_w] = stokes_sw[:,ok_w]/stokes_sw[0,ok_w]      # normalized (degree)
        nvar_sw[:,ok_w] = var_sw[:,ok_w]/stokes_sw[0,ok_w]**2
        wt_w[ok_w] = 1./(nvar_sw[1:stokess,ok_w].mean(axis=0))
        wtavnvar_s0 = np.expand_dims((nvar_sw[:,ok_w]*wt_w[ok_w]**2).sum(axis=1)/    \
            (wt_w[ok_w]).sum()**2,axis=1)
        wtavnstokes_s0 = np.expand_dims((nstokes_sw[:,ok_w]*wt_w[ok_w]).sum(axis=1)/    \
            wt_w[ok_w].sum(),axis=1)
        ok_s0 = np.ones((stokess,1),dtype=bool)

        wtavview_s0,wtaverr_s0 = viewstokes(wtavnstokes_s0,wtavnvar_s0,ok_s0)
        if stokess > 2:
            tcenter = ((0.5*np.arctan2(wtavnstokes_s0[2],wtavnstokes_s0[1])) + np.pi) % np.pi
            wtavview_s0,wtaverr_s0 = viewstokes(wtavnstokes_s0,wtavnvar_s0,ok_s0,tcenter)
        else:
            tcenter = 0.

        if savetext:
            print ("%16s " % name),(fmt % (tuple(wtavview_s0[1:,0])+tuple(wtaverr_s0[1:,0]))),
            if hassyserr: print ('%8.3f' % hdul[0].header['SYSERR']),
            print
  
        plotcolor = plotcolor_o[obs % len(plotcolor_o)]
    # plot intensity
        label = name
        if name.count("_") ==0:             # diffsum multiplots
            label = name[-4:] 
        namelist.append(name)
        ww = -1; 
        while (bpm_sw[0,ww+1:]==0).sum() > 0:
            w = ww+1+np.where(bpm_sw[0,ww+1:]==0)[0][0]
            ww = wavs
            dw = np.where(bpm_sw[0,w:]>0)[0]  
            if dw.size: ww = w + dw[0] - 1 
            plot_s[0].plot(wav_w[w:ww],stokes_sw[0,w:ww],color=plotcolor,label=label)
            label = '_'+name    

        if bintype == 'unbin':
            nstokes_sw, nerr_sw = viewstokes(stokes_sw,var_sw,ok_sw,tcenter)

        # show gaps in plot, remove them from text
            for s in range(1,stokess):
                ww = -1; 
                while (bpm_sw[s,ww+1:]==0).sum() > 0:
                    w = ww+1+np.where(bpm_sw[s,ww+1:]==0)[0][0]
                    ww = wavs
                    dw = np.where(bpm_sw[s,w:]>0)[0]  
                    if dw.size: ww = w + dw[0] - 1             
                    plot_s[s].plot(wav_w[w:ww],nstokes_sw[s,w:ww],color=plotcolor,label=label)

            wav_v = wav_w[ok_sw[0]]
            nstokes_sv = nstokes_sw[:,ok_sw[0]]
            nerr_sv = nerr_sw[:,ok_sw[0]]

        else:
        # Set up bins, blocked, or binned to error based on stokes 1 or on linear stokes p
            if bintype == 'wavl':
                bin_w = (wav_w / blk -0.5).astype(int) - int((wav_w / blk -0.5).min())
                Bins = bin_w.max()
                bin_w[~ok_sw[1]] = -1
            else:
                allowedgap = 5
                wgap0_g = np.where((bpm_sw[0,:-1]==0) & (bpm_sw[0,1:]<>0))[0] + 1
                wgap1_g = np.where((bpm_sw[0,wgap0_g[0]:-1]<>0) & (bpm_sw[0,wgap0_g[0]+1:]==0))[0] \
                    +  wgap0_g[0] + 1
                wgap0_g = wgap0_g[0:wgap1_g.shape[0]]
                isbad_g = ((wgap1_g - wgap0_g) > allowedgap)
                nstokes_sw, nerr_sw = viewstokes(stokes_sw,var_sw,ok_sw,tcenter)
                binvar_w = nerr_sw[1]**2
                ww = -1; b = 0;  bin_w = -1*np.ones((wavs))
                while (bpm_sw[0,ww+1:]==0).sum() > 0:
                    w = ww+1+np.where(bpm_sw[0,ww+1:]==0)[0][0]
                    cumsvar_w = np.cumsum(binvar_w[w:]*(bpm_sw[0,w:]==0))    \
                                /np.cumsum((bpm_sw[0,w:]==0))**2
                    err_w = np.sqrt(cumsvar_w)
                    if debug: np.savetxt("err_"+str(w)+".txt",err_w,fmt="%10.3e")
                    ww = wavs                                       # stopping point override: end
                    nextbadgap = np.where(isbad_g & (wgap0_g > w))[0]
                    if nextbadgap.size: ww = wgap0_g[nextbadgap[0]] - 1   # stopping point override: before bad gap
                    dw = np.where(err_w[:ww-w] < errbin)[0]
                    if dw.size: ww = w + dw[0]                      # err goal is reached first
                    bin_w[w:ww+1] = b
                    b += 1
                bin_w[bpm_sw[0]>0] = -1
                Bins  = b
                if debug_out: 
                    np.savetxt(name+'_'+bincode+'_binid.txt',np.vstack((wav_w,bin_w)).T,fmt="%8.2f %5i")

        # calculate binned data. _V = possible Bins, _v = good bins
            bin_V = np.arange(Bins)                   
            bin_Vw = (bin_V[:,None] == bin_w[None,:])
            stokes_sV = (stokes_sw[:,None,:]*bin_Vw).sum(axis=2)
            var_sV = (var_sw[:,None,:]*bin_Vw).sum(axis=2) 
            bpm_sV = ((bpm_sw[:,None,:]*bin_Vw).sum(axis=2)==bin_Vw.sum(axis=1)).astype(int)
            ok_sV = (bpm_sV == 0)
            ok_V = ok_sV.all(axis=0)
            bin_vw = bin_Vw[ok_V]
            wav_v = (wav_w[None,:]*bin_vw).sum(axis=1)/bin_vw.sum(axis=1)
            dwavleft_v = wav_v - wav_w[(np.argmax((wav_w[None,:]*bin_vw)>0,axis=1))] + dwav/2.
            dwavright_v = wav_w[wavs-1-(np.argmax((wav_w[None,::-1]*bin_vw[:,::-1])>0,axis=1))] - wav_v - dwav/2.
            nstokes_sV, nerr_sV = viewstokes(stokes_sV,var_sV,ok_sV,tcenter)          
            nstokes_sv, nerr_sv = nstokes_sV[:,ok_V], nerr_sV[:,ok_V]
            for s in range(1,stokess):
                if debug_out: np.savetxt('errbar_'+str(s)+'.txt', \
                    np.vstack((wav_v,nstokes_sv[s],nerr_sv[s],dwavleft_v,dwavright_v)).T,fmt = "%10.4f")
                plot_s[s].errorbar(wav_v,nstokes_sv[s],color=plotcolor,fmt='.',    \
                    yerr=nerr_sv[s],xerr=(dwavleft_v,dwavright_v),capsize=0)

        textfile = sys.stdout
        if savetext: 
            textfile = open(name+'_'+bincode+'.txt','a')
            textfile.truncate(0)
        else: print >>textfile

        print >>textfile, name+'   '+obsdate+'\n\n'+'Wavelen   '+(4*" ").join(stokeslist[1:stokess])+(2*" ")+   \
                " Err   ".join(stokeslist[1:stokess])+' Err '+'   Syserr'
        print >>textfile, ((' wtdavg  '+fmt) % (tuple(wtavview_s0[1:,0])+tuple(wtaverr_s0[1:,0]))),
        if hassyserr: print >>textfile,('%8.3f' % hdul[0].header['SYSERR']),
        print >>textfile,'\n'
        np.savetxt(textfile,np.vstack((wav_v,nstokes_sv[1:],nerr_sv[1:])).T, fmt=(fmt_s[0]+fmt))

    plot_s[0].set_ylim(bottom=0)                            # intensity plot baseline 0
    if stokess >2: 
        plot_s[1].set_ylim(bottom=0)                        # linear polarization % plot baseline 0
        ymin,ymax = plot_s[2].set_ylim()
        plot_s[2].set_ylim(bottom=min(ymin,(ymin+ymax)/2.-5.),top=max(ymax,(ymin+ymax)/2.+5.))

    if obss>1: 
       plot_s[0].legend(fontsize='x-small',loc='upper left')
    else: 
       plot_s[0].set_title(name+"   "+obsdate) 

    if saveplot:
        ylimlist = (raw_input('\nOptional scale (bottom-top, comma sep): ')).split(',')
        for (i,ys) in enumerate(ylimlist):
            if ((len(ys)>0) & ((i % 2)==0)): plot_s[stokess-i/2-1].set_ylim(bottom=float(ys))
            if ((len(ys)>0) & ((i % 2)==1)): plot_s[stokess-i/2-1].set_ylim(top=float(ys))
        tags = name.count("_")
        cyclelist = []
        if tags:                 # raw and final stokes files
            objlist = sorted(list(set(namelist[b].split("_")[0] for b in range(obss))))
            conflist = sorted(list(set(namelist[b].split("_")[1] for b in range(obss))))
            if tags>2: cyclelist = sorted(list(set(namelist[b].split("_")[2] for b in range(obss))))
            plotfile = '_'.join(objlist+conflist+cyclelist+list([plotname,bincode]))+'.pdf'
        else:                               # diffsum files from diffsum
            plotfile = namelist[0]+'-'+namelist[-1][-4:]+'.pdf'
        plt.savefig(plotfile,orientation='portrait')
        if os.name=='posix':
            if os.popen('ps -C evince -f').read().count(plotfile)==0: os.system('evince '+plotfile+' &')
    else: 
        plt.show(block=True)
    return

#---------------------------------------------------------------------------------------------
def viewstokes(stokes_sw,var_sw,ok_sw,tcenter=0.):
    """Compute normalized stokes parameters, converts Q-U to P-T, for viewing

    Parameters
    ----------
    stokes_sw: 2d float nparray(stokes,wavelength bin)
       unnormalized stokes parameters vs wavelength

    var_sw: 2d float nparray(stokes,wavelength bin) 
       variance for stokes_sw

    ok_sw: 2d boolean nparray(stokes,wavelength bin) 
       marking good stokes values

    Output: normalized stokes parameters and errors, linear stokes converted to pol degree, PA

    """

    stokess,wavs = stokes_sw.shape
    nstokes_sw = np.zeros((stokess,wavs))
    nerr_sw = np.zeros((stokess,wavs))

    nstokes_sw[1:,ok_sw[0]] = 100.*stokes_sw[1:,ok_sw[0]]/stokes_sw[0,ok_sw[0]]                            # in percent
    nerr_sw[1:,ok_sw[0]] = 100.*np.sqrt(var_sw[1:stokess,ok_sw[0]])/stokes_sw[0,ok_sw[0]]

    if stokes_sw.shape[0]>2:
        wavs = stokes_sw.shape[1]
        stokesp_w = np.zeros((wavs))
        stokest_w = np.zeros((wavs))
        varp_w = np.zeros((wavs))
        vart_w = np.zeros((wavs))
        varpe_dw = np.zeros((2,wavs))
        varpt_w = np.zeros((wavs))
        ok_w = ok_sw[:3].all(axis=0)
        stokesp_w[ok_w] = np.sqrt(stokes_sw[1,ok_w]**2 + stokes_sw[2,ok_w]**2)      # unnormalized linear polarization
        stokest_w[ok_w] = (0.5*np.arctan2(stokes_sw[2,ok_w],stokes_sw[1,ok_w]))     # PA in radians
        stokest_w[ok_w] = (stokest_w[ok_w]-(tcenter+np.pi/2.)+np.pi) % np.pi + (tcenter-np.pi/2.)
                                                                                    # optimal PA folding                
     # variance matrix eigenvalues, ellipse orientation
        varpe_dw[:,ok_w] = 0.5*(var_sw[1,ok_w]+var_sw[2,ok_w]                          \
            + np.array([1,-1])[:,None]*np.sqrt((var_sw[1,ok_w]-var_sw[2,ok_w])**2 + 4*var_sw[-1,ok_w]**2))
        varpt_w[ok_w] = 0.5*np.arctan2(2.*var_sw[-1,ok_w],var_sw[1,ok_w]-var_sw[2,ok_w])
     # linear polarization variance along p, PA   
        varp_w[ok_w] = varpe_dw[0,ok_w]*(np.cos(2.*stokest_w[ok_w]-varpt_w[ok_w]))**2   \
               + varpe_dw[1,ok_w]*(np.sin(2.*stokest_w[ok_w]-varpt_w[ok_w]))**2
        vart_w[ok_w] = varpe_dw[0,ok_w]*(np.sin(2.*stokest_w[ok_w]-varpt_w[ok_w]))**2   \
               + varpe_dw[1,ok_w]*(np.cos(2.*stokest_w[ok_w]-varpt_w[ok_w]))**2

        nstokes_sw[1,ok_w] = 100*stokesp_w[ok_w]/stokes_sw[0,ok_w]                  # normalized % linear polarization
        nerr_sw[1,ok_w] =  100*np.sqrt(var_sw[1,ok_w])/stokes_sw[0,ok_w]
        nstokes_sw[2,ok_w] = np.degrees(stokest_w[ok_w])                            # PA in degrees
        nerr_sw[2,ok_w] =  0.5*np.degrees(np.sqrt(var_sw[2,ok_w])/stokesp_w[ok_w])

    return nstokes_sw,nerr_sw 
 
if __name__=='__main__':
    infile_list=sys.argv[1:]
    saveoption = ''
    bincode = 'unbin'
    if infile_list[-1].count('text') | infile_list[-1].count('plot') | infile_list[-1].count('debug'):
        saveoption = infile_list.pop()
    if infile_list[-1][-5:] != '.fits':
        bincode = infile_list.pop()
    specpolview(infile_list, bincode, saveoption)

    


