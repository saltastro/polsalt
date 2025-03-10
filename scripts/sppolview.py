#! /usr/bin/env python2.7

"""
sppolview

Plot and text output of stokes grating LS or MOS data, optionally binned

"""

import os, sys, glob, shutil, inspect

import numpy as np
import zipfile
from astropy.io import fits as pyfits
from astropy.table import Table

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

from polutils import specpolrotate

import rsslog
from obslog import create_obslog
import matplotlib
matplotlib.use('PDF',warn=False)
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter  
plt.ioff()
np.set_printoptions(threshold=np.nan)
import warnings
#warnings.simplefilter("error")

#---------------------------------------------------------------------------------------------
def sppolview(infileList, *viewtargetList, **kwargs):
    """View Stokes output results

    Parameters
    ----------
    infileList: str or list
            one or more _stokes.fits files

    viewtargetList = [] (all targets default)
            list of targets (integer ids in TGT table, not CATID) (multiple targets viewed only for one infile)
    bin=    unbin (default)
            nnA (nn Angstroms)
            nn% (binned to %)
    errors= False(default) 
            True (plot errorbars)
    connect= '' (default) (if errors False, connect points)
            hist (if binned, use histogram connection)
    yxlim=  [] (default) (plot limits, list of 4 if provided)
    type=   Ipt (default) (unbinned intensity/flux, binned % pol, deg PA)
            Iqu (unbinned intensity/flux, binned %q, %u, with optional rotate)
    save=   '' (default) (text to terminal)
            text (text to file)
            plot (terminal plot and pdf file)
            textplot (both)
    debug=: False (default)
            True (debug output)
    logfile= 'sppolview.log' (default)
    """
    logfile= kwargs.pop('logfile','sppolview.log')    
    with_stdout = kwargs.pop('with_stdout',True)       
    if with_stdout: rsslog.history(logfile)
    
    if (type(infileList)==str): infileList = [infileList,]
    obss = len(infileList)    
    bin = kwargs.pop('bin','unbin')
    errorbars = (kwargs.pop('errors','False') == 'True')
    connect = kwargs.pop('connect','')
    yxlimList = kwargs.pop('yxlim',[])
    plottype = kwargs.pop('type','Ipt')
    save = kwargs.pop('save','')
    debug = (kwargs.pop('debug','False') == 'True')
    logfile= kwargs.pop('logfile','sppolview.log')
    
    bintype = bin
    if bin.count('%'): 
        bintype = 'percent'
        errbin = float(bin[ :bin.index('%')])
    elif bin.count('A'): 
        bintype = 'wavl'
        blk = float(bin[ :bin.index('A')])
    elif (bin != 'unbin'): 
        rsslog.message("unrecognized binning option, set to unbinned",logfile, with_stdout=with_stdout)
        bintype = 'unbin'
    if (len(bin)>6):              # for backwards compatibility
        errorbars = (bin[-6:]=="errors")
    if (connect not in ('','hist')):
        rsslog.message("unrecognized connect option",logfile, with_stdout=with_stdout)    
        exit()
    if (plottype not in ('Ipt','Iqu')):
        rsslog.message("unrecognized type option",logfile, with_stdout=with_stdout)       
        exit()
    savetext = (save.count('text')>0)
    saveplot = (save.count('plot')>0)
                    
    plotcolor_o = ['b','g','r','c','m','y','k']
    askpltlim = True
    tcenter = 0.            
    if (len(yxlimList)>0):
        yxlimList = yxlimList[1:-1].split(',')
        ismanlim_i = np.array([len(ys)>0 for ys in yxlimList])
        if ismanlim_i[0]:
            tcenter = np.radians((float(yxlimList[0]) + float(yxlimList[1]))/2.)                                
        askpltlim = False 
               
    trotate = 0.
    cunitfluxed = 'erg/s/cm^2/Ang'          # header keyword CUNIT3 if data is already fluxed

  # tabulate data targets and view targets
    hdul0 = pyfits.open(infileList[0])                  
    if 'TGT' in [hdul0[x].name for x in range(len(hdul0))]: 
        tgtTab = Table.read(hdul0['TGT'])
        oktgt_i = (tgtTab['CULL'] == '')
        tgtname_i = np.array(tgtTab['CATID'])                
        i_t = np.where(oktgt_i)[0]          
    else:
        tgtname_i = [""]
        i_t = [0]                
                                                     
    if len(viewtargetList) == 0:
        viewtargetList = list(i_t)
        targetstr = 'all'
    else:
        if (not np.isin(viewtargetList,i_t).all()):
            rsslog.message("Not all requested targets are in data, exitting", logfile, with_stdout=with_stdout)
            exit()
        targetstr = '_'.join(map(str,viewtargetList))       # need to use tdigits, sorted
    viewtargets = len(viewtargetList)
            
    if ((obss>1) & (viewtargets>1)):
        rsslog.message("Multiple observations with multiple viewed targets not supported, exitting",   \
            logfile, with_stdout=with_stdout)       
        exit() 

  # compute filename for output text zip and plot pdf files
    stokesList = hdul0['SCI'].header['CTYPE3'].split(',')  
    if stokesList[1]=="S": 
        plotname = name.split("_")[-2]
    else: 
        plotname = 'stokes'
    nameList=[os.path.basename(infileList[obs]).split('.')[0] for obs in range(obss)]                                     
    tags = nameList[0].count("_")
    if tags:                 # raw and final stokes files
        objList = sorted(list(set(nameList[obs].split("_")[0] for obs in range(obss))))
        confcycleList = sorted(list(set(nameList[obs].replace("_stokes","").split("_",1)[-1] for obs in range(obss))))
        if ((viewtargets>1) | (obss==1)):
            outfile = '_'.join([objList[0],targetstr]+confcycleList+list([plotname,bin,plottype]))
        else:
            outfile = '-'.join([objList[0],objList[-1]])+'_'+  \
                        '_'.join([targetstr,]+confcycleList+list([plotname,bin,plottype]))                   
    else:                               # diffsum files from diffsum
        outfile = nameList[0]+'-'+nameList[-1][-4:]        
     
    if (savetext & ((viewtargets > 1) | (obss > 1))):
        zipname = outfile+'.zip'
        zl = zipfile.ZipFile(zipname,mode='w')          # truncate any previous file by this name
        zl = zipfile.ZipFile(zipname,mode='a')
             
    for obs in range(obss):
        hdul = pyfits.open(infileList[obs])
        obsdate = hdul[0].header['DATE-OBS']
        stokes_Stw = hdul['SCI'].data
        var_Stw = hdul['VAR'].data
        covar_Stw = hdul['COV'].data
        bpm_Stw = hdul['BPM'].data
        isfluxed=False
        if 'CUNIT3' in hdul['SCI'].header:
            isfluxed=(hdul['SCI'].header['CUNIT3'].replace(' ','') ==cunitfluxed)
        stokess,targets,wavs = stokes_Stw.shape 

                           

        wav0 = hdul['SCI'].header['CRVAL1']
        dwav = hdul['SCI'].header['CDELT1']
        wav_w = wav0 + dwav*np.arange(wavs)
        ok_Stw = (bpm_Stw==0)
        ok_tw = ok_Stw.all(axis=0)

    # set up multiplot
        if obs==0:
            fig,plot_S = plt.subplots(stokess,1,sharex=True)
            plt.xlabel('Wavelength (Ang)')
            plot_S[0].set_ylabel(['Intensity','Flambda ('+cunitfluxed+')'][isfluxed])
            for S in range(1,stokess): plot_S[S].set_ylabel(stokesList[S]+' Polarization (%)')
            stokesList[1:] = ('  % '+stokesList[s] for s in range(1,stokess))
            if ((plottype == 'Ipt') | (plottype == 'IPt')):
                if 'PATYPE' in hdul[0].header:
                    pa_type = hdul[0].header['PATYPE']
                else:
                    pa_type = hdul[0].header['POLCAL'].split(" ")[0]    # old style
                plot_S[2].set_ylabel(pa_type+' PA (deg)')
            if (plottype == 'Ipt'):
                stokesList[1:3] = '  % P', (pa_type[:3]+' T')   
                plot_S[1].set_ylabel('Linear Polarization (%)')
            if (plottype == 'Iqu'):
                stokesList[1:3] = ('  % Q', ' % U')   
                plot_S[1].set_ylabel('Stokes Q (%)')
                plot_S[2].set_ylabel('Stokes U (%)')         
            fig.set_size_inches((8.5,11))
            fig.subplots_adjust(left=0.175)

    # calculate, print means (stokes average in unnorm space) 
        logtext = ("\n%16s %16s  Wtd mean   " % (nameList[obs],obsdate))    
        if ('SYSERR' in hdul[0].header): 
            logtext += ('Syserr: %8.3f' % hdul[0].header['SYSERR'])
        rsslog.message(logtext,logfile, with_stdout=with_stdout)          
        avstokes_Sm = np.empty((stokess,0))    
        avvar_Sm = np.empty((stokess,0))
        target_m = np.array([],dtype=int)
        avwav_m = np.array([]) 
        wav1_m = np.array([])
        wav2_m = np.array([])                        

        for i in viewtargetList:
            t = np.where(i_t==i)[0][0]
            ok_Sw = ok_Stw[:,t] 
            ok_w = ok_Sw[0]
            if ok_w.sum()==0: continue 
                
            avstokes_s, avvar_s, avwav =    \
                avstokes(stokes_Stw[:,t,ok_w],var_Stw[:-1][:,t,ok_w],covar_Stw[:,t,ok_w],wav_w[ok_w]) 
            avstokes_S = np.insert(avstokes_s,0,1.)
            avvar_S = np.insert(avvar_s,0,1.)
            avstokes_Sm = np.append(avstokes_Sm,avstokes_S[:,None],axis=1) 
            avvar_Sm = np.append(avvar_Sm,avvar_S[:,None],axis=1) 
            target_m = np.append(target_m,i)
            wav1_m = np.append(wav1_m,wav_w[ok_w][0])
            wav2_m = np.append(wav2_m,wav_w[ok_w][-1])                

          # plot intensity
            if (viewtargets>1):
                label = tgtTab['CATID'][i]
                plotcolor = plotcolor_o[t % len(plotcolor_o)]             
            else:
                label = obsdate+' '+nameList[obs]         
                plotcolor = plotcolor_o[obs % len(plotcolor_o)]                   
            if nameList[obs].count("_") ==0:             # diffsum multiplots
                label = nameList[obs][-4:] 
            ww = -1; 
            while (bpm_Stw[0,t,ww+1:]==0).sum() > 0:
                w = ww+1+np.where(bpm_Stw[0,t,ww+1:]==0)[0][0]
                ww = wavs
                dw = np.where(bpm_Stw[0,t,w:]>0)[0]  
                if dw.size: ww = w + dw[0] - 1                 
                plot_S[0].plot(wav_w[w:ww],stokes_Stw[0,t,w:ww],color=plotcolor,label=label)
                label = ""

        if (viewtargets>1):
            plot_S[0].legend(fontsize='x-small',loc='upper left',labelspacing=0.25)

        tgtname_m = tgtname_i[target_m]                    
        namelen = max(map(len,[x.strip() for x in tgtname_m]))        
        printstokestw(avstokes_Sm,avvar_Sm,target_m,tgtname_m,wav1_m,wav2_m)
    
      # get manual plot limits, resetting tcenter (PA wrap center) if necessary
        if saveplot:
            while (askpltlim):
                yxlimList = (raw_input('\nOptional scale (bottom-top, comma sep): ')).split(',')
                if len(''.join(yxlimList))==0: yxlimList = []
                ismanlim_i = np.array([len(ys)>0 for ys in yxlimList])
                if ismanlim_i.sum() == 0: 
                    askpltlim = False
                    break
                if stokess>2:
                    itbottom,ittop = [2*(stokess-3),2*(stokess-3)+1]
                    if (ismanlim_i[itbottom] != ismanlim_i[ittop]): 
                        print "set bottom plot limits for either both or neither top and bottom"
                        continue
                    if ((plottype == 'Ipt') | (plottype == 'IPt')):
                        if ismanlim_i[itbottom]:
                            tcenter = np.radians((float(yxlimList[itbottom]) + float(yxlimList[ittop]))/2.)
                    if ((plottype == 'Iqu') | (plottype == 'IQU')):
                        trotate = float(raw_input('\nOptional PA zeropoint (default 0): ') or '0')
                        if trotate:
                            plot_S[1].set_ylabel('Stokes Q (%%)  PA0= %7.1f deg' % trotate)
                            plot_S[2].set_ylabel('Stokes U (%%)  PA0= %7.1f deg' % trotate)  
                        if (len(ismanlim_i) == 2):
                            ismanlim_i = np.tile(ismanlim_i,2)
                            yxlimList += yxlimList
                                                       
                askpltlim = False
                     
      # assemble data
        
        for i in viewtargetList:
            t = np.where(i_t==i)[0][0]
            stokes_Sw = stokes_Stw[:,t]
            var_Sw = var_Stw[:,t]
            covar_Sw = covar_Stw[:,t]
            bpm_Sw = bpm_Stw[:,t]    
            ok_Sw = ok_Stw[:,t] 
            ok_w = ok_Sw[0]
            if (viewtargets>1):
                label = '_'+str(viewtargetList[t]).zfill(2)
                plotcolor = plotcolor_o[t % len(plotcolor_o)]             
            else:
                label = '_'+nameList[obs]           
                plotcolor = plotcolor_o[obs % len(plotcolor_o)]  
                
            if trotate:
                stokes_Sw,var_Sw,covar_Sw = specpolrotate(stokes_Sw,var_Sw,covar_Sw,-trotate)

            if bintype == 'unbin':
                stokes_sw = np.zeros((stokess-1,wavs))
                err_sw = np.zeros_like(stokes_sw)
                if plottype == 'Ipt':                                       # _s = %p, PA (deg)
                    stokes_sw[:,ok_w], err_sw[:,ok_w] =         \
                        viewstokes(stokes_Sw[:,ok_w],var_Sw[:,ok_w],tcenter=tcenter)  
                elif plottype =='Iqu':                                      # _s = %q, %u
                    stokes_sw[:,ok_w] = 100.*stokes_Sw[1:,ok_w]/stokes_Sw[0,ok_w]
                    err_sw[:,ok_w] = 100.*np.sqrt(var_Sw[1:-1,ok_w]/stokes_Sw[0,ok_w]**2)

              # show gaps in plot, remove them from text
                for S in range(1,stokess):
                    ww = -1; 
                    while (bpm_Sw[S,ww+1:]==0).sum() > 0:
                        w = ww+1+np.where(bpm_Sw[S,ww+1:]==0)[0][0]
                        ww = wavs
                        dw = np.where(bpm_Sw[S,w:]>0)[0]  
                        if dw.size: ww = w + dw[0] - 1             
                        plot_S[S].plot(wav_w[w:ww],stokes_sw[S-1,w:ww],color=plotcolor,label=label)

                wav_v = wav_w[ok_Sw[0]]
                stokes_sv = stokes_sw[:,ok_Sw[0]]
                err_sv = err_sw[:,ok_Sw[0]]

            else:
              # Set up bins, blocked, or binned to error based on stokes 1 or on linear stokes p
                if bintype == 'wavl':
                    bin_w = (wav_w / blk -0.5).astype(int) - int((wav_w / blk -0.5).min())
                    Bins = bin_w.max()
                    bin_w[~ok_Sw[1]] = -1
                else:
                    allowedgap = 5
                    wgap0_g = np.where((bpm_Sw[0,:-1]==0) & (bpm_Sw[0,1:]<>0))[0] + 1
                    wgap1_g = np.where((bpm_Sw[0,wgap0_g[0]:-1]<>0) & (bpm_Sw[0,wgap0_g[0]+1:]==0))[0] \
                        +  wgap0_g[0] + 1
                    wgap0_g = wgap0_g[0:wgap1_g.shape[0]]
                    isbad_g = ((wgap1_g - wgap0_g) > allowedgap)
                    if debug:
                        np.savetxt("bininput.txt",np.vstack((wav_w,ok_w,stokes_Sw,var_Sw)).T,   \
                        fmt="%7.2f %3i "+7*"%10.4e ")
                    stokes_sw, err_sw = viewstokes(stokes_Sw,var_Sw,ok_w=ok_w,tcenter=tcenter)
                    binvar_w = err_sw[0]**2
                    bincovar_w = np.zeros_like(binvar_w)
                    bincovar_w[ok_w] = binvar_w[ok_w]*covar_Sw[1,ok_w]/var_Sw[1,ok_w]
                    ww = -1; b = 0;  bin_w = -1*np.ones((wavs))
                    while (bpm_Sw[0,ww+1:]==0).sum() > 0:
                        w = ww+1+np.where(bpm_Sw[0,ww+1:]==0)[0][0]
                        cumsvar_W = np.cumsum((binvar_w[w:]+2.*bincovar_w[w:])*(bpm_Sw[0,w:]==0))    \
                                    /np.cumsum((bpm_Sw[0,w:]==0))**2
                        err_W = np.sqrt(cumsvar_W)
                        if debug: np.savetxt("err_"+str(w)+".txt",      \
                            np.vstack((wav_w[w:],bpm_Sw[0,w:],binvar_w[w:],bincovar_w[w:],err_W)).T,fmt="%10.3e")
                        ww = wavs                                       # stopping point override: end
                        nextbadgap = np.where(isbad_g & (wgap0_g > w))[0]
                        if nextbadgap.size: ww = wgap0_g[nextbadgap[0]] - 1   # stopping point override: before bad gap
                        dw = np.where(err_W[:ww-w] < errbin)[0]
                        if dw.size: ww = w + dw[0]                      # err goal is reached first
                        bin_w[w:ww+1] = b
                        b += 1
                    bin_w[bpm_Sw[0]>0] = -1
                    Bins  = b
                    if debug: 
                        np.savetxt(name+'_'+bin+'_binid.txt',np.vstack((wav_w,bin_w)).T,fmt="%8.2f %5i")

              # calculate binned data. _V = possible Bins, _v = good bins
                bin_V = np.arange(Bins)
                bin_Vw = (bin_V[:,None] == bin_w[None,:])
                stokes_SV = (stokes_Sw[:,None,:]*bin_Vw).sum(axis=2)
                var_SV = ((var_Sw[:stokess,None,:] + 2.*covar_Sw[:,None,:])*bin_Vw).sum(axis=2)  
                bpm_SV = ((bpm_Sw[:,None,:]*bin_Vw).sum(axis=2)==bin_Vw.sum(axis=1)).astype(int)           
                ok_SV = (bpm_SV == 0)
                ok_V = ok_SV.all(axis=0)
                if isfluxed:
                    stokes_SV[:,ok_V] = stokes_SV[:,ok_V]/bin_Vw[ok_V].sum(axis=1)[None,:]
                    var_SV[:,ok_V] = var_SV[:,ok_V]/((bin_Vw[ok_V].sum(axis=1))**2)[None,:]             
                bin_vw = bin_Vw[ok_V]
                wav_v = (wav_w[None,:]*bin_vw).sum(axis=1)/bin_vw.sum(axis=1)
                dwavleft_v = wav_v - wav_w[(np.argmax((wav_w[None,:]*bin_vw)>0,axis=1))-1] + dwav/2.
                dwavright_v = wav_w[wavs-1-(np.argmax((wav_w[None,::-1]*bin_vw[:,::-1])>0,axis=1))] - wav_v - dwav/2.
                if plottype == 'Ipt':                                       # _s = %p, PA (deg)
                    stokes_sv, err_sv = viewstokes(stokes_SV[:,ok_V],var_SV[:,ok_V],tcenter=tcenter)   
                elif plottype =='Iqu':                                      # _s = %q, %u
                    stokes_sv = 100.*stokes_SV[1:,ok_V]/stokes_SV[0,ok_V]
                    err_sv = 100.*np.sqrt(var_SV[1:,ok_V]/stokes_SV[0,ok_V]**2)
      
                lwdefault = matplotlib.rcParams['lines.linewidth']
                for S in range(1,stokess):
                    if debug: np.savetxt('errbar_'+str(S)+'.txt', \
                        np.vstack((wav_v,stokes_SV[S-1],var_SV[S-1],stokes_sv[S-1],err_sv[S-1],   \
                        dwavleft_v,dwavright_v)).T, fmt = "%10.4f")
                    if ((connect != 'hist') & (not errorbars)):
                        if (bintype=='unbin'): marker = 'none'
                        else: marker = '.'
                        plot_S[S].plot(wav_v,stokes_sv[S-1],color=plotcolor,marker='.',label=label)
                    if errorbars:                                       # plot y error bars
                        plot_S[S].errorbar(wav_v,stokes_sv[S-1],yerr=err_sv[S-1],   \
                            color=plotcolor,marker='None',linewidth=0.,elinewidth=lwdefault,capsize=0)     
                    if (connect == 'hist'):                             # plot vertical bars in histogram
                        wavright_v = wav_v + dwavright_v
                        doconnect_v = (wavright_v[:-1] == (wav_v[1:]-dwavleft_v[1:]))
                        vbm_sv = 0.5*(stokes_sv[:,:-1] + stokes_sv[:,1:])
                        vbe_sv = 0.5*np.abs(stokes_sv[:,:-1] - stokes_sv[:,1:])
                        plot_S[S].errorbar(wavright_v[:-1][doconnect_v],vbm_sv[S-1][doconnect_v],    \
                            yerr=vbe_sv[S-1][doconnect_v],color=plotcolor,marker='None',    \
                            linewidth=0.,elinewidth=lwdefault,capsize=0)         
                    if ((connect == 'hist') | errorbars):               # plot horizontal bars in histogram   
                        plot_S[S].errorbar(wav_v,stokes_sv[S-1],xerr=(dwavleft_v,dwavright_v),  \
                            color=plotcolor,marker='None',linewidth=0.,elinewidth=lwdefault,capsize=0)                                                  

          # Printing for observation
            textFile = sys.stdout
            if savetext:
                textfilename = nameList[obs]+'_'+("%02s" % i)+'_'+bin+'.txt' 
                textFile = open(textfilename,'ab')
                textFile.truncate(0)              
            else: print >>textFile

            if stokess > 2:
                if tcenter == 0:                 # tcenter has not been set by manual plot limits
                    tcenter = ((0.5*np.arctan2(avstokes_s[1],avstokes_s[0])) + np.pi) % np.pi
            else:
                tcenter = np.pi/2.

            textline = ("\n%16s %16s    " % (nameList[obs],obsdate))    
            if ('SYSERR' in hdul[0].header): 
                textline += ('Syserr: %8.3f' % hdul[0].header['SYSERR'])
            print >>textFile, textline 

            if bintype == 'unbin':
                printstokes(stokes_Sw[:,ok_w],var_Sw[:stokess,ok_w],wav_w[ok_w],   \
                    tcenter=tcenter,isfluxed=isfluxed,textFile=textFile)
            else:         
                printstokes(stokes_SV[:,ok_V],var_SV[:stokess,ok_V],wav_v,         \
                    tcenter=tcenter,isfluxed=isfluxed,textFile=textFile)
                    
            if (savetext & ((viewtargets > 1) | (obss > 1))):
                textFile.close()
                zl.write(textfilename)
                os.remove(textfilename)

    if (savetext & ((viewtargets > 1) | (obss > 1))):
        zl.close()

  # Plotting of stacked observations
    if saveplot:
        plot_S[0].set_ylim(bottom=0)                # intensity plot default baseline 0
        if stokess >2:
            if plottype == 'Ipt': 
                plot_S[1].set_ylim(bottom=0)            # linear polarization % plot default baseline 0
                ymin,ymax = plot_S[2].set_ylim()        # linear polarization PA plot default 5 degree pad
                plot_S[2].set_ylim(bottom=min(ymin,(ymin+ymax)/2.-5.),top=max(ymax,(ymin+ymax)/2.+5.))
        if len(yxlimList)>0:
            for (i,ys) in enumerate(yxlimList):
                if (i>3): continue
                S = stokess-i/2-1
                if (ismanlim_i[i] & ((i % 2)==0)): plot_S[S].set_ylim(bottom=float(ys))
                if (ismanlim_i[i] & ((i % 2)==1)): plot_S[S].set_ylim(top=float(ys))
        if len(yxlimList)>4:
            if (ismanlim_i[4]): plot_S[S].set_xlim(left=float(yxlimList[4]))            
            if (ismanlim_i[5]): plot_S[S].set_xlim(right=float(yxlimList[5]))    

        if (obss>1):
            plot_S[0].set_title("Target  "+str(viewtargetList[0]).zfill(2))
            plot_S[0].legend(fontsize='x-small',loc='upper left',labelspacing=0.25)
        elif (viewtargets>1): 
            plot_S[0].set_title(nameList[0]+"   "+obsdate)
        else:
            if (targets>1):
                plot_S[0].set_title(nameList[0]+"   "+obsdate+"  Target  "+str(viewtargetList[0]).zfill(2))
            else:
                plot_S[0].set_title(nameList[0]+"   "+obsdate)

        plotfile = outfile+'.pdf'                                     
        plt.savefig(plotfile,orientation='portrait')
        
        if ((os.name=='posix')&(inspect.stack()[0][1]==inspect.stack()[1][1])):     # plot from unix cmd line
            if os.popen('ps -C evince -f').read().count(plotfile)==0: os.system('evince '+plotfile+' &')
            
        plt.close('all')
    else: 
        plt.show(block=True)
    return

#---------------------------------------------------------------------------------------------
def viewstokes(stokes_Sw,err2_Sw,ok_w=[True],tcenter=0.):
    """Compute normalized stokes parameters, converts Q-U to P-T, for viewing

    Parameters
    ----------
    stokes_Sw: 2d float nparray(stokes,wavelength bin)
       unnormalized stokes parameters vs wavelength

    var_Sw: 2d float nparray(stokes,wavelength bin) 
       variance for stokes_sw

    ok_w: 1d boolean nparray(stokes,wavelength bin) 
       marking good stokes values. default all ok.

    Output: normalized stokes parameters and errors, linear stokes converted to pol %, PA
       Ignore covariance.  Assume binned first if necessary.

    """
    warnings.simplefilter("error")
    stokess,wavs = stokes_Sw.shape
    stokes_vw = np.zeros((stokess-1,wavs))
    err_vw = np.zeros((stokess-1,wavs))
    if (len(ok_w) == 1): ok_w = np.ones(wavs,dtype=bool)

    stokes_vw[:,ok_w] = 100.*stokes_Sw[1:,ok_w]/stokes_Sw[0,ok_w]               # in percent
    err_vw[:,ok_w] = 100.*np.sqrt(err2_Sw[1:stokess,ok_w])/stokes_Sw[0,ok_w]     # error bar ignores covariance

    if (stokess >2):
        stokesP_w = np.zeros((wavs))
        stokesT_w = np.zeros((wavs))
        varP_w = np.zeros((wavs))
        varT_w = np.zeros((wavs))
        varpe_dw = np.zeros((2,wavs))
        varpt_w = np.zeros((wavs))
        stokesP_w[ok_w] = np.sqrt(stokes_Sw[1,ok_w]**2 + stokes_Sw[2,ok_w]**2)      # unnormalized linear polarization
        stokesT_w[ok_w] = (0.5*np.arctan2(stokes_Sw[2,ok_w],stokes_Sw[1,ok_w]))     # PA in radians
        stokesT_w[ok_w] = (stokesT_w[ok_w]-(tcenter+np.pi/2.)+np.pi) % np.pi + (tcenter-np.pi/2.)
                                                                                    # optimal PA folding                
     # variance matrix eigenvalues, ellipse orientation
        varpe_dw[:,ok_w] = 0.5*(err2_Sw[1,ok_w]+err2_Sw[2,ok_w]                          \
            + np.array([1,-1])[:,None]*np.sqrt((err2_Sw[1,ok_w]-err2_Sw[2,ok_w])**2 + 4*err2_Sw[-1,ok_w]**2))
        varpt_w[ok_w] = 0.5*np.arctan2(2.*err2_Sw[-1,ok_w],err2_Sw[1,ok_w]-err2_Sw[2,ok_w])
     # linear polarization variance along p, PA   
        varP_w[ok_w] = varpe_dw[0,ok_w]*(np.cos(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2   \
               + varpe_dw[1,ok_w]*(np.sin(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2
        varT_w[ok_w] = varpe_dw[0,ok_w]*(np.sin(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2   \
               + varpe_dw[1,ok_w]*(np.cos(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2

        stokes_vw[0,ok_w] = 100*stokesP_w[ok_w]/stokes_Sw[0,ok_w]                  # normalized % linear polarization
        err_vw[0,ok_w] =  100*np.sqrt(err2_Sw[1,ok_w])/stokes_Sw[0,ok_w]
        stokes_vw[1,ok_w] = np.degrees(stokesT_w[ok_w])                            # PA in degrees
        err_vw[1,ok_w] =  0.5*np.degrees(np.sqrt(err2_Sw[2,ok_w])/stokesP_w[ok_w])

    return stokes_vw,err_vw
 
#---------------------------------------------------------------------------------------------
def avstokes(stokes_Sw,var_Sw,covar_Sw,wav_w):
    """Computed average normalized stokes parameters
    Average is unnormalized stokes (intensity weighted), then normalized.

    Parameters
    ----------
    stokes_Sw: 2d float nparray(unnormalized stokes,wavelength bin)
       unnormalized stokes parameters vs wavelength

    var_Sw: 2d float nparray(unnormalized stokes,wavelength bin) 
       variance for stokes_Sw

    covar_Sw: 2d float nparray(unnormalized stokes,wavelength bin) 
       covariance for stokes_Sw

    wav_w: 1d float ndarray(wavelength bin)

    Output: avg normalized stokes, err, wavelength

    """
    stokess = stokes_Sw.shape[0]
    ok_w = (var_Sw != 0).all(axis=0)

    avstokes_S = stokes_Sw[:,ok_w].sum(axis=1)/ ok_w.sum()
    avvar_S = (var_Sw[:,ok_w] + 2.*covar_Sw[:,ok_w]).sum(axis=1)/ ok_w.sum()**2
    avwav = wav_w[ok_w].sum()/ok_w.sum()
    avstokes_s  = avstokes_S[1:]/avstokes_S[0]
    avvar_s  = avvar_S[1:]/avstokes_S[0]**2
                           
    return avstokes_s, avvar_s, avwav

#---------------------------------------------------------------------------------------------
def printstokes(stokes_Sw,var_Sw,wav_w,textFile=sys.stdout,tcenter=np.pi/2.,isfluxed=False):
    """Print intensity (if not=1) and normalized stokes parameters, plus (if stokes includes Q,U) P,T.

    Parameters
    ----------
    stokes_Sw: 2d float nparray(unnormalized stokes,wavelength bin)
       unnormalized stokes parameters vs wavelength

    var_Sw: 2d float nparray(stokes,wavelength bin) 
       variance for stokes_sw

    wav_w: 1d float ndarray(wavelength bin)

    textFile: optional file object for output, else stdout
    tcenter: optional float PA center (in radians) for linear stokes theta output

    Output: None

    """
    if stokes_Sw.ndim < 2:
        stokes_Sw = np.expand_dims(stokes_Sw,axis=1)
        var_Sw = np.expand_dims(var_Sw,axis=1)
        wav_w = np.expand_dims(wav_w,axis=0)
        
    stokess,wavs = stokes_Sw.shape
    stokesList = [[],['% S'],['% Q','% U'],['% Q','% U','% V']][stokess-1]

    ok_w = (stokes_Sw != 0).all(axis=0)
    stokes_sW = stokes_Sw[1:,ok_w]/stokes_Sw[0,ok_w]                            
    err_sW = np.sqrt(var_Sw[1:stokess,ok_w])/stokes_Sw[0,ok_w]
    wav_W = wav_w[ok_w]

    if (stokes_Sw[0][ok_w].mean()==1.):                    
        fmt = "   %8.2f "+(stokess-1)*(' %9.4f')+(stokess-1)*(' %8.4f')
        label = '\n   Wavelen     '+(7*" ").join(stokesList)+(6*" ")+" Err  ".join(stokesList)+' Err '
        output_vW = np.vstack((wav_W,100.*stokes_sW,100.*err_sW))
    else:
        fmt = "   %8.2f "+["%11.2f ","%11.3e "][isfluxed]+(stokess-1)*(' %9.4f')+(stokess-1)*(' %8.4f')
        label = '\n   Wavelen    '+["Intensity"," Flambda "][isfluxed]+'   '+(7*" ").\
            join(stokesList)+(6*" ")+" Err  ".join(stokesList)+' Err '
        output_vW = np.vstack((wav_W,stokes_Sw[0,ok_w],100.*stokes_sW,100.*err_sW))
       
    if stokess>2:                                   # Q,U, or Q,U,V - add P,T output                  
        stokes_vw, err_vw = viewstokes(stokes_Sw,var_Sw,ok_w,tcenter)
        output_vW = np.vstack((output_vW,stokes_vw[:,ok_w],err_vw[:,ok_w]))
        fmt += (' '+2*('%8.4f %8.3f'))
        ptstokesList = ['% P','PA ']
        label += ('   '+(6*" ").join(ptstokesList)+(4*" ")+" Err  ".join(ptstokesList)+' Err ')

    np.savetxt(textFile, output_vW.T, fmt=fmt, header=label, comments='')

    return
    
#---------------------------------------------------------------------------------------------
def printstokestw(stokes_Sm,var_Sm,target_m,tgtname_m,wav1_m,wav2_m,chi2_m=[],wavs_m=[],culls_m=[],  \
    logfile='sppolview.log',tcenter=np.pi/2.,isfluxed=False,with_stdout=True):
    """Print intensity (if not=1) and normalized stokes parameters, plus (if stokes includes Q,U) P,T.

    Parameters
    ----------
    stokes_Sm: 2d float nparray(unnormalized stokes,target/wavelength sample)
        unnormalized stokes parameters vs sample

    var_Sm: 2d float nparray
        variance for stokes_Sm

    target_m: 1d integer nparray (scalar ok for single sample)
        target number for sample m
        
    tgtname_m: 1d string nparray (scalar ok for single sample)
        target CATID for sample m        
        
    wav1_m, wav2_m: 1d float ndarrays (scalar ok for single wavelength)
        wavelength range for sample m

    chi2_m, wavs_m, culls_m: 1d nparrays. float, in, int
        optional information for mean value printouts

    textFile: optional file object for output, else stdout
    tcenter: optional float PA center (in radians) for linear stokes theta output

    Output: None

    """
    if (np.isscalar(target_m) & np.isscalar(wav1_m)): 
        target_m = np.array([target_m,])
        tgtname_m = np.array([tgtname_m,])        
        wav1_m = np.array([wav1_m,])
        wav2_m = np.array([wav2_m,])         
    elif np.isscalar(target_m):
        target_m = np.repeat(target_m,wav1_m.shape[0])
        tgtname_m = np.repeat(tgtname_m,wav1_m.shape[0])        
    elif np.isscalar(wav1_m):
        wav1_m = np.repeat(wav1_m,target_m.shape[0]) 
        wav2_m = np.repeat(wav2_m,target_m.shape[0])                       
    samples = target_m.shape[0]

    if stokes_Sm.ndim < 2:
        stokes_Sm = stokes_Sm.reshape((-1,samples))
        var_Sm = var_Sm.reshape((-1,samples))
        
    stokess = stokes_Sm.shape[0]
    stokeslist = [[],[' %S'],[' %Q',' %U'],[' %Q',' %U',' %V']][stokess-1]

    ok_m = (stokes_Sm != 0).all(axis=0)
    stokes_sM = stokes_Sm[1:,ok_m]/stokes_Sm[0,ok_m]                            
    err_sM = np.sqrt(var_Sm[1:stokess,ok_m])/stokes_Sm[0,ok_m]
    tgtname_M = tgtname_m[ok_m]    
    namelen = max(map(len,[x.strip() for x in tgtname_M]))

    if (stokes_Sm[0][ok_m].mean()==1.):                    
        fmt = '%'+str(namelen)+'s %4i %7.1f %7.1f '+(stokess-1)*(' %9.4f')+(stokess-1)*(' %7.4f')
        label = '\n CATID'+(namelen-4)*' '+' Tgt  Wave1   Wave2       '+ \
            (7*" ").join(stokeslist)+"  "+"_Err ".join(stokeslist)+'_Err '
        output_vM = np.vstack((target_m[ok_m],wav1_m[ok_m],wav2_m[ok_m],100.*stokes_sM,100.*err_sM))
    else:
        fmt = '%'+str(namelen)+'s %4i %7.1f %7.1f '+[" %11.2f ","%11.3e "][isfluxed]+(stokess-1)*(' %9.4f')+(stokess-1)*(' %7.4f')
        label = '\n CATID'+(namelen-4)*' '+' Tgt  Wave1   Wave2       '+[" Intensity "," Flambda "][isfluxed]+'   '+(7*" ").\
            join(stokeslist)+(4*" ")+"_Err  ".join(stokeslist)+'_Err '
        output_vM = np.vstack((target_m[ok_m],wav1_m[ok_m],wav2_m[ok_m],stokes_Sm[0,ok_m],100.*stokes_sM,100.*err_sM))
       
    if stokess>2:                                   # Q,U, or Q,U,V - add P,T output                  
        stokes_vm, err_vm = viewstokes(stokes_Sm,var_Sm,ok_m,tcenter)
        output_vM = np.vstack((output_vM,stokes_vm[:,ok_m],err_vm[:,ok_m]))
        fmt += (' '+2*('%8.4f %8.3f'))
        ptstokeslist = [' %P',' PA']
        label += ('   '+(6*" ").join(ptstokeslist)+(3*" ")+"_Err  ".join(ptstokeslist)+'_Err ')

    if len(chi2_m):
        output_vM = np.vstack((output_vM,chi2_m[ok_m]))
        fmt += (' %8.2f')
        label += ("  Chisq")
    if len(wavs_m):
        output_vM = np.vstack((output_vM,wavs_m[ok_m]))
        fmt += (' %5i')
        label += (" Wavs")
    if len(culls_m):
        output_vM = np.vstack((output_vM,culls_m[ok_m]))
        fmt += (' %4i')
        label += (" Culls")               

    rsslog.message(label,logfile, with_stdout=with_stdout) 
    for M in range(len(tgtname_M)):
        rsslog.message((fmt % ((tgtname_M[M],)+tuple(output_vM[:,M]))),logfile, with_stdout=with_stdout) 

    return     
          
#---------------------------------------------------------------------------------------------
 
if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    viewtargetList = [int(x) for x in sys.argv[2:] if x.isdigit()]    
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if (x.count('.fits')==0) & (not x.isdigit()))  
    sppolview(infileList, *viewtargetList, **kwargs)

    


