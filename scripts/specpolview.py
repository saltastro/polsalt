#! /usr/bin/env python

"""
specpolview

Plot and text output of stokes data, optionally binned

"""

import os, sys, glob, shutil, inspect

import numpy as np
import pyfits

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

import specpolfinalstokes as spf

import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter  
plt.ioff()
np.set_printoptions(threshold=np.nan)

#---------------------------------------------------------------------------------------------
def specpolview(infile_list, **kwargs):
    """View Stokes output results

    Parameters
    ----------
    infile_list: list
       one or more _stokes.fits files

    bin=    unbin (default)
            nnA (nn Angstroms)
            nn% (binned to %)
    errors= False(default) 
            True (plot errorbars)
    connect= '' (default) (if errors False, connect points)
            hist (if binned, use histogram connection)
    type=   Ipt (default) (unbinned intensity/flux, binned % pol, deg PA)
            Iqu (unbinned intensity/flux, binned %q, %u, with optional rotate)
    save=   '' (default) (text to terminal)
            text (text to file)
            plot (terminal plot and pdf file)
            textplot (both)
    debug=: False (default)
            True (debug output)
    """
    obss = len(infile_list)
    bin = kwargs.pop('bin','unbin')
    errorbars = (kwargs.pop('errors','False') == 'True')
    connect = kwargs.pop('connect','')
    plottype = kwargs.pop('type','Ipt')
    save = kwargs.pop('save','')
    debug = (kwargs.pop('debug','False') == 'True')

    bintype = bin
    if bin.count('%'): 
        bintype = 'percent'
        errbin = float(bin[ :bin.index('%')])
    elif bin.count('A'): 
        bintype = 'wavl'
        blk = int(bin[ :bin.index('A')])
    elif (bin != 'unbin'): 
        print "unrecognized binning option, set to unbinned"
        bintype = 'unbin'
    if (len(bin)>6):              # for backwards compatibility
        errorbars = (bin[-6:]=="errors")
    if (connect not in ('','hist')): 
        print "unrecognized connect option"
        exit()
    if (plottype not in ('Ipt','Iqu')): 
        print "unrecognized type option"
        exit()
    savetext = (save.count('text')>0)
    saveplot = (save.count('plot')>0)

    plotcolor_o = ['b','g','r','c','m','y','k']
    askpltlim = True
    tcenter = 0.
    trotate = 0.
    cunitfluxed = 'erg/s/cm^2/Ang'          # header keyword CUNIT3 if data is already fluxed 
 
    for obs in range(obss):
        hdul = pyfits.open(infile_list[obs])
        name = os.path.basename(infile_list[obs]).split('.')[0]
        obsdate = hdul[0].header['DATE-OBS']
        stokes_Sw = hdul['SCI'].data[:,0,:]
        var_Sw = hdul['VAR'].data[:,0,:]
        covar_Sw = hdul['COV'].data[:,0,:]
        bpm_Sw = hdul['BPM'].data[:,0,:]
        isfluxed=False
        if 'CUNIT3' in hdul['SCI'].header:
            isfluxed=(hdul['SCI'].header['CUNIT3'].replace(' ','') ==cunitfluxed)
        stokess,wavs = stokes_Sw.shape
        wav0 = hdul['SCI'].header['CRVAL1']
        dwav = hdul['SCI'].header['CDELT1']
        wav_w = wav0 + dwav*np.arange(wavs)
        ok_Sw = (bpm_Sw==0)
        ok_w = ok_Sw.all(axis=0)

    # set up multiplot
        if obs==0:
            stokeslist = hdul['SCI'].header['CTYPE3'].split(',')
            fig,plot_S = plt.subplots(stokess,1,sharex=True)
            plt.xlabel('Wavelength (Ang)')
            plot_S[0].set_ylabel(['Intensity','Flambda ('+cunitfluxed+')'][isfluxed])
            for S in range(1,stokess): plot_S[S].set_ylabel(stokeslist[S]+' Polarization (%)')
            if stokeslist[1]=="S": plotname = name.split("_")[-2]
            else: plotname = 'stokes'
            stokeslist[1:] = ('  % '+stokeslist[s] for s in range(1,stokess))
            if ((plottype == 'Ipt') | (plottype == 'IPt')):
                if 'PATYPE' in hdul[0].header:
                    pa_type = hdul[0].header['PATYPE']
                else:
                    pa_type = hdul[0].header['POLCAL'].split(" ")[0]    # old style
                plot_S[2].set_ylabel(pa_type+' PA (deg)')
            if (plottype == 'Ipt'):
                stokeslist[1:3] = '  % P', (pa_type[:3]+' T')   
                plot_S[1].set_ylabel('Linear Polarization (%)')
            if (plottype == 'Iqu'):
                stokeslist[1:3] = ('  % Q', ' % U')   
                plot_S[1].set_ylabel('Stokes Q (%)')
                plot_S[2].set_ylabel('Stokes U (%)')         
            fig.set_size_inches((8.5,11))
            fig.subplots_adjust(left=0.175)
            namelist=[]

    # calculate, print means (stokes average in unnorm space)
        hassyserr = hdul[0].header.has_key('SYSERR')

        avstokes_s, avvar_s, avwav = avstokes(stokes_Sw[:,ok_w],var_Sw[:-1][:,ok_w],covar_Sw[:,ok_w],wav_w[ok_w]) 
        avstokes_S = np.insert(avstokes_s,0,1.)
        avvar_S = np.insert(avvar_s,0,1.)

        print ("\n%16s %16s  Wtd mean   " % (name,obsdate)),
        if hassyserr: print ('Syserr: %8.3f' % hdul[0].header['SYSERR']),
        print           
        printstokes(avstokes_S,avvar_S,avwav)
 
        plotcolor = plotcolor_o[obs % len(plotcolor_o)]
    # plot intensity
        label = name
        if name.count("_") ==0:             # diffsum multiplots
            label = name[-4:] 
        namelist.append(name)
        ww = -1; 
        while (bpm_Sw[0,ww+1:]==0).sum() > 0:
            w = ww+1+np.where(bpm_Sw[0,ww+1:]==0)[0][0]
            ww = wavs
            dw = np.where(bpm_Sw[0,w:]>0)[0]  
            if dw.size: ww = w + dw[0] - 1 
            plot_S[0].plot(wav_w[w:ww],stokes_Sw[0,w:ww],color=plotcolor,label=label)
            label = '_'+name    

      # get manual plot limits, resetting tcenter (PA wrap center) if necessary
        if saveplot:
            while (askpltlim):
                ylimlisti = (raw_input('\nOptional scale (bottom-top, comma sep): ')).split(',')
                if len(''.join(ylimlisti))==0: ylimlisti = []
                ismanlim_i = np.array([len(ys)>0 for ys in ylimlisti])
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
                            tcenter = np.radians((float(ylimlisti[itbottom]) + float(ylimlisti[ittop]))/2.)
                    if ((plottype == 'Iqu') | (plottype == 'IQU')):
                        trotate = float(raw_input('\nOptional PA zeropoint (default 0): ') or '0')
                        if trotate:
                            plot_S[1].set_ylabel('Stokes Q (%%)  PA0= %7.1f deg' % trotate)
                            plot_S[2].set_ylabel('Stokes U (%%)  PA0= %7.1f deg' % trotate)  
                        if (len(ismanlim_i) == 2):
                            ismanlim_i = np.tile(ismanlim_i,2)
                            ylimlisti += ylimlisti
                                                       
                askpltlim = False
                     
      # assemble data
        if trotate:
            stokes_Sw,var_Sw,covar_Sw = spf.specpolrotate(stokes_Sw,var_Sw,covar_Sw,-trotate)

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
                stokes_sw, err_sw = viewstokes(stokes_Sw,var_Sw,ok_w,tcenter)
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
        textfile = sys.stdout
        if savetext: 
            textfile = open(name+'_'+bin+'.txt','ab')
            textfile.truncate(0)
        else: print >>textfile

        if stokess > 2:
            if tcenter == 0:                 # tcenter has not been set by manual plot limits
                tcenter = ((0.5*np.arctan2(avstokes_s[1],avstokes_s[0])) + np.pi) % np.pi
        else:
            tcenter = np.pi/2.

        print >>textfile, ("\n%16s %16s    " % (name,obsdate)),
        if hassyserr: print >>textfile, ('Syserr: %8.3f' % hdul[0].header['SYSERR']),
        print >>textfile 

        if bintype == 'unbin':
            printstokes(stokes_Sw[:,ok_w],var_Sw[:stokess,ok_w],wav_w[ok_w],   \
                tcenter=tcenter,textfile=textfile,isfluxed=isfluxed)
        else:         
            printstokes(stokes_SV[:,ok_V],var_SV[:stokess,ok_V],wav_v,         \
                tcenter=tcenter,textfile=textfile,isfluxed=isfluxed)

  # Plotting of stacked observations
    if saveplot:
        plot_S[0].set_ylim(bottom=0)                # intensity plot default baseline 0
        if stokess >2:
            if plottype == 'Ipt': 
                plot_S[1].set_ylim(bottom=0)            # linear polarization % plot default baseline 0
                ymin,ymax = plot_S[2].set_ylim()        # linear polarization PA plot default 5 degree pad
                plot_S[2].set_ylim(bottom=min(ymin,(ymin+ymax)/2.-5.),top=max(ymax,(ymin+ymax)/2.+5.))
        if len(ylimlisti)>0:
            for (i,ys) in enumerate(ylimlisti):
                S = stokess-i/2-1
                if (ismanlim_i[i] & ((i % 2)==0)): plot_S[S].set_ylim(bottom=float(ys))
                if (ismanlim_i[i] & ((i % 2)==1)): plot_S[S].set_ylim(top=float(ys))

        if obss>1: 
            plot_S[0].legend(fontsize='x-small',loc='upper left')
        else: 
            plot_S[0].set_title(name+"   "+obsdate) 
        tags = name.count("_")
        cyclelist = []
        if tags:                 # raw and final stokes files
            objlist = sorted(list(set(namelist[b].split("_")[0] for b in range(obss))))
            confcyclelist = sorted(list(set(namelist[b].replace("_stokes","").split("_",1)[-1] for b in range(obss))))
            plotfile = '_'.join(objlist+confcyclelist+list([plotname,bin,plottype]))+'.pdf'
        else:                               # diffsum files from diffsum
            plotfile = namelist[0]+'-'+namelist[-1][-4:]+'.pdf'
        plt.savefig(plotfile,orientation='portrait')
        if os.name=='posix':
            if os.popen('ps -C evince -f').read().count(plotfile)==0: os.system('evince '+plotfile+' &')
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
def printstokes(stokes_Sw,var_Sw,wav_w,textfile=sys.stdout,tcenter=np.pi/2.,isfluxed=False):
    """Print intensity (if not=1) and normalized stokes parameters, plus (if stokes includes Q,U) P,T.

    Parameters
    ----------
    stokes_Sw: 2d float nparray(unnormalized stokes,wavelength bin)
       unnormalized stokes parameters vs wavelength

    var_Sw: 2d float nparray(stokes,wavelength bin) 
       variance for stokes_sw

    wav_w: 1d float ndarray(wavelength bin)

    textfile: optional file object for output, else stdout
    tcenter: optional float PA center (in radians) for linear stokes theta output

    Output: None

    """
    if stokes_Sw.ndim < 2:
        stokes_Sw = np.expand_dims(stokes_Sw,axis=1)
        var_Sw = np.expand_dims(var_Sw,axis=1)
        wav_w = np.expand_dims(wav_w,axis=0)
        
    stokess,wavs = stokes_Sw.shape
    stokeslist = [[],['% S'],['% Q','% U'],['% Q','% U','% V']][stokess-1]

    ok_w = (stokes_Sw != 0).all(axis=0)
    stokes_sW = stokes_Sw[1:,ok_w]/stokes_Sw[0,ok_w]                            
    err_sW = np.sqrt(var_Sw[1:stokess,ok_w])/stokes_Sw[0,ok_w]
    wav_W = wav_w[ok_w]

    if (stokes_Sw[0][ok_w].mean()==1.):                    
        fmt = "   %8.2f "+2*(stokess-1)*(' %8.4f')
        label = '\n   Wavelen     '+(6*" ").join(stokeslist)+(5*" ")+" Err  ".join(stokeslist)+' Err '
        output_vW = np.vstack((wav_W,100.*stokes_sW,100.*err_sW))
    else:
        fmt = "   %8.2f "+["%11.2f ","%11.3e "][isfluxed]+2*(stokess-1)*(' %8.4f')
        label = '\n   Wavelen    '+["Intensity"," Flambda "][isfluxed]+'   '+(6*" ").\
            join(stokeslist)+(5*" ")+" Err  ".join(stokeslist)+' Err '
        output_vW = np.vstack((wav_W,stokes_Sw[0,ok_w],100.*stokes_sW,100.*err_sW))
       
    if stokess>2:                                   # Q,U, or Q,U,V - add P,T output                  
        stokes_vw, err_vw = viewstokes(stokes_Sw,var_Sw,ok_w,tcenter)
        output_vW = np.vstack((output_vW,stokes_vw[:,ok_w],err_vw[:,ok_w]))
        fmt += (' '+2*('%8.4f %8.3f'))
        ptstokeslist = ['% P','PA ']
        label += ('   '+(6*" ").join(ptstokeslist)+(4*" ")+" Err  ".join(ptstokeslist)+' Err ')

    np.savetxt(textfile, output_vW.T, fmt=fmt, header=label, comments='')

    return     
#---------------------------------------------------------------------------------------------
 
if __name__=='__main__':
    infilelist=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)  
    specpolview(infilelist, **kwargs)

    


