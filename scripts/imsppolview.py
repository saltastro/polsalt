#! /usr/bin/env python

"""
imsppolview

Plot and text output of impsppol stokes data (unfiltered), optionally binned

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
from sppolview import avstokes, viewstokes, printstokestw

import matplotlib
matplotlib.use('PDF',warn=False)
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter  
plt.ioff()
np.set_printoptions(threshold=np.nan)
import warnings
#warnings.simplefilter("error")

#---------------------------------------------------------------------------------------------
def imsppolview(infile, *targetList, **kwargs):
    """View Stokes output results

    Parameters
    ----------
    infile: text
       one imsppol _stokes.fits file

    targets= [] (all targets default)
            list of targets (integer ids in TGT table)
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
        blk = float(bin[ :bin.index('A')])
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
 
    hdul = pyfits.open(infile)
    name = os.path.basename(infile).split('.')[0]
    obsdate = hdul[0].header['DATE-OBS']
    stokes_Stw = hdul['SCI'].data
    var_Stw = hdul['VAR'].data
    covar_Stw = hdul['COV'].data
    bpm_Stw = hdul['BPM'].data
    tgtTab = Table.read(hdul['TGT'])
    entries = len(tgtTab['CATID'])
    oktgt_i = (tgtTab['CULL'] == '')
    i_t = np.where(oktgt_i)[0]   
    targets = oktgt_i.sum()        
    tgtname_t = np.array(tgtTab['CATID'])[oktgt_i]         
    stokess,targets,wavs = stokes_Stw.shape
    tdigits = len(str(targets))
    if len(targetList) == 0:
        targetList = range(targets)
        targetstr = 'all'
    else:
        targetstr = '_'.join(map(str,targetList))       # need to use tdigits, sorted
        targets = len(targetList)        
    if 'WCSDVARR' not in [hdul[x].name for x in range(len(hdul))]:
        print "Not an Imsppol file"
        exit()
        
    wedge_W = hdul['WCSDVARR'].data
    dwav = hdul['SCI'].header['CDELT1']     # number of unbinned wavelength bins in bin
    wedge_w = wedge_W[:(wavs+1)*dwav:dwav]  # shape wavs+1     
    wav_w = (wedge_w[:-1] + wedge_w[1:])/2.
    wavwidth_w =  wedge_w[1:] - wedge_w[:-1]  
    ok_Stw = (bpm_Stw==0)
 
    # set up multiplot
    stokeslist = hdul['SCI'].header['CTYPE3'].split(',')
    fig,plot_S = plt.subplots(stokess,1,sharex=True)
    plt.xlabel('Wavelength (Ang)')
    plot_S[0].set_ylabel('Intensity')
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

  # calculate, print means (stokes average in unnorm space)
    hassyserr = ('SYSERR' in hdul[0].header)
    print ("\n%16s %16s  Wtd mean   " % (name,obsdate)),
    if hassyserr: print ('Syserr: %8.3f' % hdul[0].header['SYSERR']),
    print 
    avstokes_Sm = np.empty((stokess,0))    
    avvar_Sm = np.empty((stokess,0))
    target_m = np.array([],dtype=int)
    wav_m = np.array([])
        
    for T,t in enumerate(targetList):
        ok_Sw = ok_Stw[:,t] 
        ok_w = ok_Sw[0]
        if ok_w.sum()==0: continue      
        avstokes_s, avvar_s, avwav =    \
            avstokes(stokes_Stw[:,t,ok_w],var_Stw[:-1][:,t,ok_w],covar_Stw[:,t,ok_w],wav_w[ok_w]) 
        avstokes_S = np.insert(avstokes_s,0,1.)
        avvar_S = np.insert(avvar_s,0,1.)
        avstokes_Sm = np.append(avstokes_Sm,avstokes_S[:,None],axis=1) 
        avvar_Sm = np.append(avvar_Sm,avvar_S[:,None],axis=1) 
        target_m = np.append(target_m,t)
        wav_m = np.append(wav_m,avwav)                          

      # plot intensity (equal wavelength bins)
        plotcolor = plotcolor_o[T % len(plotcolor_o)]
        label = str(t) 
        ww = -1; 
        while (ok_w[ww+1:].sum() > 0):                # leaving gaps where appropriate
            w = ww+1+np.where(ok_w[ww+1:])[0][0]
            ww = wavs
            dw = np.where(~ok_w[w:]>0)[0]  
            if dw.size: ww = w + dw[0] - 1 
            plot_S[0].plot(wav_w[w:ww],(stokes_Stw[0,t]/wavwidth_w)[w:ww],color=plotcolor,label=label)
            label = '_'+name    

    tgtname_m = tgtname_t[target_m]
    namelen = max(map(len,[x.strip() for x in tgtname_m]))        
    printstokestw(avstokes_Sm,avvar_Sm,target_m,tgtname_m,wav_m)

  # get manual plot limits, resetting tcenter (PA wrap center) if necessary
    if saveplot:
        while (askpltlim):
            yxlimlisti = (raw_input('\nOptional scale (bottom-top, comma sep): ')).split(',')
            if len(''.join(yxlimlisti))==0: yxlimlisti = []
            ismanlim_i = np.array([len(ys)>0 for ys in yxlimlisti])
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
                        tcenter = np.radians((float(yxlimlisti[itbottom]) + float(yxlimlisti[ittop]))/2.)
                if ((plottype == 'Iqu') | (plottype == 'IQU')):
                    trotate = float(raw_input('\nOptional PA zeropoint (default 0): ') or '0')
                    if trotate:
                        plot_S[1].set_ylabel('Stokes Q (%%)  PA0= %7.1f deg' % trotate)
                        plot_S[2].set_ylabel('Stokes U (%%)  PA0= %7.1f deg' % trotate)  
                    if (len(ismanlim_i) == 2):
                        ismanlim_i = np.tile(ismanlim_i,2)
                        yxlimlisti += yxlimlisti
                                                       
            askpltlim = False
                     
  # assemble data
    if (savetext & (len(targetList)>1)):
        zipname = name+'_'+bin+'_'+targetstr+'.zip'
        zl = zipfile.ZipFile(zipname,mode='w')          # truncate any previous file by this name
        zl = zipfile.ZipFile(zipname,mode='a')        
    for T,t in enumerate(targetList):
        stokes_Sw = stokes_Stw[:,t]
        var_Sw = var_Stw[:,t]
        covar_Sw = covar_Stw[:,t]
        bpm_Sw = bpm_Stw[:,t]    
        ok_Sw = ok_Stw[:,t] 
        ok_w = ok_Sw[0]
        plotcolor = plotcolor_o[T % len(plotcolor_o)]                        
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
            wav_v = wav_w[ok_w]
            stokes_sv = stokes_sw[:,ok_w]
            err_sv = err_sw[:,ok_w]            
            dwavleft_v = np.diff(wedge_w)[ok_w]/2.
            dwavright_v = dwavleft_v           

        else:
        # Set up bins, blocked, or binned to error based on stokes 1 or on linear stokes p
            if bintype == 'wavl':
                bin_w = (wav_w / blk -0.5).astype(int) - int((wav_w / blk -0.5).min())
                Bins = bin_w.max()
                bin_w[~ok_Sw[1]] = -1
            else:
                allowedgap = 5
                wgap0_g = np.where(ok_Sw[0,:-1] & (~ok_Sw[0,1:]))[0] + 1
                wgap1_g = np.where((~ok_Sw[wgap0_g[0]:-1]) & ok_w[wgap0_g[0]+1:])[0] \
                    +  wgap0_g[0] + 1
                wgap0_g = wgap0_g[0:wgap1_g.shape[0]]
                isbad_g = ((wgap1_g - wgap0_g) > allowedgap)
                stokes_sw, err_sw = viewstokes(stokes_Sw,var_Sw,ok_w=ok_w,tcenter=tcenter)
                binvar_w = err_sw[0]**2
                bincovar_w = np.zeros_like(binvar_w)
                bincovar_w[ok_w] = binvar_w[ok_w]*covar_Sw[1,ok_w]/var_Sw[1,ok_w]
                ww = -1; b = 0;  bin_w = -1*np.ones((wavs))
                while (ok_w[ww+1:].sum() > 0):
                    w = ww+1+np.where(ok_w[ww+1:])[0][0]
                    cumsvar_W = np.cumsum((binvar_w[w:]+2.*bincovar_w[w:])*ok_w[w:])    \
                                /np.cumsum(ok_w[w:])**2
                    err_W = np.sqrt(cumsvar_W)
                    ww = wavs                                       # stopping point override: end
                    nextbadgap = np.where(isbad_g & (wgap0_g > w))[0]
                    if nextbadgap.size: ww = wgap0_g[nextbadgap[0]] - 1   # stopping point override: before bad gap
                    dw = np.where(err_W[:ww-w] < errbin)[0]
                    if dw.size: ww = w + dw[0]                      # err goal is reached first
                    bin_w[w:ww+1] = b
                    b += 1
                bin_w[~ok_w] = -1
                Bins  = b

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
                np.vstack((wav_v,stokes_sv[S-1],err_sv[S-1],   \
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
            textfilename = name+'_'+(("%"+str(tdigits)+"s") % t)+'_'+bin+'.txt' 
            textfile = open(textfilename,'ab')
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
        tgtname = (('%'+str(namelen)+'s') % tgtname_t[t] )  # ensure all target tables are lined up

        if bintype == 'unbin':
            printstokestw(stokes_Sw[:,ok_w],var_Sw[:stokess,ok_w],t,tgtname,wav_w[ok_w],   \
                tcenter=tcenter,textfile=textfile)
        else:
            printstokestw(stokes_SV[:,ok_V],var_SV[:stokess,ok_V],t,tgtname,wav_v,         \
                tcenter=tcenter,textfile=textfile)
        if (savetext & (len(targetList) > 1)):
            textfile.close()
            zl.write(textfilename)
            os.remove(textfilename)
                
    if (savetext & (len(targetList) > 1)):
        zl.close() 

  # Plotting of stacked observations
    if saveplot:
        plot_S[0].set_ylim(bottom=0)                # intensity plot default baseline 0
        if stokess >2:
            if plottype == 'Ipt': 
                plot_S[1].set_ylim(bottom=0)            # linear polarization % plot default baseline 0
                ymin,ymax = plot_S[2].set_ylim()        # linear polarization PA plot default 5 degree pad
                plot_S[2].set_ylim(bottom=min(ymin,(ymin+ymax)/2.-5.),top=max(ymax,(ymin+ymax)/2.+5.))
        if len(yxlimlisti)>0:
            for (i,ys) in enumerate(yxlimlisti):
                if (i>3): continue
                S = stokess-i/2-1
                if (ismanlim_i[i] & ((i % 2)==0)): plot_S[S].set_ylim(bottom=float(ys))
                if (ismanlim_i[i] & ((i % 2)==1)): plot_S[S].set_ylim(top=float(ys))
        if len(yxlimlisti)>4:
            if (ismanlim_i[4]): plot_S[S].set_xlim(left=float(yxlimlisti[4]))            
            if (ismanlim_i[5]): plot_S[S].set_xlim(right=float(yxlimlisti[5]))    

        if ((targets>1)&(targets<10)): 
            plot_S[0].legend(fontsize='x-small',loc='upper left')

        plot_S[0].set_title(name+"   "+obsdate)
        if (plottype !='Ipt'): name = name+'_'+plottype
        plotfile = name+'_'+targetstr+'.pdf'
        plt.savefig(plotfile,orientation='portrait')
        if os.name=='posix':
            if os.popen('ps -C evince -f').read().count(plotfile)==0: os.system('evince '+plotfile+' &')
    else: 
        plt.show(block=True)
    return

#---------------------------------------------------------------------------------------------
 
if __name__=='__main__':
    infile=sys.argv[1]
    targets = [int(x) for x in sys.argv[2:] if x.isdigit()]
    kwargs = dict(x.split('=', 1) for x in sys.argv[2+len(targets):]) 
    imsppolview(infile, *targets, **kwargs)

# debug:
# M30
# cd /d/pfis/khn/20161023/sci
# python script.py imsppolview.py M30_c0_1_stokes.fits 1 2 6 7 save=textplot debug=True




