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
def specpolview(infile_list, bin='unbin', save = '', debug=False):
    """View output results

    Parameters
    ----------
    infile_list: list
       one or more _stokes.fits files

    bin  
       unbin, nnA (nn Angstroms), nn% (binned to %)

   save  
       '' (text to terminal), text (text to file), plot (terminal plot and pdf file), textplot (both)

    """
    obss = len(infile_list)
    bintype = 'unbin'
    errbars = False
    if len(bin):
        if bin.count('%'): 
            bintype = 'percent'
            errbin = float(bin[ :bin.index('%')])
        elif bin.count('A'): 
            bintype = 'wavl'
            blk = int(bin[ :bin.index('A')])
        elif bin != 'unbin': 
            print "unrecognized binning option, set to unbinned"
            bintype = 'unbin'
    if len(bin)>6:
        errbars = (bin[-6:]=="errors")

    savetext = (save.count('text')>0)
    saveplot = (save.count('plot')>0)
    plotcolor_o = ['b','g','r','c','m','y','k']
    askpltlim = True
    tcenter = 0.
    cunitfluxed = 'erg/s/cm^2/Ang'          # header keyword CUNIT3 if data is already fluxed 
 
    for obs in range(obss):
        hdul = pyfits.open(infile_list[obs])
        name = os.path.basename(infile_list[obs]).split('.')[0]
        obsdate = hdul[0].header['DATE-OBS']
        stokes_Sw = hdul['SCI'].data[:,0,:]
        var_Sw = hdul['VAR'].data[:,0,:]
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
            if stokess > 2:
                if 'PATYPE' in hdul[0].header:
                    pa_type = hdul[0].header['PATYPE']
                else:
                    pa_type = hdul[0].header['POLCAL'].split(" ")[0]    # old style
                stokeslist[1:3] = '  % P', (pa_type[:3]+' T')   
                plot_S[1].set_ylabel('Linear Polarization (%)')
                plot_S[2].set_ylabel(pa_type+' PA (deg)')         
            fig.set_size_inches((8.5,11))
            fig.subplots_adjust(left=0.175)
            namelist=[]

    # calculate, print means (stokes wtd in norm space by 1/mean var)
        hassyserr = hdul[0].header.has_key('SYSERR')

        wtavstokes_s, wtavvar_s, wtavwav = wtavstokes(stokes_Sw[:,ok_w],var_Sw[:,ok_w],wav_w[ok_w]) 
        wtavstokes_S = np.insert(wtavstokes_s,0,1.)
        wtavvar_S = np.insert(wtavvar_s,0,1.)

        print ("\n%16s %16s  Wtd mean   " % (name,obsdate)),
        if hassyserr: print ('Syserr: %8.3f' % hdul[0].header['SYSERR']),
        print           
        printstokes(wtavstokes_S,wtavvar_S,wtavwav)
 
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
                        print "set PA limits for either both or neither top and bottom"
                        continue
                    if ismanlim_i[itbottom]:
                        tcenter = np.radians((float(ylimlisti[itbottom]) + float(ylimlisti[ittop]))/2.)
                askpltlim = False
                     
      # assemble data
        if bintype == 'unbin':
            stokes_sw, err_sw = viewstokes(stokes_Sw,var_Sw,ok_w,tcenter)

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
                binvar_w = err_sw[1]**2
                ww = -1; b = 0;  bin_w = -1*np.ones((wavs))
                while (bpm_Sw[0,ww+1:]==0).sum() > 0:
                    w = ww+1+np.where(bpm_Sw[0,ww+1:]==0)[0][0]
                    cumsvar_w = np.cumsum(binvar_w[w:]*(bpm_Sw[0,w:]==0))    \
                                /np.cumsum((bpm_Sw[0,w:]==0))**2
                    err_w = np.sqrt(cumsvar_w)
                    if debug: np.savetxt("err_"+str(w)+".txt",err_w,fmt="%10.3e")
                    ww = wavs                                       # stopping point override: end
                    nextbadgap = np.where(isbad_g & (wgap0_g > w))[0]
                    if nextbadgap.size: ww = wgap0_g[nextbadgap[0]] - 1   # stopping point override: before bad gap
                    dw = np.where(err_w[:ww-w] < errbin)[0]
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
            var_SV = (var_Sw[:,None,:]*bin_Vw).sum(axis=2) 
            bpm_SV = ((bpm_Sw[:,None,:]*bin_Vw).sum(axis=2)==bin_Vw.sum(axis=1)).astype(int)
            ok_SV = (bpm_SV == 0)
            ok_V = ok_SV.all(axis=0)
            bin_vw = bin_Vw[ok_V]
            wav_v = (wav_w[None,:]*bin_vw).sum(axis=1)/bin_vw.sum(axis=1)
            dwavleft_v = wav_v - wav_w[(np.argmax((wav_w[None,:]*bin_vw)>0,axis=1))] + dwav/2.
            dwavright_v = wav_w[wavs-1-(np.argmax((wav_w[None,::-1]*bin_vw[:,::-1])>0,axis=1))] - wav_v - dwav/2.
            stokes_sV, err_sV = viewstokes(stokes_SV,var_SV,ok_V,tcenter)          
            stokes_sv, err_sv = stokes_sV[:,ok_V], err_sV[:,ok_V]
            for S in range(1,stokess):
                if debug: np.savetxt('errbar_'+str(S)+'.txt', \
                    np.vstack((wav_v,stokes_SV[S-1],var_SV[S-1],stokes_sv[S-1],err_sv[S-1],dwavleft_v,dwavright_v)).T,  \
                    fmt = "%10.4f")
                if errbars:
                    plot_S[S].errorbar(wav_v,stokes_sv[S-1],color=plotcolor,fmt='.',    \
                        yerr=err_sv[S-1],xerr=(dwavleft_v,dwavright_v),capsize=0)
                else:
                    plot_S[S].plot(wav_v,stokes_sv[S-1],color=plotcolor,label=label)

      # Printing for observation
        textfile = sys.stdout
        if savetext: 
            textfile = open(name+'_'+bin+'.txt','ab')
            textfile.truncate(0)
        else: print >>textfile

        if stokess > 2:
            if tcenter == 0:                 # tcenter has not been set by manual plot limits
                tcenter = ((0.5*np.arctan2(wtavstokes_s[1],wtavstokes_s[0])) + np.pi) % np.pi
        else:
            tcenter = np.pi()/2.

        print >>textfile, ("\n%16s %16s    " % (name,obsdate)),
        if hassyserr: print >>textfile, ('Syserr: %8.3f' % hdul[0].header['SYSERR']),
        print >>textfile 

        if bintype == 'unbin':
            printstokes(stokes_Sw[:,ok_w],var_Sw[:,ok_w],wav_w[ok_w],   \
                tcenter=tcenter,textfile=textfile,isfluxed=isfluxed)
        else:         
            printstokes(stokes_SV[:,ok_V],var_SV[:,ok_V],wav_v,         \
                tcenter=tcenter,textfile=textfile,isfluxed=isfluxed)

  # Plotting of stacked observations
    if saveplot:
        plot_S[0].set_ylim(bottom=0)                # intensity plot default baseline 0
        if stokess >2: 
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
            plotfile = '_'.join(objlist+confcyclelist+list([plotname,bin]))+'.pdf'
        else:                               # diffsum files from diffsum
            plotfile = namelist[0]+'-'+namelist[-1][-4:]+'.pdf'
        plt.savefig(plotfile,orientation='portrait')
        if os.name=='posix':
            if os.popen('ps -C evince -f').read().count(plotfile)==0: os.system('evince '+plotfile+' &')
    else: 
        plt.show(block=True)
    return

#---------------------------------------------------------------------------------------------
def viewstokes(stokes_Sw,var_Sw,ok_w,tcenter=0.):
    """Compute normalized stokes parameters, converts Q-U to P-T, for viewing

    Parameters
    ----------
    stokes_Sw: 2d float nparray(stokes,wavelength bin)
       unnormalized stokes parameters vs wavelength

    var_Sw: 2d float nparray(stokes,wavelength bin) 
       variance for stokes_sw

    ok_w: 1d boolean nparray(stokes,wavelength bin) 
       marking good stokes values

    Output: normalized stokes parameters and errors, linear stokes converted to pol %, PA

    """

    stokess,wavs = stokes_Sw.shape
    stokes_vw = np.zeros((stokess-1,wavs))
    err_vw = np.zeros((stokess-1,wavs))

    stokes_vw[:,ok_w] = 100.*stokes_Sw[1:,ok_w]/stokes_Sw[0,ok_w]                            # in percent
    err_vw[:,ok_w] = 100.*np.sqrt(var_Sw[1:stokess,ok_w])/stokes_Sw[0,ok_w]

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
        varpe_dw[:,ok_w] = 0.5*(var_Sw[1,ok_w]+var_Sw[2,ok_w]                          \
            + np.array([1,-1])[:,None]*np.sqrt((var_Sw[1,ok_w]-var_Sw[2,ok_w])**2 + 4*var_Sw[-1,ok_w]**2))
        varpt_w[ok_w] = 0.5*np.arctan2(2.*var_Sw[-1,ok_w],var_Sw[1,ok_w]-var_Sw[2,ok_w])
     # linear polarization variance along p, PA   
        varP_w[ok_w] = varpe_dw[0,ok_w]*(np.cos(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2   \
               + varpe_dw[1,ok_w]*(np.sin(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2
        varT_w[ok_w] = varpe_dw[0,ok_w]*(np.sin(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2   \
               + varpe_dw[1,ok_w]*(np.cos(2.*stokesT_w[ok_w]-varpt_w[ok_w]))**2

        stokes_vw[0,ok_w] = 100*stokesP_w[ok_w]/stokes_Sw[0,ok_w]                  # normalized % linear polarization
        err_vw[0,ok_w] =  100*np.sqrt(var_Sw[1,ok_w])/stokes_Sw[0,ok_w]
        stokes_vw[1,ok_w] = np.degrees(stokesT_w[ok_w])                            # PA in degrees
        err_vw[1,ok_w] =  0.5*np.degrees(np.sqrt(var_Sw[2,ok_w])/stokesP_w[ok_w])

    return stokes_vw,err_vw
 
#---------------------------------------------------------------------------------------------
def wtavstokes(stokes_Sw,var_Sw,wav_w):
    """Computed weighted average normalized stokes parameters
    Weight is 1/sqrt(variance mean across stokes)

    Parameters
    ----------
    stokes_Sw: 2d float nparray(unnormalized stokes,wavelength bin)
       unnormalized stokes parameters vs wavelength

    var_Sw: 2d float nparray(unnormalized stokes,wavelength bin) 
       variance for stokes_Sw

    wav_w: 1d float ndarray(wavelength bin)

    Output: wtd avg normalized stokes, err, wavelength

    """
    stokess = stokes_Sw.shape[0]
    ok_w = (var_Sw != 0).all(axis=0)
    stokes_sW = stokes_Sw[1:,ok_w]/stokes_Sw[0,ok_w]                            
    var_sW = var_Sw[1:,ok_w]/stokes_Sw[0,ok_w]**2
    wav_W = wav_w[ok_w]
    wt_W = 1./(var_sW[1:stokess].mean(axis=0))

    wtavvar_s = (var_sW*wt_W**2).sum(axis=1)/ wt_W.sum()**2
    wtavstokes_s = (stokes_sW*wt_W).sum(axis=1)/ wt_W.sum()
    wtavwav = (wav_W*wt_W).sum()/wt_W.sum()

    return wtavstokes_s, wtavvar_s, wtavwav

#---------------------------------------------------------------------------------------------
def printstokes(stokes_Sw,var_Sw,wav_w,textfile=sys.stdout,tcenter=0,isfluxed=False):
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

    


