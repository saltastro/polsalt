#! /usr/bin/env python

"""
impolview

Plot and text output of stokes ImPol data

"""

import os, sys, glob, shutil, inspect
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter, FormatStrFormatter  
plt.ioff()

import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter("error")

from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.coordinates import Latitude,Longitude
import astropy.table as ta

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

import specpolview as spv
np.set_printoptions(threshold=np.nan)

#---------------------------------------------------------------------------------------------
def impolview(fileList, **kwargs):
    """View Stokes output results

    Parameters
    ----------
    fileList: list
       one or more stokes fits files

    debug=: False (default)
            True (debug output)
    """
    files = len(fileList)
    debug = (kwargs.pop('debug','False') == 'True')
    wav = float(kwargs.pop('wav','0')) 

    plotcolor_f = ['b','g','r','c','m','y','k']
    askpltlim = True
 
    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)
        stokes_Stw = hdul['SCI'].data
        var_Stw = hdul['VAR'].data
        covar_Stw = hdul['COV'].data        
        bpm_Stw = hdul['BPM'].data
        ok_tw = (bpm_Stw[0]==0)
        pmapTab = hdul['TGT'].data
        RAd0 = Longitude(hdul[0].header['RA0']+' hours').degree
        DECd0 = Latitude(hdul[0].header['DEC0']+' degrees').degree
        PA0 = float(hdul[0].header['PA0'])
        name = os.path.basename(file).split('.')[0]

        if f==0:                
          # get manual plot limits
            pscale = 0.01                       # 1%/20-arcsec default
            while (askpltlim):
                ismanlim_s = np.zeros(5,dtype=bool)
                limList = (raw_input('\nOptional scale (%,y0,y1,x0,x1(arcmin), comma sep): ')).split(',')
                if len(''.join(limList))==0: 
                    askpltlim = False
                    break  
                ismanlim_s[:len(limList)] = np.array([len(s)>0 for s in limList])                                                      
                askpltlim = False
            if ismanlim_s[0]: pscale = 0.01*float(limList[0])
            lim_s = np.array([-2.,2.,-4.,4.])
            if ismanlim_s[1]:
                lim_s[ismanlim_s[1:]] = np.array(limList[1:])[ismanlim_s[1:]]
            ymin,ymax,xmin,xmax = lim_s + np.array([0.,0.3,0.,0.])  # for label

          # set up multiplot
            fig = plt.figure(figsize=(7.,11.))
            ax2d = fig.add_subplot(3, 1, 1) 
            ax2d.set_xlabel('X (arcmin)')
            ax2d.set_ylabel('Y (arcmin)')
            ax2d.axis('equal')
            axsList = [fig.add_subplot(3, 1, s+2, projection='3d') for s in (0,1) ]
        plotcolor = plotcolor_f[f % len(plotcolor_f)]
                     
      # assemble data
        wavs = stokes_Stw.shape[2]
        ra_t, dec_t = pmapTab['RA'], pmapTab['DEC']
        sinpa, cospa = np.sin(np.radians(PA0)), np.cos(np.radians(PA0))
        y_t = (cospa*(dec_t-DECd0) + sinpa*(ra_t-RAd0)/np.cos(np.radians(dec_t)))*60.
        x_t = (-sinpa*(dec_t-DECd0) - cospa*(ra_t-RAd0)/np.cos(np.radians(dec_t)))*60.     
        ok_t = (ok_tw.any(axis=1) & (y_t > ymin) & (y_t < ymax) & (x_t > xmin) & (x_t < xmax))                        
        yx_dT = np.array([y_t[ok_t],x_t[ok_t]])
        stokes_sTw = stokes_Stw[1:,ok_t]/stokes_Stw[0,ok_t]
        err_sTw = var_Stw[1:,ok_t]/stokes_Stw[0,ok_t]**2        
        Targets = ok_t.sum()
        t_T = np.where(ok_t)[0]
                
        title = name
        if (wavs > 1):        
            wedge_W = hdul['WCSDVARR'].data
            dwav = hdul['SCI'].header['CDELT1']     # number of unbinned wavelength bins in bin
            wedge_w = wedge_W[:(wavs+1)*dwav:dwav]  # shape wavs+1     
            wav_w = (wedge_w[:-1] + wedge_w[1:])/2.                 
            if (wav>0):        
                widx = np.argmin(np.abs(wav_w - wav))
                wav = wav_w[widx]
                qu_dT = stokes_sTw[:2,:,widx]
                title = name+'\n'+(str(wav)+' Ang').center(len(name))
                name = name+'_'+str(wav)
            else:
                qu_dT = np.zeros((2,Targets))
                for T,t in enumerate(t_T):            
                    qu_dT[:,T] = spv.avstokes(stokes_Stw[:,t,ok_tw[t]],    \
                            var_Stw[:-1,t,ok_tw[t]],covar_Stw[:,t,ok_tw[t]],wav_w[ok_tw[t]])[0] 
        else:
            qu_dT = stokes_sTw[:2,:,0]                                    
        
      # 0: vector plot      
        p_T = np.sqrt((qu_dT**2).sum(axis=0))
        PA_T = 0.5*np.arctan2(qu_dT[1],qu_dT[0])                # radians
        pvec_vdT = yx_dT[None,:,:] + np.array([0.5,-0.5])[:,None,None]*(p_T[None,None,:]/(3.*pscale))* \
            np.array([np.cos(PA_T),-np.sin(PA_T)])[None,:,:]
           
        ax2d.plot(yx_dT[1],yx_dT[0],'o',ms=3,color=plotcolor)
        for T in range(Targets):
            ax2d.plot(pvec_vdT[:,1,T],pvec_vdT[:,0,T],'-',lw=2,color=plotcolor)
        yxlab_d = np.array([ymax-0.2,xmax-0.2])
        PAlab = 0.
        pveclab_vd = yxlab_d[None,:] + np.array([0.5,-0.5])[:,None]*0.01/(3.*pscale)* \
            np.array([np.sin(PAlab),np.cos(PAlab)])[None,:]
        ax2d.plot(pveclab_vd[:,1],pveclab_vd[:,0],'-',lw=3,color='k')
        ax2d.annotate(("%4.1f%%" % (100.*pscale)),tuple(yxlab_d[::-1]),xytext=(yxlab_d[1]-0.3,yxlab_d[0]-0.3))

        if f==0:
            limsize = max((ymax-ymin),(xmax-xmin))
            dtick = int((limsize/8.)/0.1) /10.                            
            ax2d.set_ylim(ymin=ymin,ymax=ymax)
            ax2d.set_xlim(xmin=xmin,xmax=xmax)
            ax2d.set_yticks(np.arange(ymin - (-ymin % dtick),ymax - (ymax % dtick) + dtick, dtick))
            ax2d.set_xticks(np.arange(xmin - (-xmin % dtick),xmax - (xmax % dtick) + dtick, dtick))  
            plt.suptitle(title)

      # 1,2: q,u surface plots
        zlimsize = 100.*(qu_dT.max(axis=1) - qu_dT.min(axis=1)).max()
        dztick = int(((zlimsize)/5.)/0.1) /10.
        zticks = int(zlimsize/dztick)
        for s in (0,1):
            zmin = 100.*qu_dT[s].min()
            zmax = zmin + zlimsize
            ztick0 = zmin - (-zmin % dztick)            
            axsList[s].plot_trisurf(yx_dT[1],yx_dT[0],100.*qu_dT[s]) 
            axsList[s].set_ylim(ymin=ymin,ymax=ymax)
            axsList[s].set_xlim(xmin=xmin,xmax=xmax)
            axsList[s].set_zlim(zmin=zmin,zmax=zmax)
            axsList[s].set_yticks(np.arange(ymin - (-ymin % dtick),ymax - (ymax % dtick) + dtick, dtick))
            axsList[s].set_xticks(np.arange(xmin - (-xmin % dtick),xmax - (xmax % dtick) + dtick, dtick))
            axsList[s].set_zticks(np.arange(ztick0,ztick0+(zticks+1)*dztick, dztick))            
            axsList[s].zaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))

  # Save plot 
    plotfile = name+'.pdf'
    plt.savefig(plotfile,orientation='portrait')
    if os.name=='posix':
        if os.popen('ps -C evince -f').read().count(plotfile)==0: os.system('evince '+plotfile+' &')
    else:
        plt.show(block=True)

    return
  
#---------------------------------------------------------------------------------------------
 
if __name__=='__main__':
    fileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)  
    impolview(fileList, **kwargs)

    


