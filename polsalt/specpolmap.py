
"""
specpolmap

Split O and E and produce wavelength map for spectropolarimetric data

"""

import os, sys, glob, shutil, inspect

import numpy as np
import pyfits
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
from scipy import linalg as la

from pyraf import iraf
from iraf import pysalt

from saltobslog import obslog

from specidentify import specidentify
from saltsafelog import logging

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"
np.set_printoptions(threshold=np.nan)
debug = True

def specpolmap(infile_list, propcode=None, inter=True, automethod='Matchlines',logfile='salt.log'):
    obsdate=os.path.basename(infile_list[0])[7:15]

    with logging(logfile, debug) as log:
    #create the observation log
        obs_dict=obslog(infile_list)

    # eliminate inapplicable images
        for i in range(len(infile_list)):
            if obs_dict['PROPID'][i].upper().strip()!=propcode or int(obs_dict['BS-STATE'][i][1])!=2: 
                del infile_list[i]
        obs_dict=obslog(infile_list)

    # use arc to make first-guess wavecal from model, locate beamsplitter split point
        for i in range(len(infile_list)):
            if obs_dict['OBJECT'][i].upper().strip()=='ARC' : arcidx = i
        rbin,cbin = np.array(obs_dict["CCDSUM"][arcidx].split(" ")).astype(int)
        grating = obs_dict['GRATING'][arcidx].strip()
        grang = float(obs_dict['GR-ANGLE'][arcidx])
        artic = float(obs_dict['CAMANG'][arcidx])
        hduarc = pyfits.open(infile_list[arcidx])
        arc_rc = hduarc['SCI'].data
        rows,cols = arc_rc.shape
        lam_c = rssmodelwave(grating,grang,artic,cbin,cols)
        arc_r = arc_rc.sum(axis=1)

        lam_m = np.loadtxt(datadir+"wollaston.txt",dtype=float,usecols=(0,))
        rpix_om = np.loadtxt(datadir+"wollaston.txt",dtype=float,unpack=True,usecols=(1,2))
        expectrow_o = ((2052 + interp1d(lam_m,rpix_om,kind='cubic') \
                        (lam_c[cols/2-cols/16:cols/2+cols/16])).mean(axis=1)/rbin).astype(int)
        foundrow_o = np.zeros((2),dtype=int)
        for o in (0,1):
            foundrow_o[o] = expectrow_o[o]-100/rbin \
                    + np.argmax(arc_r[expectrow_o[o]-100/rbin:expectrow_o[o]+100/rbin])
            arcsignal = arc_r[foundrow_o[o]]
            botedge = foundrow_o[o] + np.argmax(arc_r[foundrow_o[o]:] <  arcsignal/2.) 
            topedge = foundrow_o[o] - np.argmax(arc_r[foundrow_o[o]::-1] <  arcsignal/2.)
            foundrow_o[o] = (botedge+topedge)/2.
        splitrow = foundrow_o.mean()
        offset = int(splitrow - rows/2)
        log.message('Split Row: '+("%4i " % splitrow), with_header=False)

    # split arc into o/e images
        padbins = (np.indices((rows,cols))[0]<offset) | (np.indices((rows,cols))[0]>rows+offset)
        arc_rc = np.roll(arc_rc,-offset,axis=0)
        arc_rc[padbins] = 0.
        arc_orc =  arc_rc.reshape((2,rows/2,cols))
    
    # for O,E arc straighten spectrum, identify for each, form (unstraightened) wavelength map   

        lamp=obs_dict['LAMPID'][arcidx].strip().replace(' ', '')
        if lamp == 'NONE': lamp='CuAr'
        hduarc[0].header.update('MASKTYP','LONGSLIT')
        hduarc[0].header.update('MASKID','PL0100N001')
        del hduarc['VAR']
        del hduarc['BPM']
        lampfile=iraf.osfn("pysalt$data/linelists/%s.wav" % lamp)
        rpix_oc = interp1d(lam_m, rpix_om,kind ='cubic')(lam_c)

        log.message('\nARC: image '+infile_list[i].split('.')[0][-4:]+' GRATING '+grating\
            +' GRANG '+("%8.3f" % grang)+' ARTIC '+("%8.3f" % artic)+' LAMP '+lamp, with_header=False)

        wavmap_orc = np.zeros_like(arc_orc)
        for o in (0,1):
            foundrow_o[o] += -offset + o*rows/2
            arc_Rc = np.zeros((rows/2,cols),dtype='float32')
            for c in range(cols): 
                shift(arc_orc[o,:,c],  \
                    -(rpix_oc[o,c]-rpix_oc[o,cols/2])/rbin,arc_Rc[:,c])
            hduarc['SCI'].data = arc_Rc
            arcimage = "arc_"+str(o)+".fits"
            hduarc.writeto(arcimage,clobber=True)
            Rstart = foundrow_o[o]
            order = 3
            dbfilename = "arcdb_"+str(o)+".txt"
            if os.path.isfile(dbfilename):
#                    guessfile = "guess_"+str(o)+".txt"
#                    os.rename(dbfilename,guessfile)
                guesstype = 'file'
            else:
                guessfile = ''
                guesstype = 'rss'
                specidentify(arcimage, lampfile, dbfilename, guesstype=guesstype,
                    guessfile=guessfile, automethod=automethod,  function='legendre',  order=order,
                    rstep=10, rstart=Rstart, mdiff=20, thresh=3, niter=5, smooth=3,
                    inter=inter, clobber=True, logfile=logfile, verbose=True)

            wavlegR_y = np.loadtxt(dbfilename,dtype=float,usecols=(0,))
            wavlegcof_ly = np.loadtxt(dbfilename,unpack=True,dtype=float,usecols=range(1,order+2))
            wavlegcof_ly = np.delete(wavlegcof_ly,np.where(wavlegR_y<0)[0],axis=1)
            wavlegR_y = np.delete(wavlegR_y,np.where(wavlegR_y<0)[0],axis=0)
            wavmap_yc = np.polynomial.legendre.legval(np.arange(cols),wavlegcof_ly)

            mediancof_l = np.median(wavlegcof_ly,axis=1)
            rms_l = np.sqrt(np.median((wavlegcof_ly - mediancof_l[:,None])**2,axis=1))
            sigma_ly = (wavlegcof_ly - mediancof_l[:,None])/rms_l[:,None]
            usey = (sigma_ly[0]<3) & (sigma_ly[1]<3)
            wavmap_Rc = np.zeros((rows/2,cols),dtype='float32')
            R_y = wavlegR_y[usey]
            a = np.vstack((R_y**3,R_y**2,R_y,np.ones_like(R_y))).T
            for c in range(cols):
                polycofs = la.lstsq(a,wavmap_yc[usey,c])[0]
                wavmap_orc[o,:,c]  \
                    = np.polyval(polycofs,range(rows/2)+(rpix_oc[o,c]-rpix_oc[o,cols/2])/rbin)

    # save split data along third fits axis, add wavmap extension, save as 'w' file

        hduwav = pyfits.ImageHDU(data=wavmap_orc, header=hduarc['SCI'].header, name='WAV')          
        for i in range(len(infile_list)):  
            hdu = pyfits.open(infile_list[i])
            image_rc = np.roll(hdu['SCI'].data,-offset,axis=0)
            image_rc[padbins] = 0.
            hdu['SCI'].data = image_rc.reshape((2,rows/2,cols))
            var_rc = np.roll(hdu['VAR'].data,-offset,axis=0)
            var_rc[padbins] = 0.
            hdu['VAR'].data = var_rc.reshape((2,rows/2,cols))
            bpm_rc = np.roll(hdu['BPM'].data,-offset,axis=0)
            bpm_rc[padbins] = 1
            hdu['BPM'].data = bpm_rc.reshape((2,rows/2,cols))
            hdu.append(hduwav)
            for f in ('SCI','VAR','BPM','WAV'): hdu[f].header.update('CTYPE3','O,E')
            hdu.writeto('w'+infile_list[i],clobber='True')
            log.message('Output file '+'w'+infile_list[i] , with_header=False)
 
def rssmodelwave(grating,grang,artic,cbin,cols):
#   compute wavelengths from model (this can probably be done using pyraf spectrograph model)
    spec=np.loadtxt(datadir+"spec.txt",usecols=(1,))
    Grat0,Home0,ArtErr,T2Con,T3Con=spec[0:5]
    FCampoly=spec[5:11]
    grname=np.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=np.loadtxt(datadir+"gratings.txt",usecols=(1,2),unpack=True)

    grnum = np.where(grname==grating)[0][0]
    lmm = grlmm[grnum]
    alpha_r = np.radians(grang+Grat0)
    beta0_r = np.radians(artic*(1+ArtErr)+Home0)-alpha_r
    gam0_r = np.radians(grgam0[grnum])
    lam0 = 1e7*np.cos(gam0_r)*(np.sin(alpha_r) + np.sin(beta0_r))/lmm
    ww = lam0/1000. - 4.
    fcam = np.polyval(FCampoly,ww)
    disp = (1e7*np.cos(gam0_r)*np.cos(beta0_r)/lmm)/(fcam/.015)
    dfcam = 3.162*disp*np.polyval([FCampoly[x]*(5-x) for x in range(5)],ww)
    T2 = -0.25*(1e7*np.cos(gam0_r)*np.sin(beta0_r)/lmm)/(fcam/47.43)**2 + T2Con*disp*dfcam
    T3 = (-1./24.)*3162.*disp/(fcam/47.43)**2 + T3Con*disp
    T0 = lam0 + T2 
    T1 = 3162.*disp + 3*T3
    X = (np.array(range(cols))+1-cols/2)*cbin/3162.
    lam_X = T0+T1*X+T2*(2*X**2-1)+T3*(4*X**3-3*X)
    return lam_X
    


