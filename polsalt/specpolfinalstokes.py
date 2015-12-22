
"""
specpolfinalstokes

Correct raw stokes for track, combine, and calibrate to form final stoes.

"""

import os, sys, glob, shutil, inspect
import operator

import numpy as np
import pyfits

from scipy.interpolate import interp1d
from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"
np.set_printoptions(threshold=np.nan)
debug = True

def specpolfinalstokes(infile_list,polcal='polcal.txt',logfile='salt.log'):
    """Combine the raw stokes and apply the polarimetric calibrations

    Parameters
    ----------
    infile_list: list
        List of filenames that include an extracted spectra

    polcal: str
        File with polarimetric calibration values

    logfile: str
        Name of file for logging


    """

    patternlist = open(datadir+'wppaterns.txt','r').readlines()
    patternpairs = dict();  patternstokes = dict()
    for p in patternlist:
        if p.split()[0] == '#': continue
        patternpairs[p.split()[0]]=(len(p.split())-3)/2
        patternstokes[p.split()[0]]=int(p.split()[1])
    wav_l,heff_l,hpa_l,qeff_l = np.loadtxt(datadir+polcal,dtype=float,unpack=True)
    calversion = open(datadir+polcal, 'r').readlines()[1][2:].rstrip()

    with logging(logfile, debug) as log:
        
    # organize data using names
        files = len(infile_list)
        allrawlist = []
        for i in range(files):
            object,config,wvplt,count = os.path.basename(infile_list[i]).split('.')[0].rsplit('_',4)
            if (config[0]!='c')|(wvplt[0]!='h')|(not count.isdigit()):
                log.message('File '+infile_list[i]+' is not a raw stokes file.'  , with_header=False) 
                continue
            allrawlist.append([i,object,config,wvplt,count])
        configlist = list(set(ele[2] for ele in allrawlist))        # unique configs

    # correct raw stokes for track (TBS)

    # do one config at a time, since different configs may have different number of wavelengths
        for conf in configlist:
            rawlist = [entry for entry in allrawlist if entry[2]==conf]
            for col in (4,3,1,2): rawlist = sorted(rawlist,key=operator.itemgetter(col))            
            rawstokes = len(rawlist)
            cols = pyfits.open(infile_list[rawlist[0][0]])['SCI'].data.shape[-1]
            stokes_jsw = np.zeros((rawstokes,2,cols)); 
            var_jsw = np.zeros_like(stokes_jsw); bpm_jsw = np.zeros_like(stokes_jsw).astype(int)
            wav_jw = np.zeros((rawstokes,cols))
            comblist = []
        # get data
            for j in range(rawstokes):
                i,object,config,wvplt,count = rawlist[j]
                if j==0:
                    lampid = pyfits.getheader(infile_list[i],0)['LAMPID'].strip().upper()
                    telpa = float(pyfits.getheader(infile_list[i],0)['TELPA'])
                    if lampid=="NONE":
                        pacaltype = "Equatorial"
                        hpa_l -= (telpa % 180)
                    else:
                        pacaltype ="Instrumental"
                    calversion = (pacaltype+'  '+calversion)
                    log.message('Calibration Version: '+calversion, with_header=False) 
           
                wppat = pyfits.getheader(infile_list[i],0)['WPPATERN']
                wav0 = pyfits.getheader(infile_list[i],'SCI')['CRVAL1']
                dwav = pyfits.getheader(infile_list[i],'SCI')['CDELT1']
                stokes_jsw[j] = pyfits.open(infile_list[i])['SCI'].data.reshape((2,-1))
                var_jsw[j] = pyfits.open(infile_list[i])['VAR'].data.reshape((2,-1))
                bpm_jsw[j] = pyfits.open(infile_list[i])['BPM'].data.reshape((2,-1))
                wav_jw[j] = np.mgrid[wav0:(wav0+cols*dwav):dwav]
                if int(count)==1:
                    comblist.append((j,object,config,wvplt,count,wppat))
                else:
                    comblist[-1] = (j,object,config,wvplt,count,wppat)

        # combine multiple instances (count > 1)
            combstokes = len(comblist)
            stokes_ksw = np.zeros((combstokes,2,cols)); 
            var_ksw = np.zeros_like(stokes_ksw)
            bpm_ksw = np.zeros_like(stokes_ksw).astype(int)
            wav_kw = np.zeros((combstokes,cols))
            chisqstokes_kw = np.zeros_like(wav_kw)
            obslist = [];   obsobject = ''; obsconfig = ''
            for k in range(combstokes):
                j,object,config,wvplt,count,wppat = comblist[k]
                stokes_ksw[k] =  stokes_jsw[j-int(count)+1:j+1].sum(axis=0)
                var_ksw[k] =  var_jsw[j-int(count)+1:j+1].sum(axis=0)   
                bpm_ksw[k] =  (bpm_jsw[j-int(count)+1:j+1].sum(axis=0) > 0).astype(int)
                wav_kw[k] = wav_jw[j]

            # compute chisq/dof for multiple instances
                if int(count) > 1:
                    combstokes_w = np.zeros(cols)
                    bok = (bpm_ksw[k,1] == 0) 
                    combstokes_w[bok] = stokes_ksw[k,1,bok]/stokes_ksw[k,0,bok]
                    for jj in range(j-int(count)+1,j+1):
                        stokes_w = np.zeros(cols);  errstokes_w = np.zeros_like(stokes_w)
                        stokes_w[bok] = stokes_jsw[jj,1,bok]/stokes_jsw[jj,0,bok]
                        errstokes_w[bok] =  np.sqrt(var_jsw[jj,1,bok]/(stokes_jsw[jj,0,bok])**2)
                        chisqstokes_kw[k,bok] += ((stokes_w[bok]-combstokes_w[bok])/errstokes_w[bok])**2
                    chisqstokes_kw[k] /= int(count)-1
                    chisqstokes = chisqstokes_kw[k].sum()/bok.sum()
                    log.message("Chisq/dof Filter Pair %s: %7.2f" % (wvplt,chisqstokes), with_header=False)
                if ((object != obsobject) | (config != obsconfig)):
                    obslist.append([k,object,config,wppat,1])
                    obsobject = object; obsconfig = config
                else:
                    obslist[-1][4] +=1
                                                                     
        # for each obs combine stokes, apply efficiency and PA calibration as appropriate for pattern
            obss = len(obslist)
            for obs in range(obss):
                k,object,config,wppat,pairs = obslist[obs]
#                print k,object,config,wppat,pairs
                finstokes = patternstokes[wppat]
                if pairs != patternpairs[wppat]:
                    log.message('Not a complete pattern, skipping observation', with_header=False) 
                    continue
                stokes_fw = np.zeros((finstokes,cols)); var_fw = np.zeros_like(stokes_fw)
                wok = bpm_ksw[k:k+pairs,:].sum(axis=0).sum(axis=0) == 0
                bpm_fw = np.repeat((np.logical_not(wok))[None,:],finstokes,axis=0)
                stokes_fw[0] = stokes_ksw[k:k+pairs,0].sum(axis=0)/pairs
                var_fw[0] = var_ksw[k:k+pairs,0].sum(axis=0)/pairs**2           

                if wppat.count('Linear'):
                    var_fw = np.vstack((var_fw,np.zeros(cols)))           # add QU covariance
                    if wppat=='Linear':
                        stokes_fw[1:,wok] = stokes_ksw[k:k+2,1,wok]*(stokes_fw[0,wok]/stokes_ksw[k:k+2,0,wok])
                        var_fw[1:3,wok] = var_ksw[k:k+2,1,wok]*(stokes_fw[0,wok]/stokes_ksw[k:k+2,0,wok])**2
                    elif wppat=='Linear-Hi':
                # for Linear-Hi, must go to normalized stokes in order for the pair combination to cancel systematic errors
                        nstokes_pw = np.zeros((pairs,cols)); nvar_pw = np.zeros((pairs,cols))
                        nstokes_fw = np.zeros((finstokes,cols)); nvar_fw = np.zeros((finstokes+1,cols))
                        nstokes_pw[:,wok] = stokes_ksw[k:k+pairs,1,wok]/stokes_ksw[k:k+pairs,0,wok]
                        nvar_pw[:,wok] = var_ksw[k:k+pairs,1,wok]/(stokes_ksw[k:k+pairs,0,wok])**2
                        np.savetxt("nstokes.txt",np.vstack((wok.astype(int),nstokes_pw)).T,fmt="%3i "+4*"%10.6f ")
                        np.savetxt("nvar.txt",np.vstack((wok.astype(int),nvar_pw)).T,fmt="%3i "+4*"%14.9f ")
                        nstokes_fw[1] = 0.5*(nstokes_pw[0] + (nstokes_pw[1]-nstokes_pw[3])/np.sqrt(2.))
                        nstokes_fw[2] = 0.5*(nstokes_pw[2] + (nstokes_pw[1]+nstokes_pw[3])/np.sqrt(2.))
                        nvar_fw[1] = 0.25*(nvar_pw[0] + (nvar_pw[1]+nvar_pw[3])/2.)
                        nvar_fw[2] = 0.25*(nvar_pw[2] + (nvar_pw[1]+nvar_pw[3])/2.)
                        nvar_fw[3] = 0.25*((nvar_pw[1] - nvar_pw[3])/2.)
                        stokes_fw[1:] = nstokes_fw[1:]*stokes_fw[0]
                        var_fw[1:] = nvar_fw[1:]*stokes_fw[0]**2
                        chisqq = ((nstokes_pw[0,wok] - nstokes_fw[1,wok])**2/nvar_fw[1,wok]).sum()/wok.sum() 
                        chisqu = ((nstokes_pw[2,wok] - nstokes_fw[2,wok])**2/nvar_fw[2,wok]).sum()/wok.sum()
                        log.message("Chisq/dof Linear-Hi Q,U: %7.2f %7.2f" % (chisqq,chisqu), with_header=False) 
                    heff_w = interp1d(wav_l,heff_l,kind='cubic')(wav_kw[k])
                    par_w = -interp1d(wav_l,hpa_l,kind='cubic')(wav_kw[k])
                    c_w = np.cos(2.*np.radians(par_w)); s_w = np.sin(2.*np.radians(par_w))
                    stokes_fw[1:] /= heff_w
                    var_fw[1:] /= heff_w**2
                    stokes_fw[1:] = stokes_fw[1]*c_w - stokes_fw[2]*s_w ,    \
                                    stokes_fw[1]*s_w + stokes_fw[2]*c_w
                    var_fw[1:3] =  var_fw[1]*c_w**2 + var_fw[2]*s_w**2 ,    \
                                    var_fw[1]*s_w**2 + var_fw[2]*c_w**2
                    var_fw[3] =  c_w*s_w*(var_fw[1] - var_fw[2]) + (c_w**2-s_w**2)*var_fw[3]

                # save final stokes fits file
                    infile = infile_list[rawlist[comblist[k][0]][0]]
                    hduout = pyfits.open(infile)
                    hduout['SCI'].data = stokes_fw.astype('float32').reshape((3,1,-1))
                    hduout['SCI'].header.update('CTYPE3','I,Q,U')
                    hduout['VAR'].data = var_fw.astype('float32').reshape((4,1,-1))
                    hduout['VAR'].header.update('CTYPE3','I,Q,U,QU')
                    hduout['BPM'].data = bpm_fw.astype('uint8').reshape((3,1,-1))
                    hduout['BPM'].header.update('CTYPE3','I,Q,U')
                    hduout[0].header.update('POLCAL',calversion)
                    outfile = object+'_'+config+'_stokes.fits'
                    hduout.writeto(outfile,clobber=True,output_verify='warn')
                    log.message('\n'+outfile+' Stokes I,Q,U', with_header=False)
                     
#               elif wppat.count('Circular'):  TBS 

#               elif wppat=='All-Stokes':  TBS

    return
