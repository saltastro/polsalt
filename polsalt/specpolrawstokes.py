
"""
specpolrawstokes

Form spectropolarimetry filterpair E-O raw stokes diffsums.

"""

import os, sys, glob, shutil, inspect

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

def specpolrawstokes(infile_list, logfile='salt.log'):
    #set up some files that will be needed
    obsdate=os.path.basename(infile_list[0]).split('.')[0][-12:-4]

    logfile='specpol'+obsdate+'.log'
    patternfile=open(datadir+'wppaterns.txt','r')
    with logging(logfile, debug) as log:

        #create the observation log
        obs_dict=obslog(infile_list)
        images = len(infile_list)

        hsta_i = np.array([int(float(s)/11.25) for s in obs_dict['HWP-ANG']])
        qsta_i = np.array([int(float(s)/11.25) for s in obs_dict['QWP-STA']])
        img_i = np.array([int(os.path.basename(s).split('.')[0][-4:]) for s in infile_list])
        wpstate_i = [['unknown','out','qbl','hw','hqw'][int(s[1])] for s in obs_dict['WP-STATE']]
        #    wppat_i = obs_dict['WPPATERN']               
        wppat_i = ['UNKNOWN' for i in range(images)]        # until WPPATERN is put in obslog

        object_i = obs_dict['OBJECT']
        config_i = np.zeros(images,dtype='int')
        obs_i = -np.ones(images,dtype='int')

    # make table of observations

        configs = 0; obss = 0
        for i in range(images):
            if (wpstate_i[i] == 'unknown'):
                log.message('Warning: Image %s WP-STATE UNKNOWN, assume it is 3 (HW)' % img_i[i] , with_header=False)
                wpstate_i[i] = 'hw'
            elif (wpstate_i[i] == 'out'):
                log.message('Image %i not in a WP pattern, will skip' % img_i[i] , with_header=False)
                continue                  
            if object_i[i].count('NONE'): object_i[i] = obs_dict['LAMPID'][i]
            object_i[i] = object_i[i].replace(' ','')
            cbin,rbin = np.array(obs_dict["CCDSUM"][i].split(" ")).astype(int)
            grating = obs_dict['GRATING'][i].strip()
            grang = float(obs_dict['GR-ANGLE'][i])
            artic = float(obs_dict['CAMANG'][i])
            confdat_d = [rbin,cbin,grating,grang,artic,wppat_i[i]]
            obsdat_d = [object_i[i],rbin,cbin,grating,grang,artic,wppat_i[i]]
            if configs==0: 
                confdat_cd = [confdat_d]
                obsdat_od = [obsdat_d]
            configs = len(confdat_cd); config = 0
            while config<configs:
                if confdat_d == confdat_cd[config]: break
                config += 1
            if config == configs: confdat_cd.append(confdat_d)
            config_i[i] = config
            obss = len(obsdat_od); obs = 0
            while obs<obss:
                if obsdat_d == obsdat_od[obs]: break
                obs += 1
            if obs == obss: obsdat_od.append(obsdat_d)
            obs_i[i] = obs

        patternlist = patternfile.readlines()
 
        log.message('Raw Stokes File     OBS CCDSUM GRATING  GR-ANGLE  CAMANG    WPPATERN'  , with_header=False)

    # Compute E-O raw stokes

        for obs in range(obss):
            idx_j = np.where(obs_i == obs)
            i0 = idx_j[0][0]
            name_n = []
            if wppat_i[i0].count('UNKNOWN'):                
                if (hsta_i[idx_j] % 2).max() == 0: wppat = "Linear"
                else: wppat = "Linear-Hi"
                for i in idx_j[0]: wppat_i[i] = wppat
            if not(((wpstate_i[i0]=='hw') & (wppat_i[i0] in ('Linear','Linear-Hi')) 
                | (wpstate_i[i0]=='hqw')& (wppat_i[i0] in ('Circular','Circular-Hi','All-Stokes')))):
                print "Observation",obs,": wpstate ",wpstate_i[i0], \
                    " and wppattern ",wppat_i[i0], "not consistent"
                continue
            for p in patternlist:
                if (p.split()[0]==wppat_i[i0])&(p.split()[2]=='hwp'): 
                    wpat_p = np.array(p.split()[3:]).astype(int)
                if (p.split()[0]==wppat_i[i0])&(p.split()[2]=='qwp'): 
                    wpat_dp = np.vstack(wpat_p,np.array(p.split()[3:])) 
            stokes=0; j=-1

            while j < (len(idx_j[0])-2):
                j += 1
                i = idx_j[0][j]
                if (wpstate_i[i]=='hw'):
                    if (np.where(wpat_p[0::2]==hsta_i[i])[0].size > 0): 
                        idxp = np.where(wpat_p==hsta_i[i])[0][0]
                        if hsta_i[i+1] != wpat_p[idxp+1]: continue
                    else: continue
                if (wpstate_i[i]=='hqw'):
                    if (np.where(wpat_dp[0::2]==(hsta_i[i],qsta_i[i]))[0].size > 0): 
                        idxp = np.where(wpat_dp==(hsta_i[i],qsta_i[i]))[0][0]
                        if (hsta_i[i+1],qsta_i[i+1]) != wpat_dp[None,idxp+1]: continue
                    else: continue
    
                if stokes==0:
                    wavs = pyfits.getheader(infile_list[i],'SCI',1)['NAXIS1']
                sci_fow = np.zeros((2,2,wavs)); var_fow = np.zeros_like(sci_fow);   \
                                                bpm_fow = np.zeros_like(sci_fow) 
                for f in (0,1):
                    hdulist = pyfits.open(infile_list[i+f])
                    sci_fow[f] = hdulist['sci'].data.reshape((2,-1))
                    var_fow[f] = hdulist['var'].data.reshape((2,-1))
                    bpm_fow[f] = hdulist['bpm'].data.reshape((2,-1))

            # compute intensity, E-O stokes spectrum, VAR, BPM.
            # fits out: unnormalized (int,stokes),(length 1) spatial,wavelength
            # wavelength marked bad if it is bad in either filter or order  

                bpm_w = (bpm_fow.sum(axis=0).sum(axis=0) > 0).astype(int)
                wok = (bpm_w==0)

                stokes_sw = np.zeros((2,wavs),dtype='float32');  var_sw = np.zeros_like(stokes_sw)
                stokes_sw[0,wok] = 0.5*sci_fow[:,:,wok].reshape((2,2,-1)).sum(axis=0).sum(axis=0)
                var_sw[0,wok] = 0.25*var_fow[:,:,wok].reshape((2,2,-1)).sum(axis=0).sum(axis=0)
                stokes_sw[1,wok] = 0.5*((sci_fow[0,1,wok]-sci_fow[1,1,wok])/(sci_fow[0,1,wok]+sci_fow[1,1,wok]) \
                           - (sci_fow[0,0,wok]-sci_fow[1,0,wok])/(sci_fow[0,0,wok]+sci_fow[1,0,wok]))
                var_sw[1,wok] = 0.5*((var_fow[0,1,wok]+var_fow[1,1,wok])/(sci_fow[0,1,wok]+sci_fow[1,1,wok])**2 \
                        + (var_fow[0,0,wok]+var_fow[1,0,wok])/(sci_fow[0,0,wok]+sci_fow[1,0,wok])**2)

                stokes_sw[1] *= stokes_sw[0]
                var_sw[1] *= stokes_sw[0]**2
                bpm_sw = np.array([bpm_w,bpm_w],dtype='uint8').reshape((2,wavs))

                name = object_i[i] + '_c' + str(config_i[i]) + '_h' + str(hsta_i[i]) + str(hsta_i[i+1])
                if (wpstate_i[i]=='hqw'):
                    name += 'q'+['m','p'][qsta_i[i]==4]+['m','p'][qsta_i[i+1]==4]

                count = " ".join(name_n).count(name)
                name += ('_%02i' % (count+1))
 
                log.message('%20s  %1i  %1i %1i %8s %8.2f %8.2f %12s' % \
                    (name,obs,rbin,cbin,grating,grang,artic,wppat_i[i]), with_header=False)

                hduout = pyfits.PrimaryHDU(header=hdulist[0].header)    
                hduout = pyfits.HDUList(hduout)
                hduout[0].header.update('WPPATERN',wppat_i[i])
                header=hdulist['SCI'].header.copy()
                header.update('VAREXT',2)
                header.update('BPMEXT',3)
                header.update('CTYPE3','I,S')
                hduout.append(pyfits.ImageHDU(data=stokes_sw.reshape((2,1,wavs)), header=header, name='SCI'))
                header.update('SCIEXT',1,'Extension for Science Frame',before='VAREXT')
                hduout.append(pyfits.ImageHDU(data=var_sw.reshape((2,1,wavs)), header=header, name='VAR'))
                hduout.append(pyfits.ImageHDU(data=bpm_sw.reshape((2,1,wavs)), header=header, name='BPM'))

                hduout.writeto(name+'.fits',clobber=True,output_verify='warn')
                name_n.append(name)
                i += 1
                stokes += 1

    return
 
 

    



