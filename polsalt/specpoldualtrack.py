
"""
specpoldualtrack

Do finalstokes analysis for a dual-track observation (linear pol only, one config)

"""

import os, sys, glob, shutil, inspect
import operator

import numpy as np
from astropy.io import fits as pyfits

from scipy.interpolate import interp1d
from scipy import linalg as la
from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging
from specpolfinalstokes import datedfile, specpolrotate

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

np.set_printoptions(threshold=np.nan)


# -------------------------------------
def specpoldualtrack(infilelist1,infilelist2,logfile='salt.log',debug=False):
    """Fit normalized raw stokes as function of track position, derive instrumental linear 
    polarization vs track and object polarization

    Parameters
    ----------
    infilelist1,2: list
        Separate lists of filenames for each track

    logfile: str
        Name of file for logging
    """

#    _l: line in calibration file
#    _o: observation/track (0,1)

    with logging(logfile, debug) as log:

    # input calibration files
        obsdict=obslog(infilelist1)
        dateobs = obsdict['DATE-OBS'][0].replace('-','')
        HWCalibrationfile = datedfile(datadir+"RSSpol_HW_Calibration_yyyymmdd.txt",dateobs)
        hwav_l,heff_l,hpa_l = np.loadtxt(HWCalibrationfile,dtype=float,unpack=True,ndmin=2)
        tdate_l,t0_l = np.loadtxt(datadir+"RSSpol_Linear_PAZeropoint.txt",dtype=float,unpack=True,ndmin=2)
        t0 = t0_l[0]
        for (l,td) in enumerate(tdate_l):
            if dateobs < td: break
            tdate = td
            t0 = t0_l[l]

      # find common calibratable wavelengths
        wav0 = hwav_l.min()
        wav1 = hwav_l.max()
        dwav = pyfits.getheader(infilelist1[0],'SCI')['CDELT1']
        for file in (infilelist1+infilelist2):
            fwav0 = pyfits.getheader(file,'SCI')['CRVAL1']
            fwav1 = fwav0 + dwav*pyfits.getheader(file,'SCI')['NAXIS1']
            wav0 = max(wav0,fwav0)
            wav1 = min(wav1,fwav1)
        wavs = int((wav1 - wav0)/dwav)
        wav_w = np.linspace(wav0,wav0+dwav*wavs,wavs,endpoint=False)
        heff_w = interp1d(hwav_l,heff_l,kind='cubic',bounds_error=False)(wav_w)
        hpar_w = interp1d(hwav_l,hpa_l,kind='cubic',bounds_error=False)(wav_w)

      # evaluate track dependence in telescope coordinates of final stokes (vs time) for each track
        stokes_oFw = np.empty((2,3,wavs))
        var_oFw = np.empty((2,4,wavs))
        ok_ow = np.empty((2,wavs),dtype=bool)
        oktr_ow = np.empty((2,wavs),dtype=bool)
        trkx_od = np.empty((2,4))
        trky_od = np.empty((2,4))
        trkrho_od = np.empty((2,4))
        nstokes_oCfw = np.empty((2,3,2,wavs))   
        nvar_oCfw = np.empty((2,3,3,wavs))
        telpa_o = np.zeros(2)
        for o,infilelist in enumerate((infilelist1,infilelist2)):
            rawlist = []
            files = len(infilelist)
            for i in range(files):
                rawlist.append(os.path.basename(infilelist[i]).split('.')[0].rsplit('_',3))
            objectset = set(ele[0] for ele in rawlist)       # unique objects
            configset = set(ele[1] for ele in rawlist)       # unique configs
            if (len(objectset)>1):
                log.message('More than 1 object: '+str(objectset),with_header=False)
                exit()

            stokes_oFw[o],var_oFw[o],ok_w,nstokes_oCfw[o],nvar_oCfw[o],oktr_w,  \
                trkx_od[o],trky_od[o],trkrho_od[o] = specpoltrack(infilelist,wav0,wavs,debug=debug) 
          
            ok_ow[o] = ok_w
            oktr_ow[o] = oktr_w

        obsname = list(objectset)[0]+"_"+("".join(sorted(list(configset))))
        calhistorylist = [os.path.basename(HWCalibrationfile)]
        ok_w = ok_ow.all(axis=0)
        Cofs_o = (nstokes_oCfw.sum(axis=(2,3))!=0).sum(axis=1)   # Cofs = 2 or 3

      # find point tc_o where tracks are closest
        t_t = np.linspace(-1.,1.,51)
        dtrk_tt = np.sqrt((np.polyval(trkx_od[0],t_t) - np.polyval(trkx_od[1],t_t)[:,None])**2 \
                    + (np.polyval(trky_od[0],t_t) - np.polyval(trky_od[1],t_t)[:,None])**2)
        tc_o = (np.array(np.unravel_index(np.argmin(dtrk_tt),(51,51)))-25.)/25.

      # _f normalized final stokes q,u = 0,1
      # combine normalized observations at that point, giving target and telescope pol there
        nstokes_ofw = np.zeros((2,2,wavs))
        nvar_ofw = np.zeros((2,3,wavs))
        nstokesmean_Cf = np.zeros((3,2))
        phi_o = np.zeros(2)
        for o in range(2):       # use q,u wavelength mean for track variability: C=1,Cofs; f=0,1
            nstokesmean_Cf[:Cofs_o[o]] = (nstokes_oCfw[o,1:Cofs_o[o]][:,:,oktr_ow[o]]/    \
                np.sqrt(nvar_oCfw[o,1:Cofs_o[o],:2][:,:,oktr_ow[o]])).sum(axis=2) /   \
                (1./np.sqrt(nvar_oCfw[o,1:Cofs_o[o],:2][:,:,oktr_ow[o]])).sum(axis=2)
            nstokes_ofw[o,:][:,ok_w] = nstokes_oCfw[o,0,:][:,ok_w] + \
                (nstokesmean_Cf[0]*tc_o[o] + nstokesmean_Cf[1]*(2.*tc_o[o]**2-1.))[:,None] 
            nvar_ofw[o] = nvar_oCfw[o,0]
            phi_o[o] = telpa_o[o] - np.polyval(trkrho_od[o],tc_o[o])

        dnstokes_fw = nstokes_ofw[1] - nstokes_ofw[0]
        dnvar_fw = nvar_ofw.sum(axis=0)
        dcosphi = np.cos(np.radians(phi_o[1])) - np.cos(np.radians(phi_o[0]))
        dsinphi = np.sin(np.radians(phi_o[1])) - np.sin(np.radians(phi_o[0]))
        norm = dsinphi**2 + dcosphi**2

        ntelstokes_fw = np.zeros((2,wavs))
        ntelvar_fw = np.zeros((3,wavs))
        ntelstokes_fw[0,ok_w] = (dnstokes_fw[0,ok_w]*dcosphi + dnstokes_fw[1,ok_w]*dsinphi)/norm
        ntelstokes_fw[1,ok_w] =(-dnstokes_fw[0,ok_w]*dsinphi + dnstokes_fw[1,ok_w]*dcosphi)/norm
        ntelvar_fw[0,ok_w] = (dnvar_fw[0,ok_w]*dcosphi**2 + dnvar_fw[1,ok_w]*dsinphi**2)/norm**2
        ntelvar_fw[1,ok_w] = (dnvar_fw[0,ok_w]*dsinphi**2 + dnvar_fw[1,ok_w]*dcosphi**2)/norm**2

        ntelstokes_ofw = np.ones((2,2,wavs))
        ntelvar_ofw = np.ones((2,3,wavs))
      # rotate back to equatorial for each track
        for o in range(2):
            ntelstokes_ofw[o],ntelvar_ofw[o] = \
                specpolrotate(ntelstokes_fw,ntelvar_fw,phi_o[o],normalized=True)

      # correct each obs for telescope (wavelength mean) at that point, combine to give target pol
        ntelmean_of = (ntelstokes_ofw[:,:,ok_w]/np.sqrt(ntelvar_ofw[:,:2][:,:,ok_w])).sum(axis=2) /   \
            (1./np.sqrt(ntelvar_ofw[:,:2][:,:,ok_w])).sum(axis=2)
        ntargstokes_ofw = nstokes_ofw - ntelmean_of[:,:,None]
        ntargstokes_fw = np.ones((2,wavs))
        ntargvar_fw = np.ones((3,wavs))
        ntargstokes_fw[:2,ok_w] = ntargstokes_ofw[:,:2,ok_w].mean(axis=0)
        ntargvar_fw[:3,ok_w] = nvar_ofw[:,:3,ok_w].sum(axis=0)/4.

      # save target stokes fits file
        infile = infilelist1[0]
        hduout = pyfits.open(infile)
        targstokes_Fw = np.zeros((3,wavs))
        targvar_Fw = np.zeros((4,wavs))
        targstokes_Fw[0] = stokes_oFw[:,0].sum(axis=0)
        targstokes_Fw[1:] = ntargstokes_fw*targstokes_Fw[0]
        targvar_Fw[0] = var_oFw[:,0].sum(axis=0)
        targvar_Fw[1:] = ntargvar_fw*targvar_Fw[0]**2    
        hduout['SCI'].data = targstokes_Fw.astype('float32').reshape((3,1,-1))
        hduout['SCI'].header.update('CRVAL1',wav0)
        hduout['SCI'].header.update('CTYPE3','I,Q,U')
        hduout['VAR'].data = targvar_Fw.astype('float32').reshape((4,1,-1))
        hduout['VAR'].header.update('CRVAL1',wav0)
        hduout['VAR'].header.update('CTYPE3','I,Q,U,QU')
        hduout['BPM'].data = np.tile(~ok_w,(3,1)).astype('uint8').reshape((3,1,-1))
        hduout['BPM'].header.update('CRVAL1',wav0)
        hduout['BPM'].header.update('CTYPE3','I,Q,U')
        hduout[0].header.update('PATYPE',"Equatorial")
        calhistorylist.append("Self-Calibrated")
        hduout[0].header.add_history('POLCAL: '+' '.join(calhistorylist))
        outfile = obsname+'_dualtrack_stokes.fits'
        hduout.writeto(outfile,clobber=True,output_verify='warn')
        log.message('\n    '+outfile+' Stokes I,Q,U', with_header=False)

      # save normalized instrumental polarization (equ coords) with track variation for each track
        for o in (0,1):
            hduout = pyfits.open(infile)               
            outfile = obsname+'_instpol'+str(o)+'_stokes.fits'
            telstokes_Fw = np.vstack((np.ones(wavs),ntelstokes_ofw[o]))
            telvar_Fw = np.vstack((np.ones(wavs),ntelvar_ofw[o]))
            savetrack(telstokes_Fw,telvar_Fw,ok_w,nstokes_oCfw[o],  \
                nvar_oCfw[o],oktr_ow[o],wav_w,trkx_od[o],trky_od[o],trkrho_od[o],hduout,outfile)
            log.message('\n    '+outfile+' Stokes i,q,u,Tq,Tu', with_header=False)

    return
#---------------------------------------------------------------------------

def specpoltrack(infilelist,wav0,wavs,logfile='salt.log',debug=False):
    """Process raw stokes in a track into final stokes in equatorial coordinates,
    and fit normalized stokes track variation with smooth variation in track and wavelength

    """
#    _t: time in track (-1 - 1))
#    _i: index in file list
#    _p: pair = waveplate position pair index (enumeration within obs)
#    _s: raw pair stokes : (0,1)
#    _F: final stokes : I,Q,U = (0,1,2)
#    _f: normalized final stokes : q,u = (0,1)
#    _C: Chebyshev coefficients : (0,1,2)

    with logging(logfile, debug) as log:
    # input calibration files
        obsdict=obslog(infilelist)
        dateobs = obsdict['DATE-OBS'][0].replace('-','')

        patternlist = open(datadir+'wppaterns.txt','r').readlines()
        patternpairs = dict();  patternstokes = dict()
        for p in patternlist:
            wppatern = p.split()[0]
            if wppatern == '#': continue
            patternpairs[wppatern]=(len(p.split())-3)/2
            patternstokes[wppatern]=int(p.split()[1])
        HWCalibrationfile = datedfile(datadir+"RSSpol_HW_Calibration_yyyymmdd.txt",dateobs)
        hwav_l,heff_l,hpa_l = np.loadtxt(HWCalibrationfile,dtype=float,unpack=True,ndmin=2)

        files = len(infilelist)
        trkx_i = np.array(obsdict['TRKX'])
        trky_i = np.array(obsdict['TRKY'])
        trkrho_i = np.array(obsdict['TRKRHO'])
        pair_i = (np.array(obsdict['HWP-ANG'])/11.25).astype(int) % 4
        jd_i = np.zeros(files)
        telpa = float(pyfits.getheader(infilelist[0],0)['TELPA'])
        wppat = pyfits.getheader(infilelist[0],0)['WPPATERN'].upper()
        dwav = pyfits.getheader(infilelist[0],'SCI')['CDELT1']
        wav_w = np.linspace(wav0,wav0+dwav*wavs,wavs,endpoint=False)

      # get raw data, compute normalized stokes
        nstokes_iw = np.zeros((files,wavs)) 
        nvar_iw = np.zeros((files,wavs))
        int_w = np.zeros(wavs)
        var_w = np.zeros(wavs)
        ok_iw = np.zeros((files,wavs),dtype=bool)
        for i in range(files):
            hdul = pyfits.open(infilelist[i])
            jd_i[i] = hdul[0].header['JD']
            argw0 = int((wav0 - pyfits.getheader(infilelist[i],'SCI')['CRVAL1'])/dwav)
            stokes_sw = hdul['SCI'].data.reshape((2,-1))[:,argw0:argw0+wavs]
            var_sw = hdul['VAR'].data.reshape((2,-1))[:,argw0:argw0+wavs]
            bpm_sw = hdul['BPM'].data.reshape((2,-1))[:,argw0:argw0+wavs]
            ok_w = (bpm_sw==0).all(axis=(0))
            nstokes_iw[i,ok_w] = stokes_sw[1,ok_w]/stokes_sw[0,ok_w]
            nvar_iw[i,ok_w] = var_sw[1,ok_w]/stokes_sw[0,ok_w]**2
            int_w += stokes_sw[0]
            var_w += var_sw[0]
            ok_iw[i] = ok_w

      # compute track polynomials
        t_i = 2.*(jd_i - jd_i[0])/(jd_i[-1] - jd_i[0]) - 1.
        trkx_d = np.polyfit(t_i,trkx_i,3)
        trky_d = np.polyfit(t_i,trky_i,3)
        trkrho_d = np.polyfit(t_i,trkrho_i,3)

      # plan stokes track fit, with fallback for incomplete Linear-Hi
        Samples_p = (pair_i[:,None] == np.arange(4)).sum(axis=0)
        usepair = list(np.where(Samples_p>1)[0])
        pairs = len(usepair)
        if pairs != patternpairs[wppat]:
            if (wppat<>'LINEAR-HI')|(pairs<2):
                log.message('  Not a complete pattern, skipping observation', with_header=False)
                exit()
            else:
                wppat_fallback = ''
                if set([0,2]).issubset(usepair): 
                    wppat_fallback = '0426'
                    usepair = [0,2]
                if set([1,3]).issubset(usepair): 
                    wppat_fallback = '1537'
                    usepair = [1,3]
                pairs = 2 
                if wppat_fallback: wppat = 'LINEAR-'+wppat_fallback
                if wppat != 'LINEAR-HI':
                    log.message('  LINEAR-HI pattern truncated, using '+wppat, with_header=False)
                else:
                    log.message('  Not a complete pattern, skipping observation', with_header=False)
                    exit()
     
        minsamples = Samples_p[usepair].min()
        if minsamples<3: Cofs=2
        else: Cofs=3

      # fit normalized stokes pairs vs track time with weighted chebyshev polynomial
        nstokes_Cpw = np.zeros((3,4,wavs)) 
        nvar_Cpw = np.zeros((3,4,wavs))
        ok0_pw = np.zeros((4,wavs),dtype=bool)
        oktr_pw = np.zeros((4,wavs),dtype=bool)

        for p in usepair:
            ispair_i = (pair_i==p)
            oktr_w = ok_iw[ispair_i].sum(axis=0) >= max(Cofs,Samples_p[p]-1)   # track variation: all but 1 pairs good
            w_W = np.where(oktr_w)[0]
            for W in range(oktr_w.sum()):
                ok_i = ok_iw[:,w_W[W]] & oktr_w[w_W[W]]
                Samples = ok_i.sum()
                a_S = t_i[ok_i]
                b_S = nstokes_iw[ok_i,w_W[W]]
                wt_S = 1./np.sqrt(nvar_iw[ok_i,w_W[W]])
                b_S *= wt_S
                a_SC = wt_S[:,None]*np.vstack((np.ones_like(a_S),a_S,2.*a_S**2-1.)).T
                nstokes_Cpw[:Cofs,p,w_W[W]],sumsqerr = la.lstsq(a_SC[:,:Cofs],b_S)[0:2]
                alpha_CC = (a_SC[:,:Cofs,None]*a_SC[:,None,:Cofs]).sum(axis=0)
                eps_CC = la.inv(alpha_CC)
                nvar_Cpw[:Cofs,p,w_W[W]] = (sumsqerr/Samples)*np.diagonal(eps_CC)

            nstokesmean_C = (nstokes_Cpw[:Cofs,p,oktr_w]/np.sqrt(nvar_Cpw[:Cofs,p,oktr_w])).sum(axis=1) /   \
                (1./np.sqrt(nvar_Cpw[:Cofs,p,oktr_w])).sum(axis=1)
            oktr_pw[p] = oktr_w                             # require all pairs

          # recalculate non-variable part (C0) for any wavelength with at least one sample in pair
          # using wavelength-wtdmean of variable part
            Samples_w = ok_iw[ispair_i].sum(axis=0)
            ok0_w = (Samples_w > 0)
            if Cofs==2:
                nstokes_Cpw[0,p,ok0_w] = (nstokes_iw[ispair_i][:,ok0_w] - \
                    nstokesmean_C[1]*t_i[ispair_i,None]).sum(axis=0)/Samples_w[ok0_w]
            else:       
                nstokes_Cpw[0,p,ok0_w] = (nstokes_iw[ispair_i][:,ok0_w] - \
                    nstokesmean_C[1]*t_i[ispair_i,None] - \
                    nstokesmean_C[2]*(2*t_i[ispair_i,None]**2-1)).sum(axis=0)/Samples_w[ok0_w]
            nvar_Cpw[0,p,ok0_w] = nvar_iw[ispair_i][:,ok0_w].sum(axis=0)/Samples_w[ok0_w]**2
            ok0_pw[p] = ok0_w                               # require all pairs         

      # will be needing hw efficiency, PA and telescope PA calibration
        hpar_w = interp1d(hwav_l,hpa_l,kind='cubic',bounds_error=False)(wav_w)
        heff_w = interp1d(hwav_l,heff_l,kind='cubic',bounds_error=False)(wav_w)
        par_w = telpa-hpar_w
        ok0_w = ok0_pw[usepair].all(axis=0)
        oktr_w = oktr_pw[usepair].all(axis=0)

      # combine pairs into stokes, if necessary, and rotate
        nstokes_Cfw = np.zeros((3,2,wavs))
        nvar_Cfw = np.zeros((3,3,wavs))
        if (wppat=='LINEAR') | (wppat=='LINEAR-0426'): 
            nstokes_Cfw = nstokes_Cpw[:,usepair]
            nvar_Cfw[:,:2] = nvar_Cpw[:,usepair]
        elif wppat=='LINEAR-1537':
            nstokes_Cfw = (nstokes_Cpw[:,1] + np.array([-1,1])[:,None]*nstokes_Cpw[:,3])/sqrt(2.)
            nvar_Cfw[:,2] = (nvar_Cpw[:,1] - nvar_Cpw[:,3])/2.
            nvar_Cfw[:,:2] = (nvar_Cpw[:,1] + nvar_Cpw[:,3])/2.
        elif wppat=='LINEAR-HI':
            nstokes_Cfw[:,0] = 0.5*(nstokes_Cpw[:,0] + \
                    (nstokes_Cpw[:,1]-nstokes_Cpw[:,3])/np.sqrt(2.))
            nstokes_Cfw[:,1] = 0.5*(nstokes_Cpw[:,2] + \
                    (nstokes_Cpw[:,1]+nstokes_Cpw[:,3])/np.sqrt(2.))
            nvar_Cfw[:,0] = 0.25*(nvar_Cpw[:,0] + (nvar_Cpw[:,1]+nvar_Cpw[:,3])/2.)
            nvar_Cfw[:,1] = 0.25*(nvar_Cpw[:,2] + (nvar_Cpw[:,1]+nvar_Cpw[:,3])/2.)
            nvar_Cfw[:,2] = 0.25*((nvar_Cpw[:,1] - nvar_Cpw[:,3])/2.)
        for C in range(Cofs):
            nstokes_Cfw[C] /= heff_w
            nvar_Cfw[C] /= heff_w**2
            nstokes_Cfw[C],nvar_Cfw[C] = \
                specpolrotate(nstokes_Cfw[C],nvar_Cfw[C],par_w,normalized=True)

        stokes_Fw = np.zeros((3,wavs))
        var_Fw = np.zeros((4,wavs))
        stokes_Fw[0] = int_w
        var_Fw[0] = var_w
        stokes_Fw[1:] = nstokes_Cfw[0] * int_w
        var_Fw[1:] = nvar_Cfw[0] * int_w**2


      # debug save fits file for track
        if debug:
            infile = infilelist[0]
            hduout = pyfits.open(infile)
            obsname = '_'.join(os.path.basename(infilelist[i]).split('.')[0].rsplit('_')[0:2])                   
            outfile = obsname+'_track0_stokes.fits'
            if (os.path.exists(outfile)): outfile=outfile.replace('track0','track1')
            savetrack(stokes_Fw,var_Fw,ok0_w,nstokes_Cfw,nvar_Cfw,oktr_w,wav_w,     \
                    trkx_d,trky_d,trkrho_d,hduout,outfile)
            log.message('\n    '+outfile+' Stokes I,Q,U,TRQU', with_header=False)

        return stokes_Fw,var_Fw,ok0_w,nstokes_Cfw,nvar_Cfw,oktr_w,trkx_d,trky_d,trkrho_d
# -------------------------------------

def savetrack(stokes_Fw,var_Fw,ok_w,nstokes_cfw,nvar_cfw,oktr_w,wav_w, \
            trkx_d,trky_d,trkrho_d,hduout,outfile):
    """Save fits file with track-center stokes info and normalized Tchebshev C1-2 variation
        X,Y,RHO time variation put in header as polynomial coefficients
    """
    wav0 = wav_w[0]
    hduout['SCI'].data = stokes_Fw.astype('float32').reshape((3,1,-1))
    hduout['SCI'].header.update('CRVAL1',wav0)
    hduout['SCI'].header.update('CTYPE3','I,Q,U')
    hduout['VAR'].data = var_Fw.astype('float32').reshape((4,1,-1))
    hduout['VAR'].header.update('CRVAL1',wav0)
    hduout['VAR'].header.update('CTYPE3','I,Q,U,QU')
    hduout['BPM'].data = np.tile(~ok_w,(3,1)).astype('uint8').reshape((3,1,-1))
    hduout['BPM'].header.update('CRVAL1',wav0)
    hduout['BPM'].header.update('CTYPE3','I,Q,U')
    cofs = (nstokes_cfw.sum(axis=(1,2))!= 0).sum()
    hduout.append(pyfits.ImageHDU(  \
        data=nstokes_cfw[1:cofs].astype('float32').reshape((cofs-1,2,1,-1)),name='TRSCI'))
    hduout['TRSCI'].header.update('CRVAL1',wav0)
    hduout['TRSCI'].header.update('CTYPE3','q,u')
    hduout.append(pyfits.ImageHDU(  \
        data=nvar_cfw[1:cofs].astype('float32').reshape((cofs-1,3,1,-1)),name='TRVAR'))
    hduout['TRVAR'].header.update('CRVAL1',wav0)
    hduout['TRVAR'].header.update('CTYPE3','q,u,qu')
    hduout.append(pyfits.ImageHDU(  \
        data=(~oktr_w).astype('uint8').reshape((1,1,-1)),name='TRBPM'))
    hduout['TRBPM'].header.update('CRVAL1',wav0)
    hduout['TRBPM'].header.update('CTYPE3','qu')
    prihdr = hduout[0].header
    prihdr.update('PATYPE',"Equatorial")
    tcofs = trkx_d.shape[0]
    prihdr.update('TRKXD',(tcofs*'%8.4f ' % tuple(trkx_d)),'TRKX time dependence')
    prihdr.update('TRKYD',(tcofs*'%8.4f ' % tuple(trky_d)),'TRKX time dependence')
    prihdr.update('TRKRHOD',(tcofs*'%8.3f ' % tuple(trkrho_d)),'TRKRHO time dependence')
    hduout.writeto(outfile,clobber=True,output_verify='warn')

    return

# -------------------------------------


if __name__=='__main__':
    idx = sys.argv[1:].index("xx")
    infilelist1 = sys.argv[1:idx+1]
    infilelist2 = sys.argv[idx+2:]
    specpoldualtrack(infilelist1,infilelist2)

# Current debug,
# cd /d/pfis/khn/20160402/sci
# Pypolsalt specpoldualtrack.py ../../20160302/sci/*_c[0-1]_h*.fits xx *_c[0-1]_h*.fits
# Pypolsalt specpoldualtrack.py ../../20160302/sci/*_c[2-3]_h*.fits xx *_c[2-3]_h*.fits
# cd /d/pfis/khn/20160606/sci
# Pypolsalt specpoldualtrack.py ../sci_rising_sc/*_h* xx ../sci_setting_sc/*_h*
# cd /d/pfis/khn/20160709/sci
# Pypolsalt specpoldualtrack.py ../../20160708/sci/*_h*.fits xx *_h*.fits


