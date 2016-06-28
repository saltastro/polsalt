
"""
specpolfinalstokes

Correct raw stokes for track, combine, and calibrate to form final stoes.

"""

import os, sys, glob, shutil, inspect
import operator

import numpy as np
from astropy.io import fits as pyfits
#import pyfits

from scipy.interpolate import interp1d
from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

np.set_printoptions(threshold=np.nan)
debug = True

# ------------------------------------
def datedfile(filename,date):
    """ select file based on observation date

    Parameters
    ----------
    filename: text file name pattern, including "yyyymmdd" place holder for date
    date: yyyymmdd of observation

    Returns: file name

    """

    filelist = sorted(glob.glob(filename.replace('yyyymmdd','????????')))
    dateoffs = filename.find('yyyymmdd')
    datelist = [file[dateoffs:dateoffs+8] for file in filelist]
    file = filelist[0]
    for (f,fdate) in enumerate(datelist):
        if date < fdate: continue
        file = filelist[f] 
    return file      

# ------------------------------------
def specpolrotate(stokes_sw,var_sw,par_w):
    """ rotate linear polarization in stokes,variance cubes

    Parameters
    ----------
    stokes_sw: 2d np array
        _s = I,Q,U,(optional V) unnormalized stokes (size 3, or 4)
        _w = wavelength
    Var_sw: 2d np array (size 4, or 5)
        _s = I,Q,U,QU covariance, (optional V) variance for stokes

    Returns stokes, var (as copy)

    """

    stokes_fw = np.copy(stokes_sw)
    var_fw = np.copy(var_sw)
    c_w = np.cos(2.*np.radians(par_w)); s_w = np.sin(2.*np.radians(par_w))
    stokes_fw[1:] = stokes_fw[1]*c_w - stokes_fw[2]*s_w ,    \
        stokes_fw[1]*s_w + stokes_fw[2]*c_w
    var_fw[1:3] =  var_fw[1]*c_w**2 + var_fw[2]*s_w**2 ,    \
        var_fw[1]*s_w**2 + var_fw[2]*c_w**2
    var_fw[3] =  c_w*s_w*(var_fw[1] - var_fw[2]) + (c_w**2-s_w**2)*var_fw[3]
    return stokes_fw,var_fw

# -------------------------------------
def specpolfinalstokes(infilelist,logfile='salt.log',debug=False,  \
        HW_Cal_override=False,Linear_PolZeropoint_override=False,PAZeropoint_override=False):
    """Combine the raw stokes and apply the polarimetric calibrations

    Parameters
    ----------
    infilelist: list
        List of filenames that include an extracted spectrum

    logfile: str
        Name of file for logging

    """
    """
    _l: line in calibration file
    _i: index in file list
    _j: rawstokes = waveplate position pair index (total in config, including repeats)
    _k: combstokes = waveplate position pair index (unique, so i.e, 0-3 for Linear-Hi)
    _s: stokes within waveplate position pair: (0,1)
    """

    patternlist = open(datadir+'wppaterns.txt','r').readlines()
    patternpairs = dict();  patternstokes = dict()
    for p in patternlist:
        if p.split()[0] == '#': continue
        patternpairs[p.split()[0]]=(len(p.split())-3)/2
        patternstokes[p.split()[0]]=int(p.split()[1])

    with logging(logfile, debug) as log:
        
    # organize data using names
        obsdict=obslog(infilelist)
        files = len(infilelist)
        allrawlist = []
        for i in range(files):
            object,config,wvplt,cycle = os.path.basename(infilelist[i]).split('.')[0].rsplit('_',4)
            if (config[0]!='c')|(wvplt[0]!='h')|(not cycle.isdigit()):
                log.message('File '+infilelist[i]+' is not a raw stokes file.'  , with_header=False) 
                continue
            allrawlist.append([i,object,config,wvplt,cycle])
        configlist = sorted(list(set(ele[2] for ele in allrawlist)))       # unique configs

    # input calibration files
        dateobs = obsdict['DATE-OBS'][0].strip('-')
        HWCalibrationfile = datedfile(datadir+"RSSpol_HW_Calibration_yyyymmdd.txt",dateobs)
        hwav_l,heff_l,hpa_l = np.loadtxt(HWCalibrationfile,dtype=float,unpack=True,ndmin=2)
        PolZeropointfile = datedfile(datadir+"RSSpol_Linear_PolZeropoint_yyyymmdd.txt",dateobs)
        iwav_l,iq0_l,iu0_l = np.loadtxt(PolZeropointfile,dtype=float,unpack=True,ndmin=2)
        tdate_l,t0_l = np.loadtxt(datadir+"RSSpol_Linear_PAZeropoint.txt",dtype=float,unpack=True,ndmin=2)
        t0 = t0_l[0]
        for (l,td) in enumerate(tdate_l):
            if dateobs < td: break
            tdate = td
            t0 = t0_l[l]
            
        pacaltype = "Equatorial"
        calhistorylist = [] 
        if HW_Cal_override: 
            Linear_PolZeropoint_override=True
            PAZeropoint_override=True
            pacaltype = "Instrumental"
            calhistorylist = ["Uncalibrated"]
        elif Linear_PolZeropoint_override:
            HW_Cal_override=True
            PAZeropoint_override=True
            pacaltype = "Telescope"
            calhistorylist = ["Uncalibrated"]
        elif PAZeropoint_override: 
            calhistorylist = [os.path.basename(HWCalibrationfile),os.path.basename(PolZeropointfile)]
        else:
            calhistorylist = [os.path.basename(HWCalibrationfile),    \
                os.path.basename(PolZeropointfile),"RSSpol_Linear_PAZeropoint.txt "+str(tdate)]

        log.message('  PA type: '+pacaltype, with_header=False) 
        if len(calhistorylist): log.message('  '+'\n  '.join(calhistorylist), with_header=False) 

    # correct raw stokes for track (TBS)

    # do one config at a time, since different configs may have different number of wavelengths
        for conf in configlist:
            log.message("\nConfiguration: %s" % conf, with_header=False) 
            rawlist = [entry for entry in allrawlist if entry[2]==conf]
            for col in (4,3,1,2): rawlist = sorted(rawlist,key=operator.itemgetter(col))            
            rawstokes = len(rawlist)            # rawlist is sorted with cycle varying fastest
            cols = pyfits.open(infilelist[rawlist[0][0]])['SCI'].data.shape[-1]
            stokes_jsw = np.zeros((rawstokes,2,cols)) 
            var_jsw = np.zeros_like(stokes_jsw); bpm_jsw = np.zeros_like(stokes_jsw).astype(int)
            wav_jw = np.zeros((rawstokes,cols))
            comblist = []
        # get data
            for j in range(rawstokes):
                i,object,config,wvplt,cycle = rawlist[j]
                if j==0:
                    cycles = 1
                    lampid = pyfits.getheader(infilelist[i],0)['LAMPID'].strip().upper()
                    telpa = float(pyfits.getheader(infilelist[i],0)['TELPA'])
                    if lampid != "NONE": pacaltype ="Instrumental"
                    if pacaltype == "Equatorial": hpa_l -= (telpa % 180)
                else:
                    if rawlist[j-1][1:4] != rawlist[j][1:4]: cycles = 1
                    else: cycles += 1
           
                wppat = pyfits.getheader(infilelist[i],0)['WPPATERN']
                wav0 = pyfits.getheader(infilelist[i],'SCI')['CRVAL1']
                dwav = pyfits.getheader(infilelist[i],'SCI')['CDELT1']
                stokes_jsw[j] = pyfits.open(infilelist[i])['SCI'].data.reshape((2,-1))
                var_jsw[j] = pyfits.open(infilelist[i])['VAR'].data.reshape((2,-1))
                bpm_jsw[j] = pyfits.open(infilelist[i])['BPM'].data.reshape((2,-1))
                wav_jw[j] = wav0 + dwav*np.arange(cols)
                if cycles==1:
                    comblist.append((j,object,config,wvplt,cycles,wppat))
                else:
                    comblist[-1] = (j,object,config,wvplt,cycles,wppat)

        # combine cycles (cycles > 1)
            combstokes = len(comblist)
            stokes_ksw = np.zeros((combstokes,2,cols)); 
            var_ksw = np.zeros_like(stokes_ksw)
            bpm_ksw = np.zeros_like(stokes_ksw).astype(int)
            wav_kw = np.zeros((combstokes,cols))
            chisqstokes_kw = np.zeros_like(wav_kw)
            obslist = []
            obsobject = ''
            obsconfig = ''
            chisqlist = [[]]
            for k in range(combstokes):         
                j,object,config,wvplt,cycles,wppat = comblist[k]
                stokes_ksw[k] =  stokes_jsw[j-cycles+1:j+1].sum(axis=0) 
                var_ksw[k] =  var_jsw[j-cycles+1:j+1].sum(axis=0)   
                bpm_ksw[k] =  bpm_jsw[j-cycles+1:j+1].sum(axis=0).astype(int)
                wav_kw[k] = wav_jw[j]

            # compute chisq/dof for multiple cycles
                if cycles > 1:
                    nstokes_w = np.zeros(cols)
                    okchi = (bpm_ksw[k,1] == 0)                     # use waves where all cycles are good 
                    nstokes_w[okchi] = stokes_ksw[k,1,okchi]/stokes_ksw[k,0,okchi]

                    for jj in range(j-cycles+1,j+1):
                        nstokesj_w = np.zeros(cols)  
                        nerrstokesj_w = np.zeros_like(nstokesj_w)
                        nstokesj_w[okchi] = stokes_jsw[jj,1,okchi]/stokes_jsw[jj,0,okchi]
                        nerrstokesj_w[okchi] =  np.sqrt(var_jsw[jj,1,okchi]/(stokes_jsw[jj,0,okchi])**2)
                        chisqstokes_kw[k,okchi] += ((nstokesj_w[okchi]-nstokes_w[okchi])/nerrstokesj_w[okchi])**2
                    chisqstokes_kw[k] /= cycles-1
                    chisqstokes = chisqstokes_kw[k].sum()/okchi.sum()
                    chisqlist[-1].append(chisqstokes)
                    log.message("  Chisq/dof Waveplate Position Pair %s: %7.2f" \
                                    % (wvplt,chisqstokes), with_header=False)
                if ((object != obsobject) | (config != obsconfig)):
                    obslist.append([k,object,config,wppat,1])
                    chisqlist.append([])
                    obsobject = object; obsconfig = config
                else:
                    obslist[-1][4] +=1

            np.savetxt("chisqstokes_kw.txt",chisqstokes_kw.T,fmt="%8.5f")
                                                                     
        # for each obs combine stokes, apply efficiency and PA calibration as appropriate for pattern, and save
            obss = len(obslist)
            for obs in range(obss):
                k,object,config,wppat,pairs = obslist[obs]
                obsname = object+"_"+config
                if cycles==1: obsname += '_'+cycle
                log.message("\n  Observation: %s" % obsname, with_header=False)
                finstokes = patternstokes[wppat]
                if pairs != patternpairs[wppat]:
                    if (wppat<>'Linear-Hi')|((wppat=='Linear-Hi')&((k>0)|(pairs<2))):
                        log.message('  Not a complete pattern, skipping observation', with_header=False)
                        continue
                    wppat='Linear'
                    log.message('  Linear-Hi pattern truncated, using Linear', with_header=False)                    
                stokes_fw = np.zeros((finstokes,cols))
                var_fw = np.zeros_like(stokes_fw)
                ok_fw = (bpm_ksw[k:k+pairs,:] < cycles).sum(axis=0) == pairs
                ok_w = ok_fw.all(axis=0)    # use wavelengths with all pairs having good data in at least one cycle
                bpm_fw = np.repeat((np.logical_not(ok_w))[None,:],finstokes,axis=0)

            # normalize pairs at matching wavelengths _W: 
            # compute ratios at each wavelength, then error-weighted mean of ratio
                normint_kW = stokes_ksw[:,0,ok_w]/stokes_ksw[:,0,ok_w].mean(axis=0)
                varnorm_kW = var_ksw[:,0,ok_w]/stokes_ksw[:,0,ok_w].mean(axis=0)**2
                normint_k = (normint_kW/varnorm_kW).sum(axis=1) / (1./varnorm_kW).sum(axis=1)
                stokes_ksw /= normint_k[:,None,None]
                var_ksw /= normint_k[:,None,None]**2

            # first, the intensity
                stokes_fw[0] = stokes_ksw[k:k+pairs,0].sum(axis=0)/pairs
                var_fw[0] = var_ksw[k:k+pairs,0].sum(axis=0)/pairs**2        
            # now, the polarization stokes
                if wppat.count('Linear'):
                    var_fw = np.vstack((var_fw,np.zeros(cols)))           # add QU covariance
                    if wppat=='Linear':
                        stokes_fw[1:,ok_w] = stokes_ksw[k:k+2,1,ok_w]*(stokes_fw[0,ok_w]/stokes_ksw[k:k+2,0,ok_w])
                        var_fw[1:3,ok_w] = var_ksw[k:k+2,1,ok_w]*(stokes_fw[0,ok_w]/stokes_ksw[k:k+2,0,ok_w])**2
                    elif wppat=='Linear-Hi':
                # for Linear-Hi, must go to normalized stokes in order for the pair combination to cancel systematic errors
                        nstokes_pw = np.zeros((pairs,cols)); nvar_pw = np.zeros((pairs,cols))
                        nstokes_fw = np.zeros((finstokes,cols)); nvar_fw = np.zeros((finstokes+1,cols))
                        nstokes_pw[:,ok_w] = stokes_ksw[k:k+pairs,1,ok_w]/stokes_ksw[k:k+pairs,0,ok_w]
                        nvar_pw[:,ok_w] = var_ksw[k:k+pairs,1,ok_w]/(stokes_ksw[k:k+pairs,0,ok_w])**2
                        if debug:                        
                            np.savetxt(obsname+"_nstokes.txt",np.vstack((ok_w.astype(int),nstokes_pw)).T,fmt="%3i "+4*"%10.6f ")
                            np.savetxt(obsname+"_nvar.txt",np.vstack((ok_w.astype(int),nvar_pw)).T,fmt="%3i "+4*"%14.9f ")
                        nstokes_fw[1] = 0.5*(nstokes_pw[0] + (nstokes_pw[1]-nstokes_pw[3])/np.sqrt(2.))
                        nstokes_fw[2] = 0.5*(nstokes_pw[2] + (nstokes_pw[1]+nstokes_pw[3])/np.sqrt(2.))
                        nvar_fw[1] = 0.25*(nvar_pw[0] + (nvar_pw[1]+nvar_pw[3])/2.)
                        nvar_fw[2] = 0.25*(nvar_pw[2] + (nvar_pw[1]+nvar_pw[3])/2.)
                        nvar_fw[3] = 0.25*((nvar_pw[1] - nvar_pw[3])/2.)
                        stokes_fw[1:] = nstokes_fw[1:]*stokes_fw[0]
                        var_fw[1:] = nvar_fw[1:]*stokes_fw[0]**2
                        chisqq = ((nstokes_pw[0,ok_w] - nstokes_fw[1,ok_w])**2/nvar_fw[1,ok_w]).sum()/ok_w.sum() 
                        chisqu = ((nstokes_pw[2,ok_w] - nstokes_fw[2,ok_w])**2/nvar_fw[2,ok_w]).sum()/ok_w.sum()
                        chisqlist[obs].append(chisqq)
                        chisqlist[obs].append(chisqu)
                        log.message("    Chisq/dof Linear-Hi Q,U: %7.2f %7.2f" % (chisqq,chisqu), with_header=False) 

               # calculate, print estimated systematic error from chisq mean
                    if len(chisqlist[obs]):
                        chisqdof = np.array(chisqlist[obs]).mean()
                        dofs = float(ok_fw[0].sum())
                        chisqdoferr = np.sqrt(2./dofs)
                        syserr = 0.         # estimate systematic error using noncentral chisq distribution
                        if (chisqdof - 1.) > 3.*chisqdoferr:
                            nvar_fw = np.zeros_like(var_fw)
                            nvar_fw[:,ok_fw[0]] = var_fw[:,ok_fw[0]]/stokes_fw[0,ok_fw[0]]**2
                            syserr = np.sqrt(dofs*(chisqdof - 1.)/(1./nvar_fw[1,ok_fw[1]]).sum())
              
                        log.message(("    Mean chisq/dof: %5.2f  Estimated sys %%error: %5.2f") % \
                            (chisqdof,100.*syserr), with_header=False)

                    if not HW_Cal_override:
                    # apply hw efficiency calibration
                        heff_w = interp1d(hwav_l,heff_l,kind='cubic',bounds_error=False)(wav_kw[k])
                        ok_w &= ~np.isnan(heff_w)
                        stokes_fw[1:,ok_w] /= heff_w[ok_w]
                        var_fw[1:,ok_w] /= heff_w[ok_w]**2
                    # apply hw PA rotation calibration
                        par_w = -interp1d(hwav_l,hpa_l,kind='cubic',bounds_error=False)(wav_kw[k])
                        par_w[~ok_w] = 0.
                        stokes_fw,var_fw = specpolrotate(stokes_fw,var_fw,par_w)

                # save final stokes fits file
                    infile = infilelist[rawlist[comblist[k][0]][0]]
                    hduout = pyfits.open(infile)
                    hduout['SCI'].data = stokes_fw.astype('float32').reshape((3,1,-1))
                    hduout['SCI'].header.update('CTYPE3','I,Q,U')
                    hduout['VAR'].data = var_fw.astype('float32').reshape((4,1,-1))
                    hduout['VAR'].header.update('CTYPE3','I,Q,U,QU')

                    hduout['BPM'].data = bpm_fw.astype('uint8').reshape((3,1,-1))
                    hduout['BPM'].header.update('CTYPE3','I,Q,U')

                    hduout[0].header.update('PATYPE',pacaltype)
                    if len(calhistorylist): 
                        hduout[0].header.add_history('POLCAL: '+' '.join(calhistorylist))

                    if len(chisqlist[obs]): 
                        hduout[0].header.update('SYSERR',100.*syserr, \
                            'estimated % systematic error')
                    outfile = obsname+'_stokes.fits'
                    hduout.writeto(outfile,clobber=True,output_verify='warn')
                    log.message('\n    '+outfile+' Stokes I,Q,U', with_header=False)
                     
#               elif wppat.count('Circular'):  TBS 

#               elif wppat=='All-Stokes':  TBS

    return

if __name__=='__main__':
    infilelist=sys.argv[1:]
    specpolfinalstokes(infilelist)
