
"""
specpolfinalstokes

Correct raw stokes for track, combine, and calibrate to form final stokes.

"""

import os, sys, glob, shutil, inspect
import operator

import numpy as np
from astropy.io import fits as pyfits

from scipy.interpolate import interp1d
from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging

from specpolutils import datedfile, datedline

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

np.set_printoptions(threshold=np.nan)

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
    _j: rawstokes = waveplate position pair index (enumeration within config, including repeats)
    _k: combstokes = waveplate position pair index (enumeration within config, repeats combined)
    _K: pair = waveplate position pair index (enumeration within obs)
    _s: normalized linear stokes for zeropoint correction (0,1) = (q,u) 
    _S: unnormalized raw stokes within waveplate position pair: (0,1) = (I,Q)
    _F: unnormalized final stokes (0,1,2) = (I,Q,U)
    """
    calhistorylist = ["PolCal Model: 20170429",]

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
            object,config,wvplt,cycle = os.path.basename(infilelist[i]).rsplit('.',1)[0].rsplit('_',3)
            if (config[0]!='c')|(wvplt[0]!='h')|(not cycle.isdigit()):
                log.message('File '+infilelist[i]+' is not a raw stokes file.'  , with_header=False) 
                continue
            allrawlist.append([i,object,config,wvplt,cycle])
        configlist = sorted(list(set(ele[2] for ele in allrawlist)))       # unique configs

    # input correct HWCal and TelZeropoint calibration files
        dateobs = obsdict['DATE-OBS'][0].replace('-','')
        HWCalibrationfile = datedfile(datadir+"RSSpol_HW_Calibration_yyyymmdd_vnn.txt",dateobs)
        hwav_l,heff_l,hpa_l = np.loadtxt(HWCalibrationfile,dtype=float,unpack=True,usecols=(0,1,2),ndmin=2)
        TelZeropointfile = datedfile(datadir+"RSSpol_Linear_TelZeropoint_yyyymmdd_vnn.txt",dateobs)
        twav_l,tq0_l,tu0_l,err_l = np.loadtxt(TelZeropointfile,dtype=float,unpack=True,ndmin=2)

    # input PAZeropoint file and get correct entry
        dpadatever,dpa = datedline(datadir+"RSSpol_Linear_PAZeropoint.txt",dateobs).split()
        dpa = float(dpa)

    # prepare calibration keyword documentation            
        pacaltype = "Equatorial"
        if HW_Cal_override: 
            Linear_PolZeropoint_override=True
            PAZeropoint_override=True
            pacaltype = "Instrumental"
            calhistorylist.append("HWCal: Uncalibrated")
        elif Linear_PolZeropoint_override:
            PAZeropoint_override=True
            calhistorylist.extend(["HWCal: "+os.path.basename(HWCalibrationfile),"PolZeropoint: Null"])
        elif PAZeropoint_override: 
            calhistorylist.extend(["HWCal: "+os.path.basename(HWCalibrationfile),  \
                "PolZeropoint: "+os.path.basename(TelZeropointfile), "PAZeropoint: Null"])
        else:
            calhistorylist.extend(["HWCal: "+os.path.basename(HWCalibrationfile),    \
                "PolZeropoint: "+os.path.basename(TelZeropointfile), \
                "PAZeropoint: RSSpol_Linear_PAZeropoint.txt "+str(dpadatever)+" "+str(dpa)])

        log.message('  PA type: '+pacaltype, with_header=False) 
        if len(calhistorylist): log.message('  '+'\n  '.join(calhistorylist), with_header=False) 

    # do one config at a time
        for conf in configlist:
            log.message("\nConfiguration: %s" % conf, with_header=False) 
            rawlist = [entry for entry in allrawlist if entry[2]==conf]
            for col in (4,3,1,2): rawlist = sorted(rawlist,key=operator.itemgetter(col))            
            rawstokes = len(rawlist)            # rawlist is sorted with cycle varying fastest
            wav0 = pyfits.getheader(infilelist[rawlist[0][0]],'SCI')['CRVAL1']
            dwav = pyfits.getheader(infilelist[rawlist[0][0]],'SCI')['CDELT1']
            wavs = pyfits.getheader(infilelist[rawlist[0][0]],'SCI')['NAXIS1']
            wav_w = wav0 + dwav*np.arange(wavs)

        # interpolate HW, telZeropoint calibration wavelength dependence for this config
            okcal_w = np.ones(wavs).astype(bool)
            if not HW_Cal_override:
                heff_w = interp1d(hwav_l,heff_l,kind='cubic',bounds_error=False)(wav_w) 
                hpar_w = -interp1d(hwav_l,hpa_l,kind='cubic',bounds_error=False)(wav_w)
                okcal_w &= ~np.isnan(heff_w) 
                hpar_w[~okcal_w] = 0.
            if not Linear_PolZeropoint_override: 
                tel0_sw = interp1d(twav_l,np.array([tq0_l,tu0_l]),kind='cubic',bounds_error=False)(wav_w)
                okcal_w &= ~np.isnan(tel0_sw[0])
                tel0_sw /= 100.     # table is in % 
          
        # get spectrograph calibration file, spectrograph coordinates 
            grating = pyfits.getheader(infilelist[rawlist[0][0]])['GRATING']
            grang = pyfits.getheader(infilelist[rawlist[0][0]])['GR-ANGLE'] 
            artic = pyfits.getheader(infilelist[rawlist[0][0]])['AR-ANGLE'] 
            SpecZeropointfile = datedfile(datadir+ 
                "RSSpol_Linear_SpecZeropoint_"+grating+"_yyyymmdd_vnn.txt",dateobs)
            if len(SpecZeropointfile): calhistorylist.append(SpecZeropointfile)
          
        # get rawstokes data
            stokes_jSw = np.zeros((rawstokes,2,wavs)) 
            var_jSw = np.zeros_like(stokes_jSw)
            bpm_jSw = np.zeros_like(stokes_jSw).astype(int)
            comblist = []
            for j in range(rawstokes):
                i,object,config,wvplt,cycle = rawlist[j]
                if j==0:
                    cycles = 1
                    lampid = pyfits.getheader(infilelist[i],0)['LAMPID'].strip().upper()
                    telpa = float(pyfits.getheader(infilelist[i],0)['TELPA'])
                    if lampid != "NONE": pacaltype ="Instrumental"
                    if pacaltype == "Equatorial": eqpar_w = hpar_w + (telpa % 180)
                else:
                    if rawlist[j-1][1:4] != rawlist[j][1:4]: cycles = 1
                    else: cycles += 1
                wppat = pyfits.getheader(infilelist[i])['WPPATERN'].upper()
                stokes_jSw[j] = pyfits.open(infilelist[i])['SCI'].data.reshape((2,-1))
                var_jSw[j] = pyfits.open(infilelist[i])['VAR'].data.reshape((2,-1))
                bpm_jSw[j] = pyfits.open(infilelist[i])['BPM'].data.reshape((2,-1))

            # apply telescope zeropoint calibration, q rotated to raw coordinates
                if not Linear_PolZeropoint_override:
                    trkrho = pyfits.getheader(infilelist[i])['TRKRHO']
                    dpatelraw_w = -(22.5*float(wvplt[1]) + hpar_w + trkrho) 
                    rawtel0_sw =    \
                        specpolrotate(tel0_sw,np.zeros((3,wavs)),dpatelraw_w,normalized=True)[0]
                    rawtel0_sw[:,okcal_w] *= heff_w[okcal_w]
                    stokes_jSw[j,1,okcal_w] -= stokes_jSw[j,0,okcal_w]*rawtel0_sw[0,okcal_w]     
                if cycles==1:
                    comblist.append((j,object,config,wvplt,cycles,wppat))
                else:
                    comblist[-1] = (j,object,config,wvplt,cycles,wppat)

        # combine multiple cycles as necessary.  Absolute stokes is on a per cycle basis.
            combstokes = len(comblist)
            stokes_kSw = np.zeros((combstokes,2,wavs)) 
            var_kSw = np.zeros_like(stokes_kSw)
            cycles_kw = np.zeros((combstokes,wavs)).astype(int)
            chisqstokes_kw = np.zeros((combstokes,wavs))
            obslist = []
            obsobject = ''
            obsconfig = ''
            chisqlist = [[]]
            for k in range(combstokes):         
                j,object,config,wvplt,cycles,wppat = comblist[k]
                cycles_kw[k] =  (1-bpm_jSw[j-cycles+1:j+1,0]).sum(axis=0).astype(int)
                ok_w = (cycles_kw[k] > 0)
                stokes_kSw[k] = stokes_jSw[j-cycles+1:j+1].sum(axis=0)
                var_kSw[k] = var_jSw[j-cycles+1:j+1].sum(axis=0)
                stokes_kSw[k,:,ok_w] /= cycles_kw[k,None,ok_w] 
                var_kSw[k,:,ok_w] /= cycles_kw[k,None,ok_w]**2

            # compute chisq/dof for multiple cycles. Use wavelengths where all cycles are good
                okall_w = (cycles_kw[k] == cycles)
                if cycles > 1:
                    nstokes_w = np.zeros(wavs)
                    nstokes_w[okall_w] = stokes_kSw[k,1,okall_w]/stokes_kSw[k,0,okall_w]

                    for jj in range(j-cycles+1,j+1):
                        nstokesj_w = np.zeros(wavs)  
                        nerrstokesj_w = np.zeros_like(nstokesj_w)
                        nstokesj_w[okall_w] = stokes_jSw[jj,1,okall_w]/stokes_jSw[jj,0,okall_w]
                        nerrstokesj_w[okall_w] =  np.sqrt(var_jSw[jj,1,okall_w]/(stokes_jSw[jj,0,okall_w])**2)
                        chisqstokes_kw[k,okall_w] += \
                            ((nstokesj_w[okall_w]-nstokes_w[okall_w])/nerrstokesj_w[okall_w])**2
                    chisqstokes_kw[k] /= cycles-1
                    chisqstokes = chisqstokes_kw[k].sum()/okall_w.sum()
                    chisqlist[-1].append(chisqstokes)
                    log.message("  Chisq/dof Waveplate Position Pair %s: %7.2f" \
                                    % (wvplt,chisqstokes), with_header=False)
                if ((object != obsobject) | (config != obsconfig)):
                    obslist.append([k,object,config,wppat,1])
                    chisqlist.append([])
                    obsobject = object; obsconfig = config
                else:
                    obslist[-1][4] +=1

#            np.savetxt("chisqstokes_kw.txt",chisqstokes_kw.T,fmt="%8.5f")
                                                                     
        # for each obs combine raw stokes, apply efficiency and PA calibration as appropriate for pattern, and save
            obss = len(obslist)

            for obs in range(obss):
                k,object,config,wppat,pairs = obslist[obs]
                obsname = object+"_"+config
                cycles_K = np.array([comblist[i][4] for i in range(k,k+pairs)])
                if cycles_K.max()==1: obsname += '_'+rawlist[comblist[k][0]][4]
                log.message("\n  Observation: %s" % obsname, with_header=False)
                finstokes = patternstokes[wppat]   

            # fallback for incomplete Linear-Hi
                wppat_fallback = ''
                if pairs != patternpairs[wppat]:
                    if (wppat<>'LINEAR-HI')|(pairs<2):
                        log.message('  Not a complete pattern, skipping observation', with_header=False)
                        continue
                    else:
                        hwv_h = [int(comblist[k+h][3][1]) for h in range(pairs)]
                        if hwv_h[0:2] == [0,2]: 
                            wppat_fallback = '0426'
                        if hwv_h[-2:] == [1,3]: 
                            wppat_fallback = '1537'
                            k += pairs-2
                        pairs = 2
                        cycles_K = np.array([comblist[i][4] for i in range(k,k+pairs)])
                        if wppat_fallback: wppat = 'LINEAR-'+wppat_fallback
                        if wppat != 'LINEAR-HI':
                            log.message('  LINEAR-HI pattern truncated, using '+wppat, with_header=False)
                        else:
                            log.message('  Not a complete pattern, skipping observation', with_header=False)

                stokes_fw = np.zeros((finstokes,wavs))
                var_fw = np.zeros_like(stokes_fw)
            # evaluate wavelengths with all pairs having good, calibratable data in at least one cycle
                ok_w = okcal_w & (cycles_kw[k:k+pairs] > 0).all(axis=0) 
                bpm_fw = np.repeat((np.logical_not(ok_w))[None,:],finstokes,axis=0)

            # normalize pairs in obs at wavelengths _W where all pair/cycles have data:
                okall_w = okcal_w & (cycles_kw[k:k+pairs] == cycles_K[:,None]).all(axis=0)     
                normint_K = stokes_kSw[k:k+pairs,0,okall_w].sum(axis=1)
                normint_K /= np.mean(normint_K)
                stokes_kSw[k:k+pairs] /= normint_K[:,None,None]
                var_kSw[k:k+pairs] /= normint_K[:,None,None]**2

            # first, the intensity
                stokes_fw[0] = stokes_kSw[k:k+pairs,0].sum(axis=0)/pairs
                var_fw[0] = var_kSw[k:k+pairs,0].sum(axis=0)/pairs**2        
            # now, the polarization stokes
                if wppat.count('LINEAR'):
                    var_fw = np.vstack((var_fw,np.zeros(wavs)))           # add QU covariance
                    if (wppat=='LINEAR') | (wppat=='LINEAR-0426') | (wppat=='LINEAR-1537'):
                        stokes_fw[1:,ok_w] = stokes_kSw[k:k+2,1,ok_w]*(stokes_fw[0,ok_w]/stokes_kSw[k:k+2,0,ok_w])
                        var_fw[1:3,ok_w] = var_kSw[k:k+2,1,ok_w]*(stokes_fw[0,ok_w]/stokes_kSw[k:k+2,0,ok_w])**2
                        if wppat=='LINEAR-1537':
                            stokes_fw[1:] = (stokes_fw[1] + np.array([-1,1])[:,None]*stokes_fw[2])/np.sqrt(2.)
                            var_fw[3] =   (var_fw[1] - var_fw[2])/2.
                            var_fw[1:3] = (var_fw[1] + var_fw[2])/2.

                    elif wppat=='LINEAR-HI':
                    # for Linear-Hi, must go to normalized stokes in order for the pair combination to cancel systematic errors
                        nstokes_pw = np.zeros((pairs,wavs))
                        nvar_pw = np.zeros((pairs,wavs))
                        nstokes_fw = np.zeros((finstokes,wavs))
                        nvar_fw = np.zeros((finstokes+1,wavs))
                        nstokes_pw[:,ok_w] = stokes_kSw[k:k+pairs,1,ok_w]/stokes_kSw[k:k+pairs,0,ok_w]
                        nvar_pw[:,ok_w] = var_kSw[k:k+pairs,1,ok_w]/(stokes_kSw[k:k+pairs,0,ok_w])**2
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
                        chisqq = ((nstokes_pw[0,okall_w] - nstokes_fw[1,okall_w])**2/nvar_fw[1,okall_w]).sum()/okall_w.sum() 
                        chisqu = ((nstokes_pw[2,okall_w] - nstokes_fw[2,okall_w])**2/nvar_fw[2,okall_w]).sum()/okall_w.sum()
                        chisqlist[obs].append(chisqq)
                        chisqlist[obs].append(chisqu)
                        log.message("    Chisq/dof LINEAR-HI Q,U: %7.2f %7.2f" % (chisqq,chisqu), with_header=False) 

               # calculate, print estimated systematic error from chisq mean
                    if len(chisqlist[obs]):
                        chisqdof = np.array(chisqlist[obs]).mean()
                        dofs = float(okall_w.sum())
                        chisqdoferr = np.sqrt(2./dofs)
                        syserr = 0.         # estimate systematic error using noncentral chisq distribution
                        if (chisqdof - 1.) > 3.*chisqdoferr:
                            nvar_fw = np.zeros_like(var_fw)
                            nvar_fw[:,okall_w] = var_fw[:,okall_w]/stokes_fw[0,okall_w]**2
                            syserr = np.sqrt(dofs*(chisqdof - 1.)/(1./nvar_fw[1,okall_w]).sum())
              
                        log.message(("    Mean chisq/dof: %5.2f  Estimated sys %%error: %5.2f") % \
                            (chisqdof,100.*syserr), with_header=False)

                    if not HW_Cal_override:
                # apply hw efficiency, equatorial PA rotation calibration
                        stokes_fw[1:,ok_w] /= heff_w[ok_w]
                        var_fw[1:,ok_w] /= heff_w[ok_w]**2

                        np.savetxt("stokes_fw_0.txt",np.vstack((wav_w,stokes_fw)).T,fmt="%10.3f")

                        stokes_fw,var_fw = specpolrotate(stokes_fw,var_fw,eqpar_w)

                        np.savetxt("stokes_fw_1.txt",np.vstack((wav_w,stokes_fw)).T,fmt="%10.3f")

                # save final stokes fits file for this observation
                    infile = infilelist[rawlist[comblist[k][0]][0]]
                    hduout = pyfits.open(infile)
                    hduout['SCI'].data = stokes_fw.astype('float32').reshape((3,1,-1))
                    hduout['SCI'].header['CTYPE3'] = 'I,Q,U'
                    hduout['VAR'].data = var_fw.astype('float32').reshape((4,1,-1))
                    hduout['VAR'].header['CTYPE3'] = 'I,Q,U,QU'

                    hduout['BPM'].data = bpm_fw.astype('uint8').reshape((3,1,-1))
                    hduout['BPM'].header['CTYPE3'] = 'I,Q,U'

                    hduout[0].header['WPPATERN'] = wppat
                    hduout[0].header['PATYPE'] = pacaltype
                    if len(calhistorylist):
                        for line in calhistorylist: hduout[0].header.add_history(line)

                    if len(chisqlist[obs]): 
                        hduout[0].header['SYSERR'] = (100.*syserr,'estimated % systematic error')
                    
                    outfile = obsname+'_'+wppat_fallback+'stokes.fits'
                    hduout.writeto(outfile,clobber=True,output_verify='warn')
                    log.message('\n    '+outfile+' Stokes I,Q,U', with_header=False)
                     
#               elif wppat.count('CIRCULAR'):  TBS 

#               elif wppat=='ALL-STOKES':  TBS
            # end of obs loop
        # end of config loop
    return 

# ------------------------------------
def specpolrotate(stokes_Sw,var_Sw,par_w,normalized=False):
    """ rotate linear polarization in stokes,variance cubes

    Parameters
    ----------
    stokes_Sw: 2d np array
        _S = I,Q,U,(optional V) unnormalized stokes (size 3, or 4)
        _w = wavelength
    var_Sw: 2d np array (size 4, or 5)
        _S = I,Q,U,QU covariance, (optional V) variance for stokes
    par_w: 1d np array 
        PA(degrees) to rotate
    normalized: if True, there is no I

    Returns stokes, var (as copy)

    """

    Qarg = int(not normalized)
    stokes_Fw = np.copy(stokes_Sw)
    var_Fw = np.copy(var_Sw)
    c_w = np.cos(2.*np.radians(par_w))
    s_w = np.sin(2.*np.radians(par_w))
    stokes_Fw[Qarg:] = stokes_Fw[Qarg]*c_w - stokes_Fw[Qarg+1]*s_w ,    \
        stokes_Fw[Qarg]*s_w + stokes_Fw[Qarg+1]*c_w
    var_Fw[Qarg:Qarg+2] =  var_Fw[Qarg]*c_w**2 + var_Fw[Qarg+1]*s_w**2 ,    \
        var_Fw[Qarg]*s_w**2 + var_Fw[Qarg+1]*c_w**2
    var_Fw[Qarg+2] =  c_w*s_w*(var_Fw[Qarg] - var_Fw[Qarg+1]) + (c_w**2-s_w**2)*var_Fw[Qarg+2]
    return stokes_Fw,var_Fw

if __name__=='__main__':
    infilelist=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)
    if len(kwargs): kwargs = {k:bool(v) for k,v in kwargs.iteritems()}        
    specpolfinalstokes(infilelist,**kwargs)
