
"""
specpolfinalstokes

Correct raw stokes for track, combine, and calibrate to form final stokes.

"""

import os, sys, glob, shutil, inspect
import operator

import numpy as np
from astropy.io import fits as pyfits

from scipy.interpolate import interp1d
from scipy import linalg as la
from scipy.stats import norm
from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging

import warnings
# warnings.simplefilter("error")

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

import specpolview as spv
from specpolutils import datedfile, datedline
from specpolflux import specpolflux

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
    _J: cycle number idx (0,1,..) for each rawstokes
    _k: combstokes = waveplate position pair index (enumeration within config, repeats combined)
    _K: pair = waveplate position pair index (enumeration within obs)
    _p: pair = waveplate position pair # (eg 0,1,2,3 = 0 4  1 5  2 6  3 7 for LINEAR-HI, sorted in h0 order)
    _s: normalized linear stokes for zeropoint correction (0,1) = (q,u) 
    _S: unnormalized raw stokes within waveplate position pair: (eg 0,1 = I,Q)
    _F: unnormalized final stokes (eg 0,1,2 = I,Q,U)
    """
    calhistorylist = ["PolCal Model: 20170429",]

    patternlist = open(datadir+'wppaterns.txt','r').readlines()
    patternpairs = dict();  patternstokes = dict(); patterndict = dict()
    for p in patternlist:
        if p.split()[0] == '#': continue
        patterndict[p.split()[0]]=np.array(p.split()[3:]).astype(int).reshape((-1,2))
        patternpairs[p.split()[0]]=(len(p.split())-3)/2
        patternstokes[p.split()[0]]=int(p.split()[1])

    if len(glob.glob('specpol*.log')): logfile=glob.glob('specpol*.log')[0]
    with logging(logfile, debug) as log:

        log.message('specpolfinalstokes version: 20171226', with_header=False)          
    # organize data using names. 
    #   allrawlist = infileidx,object,config,wvplt,cycle for each infile.
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

        chifence_d = 2.2*np.array([6.43,4.08,3.31,2.91,2.65,2.49,2.35,2.25])    # *q3 for upper outer fence outlier for each dof

    # do one config at a time.  
    #   rawlist = infileidx,object,config,wvplt,cycle for each infile *in this config*. 
    #   rawlist is sorted with cycle varying fastest
    #   rawstokes = len(rawlist).   j is idx in rawlist.  

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
          
        # get all rawstokes data
        #   comblist = last rawlistidx,object,config,wvplt,cycles,wppat 
        #   one entry for each set of cycles that needs to be combined (i.e, one for each wvplt)
            stokes_jSw = np.zeros((rawstokes,2,wavs)) 
            var_jSw = np.zeros_like(stokes_jSw)
            covar_jSw = np.zeros_like(stokes_jSw)
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
              # if object,config,wvplt changes, start a new comblist entry
                else:   
                    if rawlist[j-1][1:4] != rawlist[j][1:4]: cycles = 1
                    else: cycles += 1
                wppat = pyfits.getheader(infilelist[i])['WPPATERN'].upper()
                stokes_jSw[j] = pyfits.open(infilelist[i])['SCI'].data.reshape((2,-1))
                var_jSw[j] = pyfits.open(infilelist[i])['VAR'].data.reshape((2,-1))
                covar_jSw[j] = pyfits.open(infilelist[i])['COV'].data.reshape((2,-1))
                bpm_jSw[j] = pyfits.open(infilelist[i])['BPM'].data.reshape((2,-1))

            # apply telescope zeropoint calibration, q rotated to raw coordinates
                if not Linear_PolZeropoint_override:
                    trkrho = pyfits.getheader(infilelist[i])['TRKRHO']
                    dpatelraw_w = -(22.5*float(wvplt[1]) + hpar_w + trkrho) 
                    rawtel0_sw =    \
                        specpolrotate(tel0_sw,0,0,dpatelraw_w,normalized=True)[0]
                    rawtel0_sw[:,okcal_w] *= heff_w[okcal_w]
                    stokes_jSw[j,1,okcal_w] -= stokes_jSw[j,0,okcal_w]*rawtel0_sw[0,okcal_w]     
                if cycles==1:
                    comblist.append((j,object,config,wvplt,1,wppat))
                else:
                    comblist[-1] = (j,object,config,wvplt,cycles,wppat)

        # combine multiple cycles as necessary.  Absolute stokes is on a per cycle basis.
        # polarimetric combination on normalized stokes basis 
        #  to avoid coupling mean syserr into polarimetric spectral features
            combstokess = len(comblist)
            stokes_kSw = np.zeros((combstokess,2,wavs)) 
            var_kSw = np.zeros_like(stokes_kSw)
            covar_kSw = np.zeros_like(stokes_kSw)
            cycles_kw = np.zeros((combstokess,wavs)).astype(int)
            chi2cycle_kw = np.zeros((combstokess,wavs))
            badcyclechi_kw = np.zeros((combstokess,wavs),dtype=bool)
            havecyclechi_k = np.zeros(combstokess,dtype=bool)

          # obslist = first comblist idx,object,config,wppat,pairs
          # k = idx in comblist

            obslist = []
            jlistk = []             # list of rawstokes idx for each comblist entry
            Jlistk = []             # list of cycle number for each comblist entry

            obsobject = ''
            obsconfig = ''
            chi2cycle_j = np.zeros(rawstokes)
            syserrcycle_j = np.zeros(rawstokes)
            iscull_jw = np.zeros((rawstokes,wavs),dtype=bool)
            stokes_kSw = np.zeros((combstokess,2,wavs))
            var_kSw = np.zeros_like(stokes_kSw)
            nstokes_kw = np.zeros((combstokess,wavs))
            nvar_kw = np.zeros_like(nstokes_kw)
            ncovar_kw = np.zeros_like(nstokes_kw)
            chi2cyclenet_k = np.zeros(combstokess)
            syserrcyclenet_k = np.zeros(combstokess)

            for k in range(combstokess):         
                j,object,config,wvplt,cycles,wppat = comblist[k]
                jlistk.append(range(j-cycles+1,j+1))                                
                Jlistk.append([int(rawlist[jj][4])-1 for jj in range(j-cycles+1,j+1)])  # J = cycle-1, counting from 0        
                nstokes_Jw = np.zeros((cycles,wavs))
                nvar_Jw = np.zeros((cycles,wavs))
                ncovar_Jw = np.zeros((cycles,wavs))
                bpm_Jw = np.zeros((cycles,wavs))
                ok_Jw = np.zeros((cycles,wavs),dtype=bool)

                for J,j in enumerate(jlistk[k]):
                    bpm_Jw[J] = bpm_jSw[j,0]
                    ok_Jw[J] = (bpm_Jw[J] ==0)
                    nstokes_Jw[J][ok_Jw[J]] = stokes_jSw[j,1][ok_Jw[J]]/stokes_jSw[j,0][ok_Jw[J]]
                    nvar_Jw[J][ok_Jw[J]] = var_jSw[j,1][ok_Jw[J]]/(stokes_jSw[j,0][ok_Jw[J]])**2
                    ncovar_Jw[J][ok_Jw[J]] = covar_jSw[j,1][ok_Jw[J]]/(stokes_jSw[j,0][ok_Jw[J]])**2

            # Culling:  for multiple cycles, compare each cycle with every other cycle (dof=1).
            # bad wavelengths flagged for P < .02% (1/2000): chisq  > 13.8  (chi2.isf(q=.0002,df=1))
            # for cycles>2, vote to cull specific pair/wavelength, otherwise cull wavelength

                cycles_kw[k] =  (1-bpm_Jw).sum(axis=0).astype(int)
                okchi_w = (cycles_kw[k] > 1)
                chi2lim = 13.8 
                havecyclechi_k[k] = okchi_w.any()
                if cycles > 1:
                    ok_Jw[J] = okchi_w & (bpm_Jw[J] ==0)
                    chi2cycle_JJw = np.zeros((cycles,cycles,wavs))
                    badcyclechi_JJw = np.zeros((cycles,cycles,wavs))
                    ok_JJw = ok_Jw[:,None,:] & ok_Jw[None,:,:] 
                    nstokes_JJw = nstokes_Jw[:,None] - nstokes_Jw[None,:]
                    nvar_JJw = nvar_Jw[:,None] + nvar_Jw[None,:]                           
                    chi2cycle_JJw[ok_JJw] = nstokes_JJw[ok_JJw]**2/nvar_JJw[ok_JJw]

                    triuidx = np.triu_indices(cycles,1)                 # _i enumeration of cycle differences
                    chi2cycle_iw = chi2cycle_JJw[triuidx]
                    badcyclechi_w = (chi2cycle_iw > chi2lim).any(axis=(0))
                    badcyclechiall_w = (badcyclechi_w & (ok_JJw[triuidx].reshape((-1,wavs)).sum(axis=0)<3))
                    badcyclechicull_w = (badcyclechi_w & np.logical_not(badcyclechiall_w))

                    wavcull_W = np.where(badcyclechicull_w)[0]          # cycles>2, cull by voting
                    if wavcull_W.shape[0]:
                        for W,w in enumerate(wavcull_W):                       
                            J_I = np.array(triuidx).T[np.argsort(chi2cycle_iw[:,w])].flatten()
                            _,idx = np.unique(J_I,return_index=True)
                            Jcull = J_I[np.sort(idx)][-1]
                            jcull = jlistk[k][Jcull] 
                            iscull_jw[jcull,w] = True                   # for reporting
                            bpm_jSw[jcull,:,w] = 1
                    else:
                        for j in jlistk[k]:
                            iscull_jw[j] = badcyclechiall_w             # for reporting
                            bpm_jSw[j][:,badcyclechiall_w] = 1
                    for J,j in enumerate(jlistk[k]):
                        bpm_Jw[J] = bpm_jSw[j,0]

                    if debug:
                        obsname = object+"_"+config 
                        ok_Jw = okchi_w[None,:] & (bpm_Jw ==0)
                        np.savetxt(obsname+"_nstokes_Jw_"+str(k)+".txt",np.vstack((wav_w,ok_Jw.astype(int),    \
                            nstokes_Jw,nvar_Jw)).T, fmt="%8.2f "+cycles*"%3i "+cycles*"%10.6f "+cycles*"%10.12f ")                        
                        np.savetxt(obsname+"_chi2cycle_iw_"+str(k)+".txt",np.vstack((wav_w,okchi_w.astype(int),    \
                            chi2cycle_iw.reshape((-1,wavs)),badcyclechi_w,ok_JJw[triuidx].reshape((-1,wavs)).sum(axis=0))).T, \
                            fmt="%8.2f %3i "+chi2cycle_iw.shape[0]*"%10.7f "+" %2i %2i") 
                        np.savetxt(obsname+"_Jcull_kw_"+str(k)+".txt",np.vstack((wav_w,okchi_w.astype(int),    \
                            iscull_jw[jlistk[k]].astype(int).reshape((-1,wavs)))).T, fmt="%8.2f %3i "+cycles*" %3i") 

                if ((object != obsobject) | (config != obsconfig)):
                    obslist.append([k,object,config,wppat,1])
                    obsobject = object; obsconfig = config
                else:
                    obslist[-1][4] +=1

          # Now combine cycles, using normalized stokes to minimize systematic errors

            # first normalize cycle members J at wavelengths where all cycles have data:
                cycles_kw[k] =  (1-bpm_Jw).sum(axis=0).astype(int)
                ok_w = (cycles_kw[k] > 0)
                okall_w = (cycles_kw[k] == cycles)
                normint_J = np.array(stokes_jSw[jlistk[k],0][:,okall_w].sum(axis=1))
                normint_J /= np.mean(normint_J)
                stokes_JSw = stokes_jSw[jlistk[k]]/normint_J[:,None,None]
                var_JSw = var_jSw[jlistk[k]]/normint_J[:,None,None]**2
                covar_JSw = covar_jSw[jlistk[k]]/normint_J[:,None,None]**2

                for J in range(cycles):
                    okJ_w = ok_w & (bpm_Jw[J] ==0)
                  # average the intensity
                    stokes_kSw[k,0,okJ_w] += stokes_JSw[J,0,okJ_w]/cycles_kw[k][okJ_w]
                    var_kSw[k,0,okJ_w] += var_JSw[J,0,okJ_w]/cycles_kw[k][okJ_w]**2
                    covar_kSw[k,0,okJ_w] += covar_JSw[J,0,okJ_w]/cycles_kw[k][okJ_w]**2
                  # now the normalized stokes
                    nstokes_kw[k][okJ_w] += (stokes_JSw[J,1][okJ_w]/stokes_JSw[J,0][okJ_w])/cycles_kw[k][okJ_w]
                    nvar_kw[k][okJ_w] += (var_JSw[J,1][okJ_w]/stokes_JSw[J,0][okJ_w]**2)/cycles_kw[k][okJ_w]**2
                    ncovar_kw[k][okJ_w] += (covar_JSw[J,1][okJ_w]/stokes_JSw[J,0][okJ_w]**2)/cycles_kw[k][okJ_w]**2
                stokes_kSw[k,1] = nstokes_kw[k]*stokes_kSw[k,0]
                var_kSw[k,1] = nvar_kw[k]*stokes_kSw[k,0]**2 
                covar_kSw[k,1] = ncovar_kw[k]*stokes_kSw[k,0]**2         
                if debug:
                    obsname = object+"_"+config 
                    np.savetxt(obsname+"_stokes_kSw_"+str(k)+".txt",np.vstack((wav_w,ok_w.astype(int),    \
                            stokes_kSw[k])).T, fmt="%8.2f %3i "+2*"%12.3f ")                    

            # compute mean chisq for each pair having multiple cycles  
                if cycles > 1:
                    nstokeserr_Jw = np.zeros((cycles,wavs))
                    nerr_Jw = np.zeros((cycles,wavs))
                    for J in range(cycles):
                        okJ_w = ok_w & (bpm_Jw[J] ==0)
                        nstokes_Jw[J][okJ_w] = stokes_JSw[J,1][okJ_w]/stokes_JSw[J,0][okJ_w]
                        nvar_Jw[J][okJ_w] = var_JSw[J,1][okJ_w]/(stokes_JSw[J,0][okJ_w])**2                    
                        nstokeserr_Jw[J] = (nstokes_Jw[J] - nstokes_kw[k])
                        nvar_w = nvar_Jw[J] - nvar_kw[k]
                        okall_w &= (nvar_w > 0.)
                        nerr_Jw[J,okall_w] = np.sqrt(nvar_w[okall_w])

                    nstokessyserr_J = np.average(nstokeserr_Jw[:,okall_w],weights=1./nerr_Jw[:,okall_w],axis=1)
                    nstokeserr_Jw -= nstokessyserr_J[:,None]                   
                    for J,j in enumerate(jlistk[k]):
                        loc,scale = norm.fit(nstokeserr_Jw[J,okall_w]/nerr_Jw[J,okall_w])
                        chi2cycle_j[j] = scale**2
                        syserrcycle_j[j] = nstokessyserr_J[J]
                    chi2cyclenet_k[k] = chi2cycle_j[jlistk[k]].mean()
                    syserrcyclenet_k[k] = np.sqrt((syserrcycle_j[jlistk[k]]**2).sum())/len(jlistk[k])

                    if debug:   
                        obsname = object+"_"+config
                        chisqanalysis(obsname,nstokeserr_Jw,nerr_Jw,okall_w)
                                                                     
        # for each obs combine raw stokes, apply efficiency and PA calibration as appropriate for pattern, and save
            obss = len(obslist)

            for obs in range(obss):

                k0,object,config,wppat,pairs = obslist[obs]
                patpairs = patternpairs[wppat]
                klist = range(k0,k0+pairs)                                      # entries in comblist for this obs
                obsname = object+"_"+config

                wplist = [comblist[k][3][1:] for k in klist]
                patwplist = sorted((patpairs*"%1s%1s " % tuple(patterndict[wppat].flatten())).split())
                plist = [patwplist.index(wplist[P]) for P in range(pairs)]

                k_p = np.zeros(patpairs,dtype=int)                              
                k_p[plist] = klist                                                # idx in klist for each pair idx
                cycles_p = np.zeros_like(k_p)
                cycles_p[plist] = np.array([comblist[k][4] for k in klist])       # number of cycles in comb
                cycles_pw = np.zeros((patpairs,wavs),dtype=int)
                cycles_pw[plist] = cycles_kw[klist]                               # of ok cycles for each wavelength
                havecyclechi_p = np.zeros(patpairs,dtype=bool)
                havecyclechi_p[plist] = havecyclechi_k[klist]

                havelinhichi_p = np.zeros(patpairs,dtype=bool)
               
              # name result to document hw cycles included
                kplist = list(k_p)
                if cycles_p.max()==cycles_p.min(): kplist = [klist[0],] 

                for p in range(len(kplist)):
                    obsname += "_"
                    j0 = comblist[k_p[p]][0] - cycles_p[p] + 1
                    for j in range(j0,j0+cycles_p[p]): obsname+=rawlist[j][4][-1]
                log.message("\n  Observation: %s  Date: %s" % (obsname,dateobs), with_header=False)
                finstokes = patternstokes[wppat]   

                if pairs != patpairs:
                    if (pairs<2):
                        log.message(('  Only %1i pair, skipping observation' % pairs), with_header=False)
                        continue
                    elif ((max(plist) < 2) | (min(plist) > 1)):
                        log.message('  Pattern not usable, skipping observation', with_header=False)
                        continue

                stokes_Fw = np.zeros((finstokes,wavs))
                var_Fw = np.zeros_like(stokes_Fw)
                covar_Fw = np.zeros_like(stokes_Fw)

            # normalize pairs in obs at wavelengths _W where all pair/cycles have data:
                okall_w = okcal_w & (cycles_pw[plist] == cycles_p[plist,None]).all(axis=0)     
                normint_K = stokes_kSw[klist,0][:,okall_w].sum(axis=1)
                normint_K /= np.mean(normint_K)
                stokes_kSw[klist] /= normint_K[:,None,None]
                var_kSw[klist] /= normint_K[:,None,None]**2
                covar_kSw[klist] /= normint_K[:,None,None]**2

            # first, the intensity
                stokes_Fw[0] = stokes_kSw[klist,0].sum(axis=0)/pairs
                var_Fw[0] = var_kSw[klist,0].sum(axis=0)/pairs**2 
                covar_Fw[0] = covar_kSw[klist,0].sum(axis=0)/pairs**2         
            # now, the polarization stokes
                if wppat.count('LINEAR'):
                    var_Fw = np.vstack((var_Fw,np.zeros(wavs)))           # add QU covariance
                    if (wppat=='LINEAR'):
                     # wavelengths with both pairs having good, calibratable data in at least one cycle
                        ok_w = okcal_w & (cycles_pw[plist] > 0).all(axis=0)
                        bpm_Fw = np.repeat((np.logical_not(ok_w))[None,:],finstokes,axis=0)
                        stokes_Fw[1:,ok_w] = stokes_kSw[klist,1][:,ok_w]*(stokes_Fw[0,ok_w]/stokes_kSw[klist,0][:,ok_w])
                        var_Fw[1:3,ok_w] = var_kSw[klist,1][:,ok_w]*(stokes_Fw[0,ok_w]/stokes_kSw[klist,0][:,ok_w])**2
                        covar_Fw[1:,ok_w] = covar_kSw[klist,1][:,ok_w]*(stokes_Fw[0,ok_w]/stokes_kSw[klist,0][:,ok_w])**2

                    elif wppat=='LINEAR-HI':
                     # for Linear-Hi, must go to normalized stokes in order for the pair combination to cancel systematic errors
                     # each pair p at each wavelength w is linear combination of pairs, including primary p and secondary sec_p
                     # linhi chisq is from comparison of primary and secondary
                     # evaluate wavelengths with at least both pairs 0,2 or 1,3 having good, calibratable data in at least one cycle: 
                        ok_pw = okcal_w[None,:] & (cycles_pw > 0)
                        ok_w = (ok_pw[0] & ok_pw[2]) | (ok_pw[1] & ok_pw[3])
                        bpm_Fw = np.repeat((np.logical_not(ok_w))[None,:],finstokes,axis=0)
                        stokespri_pw = np.zeros((patpairs,wavs))
                        varpri_pw = np.zeros_like(stokespri_pw)
                        covarpri_pw = np.zeros_like(stokespri_pw)
                        stokespri_pw[plist] = nstokes_kw[klist]
                        varpri_pw[plist] = nvar_kw[klist]
                        covarpri_pw[plist] = ncovar_kw[klist]
                        haveraw_pw = (cycles_pw > 0)
                        pricof_ppw = np.identity(patpairs)[:,:,None]*haveraw_pw[None,:,:]                      

                        qq = 1./np.sqrt(2.)
                        seccofb_pp = np.array([[ 0,1,  0,-1],[1, 0,1,  0],[  0,1, 0,1],[-1,  0,1, 0]])*qq    # both secs avail
                        seccof1_pp = np.array([[qq,1,-qq, 0],[1,qq,0, qq],[-qq,1,qq,0],[-1, qq,0,qq]])*qq    # only 1st sec                        
                        seccof2_pp = np.array([[qq,0, qq,-1],[0,qq,1,-qq],[ qq,0,qq,1],[ 0,-qq,1,qq]])*qq    # only 2nd sec
                        seclist_p = np.array([[1,3],[0,2],[1,3],[0,2]])
                        havesecb_pw = haveraw_pw[seclist_p].all(axis=1)
                        onlysec1_pw = (np.logical_not(havesecb_pw) & haveraw_pw[seclist_p][:,0] & havesecb_pw[seclist_p][:,1])
                        onlysec2_pw = (np.logical_not(havesecb_pw) & haveraw_pw[seclist_p][:,1] & havesecb_pw[seclist_p][:,0])
                        seccof_ppw = seccofb_pp[:,:,None]*havesecb_pw[:,None,:] + \
                                    seccof1_pp[:,:,None]*onlysec1_pw[:,None,:] + \
                                    seccof2_pp[:,:,None]*onlysec2_pw[:,None,:] 
                        stokessec_pw = (seccof_ppw*stokespri_pw[:,None,:]).sum(axis=0)
                        varsec_pw = (seccof_ppw**2*varpri_pw[:,None,:]).sum(axis=0)
                        covarsec_pw = (seccof_ppw**2*covarpri_pw[:,None,:]).sum(axis=0)

                        havesec_pw = (havesecb_pw | onlysec1_pw | onlysec2_pw)
                        prisec_pw = (haveraw_pw & havesec_pw)
                        onlypri_pw = (haveraw_pw & np.logical_not(havesec_pw))
                        onlysec_pw = (np.logical_not(haveraw_pw) & havesec_pw)
                        
                        cof_ppw = onlypri_pw[:,None,:]*pricof_ppw + onlysec_pw[:,None,:]*seccof_ppw +   \
                                    0.5*prisec_pw[:,None,:]*(pricof_ppw+seccof_ppw)

                    # now do the combination
                        stokes_pw = (cof_ppw*stokespri_pw[None,:,:]).sum(axis=1)
                        var_pw = (cof_ppw**2*varpri_pw[None,:,:]).sum(axis=1)
                        covar_pw = (cof_ppw**2*covarpri_pw[None,:,:]).sum(axis=1)
                        covarprisec_pw = 0.5*varpri_pw*np.logical_or(onlysec1_pw,onlysec2_pw)
                        covarqu_w = (cof_ppw[0]*cof_ppw[2]*varpri_pw).sum(axis=0)

                    # cull wavelengths based on chisq between primary and secondary
                        chi2linhi_pw = np.zeros((patpairs,wavs))
                        badlinhichi_w = np.zeros(wavs)
                        havelinhichi_p = prisec_pw.any(axis=1)
                        linhichis = havelinhichi_p.sum()
                        chi2linhi_pw[prisec_pw] = ((stokespri_pw[prisec_pw] - stokessec_pw[prisec_pw])**2 / \
                            (varpri_pw[prisec_pw] + varsec_pw[prisec_pw] - 2.*covarprisec_pw[prisec_pw]))

                        q3_p = np.percentile(chi2linhi_pw[:,okall_w].reshape((4,-1)),75,axis=1)
                        badlinhichi_w[ok_w] = ((chi2linhi_pw[:,ok_w] > (chifence_d[2]*q3_p)[:,None])).any(axis=0)               
                        ok_w &= np.logical_not(badlinhichi_w)
                        okall_w &= np.logical_not(badlinhichi_w)
                        chi2linhi_p = np.zeros(patpairs)
                        chi2linhi_p[havelinhichi_p] = (chi2linhi_pw[havelinhichi_p][:,ok_w]).sum(axis=1)/    \
                            (prisec_pw[havelinhichi_p][:,ok_w]).sum(axis=1)                        
                        syserrlinhi_pw = np.zeros((patpairs,wavs))
                        varlinhi_pw = np.zeros((patpairs,wavs))
                        syserrlinhi_p = np.zeros(patpairs)
                        syserrlinhi_pw[prisec_pw] = (stokespri_pw[prisec_pw] - stokessec_pw[prisec_pw])
                        varlinhi_pw[prisec_pw] = varpri_pw[prisec_pw] + varsec_pw[prisec_pw] - 2.*covarprisec_pw[prisec_pw]
                        syserrlinhi_p[havelinhichi_p] = np.average(syserrlinhi_pw[havelinhichi_p][:,okall_w], \
                            weights=1./np.sqrt(varlinhi_pw[havelinhichi_p][:,okall_w]),axis=1)

                        if debug:
                            np.savetxt(obsname+"_have_pw.txt",np.vstack((wav_w,ok_pw.astype(int),haveraw_pw,havesecb_pw,    \
                                onlysec1_pw,onlysec2_pw,havesec_pw,prisec_pw,onlypri_pw,onlysec_pw)).T,   \
                                fmt="%8.2f  "+9*"%2i %2i %2i %2i  ") 
                            np.savetxt(obsname+"_seccof_ppw.txt",np.vstack((wav_w,ok_pw.astype(int),seccof_ppw.reshape((16,-1)))).T,   \
                                fmt="%8.2f  "+4*"%2i "+16*" %6.3f") 
                            np.savetxt(obsname+"_cof_ppw.txt",np.vstack((wav_w,ok_pw.astype(int),cof_ppw.reshape((16,-1)))).T,   \
                                fmt="%8.2f  "+4*"%2i "+16*" %6.3f")                        
                            np.savetxt(obsname+"_stokes.txt",np.vstack((wav_w,ok_pw.astype(int),stokespri_pw,stokes_pw)).T,    \
                                fmt="%8.2f  "+4*"%2i "+8*" %10.6f")
                            np.savetxt(obsname+"_var.txt",np.vstack((wav_w,ok_pw.astype(int),varpri_pw,var_pw)).T, \
                                fmt="%8.2f  "+4*"%2i "+8*"%14.9f ")
                            np.savetxt(obsname+"_covar.txt",np.vstack((wav_w,ok_pw.astype(int),covarpri_pw,covar_pw)).T, \
                                fmt="%8.2f  "+4*"%2i "+8*"%14.9f ")                       
                            np.savetxt(obsname+"_chi2linhi_pw.txt",np.vstack((wav_w,stokes_Fw[0],ok_pw.astype(int),   \
                                chi2linhi_pw)).T,  fmt="%8.2f %10.0f "+4*"%2i "+4*"%10.4f ")

                        stokes_Fw[1:] = stokes_pw[[0,2]]*stokes_Fw[0]                        
                        var_Fw[1:3] = var_pw[[0,2]]*stokes_Fw[0]**2
                        var_Fw[3] = covarqu_w*stokes_Fw[0]**2
                        covar_Fw[1:] = covar_pw[[0,2]]*stokes_Fw[0]**2
                        bpm_Fw = ((bpm_Fw==1) | np.logical_not(ok_w)).astype(int)

                # document chisq results, combine flagoffs, compute mean chisq for observation, combine with final bpm
                    if (havecyclechi_p.any() | havelinhichi_p.any()):
                        chi2cyclenet = 0.
                        syserrcyclenet = 0.
                        chi2linhinet = 0.
                        syserrlinhinet = 0.
                        if havecyclechi_p.any():
                            log.message(("\n"+14*" "+"{:^"+str(5*patpairs)+"}{:^"+str(8*patpairs)+"}{:^"+str(6*patpairs)+"}")\
                                .format("culled","sys %err","mean chisq"), with_header=False)
                            log.message((9*" "+"HW "+patpairs*" %4s"+patpairs*" %7s"+patpairs*" %5s") \
                                % tuple(3*patwplist),with_header=False)
                            jlist = sum([jlistk[k] for k in klist],[])
                            Jlist = list(set(sum([Jlistk[k] for k in klist],[])))
                            Jmax = max(Jlist)
                            ok_pJ = np.zeros((patpairs,Jmax+1),dtype=bool)
                            for p in plist: ok_pJ[p][Jlistk[k_p[p]]] = True
 
                            syserrcycle_pJ = np.zeros((patpairs,Jmax+1))
                            syserrcycle_pJ[ok_pJ] = syserrcycle_j[jlist]
                            syserrcyclenet_p = np.zeros(patpairs)
                            syserrcyclenet_p[plist] = syserrcyclenet_k[klist]
                            syserrcyclenet = np.sqrt((syserrcyclenet_p**2).sum()/patpairs) 
                            chi2cycle_pJ = np.zeros((patpairs,Jmax+1))
                            chi2cycle_pJ[ok_pJ] = chi2cycle_j[jlist]
                            chi2cyclenet_p = np.zeros(patpairs)
                            chi2cyclenet_p[plist] = chi2cyclenet_k[klist]
                            chi2cyclenet = chi2cyclenet_p.sum()/patpairs
                            culls_pJ = np.zeros((patpairs,Jmax+1),dtype=int)
                            culls_pJ[ok_pJ] = iscull_jw[jlist].sum(axis=1)                            

                            if cycles_p.max() > 2:
                                for J in set(Jlist):
                                    log.message((("   cycle %2i: "+patpairs*"%4i "+patpairs*"%7.3f "+patpairs*"%5.2f ") %     \
                                        ((J+1,)+tuple(culls_pJ[:,J])+tuple(100.*syserrcycle_pJ[:,J])+tuple(chi2cycle_pJ[:,J]))), \
                                                with_header=False)

                            netculls_p = [iscull_jw[jlistk[k_p[p]]].all(axis=0).sum() for p in range(patpairs)]
                            log.message(("    net    : "+patpairs*"%4i "+patpairs*"%7.3f "+patpairs*"%5.2f ") %     \
                                 (tuple(netculls_p)+tuple(100*syserrcyclenet_p)+tuple(chi2cyclenet_p)), with_header=False)
                        if (havelinhichi_p.any()):
                            log.message(("\n"+14*" "+"{:^"+str(5*patpairs)+"}{:^"+str(8*patpairs)+"}{:^"+str(6*patpairs)+"}")\
                                .format("culled","sys %err","mean chisq"), with_header=False)
                            log.message((9*" "+"HW "+(4*patpairs/2)*" "+" all"+(4*patpairs/2)*" "+patpairs*" %7s"+patpairs*" %5s") \
                                % tuple(2*patwplist),with_header=False)
                            chicount = int(badlinhichi_w.sum())
                            chi2linhinet = chi2linhi_p.sum()/(havelinhichi_p.sum())
                            syserrlinhinet = np.sqrt((syserrlinhi_p**2).sum()/(havelinhichi_p.sum()))
                            log.message(("      Linhi: "+(2*patpairs)*" "+"%3i "+(2*patpairs)*" "+patpairs*"%7.3f "+patpairs*"%5.2f ") % \
                                ((chicount,)+tuple(100.*syserrlinhi_p)+tuple(chi2linhi_p)), with_header=False)

                        chi2qudof = (chi2cyclenet+chi2linhinet)/(int(chi2cyclenet>0)+int(chi2linhinet>0))
                        syserr = np.sqrt((syserrcyclenet**2+syserrlinhinet**2)/    \
                            (int(syserrcyclenet>0)+int(syserrlinhinet>0)))
              
                        log.message(("\n  Estimated sys %%error: %5.3f%%   Mean Chisq: %6.2f") % \
                            (100.*syserr,chi2qudof), with_header=False)

                    if not HW_Cal_override:
                # apply hw efficiency, equatorial PA rotation calibration
                        stokes_Fw[1:,ok_w] /= heff_w[ok_w]
                        var_Fw[1:,ok_w] /= heff_w[ok_w]**2
                        covar_Fw[1:,ok_w] /= heff_w[ok_w]**2
                        stokes_Fw,var_Fw,covar_Fw = specpolrotate(stokes_Fw,var_Fw,covar_Fw,eqpar_w)

                # save final stokes fits file for this observation.  Strain out nans.
                    infile = infilelist[rawlist[comblist[k][0]][0]]
                    hduout = pyfits.open(infile)
                    hduout['SCI'].data = np.nan_to_num(stokes_Fw.reshape((3,1,-1)))
                    hduout['SCI'].header['CTYPE3'] = 'I,Q,U'
                    hduout['VAR'].data = np.nan_to_num(var_Fw.reshape((4,1,-1)))
                    hduout['VAR'].header['CTYPE3'] = 'I,Q,U,QU'
                    hduout['COV'].data = np.nan_to_num(covar_Fw.reshape((3,1,-1)))
                    hduout['COV'].header['CTYPE3'] = 'I,Q,U,QU'

                    hduout['BPM'].data = bpm_Fw.astype('uint8').reshape((3,1,-1))
                    hduout['BPM'].header['CTYPE3'] = 'I,Q,U'

                    hduout[0].header['WPPATERN'] = wppat
                    hduout[0].header['PATYPE'] = pacaltype
                    if len(calhistorylist):
                        for line in calhistorylist: hduout[0].header.add_history(line)

                    if (havecyclechi_p.any() | havelinhichi_p.any()): 
                        hduout[0].header['SYSERR'] = (100.*syserr,'estimated % systematic error')
                    
                    outfile = obsname+'_stokes.fits'
                    hduout.writeto(outfile,overwrite=True,output_verify='warn')
                    log.message('\n    '+outfile+' Stokes I,Q,U', with_header=False)

                # apply flux calibration, if available
                    fluxcal_w = specpolflux(outfile,logfile=logfile)
                    if fluxcal_w.shape[0]>0:
                        stokes_Fw *= fluxcal_w
                        var_Fw *= fluxcal_w**2
                        covar_Fw *= fluxcal_w**2

                # calculate, print means (stokes averaged in unnorm space)
                    avstokes_f, avvar_f, avwav = spv.avstokes(stokes_Fw,var_Fw[:-1],covar_Fw,wav_w) 
                    avstokes_F = np.insert(avstokes_f,0,1.)
                    avvar_F = np.insert(avvar_f,0,1.)           
                    spv.printstokes(avstokes_F,avvar_F,avwav,tcenter=np.pi/2.,textfile='tmp.log')
                    log.message(open('tmp.log').read(), with_header=False)
                    os.remove('tmp.log')
                     
#               elif wppat.count('CIRCULAR'):  TBS 

#               elif wppat=='ALL-STOKES':  TBS

            # end of obs loop
        # end of config loop
    return 

# ------------------------------------
def specpolrotate(stokes_Sw,var_Sw,covar_Sw,par_w,normalized=False):
    """ rotate linear polarization in stokes,variance cubes

    Parameters
    ----------
    stokes_Sw: 2d np array
        _S = I,Q,U,(optional V) unnormalized stokes (size 3, or 4)
        _w = wavelength
    var_Sw: 2d np array (size 4, or 5).  If not an array, ignore errors
        _S = I,Q,U,QU variance, (optional V) QU covariance for stokes
    covar_Sw: 2d np array (size 3, or 4)
        _S = I,Q,U covariance, (optional V) covariance in wavelength for stokes
    par_w: 1d np array (if single float, expand it) 
        PA(degrees) to rotate
    normalized: if True, there is no I

    Returns stokes, var (as copy)

    """

    Qarg = int(not normalized)
    stokes_Fw = np.copy(stokes_Sw)
    var_Fw = np.copy(var_Sw)
    covar_Fw = np.copy(covar_Sw)
    if (type(par_w) is float):  par_w = np.repeat(par_w,stokes_Sw.shape[1])
    if (par_w.shape[0]==1):     par_w = np.repeat(par_w,stokes_Sw.shape[1])
    c_w = np.cos(2.*np.radians(par_w))
    s_w = np.sin(2.*np.radians(par_w))
    stokes_Fw[Qarg:] = stokes_Fw[Qarg]*c_w - stokes_Fw[Qarg+1]*s_w ,    \
        stokes_Fw[Qarg]*s_w + stokes_Fw[Qarg+1]*c_w
    if (type(var_Sw) == type(stokes_Sw)):
        var_Fw[Qarg:Qarg+2] =  var_Fw[Qarg]*c_w**2 + var_Fw[Qarg+1]*s_w**2 ,    \
            var_Fw[Qarg]*s_w**2 + var_Fw[Qarg+1]*c_w**2
        var_Fw[Qarg+2] =  c_w*s_w*(var_Fw[Qarg] - var_Fw[Qarg+1]) + (c_w**2-s_w**2)*var_Fw[Qarg+2]
        covar_Fw[Qarg:] =  covar_Fw[Qarg]*c_w**2 + covar_Fw[Qarg+1]*s_w**2 ,    \
            covar_Fw[Qarg]*s_w**2 + covar_Fw[Qarg+1]*c_w**2
    return stokes_Fw,var_Fw,covar_Fw

# ------------------------------------
def chisqanalysis(obsname,nstokeserr_Jw,nerr_Jw,okchi_w):
    # chisq analysis by quartiles in var, 
    #   as-binned data (f1) then binned again with adjacent bins (f2) and alternating bins (f3)
    # to judge errors above Poisson, and covariance

    cycles,wavs = nstokeserr_Jw.shape
    f0=open(obsname+"_chi2cycle_Jw.txt",'ab')
    f1=open(obsname+"_chi2cycle_Jq.txt",'ab')
    f2=open(obsname+"_chi2cycle2bin_Jq.txt",'ab')
    f3=open(obsname+"_chi2cyclealtbin_Jq.txt",'ab')

    np.savetxt(f0,np.vstack((np.arange(wavs),okchi_w,nstokeserr_Jw,nerr_Jw)).T,fmt="%5i %2i "+2*cycles*"%12.8f ")

    Wsort_Jw = np.argsort(nerr_Jw[:,okchi_w],axis=1)
    Wsort_JqW = Wsort_Jw[:,:(okchi_w.sum()-(okchi_w.sum() % 4))].reshape((cycles,4,-1))
    chi2cycle_qJ = np.zeros((4,cycles))
    for J in range(cycles):
        for q in range(4):
            loc,scale = norm.fit(nstokeserr_Jw[J,okchi_w][Wsort_JqW[J,q]]/nerr_Jw[J,okchi_w][Wsort_JqW[J,q]])
            chi2cycle_qJ[q,J] = scale**2
    np.savetxt(f1,np.vstack((range(cycles),chi2cycle_qJ)).T,fmt="%3i "+4*"%8.3f ")

    ok2bin_w = (okchi_w[:-1] & okchi_w[1:])
    w1binlist = np.where(ok2bin_w)[0][::2]
    w2binlist = [w+1 for w in w1binlist]
    nstokeserr2bin_Jw = np.zeros_like(nstokeserr_Jw)
    nerr2bin_Jw = np.zeros_like(nerr_Jw)
    nstokeserr2bin_Jw[:,w1binlist] = (nstokeserr_Jw[:,w1binlist]+nstokeserr_Jw[:,w2binlist])/2.
    nerr2bin_Jw[:,w1binlist] = np.sqrt((nerr_Jw[:,w1binlist]**2 + nerr_Jw[:,w2binlist]**2))/2.
    Wsort_Jw = np.argsort(nerr2bin_Jw[:,w1binlist],axis=1)
    Wsort_JqW = Wsort_Jw[:,:(len(w1binlist)-(len(w1binlist) % 4))].reshape((cycles,4,-1))
    chi2cycle_qJ = np.zeros((4,cycles))
    for J in range(cycles):
        for q in range(4):
            loc,scale = norm.fit(nstokeserr2bin_Jw[J,w1binlist][Wsort_JqW[J,q]]/    \
                nerr2bin_Jw[J,w1binlist][Wsort_JqW[J,q]])
            chi2cycle_qJ[q,J] = scale**2
    np.savetxt(f2,np.vstack((range(cycles),chi2cycle_qJ)).T,fmt="%3i "+4*"%8.3f ")

    wevenlist = list(np.where(okchi_w)[0][::2]) 
    if (len(wevenlist) % 2): wevenlist.pop()     
    woddlist = list(np.where(okchi_w)[0][1::2])
    if (len(woddlist) % 2): woddlist.pop() 
    w1binlist = wevenlist[::2]+woddlist[::2]
    w2binlist = wevenlist[1::2]+woddlist[1::2] 
    nstokeserraltbin_Jw = np.zeros_like(nstokeserr_Jw)
    nerraltbin_Jw = np.zeros_like(nerr_Jw)
    nstokeserraltbin_Jw[:,w1binlist] = (nstokeserr_Jw[:,w1binlist]+nstokeserr_Jw[:,w2binlist])/2.
    nerraltbin_Jw[:,w1binlist] = np.sqrt((nerr_Jw[:,w1binlist]**2 + nerr_Jw[:,w2binlist]**2))/2.
    Wsort_Jw = np.argsort(nerraltbin_Jw[:,w1binlist],axis=1)
    Wsort_JqW = Wsort_Jw[:,:(len(w1binlist)-(len(w1binlist) % 4))].reshape((cycles,4,-1))
    chi2cycle_qJ = np.zeros((4,cycles))
    for J in range(cycles):
        for q in range(4):
            loc,scale = norm.fit(nstokeserraltbin_Jw[J,w1binlist][Wsort_JqW[J,q]]/  \
                nerraltbin_Jw[J,w1binlist][Wsort_JqW[J,q]])
            chi2cycle_qJ[q,J] = scale**2
    np.savetxt(f3,np.vstack((range(cycles),chi2cycle_qJ)).T,fmt="%3i "+4*"%8.3f ")

    f1.close()
    f2.close()
    f3.close()

    return

if __name__=='__main__':
    infilelist=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)
    if len(kwargs): kwargs = {k:bool(v) for k,v in kwargs.iteritems()}        
    specpolfinalstokes(infilelist,**kwargs)
