#! /usr/bin/env python2.7

"""
polfinalstokes

Correct raw stokes for track, combine, and calibrate to form final stokes.
For polarimetric data of all modes
Uses Moon cal

"""

import os, sys, glob, shutil, inspect
import operator

import numpy as np
from astropy.io import fits as pyfits
from astropy.io import ascii
from scipy.interpolate import interp1d
from scipy import linalg as la
from scipy.stats import norm
from astropy.table import Table
from zipfile import ZipFile

# this is pysalt-free
import warnings
# warnings.simplefilter("error")

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
keywordfile = datadir+"obslog_config.json"
keywordsecfile = datadir+"obslogsec_config.json"
sys.path.extend((polsaltdir+'/polsalt/',))

import rsslog
from obslog import create_obslog
from sppolview import avstokes
from sppolview import printstokestw
from polutils_test import datedfile, datedline, angle_average, heffcal, heffcalcorrect, polzerocal, specpolrotate
from calassess import calaccuracy_assess
from polflux import polflux
np.set_printoptions(threshold=np.nan)

# -------------------------------------
def polfinalstokes(infileList, **kwargs):
    """Combine the raw stokes and apply the polarimetric calibrations

    Parameters
    ----------
    infileList: list
        List of filenames that include an extracted spectrum

    logfile=: str
        Name of file for logging
    """
    """
    _l: line in calibration file
    _i: index in rawstokes file list
    _j: rawstokes idx = waveplate position pair index (enumeration within config, including cycles)
    _J: cycle number idx (0,1,..) for each rawstokes
    _k: combstokes = waveplate position pair index (enumeration within config, cycles combined)
    _K: pair = waveplate position pair index (enumeration within obs)
    _p: pair = raw stokes # (eg 0,1,2,3 = 0 4  1 5  2 6  3 7 for LINEAR-HI, sorted in h0 order)
    _s: normalized linear stokes for zeropoint correction (0,1) = (q,u) 
    _S: unnormalized raw stokes within waveplate position pair: (eg 0,1 = I,Q)
    _F: unnormalized final stokes (eg 0,1,2 = I,Q,U)
    _i: mask slits (entries in xml)    
    _t: culled target index (input)
    _T: processed target index
    _w: wavelength index
    _m: sample index = tw
    """

    logfile= kwargs.pop('logfile','polfinalstokes.log')
    with_stdout = kwargs.pop('with_stdout',True)       
    if with_stdout: rsslog.history(logfile)
     
    rsslog.message(str(kwargs), logfile, with_stdout=with_stdout) 
    
    debug = (str(kwargs.pop('debug','False')).lower() == 'true')    # allow for string or boolean
    PA_Equatorial_override = (str(kwargs.pop('PA_Equatorial_override','False')).lower() == 'true')
    Heffcal_override = (str(kwargs.pop('Heffcal_override','False')).lower() == 'true')
    LinearPolZeropoint_override = (str(kwargs.pop('LinearPolZeropoint_override','False')).lower() == 'true')
    PAZeropoint_override = (str(kwargs.pop('PAZeropoint_override','False')).lower() == 'true')    
    usrHeffcalfile = kwargs.pop('usrHeffcalfile',"")      # if set, new style cal file is in cwd, not polsalt/data     
    useoldheffcal = (str(kwargs.pop('useoldheffcal','False')).lower() == 'true')
    useMPlat = kwargs.pop('useMPlat','default')
    nametag = kwargs.pop('nametag',"")
    if (useMPlat != 'default'): useMPlat = (useMPlat.lower() == 'true')
    illummask = kwargs.pop('illummask',1.)          
    patternList = open(datadir+'wppaterns.txt','r').readlines()
    patternpairs = dict();  patternstokes = dict(); patterndict = dict()
    for p in patternList:
        if p.split()[0] == '#': continue
        patterndict[p.split()[0]]=np.array(p.split()[3:]).astype(int).reshape((-1,2))
        patternpairs[p.split()[0]]=(len(p.split())-3)/2
        patternstokes[p.split()[0]]=int(p.split()[1])

    rsslog.message('polfinalstokes version: 20250225', logfile, with_stdout=with_stdout)

  # optics data
    obsDict = create_obslog(infileList,keywordfile)
    obssecDict = create_obslog(infileList,keywordsecfile,ext='SCI')    
    dateobs = obsDict['DATE-OBS'][0].replace('-','')         
    row0,col0,C0 = np.array(datedline(datadir+"RSSimgalign.txt",dateobs).split()[1:]).astype(float) 
    imgoptfile = datadir+'RSSimgopt.txt'
    distTab = ascii.read(imgoptfile,data_start=1,   \
            names=['Wavel','Fcoll','Acoll','Bcoll','ydcoll','xdcoll','Fcam','acam','alfydcam','alfxdcam'])
    FColl6000 = distTab['Fcoll'][list(distTab['Wavel']).index(6000.)]
    FCam6000 = distTab['Fcam'][list(distTab['Wavel']).index(6000.)]  
                            
  # Organize data using names. 
  #   allrawListfd = list of lists: [fileidx,object,config,wvplt,cycle] for each f rawstokes file idx.        
    files = len(infileList)
    allrawListfd = []
    for f in range(files):
        object,config,wvplt,cycle = os.path.basename(infileList[f]).rsplit('.',1)[0].rsplit('_',3)
        if (config[0]!='c')|(wvplt[0]!='h')|(not cycle.isdigit()):
            rsslog.message('File '+infileList[f]+' is not a raw stokes file.'  , logfile, with_stdout=with_stdout)
            continue
        allrawListfd.append([f,object,config,wvplt,cycle])
    configList = sorted(list(set(data[2] for data in allrawListfd)))       # unique configs
    
  # If config of same name is different (ie different number of wavelength bins), add a digit to config name  
    conf_f = np.array(tuple(zip(*allrawListfd))[2])
    
    for conf in configList:
        fcList = np.where(conf_f==conf)[0]
        lens = len(set(np.array(obssecDict['NAXIS1'])[fcList]))        
        if (lens==1): continue
        lenListF = list(np.array(obssecDict['NAXIS1'])[fcList])
        lenList = list(set(lenListF))
        lens = len(lenList)
        for lidx in range(lens):        
            FList = np.where(np.array(lenListF)==lenList[lidx])[0]
            for f in np.array(fcList)[FList]:            
                allrawListfd[f][2] += str(lidx)
    configList = sorted(list(set(data[2] for data in allrawListfd)))       # unique configs
        
    if debug:
        rsslog.message('allrawListfd:', logfile, with_stdout=with_stdout)
        for i in range(len(allrawListfd)):
            rsslog.message(repr(allrawListfd[i]), logfile, with_stdout=with_stdout)
            
  # input correct HZeroCal, and TelZeropoint calibration files
    TelZeropointfile = datedfile(datadir+"RSSpol_Linear_TelZeropoint_yyyymmdd_vnn.txt",dateobs)
    twav_l,tq0_l,tu0_l,err_l = np.loadtxt(TelZeropointfile,dtype=float,unpack=True,ndmin=2)

  # input PAZeropoint file and get correct entry
    dpadatever,dpa = datedline(datadir+"RSSpol_Linear_PAZeropoint.txt",dateobs).split()
    dpa = float(dpa)
    chifence_d = 2.2*np.array([6.43,4.08,3.31,2.91,2.65,2.49,2.35,2.25])    # *q3 for upper outer fence outlier for each dof

  # Maintenance Platform control status
  
    if (useMPlat=="default"):
        MPlatdates_dx = np.loadtxt(datadir+"MPlatdates.txt",dtype=int,usecols=(0,1),unpack=True)
        MPlaton = (((int(dateobs) >= MPlatdates_dx[0]) &  \
            (int(dateobs) <= MPlatdates_dx[1])).sum() >0)        
        if MPlaton:
            rsslog.message('\nDateobs is in Maintenance Platform date list: using correction', logfile, with_stdout=with_stdout)
    else:
        rsslog.message('\nMaintenance Platform correction turned '+['Off','on'][useMPlat], logfile, with_stdout=with_stdout)
        MPlaton = useMPlat

  # do one config at a time. For each config: 
  #   rawListjd = list of lists: infile idx,object,config,wvplt,cycle for each rawstokes file *in this config*. 
  #     rawListjd is sorted with cycle varying fastest, then hw
  #   combListkd = list of lists: last rawList idx,object,config,wvplt,cycles,wppat 
  #     one entry for each set of cycles that needs to be combined (i.e, one for each wvplt position pair)
  #   obsListod = list of lists: first combListkd idx,object,config,wppat,pairs for an object

    illum_vList = []
    for conf in configList:
        rsslog.message("\nConfiguration: %s" % conf, logfile, with_stdout=with_stdout) 
        rawListjd = [data for data in allrawListfd if data[2]==conf]
        for col in (4,3,1,2): rawListjd = sorted(rawListjd,key=operator.itemgetter(col))                        
        rawstokess = len(rawListjd)
        calhistoryList = ["PolCal Model: 20231221",]
        
        hdul0 = pyfits.open(infileList[rawListjd[0][0]])        
        wav0 = hdul0['SCI'].header['CRVAL1']
        dwav = hdul0['SCI'].header['CDELT1']
        wavs = hdul0['SCI'].header['NAXIS1']
        wcshdu = None
        if 'WCSDVARR' in [hdul0[x].name for x in range(len(hdul0))]:
            wedge_W = hdul0['WCSDVARR'].data
            wav_w = (wedge_W[:dwav*wavs:dwav] + wedge_W[dwav:(wavs+1)*dwav:dwav])/2.
        else:
            wav_w = wav0 + dwav*np.arange(wavs)
        targets = hdul0['SCI'].data.shape[1]

        if (targets > 1):
            tgtTab = Table.read(hdul0['TGT'])           
            entries = len(tgtTab['CATID'])
            if (not ('CULL' in tgtTab.colnames)):
                tgtTab.add_column(Table.Column(name='CULL',data=targets*['']))
            oktgt_i = (tgtTab['CULL'] == '')
            i_t = np.where(oktgt_i)[0]                                   
            YX_dt = np.array([tgtTab['YCE'],tgtTab['XCE']])[:,oktgt_i]
            r_t = np.sqrt((YX_dt**2).sum(axis=0))
            t00 = np.argmin(r_t)
            ok_t = np.ones(targets,dtype=bool)             
            if (useoldheffcal):
                tgtTab['CULL'][i_t] = 'NOCAL'
                tgtTab['CULL'][i_t[t00]] = ''
                ok_t = (np.arange(targets)==t00)                                                                         
            Targets = ok_t.sum()             
            tgtname_t = np.array(tgtTab['CATID'])[oktgt_i]
           
        else:
            ok_t = np.ones(1,dtype=bool)
            YX_dt = np.zeros((2,1))
            tgtname_t = np.array(['00000',])
            chi2qudof_t = np.zeros(1)
            Targets = 1
            t00 = 0
            i_t = np.zeros(1)
        T00 = np.where(np.arange(targets)[ok_t]==t00)[0][0]
        YX_dT = YX_dt[:,ok_t] 
        tgtname_T = tgtname_t[ok_t]            
        samples = Targets*wavs                     
        tw_dm = np.unravel_index(range(samples),(Targets,wavs))
        mList = np.where(tw_dm[0]==t00)[0]           # for center target debug                 

        if debug:
            rsslog.message(" center target: %2i" % t00, logfile, with_stdout=with_stdout)             

      # interpolate telZeropoint calibration wavelength dependence for this config
        okzpcal_w = np.ones(wavs).astype(bool)                
        if not LinearPolZeropoint_override:  
            tel0_sw = interp1d(twav_l,                  \
                np.array([tq0_l,tu0_l]),kind='cubic',bounds_error=False)(wav_w)/100.
            okzpcal_w &= ~np.isnan(tel0_sw[0])
            tel0_sw[:,~okzpcal_w] = 0.       
          
      # get spectrograph data
        grating = obsDict['GRATING'][rawListjd[0][0]]
        grang = obsDict['GR-ANGLE'][rawListjd[0][0]] 
        artic = obsDict['AR-ANGLE'][rawListjd[0][0]] 

      # input correct HeffCal calibration file
        label = [grating,'Im'][grating=='N/A']
        if (not Heffcal_override):        
            if usrHeffcalfile:
                Heffcalfile = usrHeffcalfile
                usenewheffcal = True                    # user-specified heffcal must be new style                   
            else:                                       # MOS calibrations will be zip, Onaxis will be txt
                Heffcalfile = datedfile(datadir+"RSSpol_Heff_"+label+"_yyyymmdd_vnn.???",dateobs)

                print 'calfile lookup'                
                print datadir+"RSSpol_Heff_"+label+"_yyyymmdd_vnn.???"
                print dateobs
                print glob.glob(datadir+"/RSSpol*.txt")
                print datedfile(datadir+"RSSpol_Heff_"+label+"_yyyymmdd_vnn.???",dateobs)
                print len(datedfile(datadir+"RSSpol_Heff_"+label+"_yyyymmdd_vnn.???",dateobs)), len(Heffcalfile)
                print useoldheffcal, (len(Heffcalfile) & (not useoldheffcal))
                print ((len(Heffcalfile)>0) & (not useoldheffcal))
                print
                
                usenewheffcal = ((len(Heffcalfile)>0) & (not useoldheffcal))   # FIX!
                if (not usenewheffcal):                        
                    Heffcalfile = datedfile(datadir+"RSSpol_HW_Calibration_yyyymmdd_vnn.txt",dateobs)                          
                    rsslog.message("\nUsing old heffcalfile: %s, target %i" % (Heffcalfile,i_t[t00]),    \
                        logfile, with_stdout=with_stdout)
            caldoc2 = ''                                            
            if (Heffcalfile.split(".")[1] == "txt"):
                calFile = open(Heffcalfile)
                caldoc = calFile.readline()[2:].rstrip('\n').rstrip('\r')      # FIX!
                caldocline2 = calFile.readline()[2:].rstrip('\n').rstrip('\r')   # FIX!
                if caldocline2.count('pacor'): caldoc2 = caldocline2[1:]
            else:         
                caldoc = ZipFile(Heffcalfile).open(ZipFile(Heffcalfile).namelist()[0]).readline()[2:-1]
          
      # get all rawstokes and track-dependent calibration data
        stokes_jSm = np.zeros((rawstokess,2,samples)) 
        var_jSm = np.zeros_like(stokes_jSm)
        covar_jSm = np.zeros_like(stokes_jSm)
        bpm_jSm = np.zeros_like(stokes_jSm).astype(int)
        okcal_jTw = np.zeros((rawstokess,Targets,wavs),dtype=bool)
        maskasec_j = np.zeros(rawstokess)        
        telpa_j = np.zeros(rawstokess)
        rho_j = np.zeros(rawstokess)
        imgtime_j = np.zeros(rawstokess)        
        obsDT_j = np.zeros(rawstokess,dtype='object')         
        pairf21_j = np.zeros(rawstokess)        
        TRKyx_dj = np.zeros((2,rawstokess)) 
        MByx_dj = np.zeros((2,rawstokess))
        heff_jm = np.zeros((rawstokess,samples))
        hpacor_jm = np.zeros((rawstokess,samples))
        okcal_jm = np.ones((rawstokess,samples),dtype=bool)                
        
        usetrk = (min(min(obsDict['MBY']),min(obsDict['MBX'])) == -999.99)
        if usetrk:
            rsslog.message("\nWARNING: some missing MB data, using TRK\n", logfile, with_stdout=with_stdout)
        calhistoryList.append("Heffcal: Moving Baffle "+['OK','using TRK'][usetrk])
                                                        
        combListkd = []

        for j in range(rawstokess):
            i,object,config,wvplt,cycle = rawListjd[j]
            hdr0 = pyfits.getheader(infileList[i],0)
            maskasec_j[j] = float(obsDict['MASKID'][i][2:6])/100. 
            telpa_j[j] = float(obsDict['TELPA'][i])
            rho_j[j] = float(obsDict['TRKRHO'][i])
            imgtime_j[j] = hdr0.get('IMGTIME',default=0.)   # this is for shutter illumination correction
            if ((imgtime_j[j] == 0.) & (not useoldheffcal)):
                imgtime_j[j] = float(obsDict['EXPTIME'][i])/2.
                rsslog.message("\nWARNING: %s IMGTIME not found, using exptime/2: %6.2f" %  \
                    (infileList[i],imgtime_j[j]), logfile, with_stdout=with_stdout)
                        
            obsDT_j[j] = np.datetime64(obsDict['DATE-OBS'][i]+'T'+obsDict['UTC-OBS'][i])                                
                                              
            if j==0:
                cycles = 1
                lampid = obsDict['LAMPID'][i].strip().upper()
                issky = (lampid == "NONE")
                havepairf21 = (obsDict['PAIRF21'][i]<>'UNKNOWN')
                    
              # prepare calibration keyword documentation for this configuration
                PAcaltype = "Equatorial"
                if PA_Equatorial_override:
                    PAcaltype = "Instrumental"
                if (not issky): 
                    PAcaltype ="Instrumental"
                if (not Heffcal_override): 
                    calhistoryList.append("Heffcal: "+os.path.basename(Heffcalfile))
                    if MPlaton:
                        calhistoryList.append("Heffcal: Using MPlat correction")
                    calhistoryList.append("Heffcal: "+caldoc)
                    if (caldoc2): calhistoryList.append("Heffcal: "+caldoc2)

                    print "calhistoryList"                    
                    print calhistoryList
                    exit()
                                                                
                elif issky:                             # for HWZero cal
                    PAcaltype = "Instrumental"
                    LinearPolZeropoint_override = True     
                    calhistoryList.append("Heffcal: Uncalibrated")                                                                                        
                else:                                   # for HWPol cal 
                    PAcaltype = "Instrumental"
                    calhistoryList.append("Heffcal: Uncalibrated")                        
                           
                if (LinearPolZeropoint_override | (not issky)):
                    calhistoryList.append("TelPolZeropoint: Null")
                else:
                    calhistoryList.append("TelPolZeropoint: "+os.path.basename(TelZeropointfile))                        
                
                if PAZeropoint_override:
                    calhistoryList.append("PAZeropoint: Null")
                else:        
                    calhistoryList.append("PAZeropoint: RSSpol_Linear_PAZeropoint.txt "+str(dpadatever)+" "+str(dpa))

                rsslog.message('  PA type: '+PAcaltype, logfile, with_stdout=with_stdout)
                if len(calhistoryList): 
                    rsslog.message('  '+'\n  '.join(calhistoryList), logfile, with_stdout=with_stdout)                                  
                if (not Heffcal_override):
                    if usenewheffcal:
                        rsslog.message('\n cy wp      TRKY       TRKX       MBY        MBX   minillum puparea pupnom', logfile,     \
                        with_stdout=with_stdout)

          # if object,config,wvplt changes, start a new combListkd entry
            else:
                if rawListjd[j-1][1:4] != rawListjd[j][1:4]: cycles = 1
                else: cycles += 1

            if usetrk:
                MByx_dj[0,j] = 0.000430 - 0.025291 * obsDict['TRKY'][i]     # Poleff/Moving Baffle Positions.xlsx              
                MByx_dj[1,j] = 0.000628 + 0.025291 * obsDict['TRKX'][i]
            else:
                MByx_dj[:,j] = np.array([obsDict['MBY'][i],obsDict['MBX'][i]]).astype(float)
                                                 
            if issky:
                TRKyx_dj[:,j] = np.array([obsDict['TRKY'][i],obsDict['TRKX'][i]]).astype(float)                         
            else:
                TRKyx_dj[0,j] =  0.03151 - 39.57633*MByx_dj[0,j]                                                 
                TRKyx_dj[1,j] = -0.02369 + 39.89163*MByx_dj[1,j]                  
                    
            wppat = obsDict['WPPATERN'][i].upper()
            if havepairf21: pairf21_j[j] = float(obsDict['PAIRF21'][i])            
            hdul = pyfits.open(infileList[i])                                                        
            stokes_jSm[j] = hdul['SCI'].data[:,ok_t].reshape((-1,samples))
            var_jSm[j] = hdul['VAR'].data[:,ok_t].reshape((-1,samples))
            covar_jSm[j] = hdul['COV'].data[:,ok_t].reshape((-1,samples))
            bpm_jSm[j] = hdul['BPM'].data[:,ok_t].reshape((-1,samples))
                                                                
          # if HeffCal used, get HWPol for track, FOV, correct for heff and tel zeropoint, converted to raw
            if (not Heffcal_override):
                if usenewheffcal:
                  # input heffcal file data. if zip, targets are stacked in zip. _p=0,1 are stacked inside each target 
                    hwcal_pTw, okcal_Tw, pacorfile, modelarea_T, modelareanom_T, minillum_T =       \
                        heffcal(Heffcalfile,YX_dT,MByx_dj[:,j],TRKyx_dj[:,j],   \
                        grang,rho_j[j],imgtime_j[j],wav_w,wvplt,  \
                            illummask=illummask,useMPlat=MPlaton,logfile=logfile,debug=debug)
                    if debug:
                        np.savetxt(object+"_"+config+"_hwcal_pw.txt",np.vstack((wav_w,hwcal_pTw[:,T00])).T, \
                            fmt=" %7.2f %8.3f %8.3f")                                              
                    heff_Tw = hwcal_pTw[0]/100.                # cal file is in % 
                    hpacor_Tw = -hwcal_pTw[1]
                    nomareamsg = ['',("%6.3f" % modelareanom_T[T00])][modelarea_T[T00] != modelareanom_T[T00]]                
                    rsslog.message( (" %2i %2s "+4*"%10.5f "+"%6.3f %6.3f %s") %  \
                        ((cycles,wvplt)+tuple(TRKyx_dj[:,j])+tuple(MByx_dj[:,j])   \
                        +(minillum_T[T00],modelarea_T[T00],nomareamsg)), logfile, with_stdout=with_stdout) 
                                             
                else:
                    hwav_l,heff_l,hpa_l = np.loadtxt(Heffcalfile,dtype=float,unpack=True,usecols=(0,1,2),ndmin=2)                
                    heff_Tw = interp1d(hwav_l,heff_l,kind='cubic',bounds_error=False)(wav_w)
                    okcal_Tw = ((wav_w >= hwav_l[0]) & (wav_w <= hwav_l[-1]))
                    hpacor_Tw = -interp1d(hwav_l,hpa_l,kind='cubic',bounds_error=False)(wav_w)                                   
                heff_m = heff_Tw.flatten()                                       
                hpacor_jm[j] = hpacor_Tw.flatten()           
                okcal_jm[j] &= okcal_Tw.flatten()                
                stokes_jSm[j,1,okcal_jm[j]] = stokes_jSm[j,1,okcal_jm[j]]/heff_m[okcal_jm[j]]                                  
                var_jSm[j,1,okcal_jm[j]] = var_jSm[j,1,okcal_jm[j]]/(heff_m[okcal_jm[j]])**2
                covar_jSm[j,1,okcal_jm[j]] = covar_jSm[j,1,okcal_jm[j]]/(heff_m[okcal_jm[j]])**2

                if ((not LinearPolZeropoint_override) & issky):                           
                    okcal_jm[j] &= np.tile(okzpcal_w,Targets)
                    tel0_sm = np.tile(tel0_sw,Targets)                    
                    dpatelraw_m = -(22.5*float(wvplt[1]) + hpacor_jm[j] + rho_j[j] + dpa)                                        
                    rawtel0_m = specpolrotate(tel0_sm,0,0,dpatelraw_m,normalized=True)[0][0]                    
                    stokes_jSm[j,1] -= rawtel0_m*(stokes_jSm[j,0])                                                             
                                             
            if cycles==1:
                combListkd.append([j,object,config,wvplt,1,wppat])
            else:
                combListkd[-1] = [j,object,config,wvplt,cycles,wppat]
        
        if debug:
            np.savetxt("hpacor_"+str(conf)+"_jm.txt",  \
                np.vstack((tw_dm[0],wav_w[tw_dm[1]],hpacor_jm)).T,  \
                fmt="%3i %8.2f "+rawstokess*"%9.3f ")

        combstokess = len(combListkd)
        obsListod = []
        obsobject = ''
        obsconfig = ''        
        for k in range(combstokess):         
            j,object,config,wvplt,cycles,wppat = combListkd[k]
            if ((object != obsobject) | (config != obsconfig)):
                obsListod.append([k,object,config,wppat,1])
                obsobject = object; obsconfig = config
            else:
                obsListod[-1][4] +=1        # number of wp pairs for observation
        obss = len(obsListod)
        
     #  If Heffcal enabled, apply delta dPA calibration to observation mean dPA: "adjusted" raw
     #     See calibrations/Poleff/rawcorn_algorithm.txt for derivation of PA correction algorithm
        okcal_om = np.ones((obss,samples),dtype=bool)     
        if (not Heffcal_override):
            hpacormean_om = np.zeros((obss,samples))          # this will be removed at the end        
            nstokes_jm = np.zeros((rawstokess,samples))
            okcal_om = np.zeros((obss,samples),dtype=bool)       
            for obs in range(obss):
                k0,object,config,wppat,hwpairs = obsListod[obs]
                j,object,config,wvplt,cycles,wppat = combListkd[k0]                         
                jListhj = [range((combListkd[k][0]-combListkd[k][4]+1),combListkd[k][0]+1)      \
                     for k in range(k0,k0+hwpairs)]         # hwpair list of j lists
                jListo = sum(jListhj,[])
                ok_jm = ((bpm_jSm.all(axis=1)==0) & np.in1d(range(rawstokess),jListo)[:,None]) 
                nstokes_jm[ok_jm] = stokes_jSm[:,1][ok_jm]/stokes_jSm[:,0][ok_jm]
                hpacormean_om[obs] = hpacor_jm[jListo].mean(axis=0)
                okcal_om[obs] = okcal_jm[jListo].all(axis=0)
                dhpar_jm = -(hpacor_jm - hpacormean_om[obs])
                qupairs = hwpairs/2 

                if debug:                
                    np.savetxt("stokes_jSm_before.txt",np.vstack((wav_w,stokes_jSm[:,:,mList].reshape((-1,wavs)))).T,    \
                        fmt=(" %8.1f"+4*cycles*" %10.2f"))
                    np.savetxt("dhpar_jm.txt", np.vstack((wav_w,dhpar_jm[:,mList])).T,fmt=" %8.1f"+2*cycles*" %10.4f")
                    np.savetxt("hpacormean_w.txt",np.vstack((wav_w,hpacormean_om[obs,mList])).T,fmt=" %8.1f %10.4f")                    
                 
                for qupair in range(qupairs):          
                    jListq = jListhj[qupair]
                    JListq = [int(rawListjd[j][4]) for j in jListq]             
                    jListu = jListhj[qupair+1]
                    JListu = [int(rawListjd[j][4]) for j in jListu]                     
                    for qidx,j in enumerate(jListq):      
                        uidx = np.argmin(np.abs(np.array(JListu)-JListq[qidx]))    # find nearest u cycle
                        ok_m = (ok_jm[j] & ok_jm[jListu[uidx]])                                                  
                        ddth_M = -np.radians(dhpar_jm[jListq[qidx]]-dhpar_jm[j].mean(axis=0))[ok_m]                                                                    
                        uqrat_M = nstokes_jm[jListu[uidx]][ok_m]/nstokes_jm[jListq[qidx]][ok_m]
                        th_M = np.degrees(np.arctan2((uqrat_M-2.*ddth_M), (1.-2.*uqrat_M*ddth_M)))/2.   \
                            - dhpar_jm[j].mean(axis=0)
                        q_M = nstokes_jm[jListq[qidx]][ok_m]*   \
                            np.cos(2.*np.radians(th_M))/np.cos(2.*np.radians(th_M+dhpar_jm[j,ok_m]))                            
                        stokes_jSm[jListq[qidx],1][ok_m] = q_M*stokes_jSm[jListq[qidx],0][ok_m]

                    if debug:
                        cordat_dm = np.zeros((4,samples))
                        cordat_dm[:,ok_m] = np.vstack((ddth_M,uqrat_M,th_M,q_M))
                        np.savetxt("cordat04_M.txt",np.vstack((wav_w,ok_m[mList],cordat_dm[:,mList])).T, \
                            fmt=" %8.1f %3i "+4*"%10.4f ")
                        
                    for uidx,j in enumerate(jListu):       
                        qidx = np.argmin(np.abs(np.array(JListq)-JListu[uidx]))    # find nearest q cycle
                        ok_m = (ok_jm[j] & ok_jm[jListq[qidx]])                          
                        ddth_M = np.radians(dhpar_jm[jListu[uidx]]-dhpar_jm[j].mean(axis=0))[ok_m]  
                        uqrat_M = nstokes_jm[jListu[uidx]][ok_m]/nstokes_jm[jListq[qidx]][ok_m]
                        th_M = np.degrees(np.arctan2((uqrat_M-2.*ddth_M), (1.-2.*uqrat_M*ddth_M)))/2.   \
                            - dhpar_jm[j].mean(axis=0)
                        u_M = nstokes_jm[jListu[uidx]][ok_m]*   \
                            np.sin(2.*np.radians(th_M))/np.sin(2.*np.radians(th_M+dhpar_jm[j,ok_m]))                                                        
                        stokes_jSm[jListu[uidx],1][ok_m] = u_M*stokes_jSm[jListu[uidx],0][ok_m]             

                    if debug:
                        cordat_dm = np.zeros((4,samples))
                        cordat_dm[:,ok_m] = np.vstack((ddth_M,uqrat_M,th_M,u_M))
                        np.savetxt("cordat26_M.txt",np.vstack((wav_w,ok_m[mList],cordat_dm[:,mList])).T, \
                            fmt=" %8.1f %3i "+4*"%10.4f ")                        
                        np.savetxt("stokes_jSm_after.txt",np.vstack((wav_w,stokes_jSm[:,:,mList].reshape((-1,wavs)))).T,    \
                            fmt=(" %8.1f"+4*cycles*" %10.2f"))
                                                       
      # combine multiple cycles as necessary.  Absolute stokes is on a per cycle basis.
      # polarimetric combination on normalized stokes basis 
      #  to avoid coupling mean syserr into polarimetric spectral features

        stokes_kSm = np.zeros((combstokess,2,samples)) 
        var_kSm = np.zeros_like(stokes_kSm)
        covar_kSm = np.zeros_like(stokes_kSm)
        cycles_km = np.zeros((combstokess,samples)).astype(int)
        chi2cycle_km = np.zeros((combstokess,samples))
        badcyclechi_km = np.zeros((combstokess,samples),dtype=bool)
        havecyclechi_k = np.zeros(combstokess,dtype=bool)

        chi2cycle_j = np.zeros(rawstokess)
        chi2cycle_jT = np.zeros((rawstokess,Targets)) 
        syserrcycle_j = np.zeros(rawstokess)
        iscull_jm = np.zeros((rawstokess,samples),dtype=bool)
        nstokes_km = np.zeros((combstokess,samples))
        nvar_km = np.zeros_like(nstokes_km)
        ncovar_km = np.zeros_like(nstokes_km)
        chi2cyclenet_k = np.zeros(combstokess)
        chi2cyclenet_kT = np.zeros((combstokess,Targets))
        chi2cycles_kT = np.zeros((combstokess,Targets),dtype=int)             
        okcyclechi_kT = np.zeros((combstokess,Targets),dtype=bool)                       
        syserrcyclenet_k = np.zeros(combstokess)
        jListk = []             # list of rawstokes idx for each combListkd entry
        JListk = []             # list of cycle number for each combListkd entry
        
        for k in range(combstokess):         
            j,object,config,wvplt,cycles,wppat = combListkd[k]
            jListk.append(range(j-cycles+1,j+1))                                
            JListk.append([int(rawListjd[jj][4])-1 for jj in range(j-cycles+1,j+1)])  # J = cycle-1, counting from 0        
            nstokesk_Jm = np.zeros((cycles,samples))
            nvark_Jm = np.zeros((cycles,samples))
            ncovark_Jm = np.zeros((cycles,samples))
            bpmk_Jm = np.zeros((cycles,samples))
            okk_Jm = np.zeros((cycles,samples),dtype=bool)

            for J,j in enumerate(jListk[k]):
                bpmk_Jm[J] = bpm_jSm[j,0]
                okk_Jm[J] = (bpmk_Jm[J] ==0)
                nstokesk_Jm[J][okk_Jm[J]] = stokes_jSm[j,1][okk_Jm[J]]/stokes_jSm[j,0][okk_Jm[J]]
                nvark_Jm[J][okk_Jm[J]] = var_jSm[j,1][okk_Jm[J]]/(stokes_jSm[j,0][okk_Jm[J]])**2
                ncovark_Jm[J][okk_Jm[J]] = covar_jSm[j,1][okk_Jm[J]]/(stokes_jSm[j,0][okk_Jm[J]])**2

          # Culling:  for multiple cycles, compare each cycle with every other cycle (dof=1).          
          # bad wavelengths flagged for P < .02% (1/2000): chisq  > 13.8  (chi2.isf(q=.0002,df=1))
          # for cycles>2, vote to cull specific pair/wavelength, otherwise cull wavelength

            cycles_km[k] =  (1-bpmk_Jm).sum(axis=0).astype(int)
            okchik_m = (cycles_km[k] > 1)                       # okchik_m: at least 2 good cycles for t,w, this k
            chi2lim = 13.8 
            havecyclechi_k[k] = okchik_m.any()
            if cycles > 1:
                okk_Jm[J] = okchik_m & (bpmk_Jm[J] ==0)
                chi2cyclek_JJm = np.zeros((cycles,cycles,samples))
                okk_JJm = okk_Jm[:,None,:] & okk_Jm[None,:,:] 
                nstokes_JJm = nstokesk_Jm[:,None,:] - nstokesk_Jm[None,:,:]
                nvar_JJm = nvark_Jm[:,None,:] + nvark_Jm[None,:,:] 
                                              
                chi2cyclek_JJm[okk_JJm] = nstokes_JJm[okk_JJm]**2/nvar_JJm[okk_JJm]
                triuidx = np.triu_indices(cycles,1)             # _i enumeration of cycle differences
                chi2cyclek_im = chi2cyclek_JJm[triuidx]                    
                badcyclechik_m = (chi2cyclek_im > chi2lim).any(axis=(0))
                badcyclechiallk_m = (badcyclechik_m & (okk_JJm[triuidx].reshape((-1,samples)).sum(axis=0)<3))
                badcyclechicullk_m = (badcyclechik_m & np.logical_not(badcyclechiallk_m))

                wavcullk_M = np.where(badcyclechicullk_m)[0]      # cycles>2, cull by voting 
                if wavcullk_M.shape[0]:
                    for M,m in enumerate(wavcullk_M):                       
                        J_I = np.array(triuidx).T[np.argsort(chi2cyclek_im[:,m])].flatten()
                        _,idx = np.unique(J_I,return_index=True)
                        Jcull = J_I[np.sort(idx)][-1]
                        jcull = jListk[k][Jcull] 
                        iscull_jm[jcull,m] = True                # for reporting
                        bpm_jSm[jcull,:,m] = 1
                else:
                    for j in jListk[k]:
                        iscull_jm[j] = badcyclechiallk_m         # for reporting
                        bpm_jSm[j,:,badcyclechiallk_m] = 1
                for J,j in enumerate(jListk[k]):
                    bpmk_Jm[J] = bpm_jSm[j,0]

                if debug:
                    obsname = object+"_"+config 
                    okk_Jm = okchik_m[None,:] & (bpmk_Jm ==0)
                    Tw_dm = np.unravel_index(range(samples),(Targets,wavs))
                    np.savetxt(obsname+"_nstokesk_Jm_"+str(k)+".txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],   \
                        okk_Jm.astype(int), nstokesk_Jm,    \
                        nvark_Jm)).T, fmt="%3i %8.2f "+cycles*"%3i "+cycles*"%10.6f "+cycles*"%10.12f ")                        
                    np.savetxt(obsname+"_chi2cyclek_im_"+str(k)+".txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]], \
                        okchik_m.astype(int),chi2cyclek_im, badcyclechik_m,okk_JJm[triuidx].sum(axis=0))).T, \
                        fmt="%3i %8.2f %3i "+chi2cyclek_im.shape[0]*"%10.3f "+" %2i %2i") 
                    np.savetxt(obsname+"_iscull_km_"+str(k)+".txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],   \
                        okchik_m.astype(int), iscull_jm[jListk[k],:].astype(int))).T,   \
                            fmt="%3i %8.2f %3i "+cycles*" %3i") 

          # Now combine cycles, using normalized stokes to minimize systematic errors

          # first normalize cycle members J at samples where all cycles have data:
            cycles_km[k] =  (1-bpmk_Jm).sum(axis=0).astype(int)
            ok_m = (cycles_km[k] > 0)
            okchik_m = (cycles_km[k] > 1)                 
            okallk_m = (cycles_km[k] == cycles)                
            normint_J = np.array(stokes_jSm[jListk[k],0][:,okallk_m].sum(axis=1))
            normint_J /= np.mean(normint_J)
            stokes_JSm = stokes_jSm[jListk[k]]/normint_J[:,None,None]
            var_JSm = var_jSm[jListk[k]]/normint_J[:,None,None]**2
            covar_JSm = covar_jSm[jListk[k]]/normint_J[:,None,None]**2

            for J in range(cycles):
                okJ_m = ok_m & (bpmk_Jm[J] ==0)
              # average the intensity
                stokes_kSm[k,0][okJ_m] += stokes_JSm[J,0][okJ_m]/cycles_km[k][okJ_m]
                var_kSm[k,0][okJ_m] += var_JSm[J,0][okJ_m]/cycles_km[k][okJ_m]**2
                covar_kSm[k,0][okJ_m] += covar_JSm[J,0][okJ_m]/cycles_km[k][okJ_m]**2
              # now the normalized stokes
                nstokes_km[k][okJ_m] +=   \
                    (stokes_JSm[J,1][okJ_m]/stokes_JSm[J,0][okJ_m])/cycles_km[k][okJ_m]
                nvar_km[k][okJ_m] +=  \
                    (var_JSm[J,1][okJ_m]/stokes_JSm[J,0][okJ_m]**2)/cycles_km[k][okJ_m]**2                    
                ncovar_km[k][okJ_m] +=    \
                    (covar_JSm[J,1][okJ_m]/stokes_JSm[J,0][okJ_m]**2)/cycles_km[k][okJ_m]**2
            stokes_kSm[k,1] = nstokes_km[k]*stokes_kSm[k,0]
            var_kSm[k,1] = nvar_km[k]*stokes_kSm[k,0]**2 
            covar_kSm[k,1] = ncovar_km[k]*stokes_kSm[k,0]**2                           

            if debug:
                obsname = object+"_"+config 
                Tw_dm = np.unravel_index(range(samples),(Targets,wavs))                    
                np.savetxt(obsname+"_stokes_kSm_"+str(k)+".txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],   \
                    ok_m.astype(int), okallk_m.astype(int), bpmk_Jm, stokes_kSm[k])).T,   \
                    fmt="%3i %8.2f %3i %3i "+cycles*"%3i "+2*"%12.3f ")                          

          # compute mean chisq by target for each pair having multiple cycles 
            okallk_Tw = okallk_m.reshape((Targets,wavs)) 
                      
            if cycles > 1:
                nstokeserr_Jm = np.zeros((cycles,samples))
                nstokeserr_JTw = nstokeserr_Jm.reshape((cycles,Targets,wavs))
                nerr_Jm = np.zeros_like(nstokeserr_Jm)
                nerr_JTw = nerr_Jm.reshape((cycles,Targets,wavs)) 
                for J in range(cycles):
                    okJ_m = ok_m & (bpmk_Jm[J] ==0)
                    nstokesk_Jm[J][okJ_m] = stokes_JSm[J,1][okJ_m]/stokes_JSm[J,0][okJ_m]
                    nvark_Jm[J][okJ_m] = var_JSm[J,1][okJ_m]/(stokes_JSm[J,0][okJ_m])**2                    
                    nstokeserr_Jm[J] = nstokesk_Jm[J] - nstokes_km[k]
                    nvar_m = nvark_Jm[J] - nvar_km[k]
                    okallk_m &= (nvar_m > 0.)
                    nerr_Jm[J][okallk_m] = np.sqrt(nvar_m[okallk_m])
                        
                nstokessyserr_J = np.average(nstokeserr_Jm[:,okallk_m].reshape((cycles,-1)),    \
                    weights=1./nerr_Jm[:,okallk_m],axis=1)
                nstokeserr_Jm -= nstokessyserr_J[:,None]
                chi2cycles_kT[k] = okallk_Tw.sum(axis=1)                    
                okcyclechi_kT[k] = (chi2cycles_kT[k] > 2)                   
                for J,j in enumerate(jListk[k]):
                    loc,scale = norm.fit(nstokeserr_Jm[J][okallk_m]/nerr_Jm[J][okallk_m])
                    chi2cycle_j[j] = scale**2
                    syserrcycle_j[j] = nstokessyserr_J[J]
                    for T in range(Targets):
                        if (~okcyclechi_kT[k,T]): continue
                        loc,scale = norm.fit(nstokeserr_JTw[J,T][okallk_Tw[T]]/nerr_JTw[J,T][okallk_Tw[T]])                            
                        chi2cycle_jT[j,T] = scale**2
                chi2cyclenet_k[k] = chi2cycle_j[jListk[k]].mean()
                chi2cyclenet_kT[k] = chi2cycle_jT[jListk[k]].mean(axis=0)
                syserrcyclenet_k[k] = np.sqrt((syserrcycle_j[jListk[k]]**2).sum())/len(jListk[k])

                if debug:   
                    obsname = object+"_"+config
                    chisqanalysis(obsname,nstokeserr_Jm,nerr_Jm,okallk_m)
                                                                     
      # for each obs combine raw stokes, mean dPA calibration as appropriate for pattern, and save

        if debug:              
            rsslog.message("\nrawListjd", logfile, with_stdout=with_stdout)
            for i in range(len(rawListjd)):
                rsslog.message(repr(rawListjd[i]), logfile, with_stdout=with_stdout)
            rsslog.message("\ncombListkd", logfile, with_stdout=with_stdout)
            for i in range(len(combListkd)):
                rsslog.message(repr(combListkd[i]), logfile)                                                
            rsslog.message(("\nobsListod"+repr(obsListod)), logfile, with_stdout=with_stdout)

        for obs in range(obss):
            k0,object,config,wppat,pairs = obsListod[obs]
            patpairs = patternpairs[wppat]
            kList = range(k0,k0+pairs)                                      # entries in combListkd for this obs            
            jList = sum([jListk[k] for k in kList],[])
            maskasec = maskasec_j[jList[0]]                                    
            telpa = angle_average(telpa_j[jList])
            TRKyx_d = TRKyx_dj[:,jList].mean(axis=1) - np.diff(TRKyx_dj[:,jList],axis=1).mean(axis=1)/2.            
            MByx_d = MByx_dj[:,jList].mean(axis=1) - np.diff(MByx_dj[:,jList],axis=1).mean(axis=1)/2.
                        
            rho = rho_j[jList].mean() - np.diff(rho_j[jList]).mean()/2.
            obsDT = obsDT_j[jList].min() + (obsDT_j[jList].max() - obsDT_j[jList].min())/2.                                              
            obsname = object+"_"+config
            wpList = [combListkd[k][3][1:] for k in kList]
            patwpList = sorted((patpairs*"%1s%1s " % tuple(patterndict[wppat].flatten())).split())
            pList = [patwpList.index(wpList[P]) for P in range(pairs)]
                        
            rawp_j = np.array([patwpList.index(rawListjd[j][3][1:]) for j in range(rawstokess)]) 
            rawJ_j = np.array([int(rawListjd[j][4]) for j in range(rawstokess)])              
                        
            k_p = np.zeros(patpairs,dtype=int)                              
            k_p[pList] = kList                                                # idx in kList for each pair idx
            cycles_p = np.zeros_like(k_p)
            cycles_p[pList] = np.array([combListkd[k][4] for k in kList])     # number of cycles in comb
            cycles_pm = np.zeros((patpairs,samples),dtype=int)
            cycles_pm[pList] = cycles_km[kList]                               # of ok cycles for each wavelength
            havecyclechi_p = np.zeros(patpairs,dtype=bool)
            havecyclechi_p[pList] = havecyclechi_k[kList]
            havelinhichi_p = np.zeros(patpairs,dtype=bool)
               
          # name result to document hw cycles included
            kpList = list(k_p)
            if cycles_p.max()==cycles_p.min(): kpList = [kList[0],] 

            for p in range(len(kpList)):
                obsname += "_"
                j0 = combListkd[k_p[p]][0] - cycles_p[p] + 1
                if (max([int(rawListjd[j][4]) for j in range(j0,j0+cycles_p[p])])>9): 
                    obsname += 'd'                                          # double-digit cycles
                    for j in range(j0,j0+cycles_p[p]): obsname+=rawListjd[j][4]
                else:
                    for j in range(j0,j0+cycles_p[p]): obsname+=rawListjd[j][4][-1]                 
            rsslog.message("\n  Observation: %s  Date: %s" % (obsname,dateobs), logfile, with_stdout=with_stdout)
            finstokess = patternstokes[wppat]   

            if pairs != patpairs:
                if (pairs<2):
                    rsslog.message(('  Only %1i pair, skipping observation' % pairs), logfile, with_stdout=with_stdout)
                    continue
                elif ((max(pList) < 2) | (min(pList) > 1)):
                    rsslog.message('  Pattern not usable, skipping observation', logfile, with_stdout=with_stdout)
                    continue

            stokes_Fm = np.zeros((finstokess,samples))
            var_Fm = np.zeros_like(stokes_Fm)
            covar_Fm = np.zeros_like(stokes_Fm)

          # normalize pairs in obs at tw samples where all pair/cycles have data:
            okall_m = okcal_om[obs] & (cycles_pm[pList] == cycles_p[pList,None]).all(axis=0)     
            normint_K = stokes_kSm[kList,0][:,okall_m].sum(axis=1)
            normint_K /= np.mean(normint_K)
            stokes_kSm[kList] /= normint_K[:,None,None]
            var_kSm[kList] /= normint_K[:,None,None]**2
            covar_kSm[kList] /= normint_K[:,None,None]**2

          # first, the intensity
            stokes_Fm[0] = stokes_kSm[kList,0].sum(axis=0)/pairs
            var_Fm[0] = var_kSm[kList,0].sum(axis=0)/pairs**2 
            covar_Fm[0] = covar_kSm[kList,0].sum(axis=0)/pairs**2
            badlinhichi_m = np.zeros(samples,dtype=bool)                         
          # now, the polarization stokes
            if wppat.count('LINEAR'):
                var_Fm = np.vstack((var_Fm,np.zeros(samples)))           # add QU covariance
                if (wppat=='LINEAR'):
                # samples with both pairs having good, calibratable data in at least one cycle
                    ok_m = okcal_om[obs] & (cycles_pm[pList] > 0).all(axis=0)
                    bpm_Fm = np.repeat((np.logical_not(ok_m))[None,:],finstokess,axis=0)
                    stokes_Fm[1:,ok_m] = stokes_kSm[kList,1][:,ok_m]*(stokes_Fm[0,ok_m]/stokes_kSm[kList,0][:,ok_m])
                    var_Fm[1:3,ok_m] = var_kSm[kList,1][:,ok_m]*(stokes_Fm[0,ok_m]/stokes_kSm[kList,0][:,ok_m])**2
                    covar_Fm[1:,ok_m] = covar_kSm[kList,1][:,ok_m]*(stokes_Fm[0,ok_m]/stokes_kSm[kList,0][:,ok_m])**2
                    if debug:                       
                        np.savetxt(obsname+"_stokes_LINm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],ok_m.astype(int),   \
                            stokes_Fm)).T, fmt="%3i %8.2f  "+"%2i "+3*" %10.6f")
                        np.savetxt(obsname+"_var_LINm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],ok_m.astype(int),    \
                            var_Fm)).T, fmt="%3i %8.2f  "+"%2i "+4*"%14.9f ")
                        np.savetxt(obsname+"_covar_LINm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],ok_m.astype(int),  \
                            covar_Fm)).T, fmt="%3i %8.2f  "+"%2i "+3*"%14.9f ")                       

                elif wppat=='LINEAR-HI':
                # for Linear-Hi, must go to normalized stokes in order for the pair combination to cancel systematic errors
                # each pair p at each sample tw is linear combination of pairs, including primary p and secondary sec_p
                # linhi chisq is from comparison of primary and secondary
                # evaluate wavelengths with at least both pairs 0,2 or 1,3 having good, calibratable data in at least one cycle: 
                    ok_pm = okcal_om[obs,None,:] & (cycles_pm > 0)
                    ok_m = (ok_pm[0] & ok_pm[2]) | (ok_pm[1] & ok_pm[3])
                    bpm_Fm = np.repeat((np.logical_not(ok_m))[None,:],finstokess,axis=0)                    
                    stokespri_pm = np.zeros((patpairs,samples))
                    varpri_pm = np.zeros_like(stokespri_pm)
                    covarpri_pm = np.zeros_like(stokespri_pm)
                    stokespri_pm[pList] = nstokes_km[kList]
                    varpri_pm[pList] = nvar_km[kList]
                    covarpri_pm[pList] = ncovar_km[kList]
                    haveraw_pm = (cycles_pm > 0)
                    pricof_ppm = np.identity(patpairs)[:,:,None]*haveraw_pm[None,:,:]                      

                    qq = 1./np.sqrt(2.)
                    seccofb_pp = np.array([[ 0,1,  0,-1],[1, 0,1,  0],[  0,1, 0,1],[-1,  0,1, 0]])*qq    # both secs avail
                    seccof1_pp = np.array([[qq,1,-qq, 0],[1,qq,0, qq],[-qq,1,qq,0],[-1, qq,0,qq]])*qq    # only 1st sec                        
                    seccof2_pp = np.array([[qq,0, qq,-1],[0,qq,1,-qq],[ qq,0,qq,1],[ 0,-qq,1,qq]])*qq    # only 2nd sec
                    secList_p = np.array([[1,3],[0,2],[1,3],[0,2]])
                    havesecb_pm = haveraw_pm[secList_p].all(axis=1)
                    onlysec1_pm = (np.logical_not(havesecb_pm) & haveraw_pm[secList_p][:,0] & havesecb_pm[secList_p][:,1])
                    onlysec2_pm = (np.logical_not(havesecb_pm) & haveraw_pm[secList_p][:,1] & havesecb_pm[secList_p][:,0])
                    seccof_ppm = seccofb_pp[:,:,None]*havesecb_pm[:,None,:] + \
                        seccof1_pp[:,:,None]*onlysec1_pm[:,None,:] + \
                        seccof2_pp[:,:,None]*onlysec2_pm[:,None,:] 
                    stokessec_pm = (seccof_ppm*stokespri_pm[:,None,:]).sum(axis=0)
                    varsec_pm = (seccof_ppm**2*varpri_pm[:,None,:]).sum(axis=0)
                    covarsec_pm = (seccof_ppm**2*covarpri_pm[:,None,:]).sum(axis=0)

                    havesec_pm = (havesecb_pm | onlysec1_pm | onlysec2_pm)
                    prisec_pm = (haveraw_pm & havesec_pm)
                    onlypri_pm = (haveraw_pm & np.logical_not(havesec_pm))
                    onlysec_pm = (np.logical_not(haveraw_pm) & havesec_pm)
                        
                    cof_ppm = onlypri_pm[:,None,:]*pricof_ppm + onlysec_pm[:,None,:]*seccof_ppm +   \
                        0.5*prisec_pm[:,None,:]*(pricof_ppm+seccof_ppm)

                  # now do the combination
                    stokes_pm = (cof_ppm*stokespri_pm[None,:,:]).sum(axis=1)
                    var_pm = (cof_ppm**2*varpri_pm[None,:,:]).sum(axis=1)
                    covar_pm = (cof_ppm**2*covarpri_pm[None,:,:]).sum(axis=1)
                    covarprisec_pm = 0.5*varpri_pm*np.logical_or(onlysec1_pm,onlysec2_pm)
                    covarqu_m = (cof_ppm[0]*cof_ppm[2]*varpri_pm).sum(axis=0)

                  # cull samples based on chisq between primary and secondary (but not for lamp cal data)
                    havelinhichi_p = prisec_pm.any(axis=1)
                    linhichis = havelinhichi_p.sum()
                    chi2linhi_pm = np.zeros((patpairs,samples))
                    chi2linhi_pm[prisec_pm] = ((stokespri_pm[prisec_pm] - stokessec_pm[prisec_pm])**2 / \
                        (varpri_pm[prisec_pm] + varsec_pm[prisec_pm] - 2.*covarprisec_pm[prisec_pm]))
                    q3_p = np.percentile(chi2linhi_pm[:,okall_m].reshape((4,-1)),75,axis=1)
                        
                    if issky:                        
                        badlinhichi_m[ok_m] =   \
                            ((chi2linhi_pm[:,ok_m] > (chifence_d[2]*q3_p)[:,None])).any(axis=0)                       
                                      
                    ok_m &= np.logical_not(badlinhichi_m)
                    okall_m &= np.logical_not(badlinhichi_m)
                    chi2linhi_p = np.zeros(patpairs)
                    chi2linhi_p[havelinhichi_p] = (chi2linhi_pm[havelinhichi_p][:,ok_m]).sum(axis=1)/    \
                        (prisec_pm[havelinhichi_p][:,ok_m]).sum(axis=1)

                    havelinhichi_pTw = prisec_pm.reshape((patpairs,Targets,wavs))
                    chi2linhi_pTw = chi2linhi_pm.reshape((patpairs,Targets,wavs))
                    oklinhichi_pTw =  havelinhichi_pTw & ok_m.reshape((Targets,wavs))[None,:,:]
                    oklinhichi_pT = oklinhichi_pTw.any(axis=2)
                    chi2linhi_pT = np.zeros((patpairs,Targets))
                    chi2linhi_pT[oklinhichi_pT] = (oklinhichi_pTw*chi2linhi_pTw).sum(axis=2)[oklinhichi_pT]/    \
                        oklinhichi_pTw.sum(axis=2)[oklinhichi_pT]
                        
                    syserrlinhi_pm = np.zeros((patpairs,samples))
                    varlinhi_pm = np.zeros((patpairs,samples))
                    syserrlinhi_p = np.zeros(patpairs)
                    syserrlinhi_pm[prisec_pm] = (stokespri_pm[prisec_pm] - stokessec_pm[prisec_pm])
                    varlinhi_pm[prisec_pm] = varpri_pm[prisec_pm] + varsec_pm[prisec_pm] - 2.*covarprisec_pm[prisec_pm]
                    syserrlinhi_p[havelinhichi_p] = np.average(syserrlinhi_pm[havelinhichi_p][:,okall_m], \
                        weights=1./np.sqrt(varlinhi_pm[havelinhichi_p][:,okall_m]),axis=1)

                    if debug:
                        rsslog.message((("   chi2linhi q3_p: "+len(q3_p)*" %6.2f ") % tuple(q3_p)),logfile, with_stdout=with_stdout)
                        np.savetxt(obsname+"_have_pm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],ok_pm.astype(int),  \
                            haveraw_pm,havesecb_pm,onlysec1_pm,onlysec2_pm,havesec_pm,prisec_pm,onlypri_pm,onlysec_pm)).T,   \
                            fmt="%3i %8.2f  "+9*"%2i %2i %2i %2i  ") 
                        np.savetxt(obsname+"_seccof_ppm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],ok_pm.astype(int),   \
                            seccof_ppm.reshape((16,-1)))).T, fmt="%3i %8.2f  "+4*"%2i "+16*" %6.3f") 
                        np.savetxt(obsname+"_cof_ppm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],ok_pm.astype(int),  \
                            cof_ppm.reshape((16,-1)))).T, fmt="%3i %8.2f  "+4*"%2i "+16*" %6.3f")                        
                        np.savetxt(obsname+"_stokes_pm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],ok_pm.astype(int),   \
                            stokespri_pm,stokes_pm)).T, fmt="%3i %8.2f  "+4*"%2i "+8*" %10.6f")
                        np.savetxt(obsname+"_var_pm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],ok_pm.astype(int),  \
                            varpri_pm,var_pm)).T, fmt="%3i %8.2f  "+4*"%2i "+8*"%14.9f ")
                        np.savetxt(obsname+"_covar_pm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],ok_pm.astype(int),    \
                            covarpri_pm,covar_pm)).T, fmt="%3i %8.2f  "+4*"%2i "+8*"%14.9f ")                       
                        np.savetxt(obsname+"_chi2linhi_pm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],stokes_Fm[0],  \
                            ok_pm.astype(int), ok_m.astype(int), chi2linhi_pm)).T,      \
                            fmt="%3i %8.2f %10.0f "+4*"%2i "+"%3i "+4*"%10.4f ")

                    stokes_Fm[1:] = stokes_pm[[0,2]]*stokes_Fm[0]                        
                    var_Fm[1:3] = var_pm[[0,2]]*stokes_Fm[0]**2
                    var_Fm[3] = covarqu_m*stokes_Fm[0]**2
                    covar_Fm[1:] = covar_pm[[0,2]]*stokes_Fm[0]**2
                    bpm_Fm = ((bpm_Fm==1) | np.logical_not(ok_m)).astype(int)

              # document chisq results, combine flagoffs, compute mean chisq for observation, combine with final bpm
                chi2qudof_T = np.zeros(Targets)                      
                if (havecyclechi_p.any() | havelinhichi_p.any()):
                    chi2cyclenet = 0.
                    chi2cyclenet_T = np.zeros(Targets)
                    okcyclechinet_T = np.zeros(Targets).astype(bool)                        
                    syserrcyclenet = 0.
                    chi2linhinet = 0.
                    chi2linhinet_T = np.zeros(Targets)
                    oklinhichinet_T = np.zeros(Targets).astype(bool)                        
                    syserrlinhinet = 0.                  
                    if havecyclechi_p.any():
                        if havepairf21:                    
                            rsslog.message(("\n"+14*" "+    \
                              "{:^"+str(5*patpairs)+"}{:^"+str(7*patpairs+2)+"}{:^"+str(6*patpairs+2)+"}{:^"+str(8*patpairs)+"}")\
                              .format("culled","sys %err","mean chisq","pair f21"), logfile, with_stdout=with_stdout)
                            rsslog.message((9*" "+"HW "+patpairs*" %4s"+patpairs*" %7s"+"  "+patpairs*" %5s"+"  "+patpairs*" %7s") \
                              % tuple(4*patwpList),logfile, with_stdout=with_stdout)
                        else:
                            rsslog.message(("\n"+14*" "+    \
                              "{:^"+str(5*patpairs)+"}{:^"+str(7*patpairs+2)+"}{:^"+str(6*patpairs)+"}")\
                              .format("culled","sys %err","mean chisq"), logfile, with_stdout=with_stdout)
                            rsslog.message((9*" "+"HW "+patpairs*" %4s"+patpairs*" %7s"+patpairs*" %6s") \
                              % tuple(3*patwpList),logfile, with_stdout=with_stdout)                                                      
                        JList = list(set(sum([JListk[k] for k in kList],[])))
                        Jmax = max(JList)
                        ok_pJ = np.zeros((patpairs,Jmax+1),dtype=bool)
                        for p in pList: ok_pJ[p][JListk[k_p[p]]] = True
                             
                        syserrcycle_pJ = np.zeros((patpairs,Jmax+1))
                        syserrcycle_pJ[ok_pJ] = syserrcycle_j[jList]
                            
                        syserrcyclenet_p = np.zeros(patpairs)
                        syserrcyclenet_p[pList] = syserrcyclenet_k[kList]
                        syserrcyclenet = np.sqrt((syserrcyclenet_p**2).sum()/patpairs) 

                        chi2cycle_pJ = np.zeros((patpairs,Jmax+1))
                        chi2cycle_pJ[ok_pJ] = chi2cycle_j[jList]
                            
                        chi2cyclenet_pT = np.zeros((patpairs,Targets))          
                        chi2cyclenet_pT[pList,:] = chi2cyclenet_kT[kList,:]

                        okcyclechi_pT = np.zeros((patpairs,Targets)).astype(bool)
                        okcyclechi_pT[pList,:] = okcyclechi_kT[kList,:]
                        okcyclechinet_T = okcyclechi_pT.any(axis=0)                                                                                                         
                        chi2cyclenet_T[okcyclechinet_T] = chi2cyclenet_pT.sum(axis=0)[okcyclechinet_T] /  \
                            okcyclechi_pT.sum(axis=0)[okcyclechinet_T]

                        okcyclechinet_p = okcyclechi_pT.any(axis=1)
                        chi2cyclenet_p = np.zeros(patpairs)
                        chi2cyclenet_p[okcyclechinet_p] = chi2cyclenet_pT.sum(axis=1)[okcyclechinet_p] /    \
                            okcyclechi_pT.sum(axis=1)[okcyclechinet_p] 
                        chi2cyclenet = chi2cyclenet_pT.sum()/okcyclechi_pT.sum()                         

                        culls_pJ = np.zeros((patpairs,Jmax+1),dtype=int)
                        culls_pJ[ok_pJ] = iscull_jm[jList].sum(axis=1) 

                        pairf21_pJ = np.zeros((patpairs,Jmax+1))
                        
                        for p in range(patpairs):                                
                            for J in range(Jmax+1):
                                if (not ok_pJ[p,J]): continue
                                j = np.where((rawp_j==p) & (rawJ_j==J+1))[0][0]                                
                                pairf21_pJ[p,J] = pairf21_j[j]

                        if cycles_p.max() > 2:                                                                         
                            for J in set(JList):
                                if havepairf21:                            
                                    rsslog.message((("   cycle %2i: "+patpairs*"%4i "+patpairs*"%7.3f "+"  "+   \
                                      patpairs*"%5.2f "+"  "+patpairs*"%7.3f ") % ((J+1,)+tuple(culls_pJ[:,J])+ \
                                      tuple(100.*syserrcycle_pJ[:,J])+tuple(chi2cycle_pJ[:,J])+     \
                                      tuple(pairf21_pJ[:,J]))),  logfile, with_stdout=with_stdout)
                                else:
                                    rsslog.message((("   cycle %2i: "+patpairs*"%4i "+patpairs*"%7.3f ") %     \
                                      ((J+1,)+tuple(culls_pJ[:,J])+tuple(100.*syserrcycle_pJ[:,J])+ \
                                      tuple(chi2cycle_pJ[:,J]))), logfile, with_stdout=with_stdout)
                                      
                        netculls_p = [iscull_jm[jListk[k_p[p]]].all(axis=0).sum() for p in range(patpairs)]
                        rsslog.message(("    net    : "+patpairs*"%4i "+patpairs*"%7.3f "+"  "+patpairs*"%5.2f ") %     \
                            (tuple(netculls_p)+tuple(100*syserrcyclenet_p)+tuple(chi2cyclenet_p)), logfile, with_stdout=with_stdout)

                        if debug:                            
                            culls_jT = iscull_jm[jList].reshape((-1,Targets,wavs)).sum(axis=2)
                            np.savetxt(obsname+'_cull_T.txt',culls_jT.reshape((-1,Targets)).T,fmt='%4i ')                                 
                            chi2cycles_pT = np.zeros((patpairs,Targets),dtype=int)
                            chi2cycles_pT[pList,:] = chi2cycles_kT[kList,:]
                            np.savetxt(obsname+'_cyclechi_T.txt', \
                                np.vstack((chi2cycles_pT,chi2cyclenet_pT,chi2cyclenet_T)).T,    \
                                fmt=patpairs*'%4i '+patpairs*'%7.2f '+'%7.2f ')
                    if (havelinhichi_p.any()):
                        chicount = int(badlinhichi_m.sum())
                        chi2linhinet = chi2linhi_p.sum()/(havelinhichi_p.sum())
                        oklinhichinet_T = oklinhichi_pTw.any(axis=2).any(axis=0)
                        chi2linhinet_T[oklinhichinet_T] = chi2linhi_pT.mean(axis=0)[oklinhichinet_T]
                        syserrlinhinet = np.sqrt((syserrlinhi_p**2).sum()/(havelinhichi_p.sum()))
                        if havepairf21:
                            pairf21_p = pairf21_j[np.argsort(rawp_j)]                            
                            rsslog.message(("\n"+14*" "+"{:^"+str(5*patpairs)+"}{:^"+str(7*patpairs+2)+"}{:^"+str(6*patpairs+2)+\
                              "}{:^"+str(7*patpairs+1)+"}").format("culled","sys %err","mean chisq","pair f21"),logfile,    \
                              with_stdout=with_stdout)
                            rsslog.message((9*" "+"HW "+(4*patpairs/2)*" "+" all"+(4*patpairs/2)*" "+patpairs*" %6s"+"  "+   \
                              patpairs*" %5s"+"  "+patpairs*" %6s") % tuple(3*patwpList),logfile, with_stdout=with_stdout)
                            rsslog.message(("      Linhi: "+(2*patpairs)*" "+"%3i "+(2*patpairs)*" "+patpairs*"%6.3f "+"  "+patpairs*"%5.2f "+\
                              "  "+patpairs*"%6.3f ") % ((chicount,)+tuple(100.*syserrlinhi_p)+tuple(chi2linhi_p)+tuple(pairf21_p)), logfile,   \
                              with_stdout=with_stdout)
                        else:
                            rsslog.message(("\n"+14*" "+"{:^"+str(5*patpairs)+"}{:^"+str(8*patpairs)+"}{:^"+str(6*patpairs)+"}")\
                              .format("culled","sys %err","mean chisq"), logfile, with_stdout=with_stdout)
                            rsslog.message((9*" "+"HW "+(4*patpairs/2)*" "+" all"+(4*patpairs/2)*" "+patpairs*" %7s"+   \
                              patpairs*" %5s") % tuple(2*patwpList),logfile, with_stdout=with_stdout)
                            rsslog.message(("      Linhi: "+(2*patpairs)*" "+"%3i "+(2*patpairs)*" "+patpairs*"%7.3f "+patpairs*"%5.2f ") % \
                              ((chicount,)+tuple(100.*syserrlinhi_p)+tuple(chi2linhi_p)), logfile, with_stdout=with_stdout)                       
                        if debug:
                            np.savetxt(obsname+"_linhicull_tw.txt",np.vstack((wav_w,   \
                            badlinhichi_m.reshape((Targets,wavs)).astype(int))).T,fmt="%8.2f "+Targets*"%2i ")

                    ok_T = ok_m.reshape((Targets,wavs)).any(axis=1)
                    if "tgtTab" in locals():
                        badtarg_i = np.in1d(np.arange(entries),i_t[~ok_t])
                        if badtarg_i.sum():
                            rsslog.message((("\n  Culled TGT entries for incomplete: "+badtarg_i.sum()*"%3i ") %    \
                                tuple(np.where(badtarg_i)[0])),logfile, with_stdout=with_stdout)
                            oktgt_i[badtarg_i] = False
                        tgtTab['CULL'][badtarg_i] = 'compl'

                    okchi2net_T = (okcyclechinet_T | oklinhichinet_T)
                    chi2qudof_T[okchi2net_T] = (chi2cyclenet_T + chi2linhinet_T)[okchi2net_T]/      \
                        (okcyclechinet_T.astype(int) + oklinhichinet_T.astype(int))[okchi2net_T]
                    chi2qudof = chi2qudof_T[okchi2net_T].mean()
                    syserr = np.sqrt((syserrcyclenet**2+syserrlinhinet**2)/    \
                        (int(syserrcyclenet>0)+int(syserrlinhinet>0)))              
                    rsslog.message(("\n  Estimated sys %%error: %5.3f%%\n  Mean Chisq:         %6.2f") % \
                        (100.*syserr,chi2qudof), logfile, with_stdout=with_stdout)

                hpacor_m = dpa*np.ones(samples)
                if (not Heffcal_override):                 # internal HW properties
                  # apply hw pa cal, equatorial PA rotation calibration averaged over track                
                    hpacor_m = (hpacormean_om[obs]  + dpa) 
                    okcal_om[obs] &= okcal_Tw.flatten()
                    ok_m &= okcal_om[obs]                                                                                    
                    stokes_Fm,var_Fm,covar_Fm = specpolrotate(stokes_Fm,var_Fm,covar_Fm,hpacor_m)

                    if (not useoldheffcal):
                        if pacorfile:
                            slitdwav = (C0*maskasec*(FCam6000/FColl6000)/15.)*(wavs*dwav/(3*2048))
                            stokes_Fm,var_Fm,covar_Fm = \
                                heffcalcorrect(datadir+pacorfile,wav_w,stokes_Fm,var_Fm,covar_Fm,ok_m,  \
                                    slitdwav=slitdwav,debug=debug)

                    if PAcaltype == "Equatorial":
                        stokes_Fm,var_Fm,covar_Fm = specpolrotate(stokes_Fm,var_Fm,covar_Fm,(telpa % 180))                          
                    ok_m &= okcal_om[obs]
                        
                    if debug:
                        np.savetxt(obsname+"_hpacor_0w.txt",np.vstack((wav_w,hpacor_m[:wavs])).T,fmt="%8.1f %9.3f")
                        np.savetxt(obsname+"_rawstokes_Fm.txt",np.vstack((Tw_dm[0],wav_w[Tw_dm[1]],bpm_Fm,   \
                            stokes_Fm)).T, fmt="%3i %8.2f  "+3*"%2i "+3*" %12.2f")               

                if debug:                                               
                    np.savetxt(obsname+"_stokes_Fm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],bpm_Fm,   \
                        stokes_Fm)).T, fmt="%3i %8.2f  "+3*"%2i "+3*" %12.2f")
                    np.savetxt(obsname+"_var_Fm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],bpm_Fm,    \
                        var_Fm)).T, fmt="%3i %8.2f  "+3*"%2i "+4*"%12.2f ")
                    np.savetxt(obsname+"_covar_Fm.txt",np.vstack((tw_dm[0],wav_w[tw_dm[1]],bpm_Fm,  \
                        covar_Fm)).T, fmt="%3i %8.2f  "+3*"%2i "+3*"%12.2f ") 

                # save final stokes fits file for this observation.  Strain out nans. Remove culled targets

                infile = infileList[rawListjd[combListkd[k][0]][0]]
                hduout = pyfits.open(infile)
                hduout['SCI'].data = np.nan_to_num(stokes_Fm.reshape((finstokess,Targets,wavs)))
                hduout['SCI'].header['CTYPE3'] = 'I,Q,U'
                hduout['VAR'].data = np.nan_to_num(var_Fm.reshape((finstokess+1,Targets,wavs)))
                hduout['VAR'].header['CTYPE3'] = 'I,Q,U,QU'
                hduout['COV'].data = np.nan_to_num(covar_Fm.reshape((finstokess,Targets,wavs)))
                hduout['COV'].header['CTYPE3'] = 'I,Q,U'
                hduout['BPM'].data = bpm_Fm.reshape((finstokess,Targets,wavs)).astype('uint8')
                hduout['BPM'].header['CTYPE3'] = 'I,Q,U'
                if (targets>1):
                    hduout['TGT'] = pyfits.table_to_hdu(tgtTab)
        
                hduout[0].header['TELPA'] =  round(telpa,4)
                hduout[0].header['TRKRHO'] =  round(rho,4)
                hduout[0].header['DATE-OBS'] =  str(obsDT).split('T')[0]
                hduout[0].header['UTC-OBS'] =  str(obsDT).split('T')[1]               
                hduout[0].header['TRKY'] =  round(TRKyx_d[0],4)
                hduout[0].header['TRKX'] =  round(TRKyx_d[1],4)
                hduout[0].header['MBY'] =  round(MByx_d[0],6)
                hduout[0].header['MBX'] =  round(MByx_d[1],6)                                                       
                hduout[0].header['WPPATERN'] = wppat
                hduout[0].header['PATYPE'] = PAcaltype
                if len(calhistoryList):
                    for line in calhistoryList: hduout[0].header.add_history(line)
                if (havecyclechi_p.any() | havelinhichi_p.any()): 
                    hduout[0].header['SYSERR'] = (100.*syserr,'estimated % systematic error')

                if nametag: nametag = "_"+nametag                    
                outfile = obsname+nametag+'_'+Heffcal_override*'raw'+'stokes.fits'
                hduout.writeto(outfile,overwrite=True,output_verify='warn')
                
                if (not Heffcal_override):
                    accmeanPA, accmeanPAerr = calaccuracy_assess(outfile, debug=debug)[:2]
                                    
                    rsslog.message("  Resid cal PA ripple: %6.3f +/- %6.3f deg" %   \
                        (accmeanPA, accmeanPAerr), logfile, with_stdout=with_stdout)
                                        
                rsslog.message('\n    '+outfile+' Stokes I,Q,U', logfile, with_stdout=with_stdout)

                # apply flux calibration for specpol, if available
                fluxcal_m = polflux(outfile,logfile=logfile, with_stdout=with_stdout)
                if fluxcal_m.shape[0]>0:
                    stokes_Fm *= fluxcal_m
                    var_Fm *= fluxcal_m**2
                    covar_Fm *= fluxcal_m**2

                # calculate, print means (target stokes averaged in unnorm space)
                stokes_FTw = stokes_Fm.reshape((-1,Targets,wavs))
                var_FTw = var_Fm.reshape((-1,Targets,wavs))
                covar_FTw = covar_Fm.reshape((-1,Targets,wavs))
                ok_Tw = (bpm_Fm[0].reshape((Targets,wavs)) == 0)
                avstokes_FT = np.zeros((finstokess,Targets))
                avvar_FT = np.zeros_like(avstokes_FT)
                avwav_T = np.zeros(Targets)
                wav1_T = np.zeros(Targets) 
                wav2_T = np.zeros(Targets)                                
                for T in np.arange(Targets):
                    avstokes_f, avvar_f, avwav_T[T] = avstokes(stokes_FTw[:,T,ok_Tw[T]],    \
                        var_FTw[:-1,T,ok_Tw[T]],covar_FTw[:,T,ok_Tw[T]],wav_w[ok_Tw[T]]) 
                    avstokes_FT[:,T] = np.insert(avstokes_f,0,1.)
                    avvar_FT[:,T] = np.insert(avvar_f,0,1.)
                    wav1_T[T] = wav_w[ok_Tw[T]][0]
                    wav2_T[T] = wav_w[ok_Tw[T]][-1]
                wavs_T=ok_Tw.sum(axis=1)
                ok_T = (wavs_T > 0)                                        
                culls_T=(iscull_jm.sum(axis=0)+badlinhichi_m).reshape((Targets,wavs))[ok_T].sum(axis=1)
                chi2_m = [[],chi2qudof_T[ok_T]][chi2qudof_T[ok_T].sum()>0]
                printstokestw(avstokes_FT[:,ok_T],avvar_FT[:,ok_T],np.where(ok_T)[0],tgtname_T[ok_T],   \
                    wav1_T[ok_T], wav2_T[ok_T], chi2_m=chi2_m, wavs_m=wavs_T , \
                    culls_m=culls_T, tcenter=np.pi/2., logfile=logfile, with_stdout=with_stdout) 
                     
#           elif wppat.count('CIRCULAR'):  TBS 

#           elif wppat=='ALL-STOKES':  TBS

      # end of obs loop
  # end of config loop
    return 
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
#-----------------------------------------------
if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)
    polfinalstokes(infileList,**kwargs)

# debug:
# > 20061112 HD14069 + Pol  DO NOT USE
# cd /d/pfis/khn/20061112/sci
# python script.py polfinalstokes.py HD14069_c0_h04_01.fits HD14069_c0_h26_01.fits
# > 20161129 HD298383
# cd /d/pfis/khn/20161129/newcalx (x=1, 2=newHeffcal, 3=include MPlat)
# python script.py polfinalstokes.py HD298383*_h*.fits debug=True
# > 20210303 WR047
# cd /d/pfis/khn/20210303/newcalx (x=1,2)
# python script.py polfinalstokes.py WR*_h*.fits debug=True 
# > 20220226 WR048
# cd /d/pfis/khn/20220226/newcalx (x=1,2)
# python script.py polfinalstokes.py WR*_h*.fits debug=True
# > 20220825 ET211B
# cd /d/pfis/khn/20220825/sci
# python script.py polfinalstokes.py ET21B-00_c0_h04_01.fits ET21B-00_c0_h26_01.fits illum=illum_vt_C.txt LinearPolZeropoint_override=True
# > 20220926 WR006
# cd /d/pfis/khn/20220926/sci_newcal
# python script.py polfinalstokes.py WR006*_h*.fits debug=True
# cd /d/pfis/khn/20211220/newcal
# python polsalt.py specpolrawstokes.py ec*.fits
# python script.py polfinalstokes.py EtaCarina*_h*.fits debug=True
# cd /d/pfis/khn/20230214/newcal_zfit
# python script.py polfinalstokes.py HD298383_1*_h*08.fits HD298383_2*_h*01.fits usrHeffcalfile=RSSpol_Heff_zfit.zip debug=True
# cd /d/pfis/khn/20241225/newcal
# polfinalstokes.py HD38949_*h??_??.fits usrHeffcalfile=RSSpol_Heff_Moon0_10_c0,cy1,cx1_shtrcor_qucor_pacor.txt
# cd /d/pfis/khn/20250214/newcal
# polfinalstokes_test.py LAMLEP_*h??_??.fits usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip illummask=5 nametag=g05calxy9
# specpolview.py LAMLEP_c0_123456_calxy9_stokes.fits LAMLEP_c0_123456_g05calxy9_stokes.fits LAMLEP_c0_123456_g05calxy9_stokes.fits bin=20A connect=hist save=plottext
