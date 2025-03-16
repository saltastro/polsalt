#! /usr/bin/env python2.7

import os, sys, glob, shutil
import numpy as np
from astropy.io import fits as pyfits

polsaltdir = '/usr/users/khn/src/salt/polsaltcurrent/'
datadir = polsaltdir+'polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',polsaltdir+'/scripts/',))
from specpolrawstokes import specpolrawstokes
from polfinalstokes import polfinalstokes
from sppolview import viewstokes
from polutils import specpolrotate, chi2sample, fence
from calassess import calaccuracy_assess
from recalibrate_raw import rawfromname
import rsslog
from obslog import create_obslog
keywordfile = datadir+"obslog_config.json"

"""
Script to recalibrate previously reduced data  in /sci_gamma or /newcal
    infiles:  file,file,file... or   text file of stokes files to recalibrate (dateobs/scidir/file)
    
"""
    
def scirecal(infiles, **kwargs):
    startdir = os.getcwd()
    logfile= startdir+'/recal.log'
    
    rsslog.history(logfile)
    rsslog.message(str(kwargs), logfile)
    rsslog.message('scirecal version: 20250315', logfile)
    
    usrHeffcalfile = kwargs.pop('usrHeffcalfile',"")      # if set, new style cal file is in cwd, not polsalt/data         
    nametag = kwargs.pop('nametag','')
    minwav = float(kwargs.pop('minwav','0.'))
    if (minwav > 0.):
        rsslog.message("\n Wavelengths restricted to >= %6.0f" % minwav, logfile)               
    debug = (str(kwargs.pop('debug','False')).lower() == 'true')
    
    if infiles.count('fits'):
        infileList = infiles.split(',')
    else:
        infileList = np.loadtxt(infiles,dtype=str)       
    obsdate_f = np.array([file.split('/')[0] for file in infileList])
    scidirList = [file.split('/')[1] for file in infileList]
    obsfileList = [file.split('/')[2] for file in infileList]  
    obss = obsdate_f.shape[0]          
    inlen = np.array(map(len,obsfileList)).max()
    objname = obsfileList[0].split('_')[0]
    if (objname[0:2] == 'WR'):
        objname = objname[:2]+str(int(objname[2:])).zfill(3)       # output puts in leading zero in WR name
    outlen = 25+len(nametag)
    RSSdir = "/d/pfis/khn/"

    sciaccmeanPA_f = np.zeros(obss)
    sciaccmeanPAerr_f = np.zeros(obss)
    accmeanPA_f = np.zeros(obss)
    accmeanPAerr_f = np.zeros(obss)
    wavmin = 10000.
    wavmax = 0.

    for fidx in range(obss):
        scidir = RSSdir+str(obsdate_f[fidx])+'/'+scidirList[fidx]
        os.chdir(scidir)
      # get information from sci fits
        if (fidx==0):
            if (not ("COV" in pyfits.open(obsfileList[fidx]))):
                rsslog.message("\n Stokes file has no COV: taken to be zero", logfile)      
            rsslog.message(("\n dateobs   scifile"+(inlen-6)*" "+"newfile"+(outlen-7)*" "+"     obsaccPA   err   newaccPA   err"), logfile)
        scihdr = pyfits.open(obsfileList[fidx])[0].header
        filepartList = obsfileList[fidx].split('_')
        fileparts = len(filepartList)
        confpartno = [p for p in range(fileparts) if (filepartList[p][0]=='c' and len(filepartList[p])==2)][0]        
        targvisit = '_'.join(filepartList[:confpartno])        # will include optional multiple-conf _n in target name
        target = targvisit.split('_')[0]
        visit = targvisit[len(target):]+" "
        sciconfig = filepartList[confpartno]
        cycle = '_'.join(filepartList[(confpartno+1):-1])
                           
        sciaccmeanPA_f[fidx], sciaccmeanPAerr_f[fidx], scimeanacc_hs, sciwavmean_h =  \
            calaccuracy_assess(obsfileList[fidx],minwav=minwav)[:4]
                      
      # tabulate which rawstokes were used     
        wppatern = scihdr['WPPATERN'].upper()        
        rawfileList = rawfromname(obsfileList[fidx],obsdate_f[fidx],wppatern)        

      # tabulate which ec*.fits were used for this config (based on GRTILT, BVISITID)
        grtilt = scihdr['GRTILT']
        if ('BVISITID' in scihdr):
            if len(scihdr['BVISITID']):
                bvisitid = scihdr['BVISITID']
            else:
                bvisitid = 0
        else:
            bvisitid = 0
        ecfileList = sorted(glob.glob('ec*.fits'))  
                
        obsDict = create_obslog(ecfileList,keywordfile)         
        bvisitid_f = np.array(obsDict['BVISITID'])        
        if (bvisitid_f=='UNKNOWN').any():
            bvisitid_f = np.zeros(len(ecfileList),dtype=int)
        
        ecfileList = [ ecfileList[f] for f in range(len(ecfileList))    \
            if ((obsDict['OBJECT'][f].replace(' ','') in target )     \
                and (obsDict['GRTILT'][f]==grtilt )                 \
                and (bvisitid_f[f]==bvisitid )) ]                

        if (len(usrHeffcalfile)>0):                                             
            newcaldir = 'newcal'
        else:
            newcaldir = 'sci_gamma'
        newcalpath = RSSdir+str(obsdate_f[fidx])+'/'+newcaldir
        if (not os.path.exists(newcalpath)): 
            os.makedirs(newcalpath) 
        else:
            okproceed = ((raw_input('\nResult will go in existing directory, OK? (y/n): ')) =='y')
            if (not okproceed): exit()
            
        os.chdir(newcalpath)
        logfile= newcalpath+'/recal.log'
        rsslog.message(("\n dateobs   scifile"+(inlen-6)*" "+"newfile"+(outlen-7)*" "+"     obsaccPA   err   newaccPA   err"), logfile)           

        ecfileList = [ '../'+scidirList[fidx]+'/'+file for file in ecfileList ] 
        echdr = pyfits.open(ecfileList[0])[0].header
        ectarget = echdr['OBJECT'].replace(' ','')
                      
        specpolrawstokes(ecfileList,with_stdout=False)
#        specpolrawstokes(ecfileList, debug=debug) 
       
        for file in glob.glob(ectarget+'_c?_*.fits'):
            os.rename(file,file.replace(ectarget,targvisit))
                
        if usrHeffcalfile:
            shutil.copy(datadir+usrHeffcalfile,os.getcwd()+'/'+usrHeffcalfile) 

        confchar = glob.glob(targvisit+'_c?_*.fits')[0].index('_c')             
        newcalconfig = glob.glob(targvisit+'_c?_h??_*.fits')[0][confchar+1:confchar+3] 
        rawfileList = [f.replace(sciconfig,newcalconfig) for f in rawfileList]
        rawfileList = [f.replace(scidirList[fidx],newcaldir) for f in rawfileList]        
                                  
        polfinalstokes(rawfileList,usrHeffcalfile=usrHeffcalfile,nametag=nametag,with_stdout=False)
        addcycle = ["",cycle+"_"][len(cycle)>0]
        nametag = ["",nametag+"_"][len(nametag)>0]                      
        newcalfile = glob.glob(targvisit+'*_'+addcycle+nametag+'stokes.fits')[0]
        accmeanPA_f[fidx], accmeanPAerr_f[fidx], meanacc_hs, wavmean_h =    \
            calaccuracy_assess(newcalfile,minwav=minwav)[:4]  
        wavmin = min(wavmin,wavmean_h.min())
        wavmax = max(wavmax,wavmean_h.max())
           
        rsslog.message((" %8s %"+str(inlen)+"s %"+str(outlen)+"s %7.3f %7.3f   %7.3f %7.3f") %   \
            (obsdate_f[fidx], obsfileList[fidx], newcalfile,        \
             sciaccmeanPA_f[fidx], sciaccmeanPAerr_f[fidx], accmeanPA_f[fidx], accmeanPAerr_f[fidx]), logfile)          

    rsslog.message("\n mean accPA median    rms     std", logfile)
    scimedmeanPA = np.median(sciaccmeanPA_f)    
    scirmsmeanPA = np.sqrt((sciaccmeanPA_f**2).mean())
    scirmsmeanPAstd = np.std(sciaccmeanPA_f)
    medmeanPA = np.median(accmeanPA_f)                
    rmsmeanPA = np.sqrt((accmeanPA_f**2).mean())
    rmsmeanPAstd = np.std(accmeanPA_f)  
    rsslog.message(" oldcal  %8.3f %8.3f %8.3f \n newcal  %8.3f %8.3f %8.3f" %    \
        (scimedmeanPA,scirmsmeanPA, scirmsmeanPAstd,medmeanPA,rmsmeanPA, rmsmeanPAstd), logfile)     
                   
    return
                   
#--------------------------------------------------------------------            
if __name__=='__main__':
    infiles=sys.argv[1]
    kwargs = dict(x.split('=', 1) for x in sys.argv[2:] )
    scirecal(infiles, **kwargs)

# cd ~/salt/polarimetry/WR/WR021
# scirecal.py WR021_files_debug.txt usrHeffcalfile=RSSpol_Heff_Moon0_7_c0,cy1,cx1.zip nametag=calxy7 dorippleplot=True
# scirecal.py WR021_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True
# cd ~/salt/polarimetry/WR/WR097
# scirecal.py WR097_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True
# scirecal.py WR097_files.txt nametag=calxy10 dorippleplot=True
# cd ~/salt/polarimetry/WR/WR048
# scirecal.py WR048_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True
# scirecal.py WR048_optPA_files.txt nametag=calxy10 dorippleplot=True
# cd ~/salt/polarimetry/WR/WR012
# scirecal.py WR012_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True
# scirecal.py WR012_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True
# cd ~/salt/polarimetry/WR/WR047
# scirecal.py WR047_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True
# cd ~/salt/polarimetry/WR/WR031
# scirecal.py WR031_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True
# cd ~/salt/polarimetry/WR/WR113
# scirecal.py WR113_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True

# cd ~/salt/polarimetry/Standards/OmiSco
# scirecal.py OmiSco_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_7_c0,cy1,cx1.zip nametag=calxy7 dorippleplot=True
# scirecal.py OmiSco_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True minwav=4000.
# scirecal.py OmiSco_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True minwav=4000.
# scirecal.py OmiSco_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_10_c0,cy1,cx1_shtrcor_qucor_pacor.txt nametag=calxy10 dorippleplot=True minwav=4000.

# cd ~/salt/polarimetry/Standards/HD298383
# scirecal.py HD298383_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_7_c0,cy1,cx1.zip nametag=calxy7 dorippleplot=True
# scirecal.py HD298383_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True minwav=4000. 
# scirecal.py HD298383_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True minwav=4000. 
# scirecal.py HD298383_files_1p25asec.txt nametag=calxy10 dorippleplot=True minwav=4000. 
# scirecal.py HD298383_files_3asec.txt nametag=calxy10 dorippleplot=True minwav=4000.

# cd ~/salt/polarimetry/Standards/Hiltner652
# scirecal.py Hiltner652_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_7_c0,cy1,cx1.zip nametag=calxy7 dorippleplot=True
# scirecal.py Hiltner652_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True minwav=4000.
# scirecal.py Hiltner652_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True minwav=4000.
# scirecal.py Hiltner652_files.txt nametag=calxy10 dorippleplot=True minwav=4000.

# cd ~/salt/polarimetry/Standards/NGC2024-1
# scirecal.py NGC2024-1_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True minwav=4000.
# scirecal.py NGC2024-1_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True minwav=4000.
# scirecal.py NGC2024-1_files.txt nametag=calxy10 dorippleplot=True minwav=4000.

# cd ~/salt/polarimetry/Standards/Vela1-95
# scirecal.py Vela1-95_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True minwav=4000.
# scirecal.py Vela1-95_files.txt nametag=calxy10 dorippleplot=True minwav=4000.

# cd ~/salt/polarimetry/Standards/Star+Pol
# scirecal.py 20110613_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True
# for 20241118, 20241123, 20241128:
# scirecal.py 20241128_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True
# specpolview.py 20241128_HD200654_?_calxy8_stokes.fits bin=20A connect=hist save=plottext
# scirecal.py 20241128_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True
# specpolview.py 20241128_HD200654_?_calxy9_stokes.fits bin=20A connect=hist save=plottext
# for 20241204:
# scirecal.py 20241204_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True
# specpolview.py 20241204_HD14374telPA0_?_calxy8_stokes.fits bin=20A connect=hist save=plottext
# scirecal.py 20241204_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True
# specpolview.py 20241204_HD14374telPA0_?_calxy9_stokes.fits bin=20A connect=hist save=plottext
# for 20241208:
# scirecal.py 20241208_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True
# specpolview.py 20241208_HD8779telPA0_calxy8_stokes.fits bin=20A connect=hist save=plottext
# scirecal.py 20241208_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True
# specpolview.py 20241208_HD8779telPA0_calxy9_stokes.fits bin=20A connect=hist save=plottext
# for 20241211:
# scirecal.py 20241211_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_8_c0,cy1,cx1_shtrcor.zip nametag=calxy8 dorippleplot=True
# specpolview.py 20241211_HD074000_calxy8_stokes.fits bin=20A connect=hist save=plottext
# scirecal.py 20241211_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9 dorippleplot=True
# specpolview.py 20241211_HD074000_calxy9_stokes.fits bin=20A connect=hist save=plottext
# for 20241214:
# scirecal.py 20241214_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9
# specpolview.py HD8648telPA0_c0_?_calxy9_stokes.fits bin=20A connect=hist save=plottext
# for 20241217:
# scirecal.py 20241217_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9
# specpolview.py 20241217_HD8648_?_calxy9_stokes.fits bin=20A connect=hist save=plottext
# for 20241225:
# scirecal.py 20241225_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9
# specpolview.py 20241225_HD38949_?_calxy9_stokes.fits bin=20A connect=hist save=plottext
# for 20250206:
# scirecal.py 20250206_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9
# specpolview.py 20241225_HD38949_?_calxy9_stokes.fits bin=20A connect=hist save=plottext
# for 20250214:
# scirecal.py 20250214_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9
# specpolview.py 20250214_LAMLEP_123456_calxy9_stokes.fits bin=20A connect=hist save=plottext
# specpolview.py 20250214_LAMLEP_?_calxy9_stokes.fits bin=20A connect=hist save=plottext
# for 20250217:
# scirecal.py 20250217_files.txt usrHeffcalfile=RSSpol_Heff_Moon0_9_c0,cy1,cx1_shtrcor_qucor.zip nametag=calxy9
# specpolview.py 20250217_HD70110telPA0_123456_calxy9_stokes.fits bin=20A connect=hist save=plottext
# specpolview.py 20250217_HD70110telPA0_?_calxy9_stokes.fits bin=20A connect=hist save=plottext
