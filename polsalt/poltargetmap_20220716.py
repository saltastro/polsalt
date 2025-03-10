
"""
poltargetmap

Compute wavmap, ymap, and target table for polarimetric data of all modes
Write out new images (wm*) with maps as new extensions, target table as embedded fits table

"""

import os, sys, glob, shutil, inspect

import numpy as np
import numpy.polynomial.chebyshev as ch
from scipy.interpolate import interp1d
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.coordinates import Latitude,Longitude
from astropy.table import Table

# this is pysalt-free

import rsslog
from obslog import create_obslog
from polutils import datedline, rssdtralign, configmap
from polmaptools import readmaskxml, readmaskgcode, YXcalc
from imslitlessmap import imfilmap, imspecslitlessmap
from immospolmap import imarcpolmap, imspecmospolmap
from sppolmap import sptilt, sparcpolmap, spmospolmap
   
datadir = os.path.dirname(__file__) + '/data/'
keywordfile = datadir+"obslog_config.json"
#np.set_printoptions(threshold=np.nan)

def poltargetmap(infileList, cutwavoverride=0., logfile='salt.log', isdiffuse=False, debug=False, pidebug=(-1,-1)):
    """derive target, wavelength, spatial maps at detector for each configuration

    Parameters 
    ----------
    infileList: list of strings

    """
    """
    _c configuration index
    _d dimension index y,x = 0,1
    _i file index in infileList
    _j index within files of an observation
    _f filter index 
    _y, _x mm coordinate
    _r, _c bin coordinate
    _t catalog index
    _s located star on image
    _S culled star target
    """

    if debug=='True': debug=True
    
    rsslog.history(logfile)
      
    # group the files together
    confitemlist = ['MASKID','GRATING','GR-ANGLE','CAMANG','BVISITID']
    
    if (len(infileList) == 0):
        rsslog.message("\ninfileList is empty, exitting ", logfile)
        exit()     
        
    obs_i,config_i,obsTab,configTab = configmap(infileList, confitemlist)    
    obss = len(obsTab)
    configs = len(configTab)
    infiles = len(infileList)

    if debug:
        print '\n', obsTab
        print '\n', configTab

    # group ARC into observation
    arcobsList = list(np.where(obsTab['object'] == 'ARC')[0])
    iarc_o = -np.ones(obss,dtype=int)    
    for oarc in arcobsList:
        config = obsTab['config'][oarc]                        
        otarget = np.where((obsTab['config']==config) & (range(obss) != oarc))[0]
        iarc_o[otarget] = np.where(obs_i==oarc)[0][0]   # for now, just use first one       
        obs_i[iarc_o[otarget]] = otarget

    for o in range(obss):
        maskid, grating, grangle, camang, bvisitid = configTab[obsTab[o]['config']]
        fileListj = [infileList[i] for i in range(infiles) if obs_i[i]==o]            
        files = len(fileListj)
        if (files==0): continue         # ARC observations are moved into target obs
        
        obsDictj = create_obslog(fileListj,keywordfile)            
        lampid = obsDictj['LAMPID'][0].replace(' ','')
        objectname = obsTab[o]['object']
        rsslog.message('\nField Name: '+objectname, logfile)

        hdul0 = pyfits.open(fileListj[0])
        rows, cols = hdul0[1].data.shape
        cbin, rbin = [int(x) for x in hdul0[0].header['CCDSUM'].split(" ")]
        trkrho = obsDictj['TRKRHO'][0]
        dateobs =  obsDictj['DATE-OBS'][0].replace('-','')
        prows = rows/2 

      # input catalogs, transform to SALT FP (mm relative to optical axis = mask center), as required
        ur0,uc0,fps = rssdtralign(dateobs,trkrho)           # ur, uc =unbinned pixels, fps =micr/arcsec
        RAd = Longitude(obsDictj['RA'][0]+' hours').degree
        DECd = Latitude(obsDictj['DEC'][0]+' degrees').degree
        PAd = obsDictj['TELPA'][0]

        catfile = glob.glob(objectname+'*cat.txt')    # 1st 5 cols must be ID,RA,Dec,mag,BP-RP
        iscat = (len(catfile) > 0)
        if iscat:
            converters = dict.fromkeys(['col2','col3','col4'],[ascii.convert_numpy(np.float)])
            converters['col1'] = [ascii.convert_numpy(np.str)]
            catTab = ascii.read(catfile[0],comment='#',format='fixed_width_two_line', \
                delimiter=' ',converters=converters)
            oldnameList = catTab.colnames
            for col in range(5): 
                catTab[oldnameList[col]].name=['CATID','RA','DEC','MAG','BP-RP'][col]
            catTab.keep_columns(['CATID','RA','DEC','MAG','BP-RP'])
            catTab.sort('MAG')
            if (type(catTab['CATID'][0]) != str):           # all catids should be strings
                catTab['CATID'] = [str(x) for x in catTab['CATID']]                    
            YX_dt = YXcalc(catTab['RA'],catTab['DEC'],RAd,DECd,PAd,fps)
            catTab['Y'],catTab['X'] = tuple(YX_dt)
          # clip off targets outside polarimetric FOV +/- 100 arcsec (20 mm)
            clipit_t = ((np.abs(YX_dt[0]) > 55.) |  \
                (np.sqrt((YX_dt**2).sum(axis=0)) > 70.))
            catTab.remove_rows(np.where(clipit_t)[0])
            targets = len(catTab)
                
        linefile = glob.glob(objectname+'*line.txt')
        if len(linefile): 
            linefile = linefile[0]
        masktyp = obsDictj['MASKTYP'][0]
              
        if (masktyp=='MOS'):        
            slitfileList = glob.glob(maskid+'.*')                         
            if len(slitfileList) == 0:
                rsslog.message('\nno MOS support file found for '+'maskid'+'\n', logfile)
                continue 
            else:
                typeList = [slitfile.split('.')[-1] for slitfile in slitfileList]                    
                if (not np.in1d(typeList,('xml','gcode')).any()):
                    rsslog.message('\nno MOS xml or gcode file found for '+'maskid'+'\n', logfile)
                    continue
                if typeList.count('xml'):                       
                    slitTab, PAd, RAd, DECd = readmaskxml(maskid+'.xml')
                    YX_dt = YXcalc(slitTab['RACE'],slitTab['DECCE'],RAd,DECd,PAd,fps)
                    slitTab['YCE'],slitTab['XCE'] = tuple(YX_dt)
                else:
                    slitTab = readmaskgcode(maskid+'.gcode')
                    slitTab['WIDTH'] = 1000.*slitTab['XLEN']/fps
                    slitTab['LENGTH'] = 1000.*slitTab['YLEN']/fps                        
                if debug:
                    filter = hdul0[0].header['FILTER']
                    for c,col in enumerate(slitTab.colnames[2:]): 
                        slitTab[col].format=(2*['%.6f']+4*['%7.3f'])[c]
                    slitTab.write(objectname+"_"+filter+"_slitTab.txt",format='ascii.fixed_width',   \
                        bookend=False, delimiter=None, overwrite=True)                                         
        elif (masktyp=='LONGSLIT'):
            slitTab = Table(names=('TYPE','CATID','RACE','DECCE','WIDTH','LENGTH'),     \
                dtype=('S7','<S64', float, float, float, float))                            
            slitTab.add_row(('target','longslit',RAd,DECd,float(mid(maskid,2,4))/100.,240.))
            YX_dt = YXcalc(slitTab['RACE'],slitTab['DECCE'],RAd,DECd,PAd,fps)
            slitTab['YCE'],slitTab['XCE'] = tuple(YX_dt)                               
        else:
            if (not iscat):
                rsslog.message('\nno cat file found for slitless data\n', logfile)
                continue           
        isslit = (len(slitTab)>0)
                
        if ((grating=='N/A') & (masktyp!='LONGSLIT')):
          # imaging polarimetry
            coltem = hdul0[0].header['COLTEM'] 
            camtem = hdul0[0].header['CAMTEM']
            filterListj = obsDictj['FILTER']                                                                                                           
            jarcList = list(np.where((np.array(obsDictj['OBJECT'])=='ARC') &  \
                                    (np.array(obsDictj['LAMPID'])=='Hg Ar'))[0])
            ismosarc = (len(jarcList)>0)
          # process arcs
            if ismosarc:
                jcal = jarcList[0]          # use the first one
                rsslog.message("\n"+objectname+"  "+filterListj[jcal]+"  HgAr ARC", logfile)                        
                imarcpolmap(fileListj[jcal],slitTab,objectname,logfile=logfile,debug=debug)
                calHdul = pyfits.open("t"+fileListj[jcal])
                isspec_j = np.in1d(np.arange(files),jarcList,invert=True)

          # process filtered images, for object finding and wavcal if MOS w/o arc, or slitless                       
            else:
                filterListf = sorted(list(set(filterListj)))
                filters = len(filterListf)
                iscal_f = np.array([filterListf[f][:3]=='PI0' for f in range(filters)])               
                if iscal_f.sum()==0:
                    rsslog.message('No Arc or Filtered Images to calibrate this configuration:', logfile)
                    continue
                                            
                jList_f = np.empty(filters,dtype=object)
                for f in range(filters):
                    jList_f[f] = [j for j in range(files) if filterListj[j]==filterListf[f]]            
                jcalList = jList_f[iscal_f].sum()                          
                calHdulList = []
                if isMOS:
                    mapTab = slitTab
                else:
                    mapTab = catTab
                for f in np.where(iscal_f)[0]:
                    rsslog.message("\n"+objectname+"  "+filterListf[f]+"  Cal filter", logfile)
                    fileList = [fileListj[j] for j in jList_f[f]]
                    imfilmap(fileList,mapTab,objectname,logfile=logfile,debug=debug)
                    calHdulList.append(pyfits.open("t"+fileList[0]))
                isspec_j = np.in1d(np.arange(files),jcalList,invert=True)                

          # process imaging spectropolarimetry
            specimages = isspec_j.sum()               
            if specimages==0:
                rsslog.message('No Spectropolarimetric Images for this configuration:', logfile)
                continue
            rsslog.message("\n"+objectname+"  "+np.array(filterListj)[isspec_j][0], logfile)
            fileList = list(np.array(fileListj)[isspec_j])
            if ismosarc:          
                imspecmospolmap(fileList,objectname,calHdul,cutwavoverride=cutwavoverride, \
                    logfile=logfile,debug=debug)
            else:
                imspecslitlessmap(fileList,catTab,objectname,calHdulList,cutwavoverride=cutwavoverride, \
                    logfile=logfile,debug=debug)                    

#       elif astrocat==object+'wavcat.txt':
#       linepolmap(objectlist_i,astromapo)

        else:
        # grating spectropolarimetry
            dataListj = [pyfits.open(file) for file in fileListj]
            filterListj = obsDictj['FILTER']
            lampListj = obsDictj['LAMPID']                                                                                                             
            jarcList = list(np.where((np.array(obsDictj['OBJECT'])=='ARC'))[0])
            if (len(jarcList)>0):
                jcal = jarcList[0]                      # use the first one, for now
                arcfile = fileListj[jcal] 
                arcdata = dataListj[jcal]
                isspec_j = np.in1d(np.arange(files),jarcList,invert=True)
            else:                                           
          # if no arc for this config look for a similar one with different config
                config = obsTab['config'][o]
                okconfig_c = ((configTab['GRATING']==configTab['GRATING'][config]) &  \
                     (np.abs(configTab['GR-ANGLE'] - configTab['GR-ANGLE'][config]) < .03) & \
                     (np.abs(configTab['CAMANG'] - configTab['CAMANG'][config]) < .05))
                okarc_o = (iarc_o[np.in1d(obsTab['config'],np.where(okconfig_c)[0])] >=0) 
                iarcList = list(iarc_o[okarc_o])               

                if (len(iarcList) > 0):
                    arcfile = infileList[iarcList[0]]     # just use the first one for now
                    rsslog.message(('Warning: using arc from different BLOCKID: %s' % arcfile), logfile)                
                else:            
                    rsslog.message('No arc for this configuration:', logfile)
                    continue                    
                arcdata = pyfits.open(arcfile)
                isspec_j = np.ones(files,dtype=bool)

            if isdiffuse:  
                rsslog.message("target fills slit",logfile)
            specimages = isspec_j.sum()               
            if specimages==0:
                rsslog.message('No Spectropolarimetric Images for this configuration:', logfile)
                continue
            spdataList = [dataListj[j] for j in np.where(isspec_j)[0]]
                                     
          # evaluate spectral tilt, row offset from spectra, process arc, if not done already
            arcoutfile = 't'+arcfile
            if os.path.exists(arcoutfile):
                arcdata = pyfits.open(arcoutfile)
            else:                
                tiltcor, droffset_pi = sptilt(spdataList,slitTab,debug=debug)           
                isdiffuse = ((np.array(lampListj)[isspec_j] != 'NONE').all() | isdiffuse)  
                rsslog.message(("\nspectral tilt(deg)    : %8.4f" % tiltcor),logfile)
                rsslog.message(("median vertical offset (bins): %8.4f" % np.median(droffset_pi)),logfile)                                
                rsslog.message("\n"+fileListj[jcal]+" "+objectname+" "+filterListj[jcal]+" "+lampListj[jcal]+" ARC", \
                    logfile)                        
                sparcpolmap(arcdata,slitTab,objectname,tiltcor,droffset_pi,isdiffuse,   \
                    logfile=logfile,debug=debug,pidebug=pidebug)

          # process grating spectropolarimetry
            rsslog.message("\n"+objectname+"  "+np.array(filterListj)[isspec_j][0], logfile)
       
            spmospolmap(spdataList,objectname,arcdata,isdiffuse,logfile=logfile,debug=debug)
                
    return

# ------------------------------------

if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)   
    poltargetmap(infileList,**kwargs)

# debug:
# M30
# cd /d/pfis/khn/20161023/sci
# python polsalt.py poltargetmap.py m*.fits debug=True
# python polsalt.py poltargetmap.py m*010[5-9].fits m*011?.fits m*0120.fits debug=True
