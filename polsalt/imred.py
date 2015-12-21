
"""
IMRED 

Reduction script for SALT data -- this is 
for science level reductions with variance frames

This includes step that are not yet included in the pipeline 
and can be used for extended reductions of SALT data. 

It does require the pysalt package to be installed 
and up to date.

"""
# polSALT: fix Pfits without XTALK
# polSALT: use local version of createbadpixel = masterbadpixel
# polSALT: fix VAR and BPM extensions after mosaic

import os, sys, glob, shutil, inspect

import numpy as np
from astropy.io import fits as pyfits
from scipy.ndimage.filters import median_filter
 
from pyraf import iraf
from iraf import pysalt

from saltsafelog import logging
from saltobslog import obslog
from saltprepare import *
from saltbias import saltbias
from saltgain import saltgain
from saltxtalk import saltxtalk
from saltcrclean import saltcrclean
from saltcombine import saltcombine
from saltflat import saltflat
from saltmosaic import saltmosaic
from saltillum import saltillum
debug = True

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

def imred(infile_list, prodir, bpmfile=None, gaindb = None, cleanup=True):
    #get the name of the files
    infiles=','.join(['%s' % x for x in infile_list])
    

    #get the current date for the files
    obsdate=os.path.basename(infile_list[0])[1:9]
    print obsdate

    #set up some files that will be needed
    logfile='im'+obsdate+'.log'
    flatimage='FLAT%s.fits' % (obsdate)
    dbfile='spec%s.db' % obsdate

    #create the observation log
    obs_dict=obslog(infile_list)

    with logging(logfile, debug) as log:
        log.message('Pysalt Version: '+pysalt.verno, with_header=False)
 
    #prepare the data
    saltprepare(infiles, '', 'p', createvar=False, badpixelimage='', clobber=True, logfile=logfile, verbose=True)

    for img in infile_list:
        hdu = pyfits.open('p'+os.path.basename(img), 'update')
        # for backwards compatibility  
        if not 'XTALK' in hdu[1].header:
            hdu[1].header.update('XTALK',1474)
            hdu[2].header.update('XTALK',1474)
            hdu[3].header.update('XTALK',1166)
            hdu[4].header.update('XTALK',1111)
            hdu[5].header.update('XTALK',1377)
            hdu[6].header.update('XTALK',1377)
        hdu.close()
        
    #bias subtract the data
    saltbias('pP*fits', '', 'b', subover=True, trim=True, subbias=False, masterbias='',  
              median=False, function='polynomial', order=5, rej_lo=3.0, rej_hi=5.0, 
              niter=10, plotover=False, turbo=False, 
              clobber=True, logfile=logfile, verbose=True)

    add_variance('bpP*fits', bpmfile)

    #gain correct the data 
    usedb = False
    if gaindb: usedb = True
    saltgain('bpP*fits', '', 'g', gaindb=gaindb, usedb=usedb, mult=True, clobber=True, logfile=logfile, verbose=True)

    #cross talk correct the data
    saltxtalk('gbpP*fits', '', 'x', xtalkfile = "", usedb=False, clobber=True, logfile=logfile, verbose=True)

    #cosmic ray clean the data
    #only clean the object data
    for i in range(len(infile_list)):
        if (obs_dict['CCDTYPE'][i].count('OBJECT') \
            and obs_dict['LAMPID'][i].count('NONE') \
            and obs_dict['INSTRUME'][i].count('RSS')):
          img='xgbp'+os.path.basename(infile_list[i])
          saltcrclean(img, img, '', crtype='edge', thresh=5, mbox=11, bthresh=5.0,
                flux_ratio=0.2, bbox=25, gain=1.0, rdnoise=5.0, fthresh=5.0, bfactor=2,
                gbox=3, maxiter=5, multithread=True,  clobber=True, logfile=logfile, verbose=True)

    #mosaic the data
    #khn: attempt to use most recent previous geometry to obsdate.  
    #NOTE: mosaicing does not do this correctly
    #geomdb = open(datadir+"RSSgeom.dat",'r')
    #for geomline in geomdb:
    #    if geomline[0]=='#': continue
    #    if (int(obsdate) > int(geomline.split(' ')[0].replace('-',''))): break
    #geomfile = "RSSgeom_obsdate.dat"
    #open(geomfile,'w').write(geomline)

    geomfile=iraf.osfn("pysalt$data/rss/RSSgeom.dat")
    
    try:
       saltmosaic('xgbpP*fits', '', 'm', geomfile, interp='linear', cleanup=True, geotran=True, clobber=True, logfile=logfile, verbose=True)
    except:
       saltmosaic('xgbpP*fits', '', 'm', geomfile, interp='linear', cleanup=True, geotran=True, clobber=True, logfile=logfile, verbose=True)
    #khn: fix mosaiced VAR and BPM extensions
    #khn: fix mosaiced bpm missing some of gap
    for img in infile_list:
        filename = 'mxgbp'+os.path.basename(img)
        hdu = pyfits.open(filename, 'update')
        hdu[2].header.update('EXTNAME','VAR')
        hdu[3].header.update('EXTNAME','BPM')
        bpm_rc = (hdu[3].data>0).astype('uint8')
        zeroscicol = hdu['SCI'].data.sum(axis=0) == 0
        bpmgapcol = bpm_rc.mean(axis=0) == 1
        addbpmcol = zeroscicol & ~bpmgapcol
        addbpmcol[np.argmax(addbpmcol)-4:np.argmax(addbpmcol)] = True    # allow for chip tilt
        bpm_rc[:,addbpmcol] = 1
        hdu[3].data = bpm_rc
        hdu.writeto(filename,clobber=True)

    #clean up the images
    if cleanup:
           for f in glob.glob('p*fits'): os.remove(f)
           for f in glob.glob('bp*fits'): os.remove(f)
           for f in glob.glob('gbp*fits'): os.remove(f)
           for f in glob.glob('xgbp*fits'): os.remove(f)

def add_variance(filenames, bpmfile):
    file_list=glob.glob(filenames)
    badpixelstruct = saltio.openfits(bpmfile)
    for f in file_list:
        struct = pyfits.open(f)
        nsciext=len(struct)-1
        nextend=nsciext
        for i in range(1, nsciext+1):
            hdu=CreateVariance(struct[i], i, nextend+i)
            hdu.header.update('EXTNAME','VAR')
            struct[i].header.update('VAREXT',nextend+i, comment='Extension for Variance Frame')
            struct.append(hdu)
        nextend+=nsciext
        for i in range(1, nsciext+1):
            hdu=masterbadpixel(struct, badpixelstruct, i, nextend+i)
            struct[i].header.update('BPMEXT',nextend+i, comment='Extension for Bad Pixel Mask')
            struct.append(hdu)
        nextend+=nsciext
        struct[0].header.update('NEXTEND', nextend)
        if os.path.isfile(f): os.remove(f)
        struct.writeto(f)

def masterbadpixel(inhdu, bphdu, sci_ext, bp_ext):
#   khn: Create the bad pixel hdu bp_ext for inhdu[sci_ext] from a master, bphdu

    if bphdu is None:
        data=np.zeros_like(inhdu[sci_ext].data).astype("uint8")
    else:
        infile=inhdu.fileinfo(0)['filename']
        bpfile=bphdu.fileinfo(0)['filename']
        masternext = len(bphdu)-1
        masterext = (sci_ext-1) % masternext + 1        # allow for windows
       
        if not saltkey.compare(inhdu[0], bphdu[0], 'INSTRUME', infile, bpfile):
            message = '%s and %s are not the same %s' % (infile,bpfile, 'INSTRUME')
            raise SaltError(message)
        else:
            rows,cols = inhdu[sci_ext].data.shape
            cbin,rbin = np.array(inhdu[sci_ext].header["CCDSUM"].split(" ")).astype(int)
            masterrows,mastercols = bphdu[masterext].data.shape
            master_rc = np.ones((masterrows+(masterrows % rbin),mastercols+(mastercols % cbin)))
            master_rc[:masterrows,:mastercols] = bphdu[masterext].data
            masterrows,mastercols=(masterrows+(masterrows % rbin),mastercols+(mastercols % cbin))
            ampsec = inhdu[sci_ext].header["AMPSEC"].strip("[]").split(",")
            r1,r2 = (np.array(ampsec[1].split(":")).astype(int) / rbin) * rbin
            c1,c2 = (np.array(ampsec[0].split(":")).astype(int) / cbin) * cbin
            if c1 > c2: c1,c2 = c2,c1
            bin_rc = (master_rc.reshape(masterrows/rbin,rbin,mastercols/cbin,cbin).sum(axis=3).sum(axis=1) > 0)
            data = bin_rc[ r1:r2, c1:c2 ].astype('uint8')
            
    header=inhdu[sci_ext].header.copy()
    header.update('EXTVER',bp_ext)
    header.update('SCIEXT',sci_ext,comment='Extension of science frame')

    return pyfits.ImageHDU(data=data, header=header, name='BPM')


if __name__=='__main__':
    rawdir=sys.argv[1]
    prodir=os.path.curdir+'/'
    bpmfile = os.path.dirname(sys.argv[0]) + '/bpm_sn.fits'
    imred(rawdir, prodir, cleanup=True, bpmfile=bpmfile)
