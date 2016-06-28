
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

import os, sys, glob, copy, shutil, inspect

import numpy as np
from astropy.io import fits as pyfits
from scipy.ndimage.filters import median_filter
 
from pyraf import iraf
from iraf import pysalt

from saltsafelog import logging
from saltobslog import obslog
from saltprepare import *
from saltbias import bias
from saltgain import gain
from saltxtalk import xtalk
from saltcrclean import multicrclean

from saltcombine import saltcombine
from saltflat import saltflat
from saltmosaic import saltmosaic
from saltillum import saltillum
debug = True

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

def imred(infilelist, prodir, bpmfile=None, gaindb = None, cleanup=True):
    #get the name of the files
    infiles=','.join(['%s' % x for x in infilelist])
    
    #get the current date for the files
    obsdate=os.path.basename(infilelist[0])[1:9]
    print "Observation Date: ",obsdate

    #set up some files that will be needed
    logfile='im'+obsdate+'.log'
    flatimage='FLAT%s.fits' % (obsdate)
    dbfile='spec%s.db' % obsdate

    #create the observation log
#    obs_dict=obslog(infilelist)

    verbose=True

    with logging(logfile, debug) as log:
        log.message('Pysalt Version: '+pysalt.verno, with_header=False)
 
    #prepare the data

        for img in infilelist:
            hdu = pyfits.open(img)

            # for backwards compatibility
            hdu = remove_duplicate_keys(hdu)  
            if not 'XTALK' in hdu[1].header:
                hdu[1].header['XTALK']=1474
                hdu[2].header['XTALK']=1474
                hdu[3].header['XTALK']=1166
                hdu[4].header['XTALK']=1111
                hdu[5].header['XTALK']=1377
                hdu[6].header['XTALK']=1377

            img = os.path.basename(img)
                                                                    
            hdu = prepare(hdu, createvar=False, badpixelstruct=None)
            if not cleanup: hdu.writeto('p'+img, clobber=True)

            hdu = bias(hdu,subover=True, trim=True, subbias=False,
                       bstruct=None, median=False, function='polynomial',
                       order=5, rej_lo=5.0, rej_hi=5.0, niter=10,
                       plotover=False, log=log, verbose=verbose)    
            if not cleanup: hdu.writeto('bp'+img, clobber=True)

            # put windowed data into full image
            exts = len(hdu)
            if exts > 7:
                rows, cols = hdu[1].data.shape
                cbin, rbin = [int(x) for x in hdu[0].header['CCDSUM'].split(" ")]
                ampsecO = hdu[1].header["AMPSEC"].strip("[]").split(",")
                ampsecE = hdu[7].header["AMPSEC"].strip("[]").split(",")
                rO = int((float(ampsecO[1].split(":")[0]) - 1.)/rbin)
                rE = int((float(ampsecE[1].split(":")[0]) - 1.)/rbin)
                keylist = ['BIASSEC','DATASEC','AMPSEC','CCDSEC','DETSEC']
                oldlist = [hdu[1].header[key].strip("[]").split(",")[1] for key in keylist]
                newlist = 2*['1:'+str(int(0.5+4102/rbin))]+3*[str(int(rbin/2))+':4102']

                for amp in range(6):
                    hduO = hdu[amp+1].copy()                    
                    hdu[amp+1].data = np.zeros((4102/rbin,cols))
                    hdu[amp+1].data[rO:rO+rows] = hduO.data
                    hdu[amp+1].data[rE:rE+rows] = hdu[amp+7].data
                    hdu[amp+1].update_header
                    for k,key in enumerate(keylist): 
                        hdu[amp+1].header[key] = \
                            hdu[amp+1].header[key].replace(oldlist[k],newlist[k])
                del hdu[7:]
                hdu[0].header['NSCIEXT'] = 6
     
            badpixelstruct = saltio.openfits(bpmfile)
            hdu = add_variance(hdu, badpixelstruct)
             
            #gain correct the data 
            if gaindb: 
                usedb = True
                dblist = saltio.readgaindb(gaindb.strip())
            else:
                usedb = False
                dblist = ''
            hdu = gain(hdu, mult=True, usedb=usedb, dblist=dblist, log=log, verbose=verbose)
            if not cleanup: hdu.writeto('gbp'+img, clobber=True)

            #cross talk correct the data
            hdu=xtalk(hdu, [], log=log, verbose=verbose)

            #cosmic ray clean the data
            #only clean the object data
            thresh = 5.0
            if hdu[0].header['GRATING'].strip()=='PG0300': thresh = 7.0

            if hdu[0].header['CCDTYPE']=='OBJECT' and \
               hdu[0].header['LAMPID']=='NONE' and \
               hdu[0].header['INSTRUME']=='RSS':
               log.message('Cleaning CR using thresh={}'.format(thresh))
               hdu = multicrclean(hdu, crtype='edge', thresh=thresh, mbox=11, bthresh=5.0,
                  flux_ratio=0.2, bbox=25, gain=1.0, rdnoise=5.0, fthresh=5.0, bfactor=2,
                  gbox=3, maxiter=5, log=log, verbose=verbose)
               for ext in range(13,19): hdu[ext].data = hdu[ext].data.astype('uint8')
            hdu.writeto('xgbp'+img, clobber=True)
            hdu.close()
        
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
    for img in infilelist:
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

def remove_duplicate_keys(hdu):
    # in case of duplicate primary header keys, remove those with blank values
    keylist = hdu[0].header.keys()
    vallist = hdu[0].header.values()
    dupkeylist = list(set([x for x in keylist if keylist.count(x)>1]))
    delarglist = []
    for key in dupkeylist:
        arglist = [i for i in range(len(keylist)) if keylist[i]==key]
        for arg in arglist:
            if len(vallist[arg].strip()) == 0: delarglist.append(arg)
    for arg in sorted(delarglist,reverse=True): del hdu[0].header[arg]
    return hdu

def add_variance_files(filenames, bpmfile):
    file_list=glob.glob(filenames)
    badpixelstruct = saltio.openfits(bpmfile)
    for f in file_list:
        struct = pyfits.open(f)
        struct = add_variance(struct, bpmstruct)
        if os.path.isfile(f): os.remove(f)
        struct.writeto(f)

def add_variance(struct, badpixelstruct):
    """Add variance and badpixel frame"""
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
    return struct 


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
            r1 = int((float(ampsec[1].split(":")[0]) - 1.)/rbin)
            c1 = int((float(ampsec[0].split(":")[0]) - 1.)/cbin)
            bin_rc = (master_rc.reshape(masterrows/rbin,rbin,mastercols/cbin,cbin).sum(axis=3).sum(axis=1) > 0)
            data = bin_rc[ r1:r1+rows, c1:c1+cols ].astype('uint8')
        
    header=inhdu[sci_ext].header.copy()
    header.update('EXTVER',bp_ext)
    header.update('SCIEXT',sci_ext,comment='Extension of science frame')

    return pyfits.ImageHDU(data=data, header=header, name='BPM')


if __name__=='__main__':
    rawdir=sys.argv[1]
    prodir=os.path.curdir+'/'
    bpmfile = os.path.dirname(sys.argv[0]) + '/bpm_sn.fits'
    imred(rawdir, prodir, cleanup=True, bpmfile=bpmfile)
