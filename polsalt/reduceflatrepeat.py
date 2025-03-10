import os, sys, glob
reddir = '/d/carol/Synched/software/SALT/polsaltcopy/polsalt/'

sys.path.append(reddir)
datadir = reddir+'data/'
import numpy as np
from astropy.io import fits as pyfits
 
from pyraf import iraf
from iraf import pysalt

from saltsafelog import logging
from saltobslog import obslog
from saltprepare import *
from saltbias import saltbias
from saltgain import saltgain
from flatrepeat import flatrepeat

logfile = '/dev/null'

obsdate = sys.argv[1]
i1,i2 = int(sys.argv[2]),int(sys.argv[3])
outfile =""
if len(sys.argv)>4: outfile = sys.argv[4]

os.chdir(obsdate)
if not os.path.isdir('sci'): os.mkdir('sci')
os.chdir('sci')

Pfilelist = glob.glob('../raw/P*fits')
imagenolist = [int(f[-9:-5]) for f in Pfilelist]
if ((imagenolist.count(i1)==0) | (imagenolist.count(i2)==0)):
    print "Images not found"
    exit()

f1,f2 = (imagenolist.index(i1),imagenolist.index(i2))
infilelist = Pfilelist[f1:f2+1]

#prepare the data
infiles=','.join(['%s' % x for x in infilelist])
saltprepare(infiles, '', 'p', createvar=False, badpixelimage='', clobber=True, logfile=logfile, verbose=True)

#bias subtract the data
saltbias('pP*fits', '', 'b', subover=True, trim=True, subbias=False, masterbias='',  
          median=False, function='polynomial', order=5, rej_lo=3.0, rej_hi=5.0, 
          niter=10, plotover=False, turbo=False, 
          clobber=True, logfile=logfile, verbose=True)

#gain correct the data 
saltgain('bpP*fits', '', 'g', usedb=False, mult=True, clobber=True, logfile=logfile, verbose=True)

#run flatrepeat routine
infilelist = sorted(glob.glob('gbpP*fits'))
flatrepeat(infilelist,outfile=outfile)

#clean up the images
for f in glob.glob('p*fits'): os.remove(f)
for f in glob.glob('bp*fits'): os.remove(f)
for f in glob.glob('gbp*fits'): os.remove(f)
