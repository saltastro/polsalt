#development reduction script

import os, sys, glob, shutil
poldir = '/usr/users/khn/src/salt/polsaltcurrent/'
reddir=poldir+'polsalt/'
scrdir=poldir+'scripts/'
sys.path.extend((reddir,scrdir,poldir))
datadir = reddir+'data/'
import numpy as np
from astropy.io import fits as pyfits
from imred import imred

from poltargetmap import poltargetmap
from polextract import polextract, findpair
from polrawstokes import polrawstokes
from polfinalstokes_20220729 import polfinalstokes
from polfinalstokes import polfinalstokes     # new moscal 

import warnings
# warnings.filterwarnings("error",category=RuntimeWarning)    # trace warnings

obsdate = sys.argv[1]
os.chdir(obsdate)
if not os.path.isdir('sci'): os.mkdir('sci')
shutil.copy(scrdir+'script.py','sci')
os.chdir('sci')

#basic image reductions, using cleaned bpm
infilelist = sorted(glob.glob('../raw/P*.fits'))
imred(infilelist, './', datadir+'bpm01_rss_11.fits', crthresh=False, cleanup=True)
exit()
#basic polarimetric reductions
logfile='specpol'+obsdate+'.log'
#target map
infilelist = sorted(glob.glob('mx*.fits'))
linelistlib=""
poltargetmap(infilelist, logfile=logfile,isdiffuse=True)

#background subtraction and extraction
infilelist = sorted(glob.glob('tm*.fits'))
polextract(infilelist,logfile=logfile)

#raw stokes
infilelist = sorted(glob.glob('etm*.fits'))
polrawstokes(infilelist, logfile=logfile)
exit()

#final stokes
infilelist = sorted(glob.glob('ET21B*_h*.fits'))

# polfinalstokes(infilelist, logfile=logfile,HWCal="")
polfinalstokes(infilelist, logfile=logfile,Heffcal_override=True)
