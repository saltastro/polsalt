#reduction script, imred only

import os, sys, glob, shutil
poldir = '/usr/users/khn/src/salt/polsaltcurrent/'
reddir=poldir+'polsalt/'
scrdir=poldir+'scripts/'
sys.path.extend((reddir,scrdir,poldir))
datadir = reddir+'data/'
import numpy as np
from astropy.io import fits as pyfits
from imred import imred

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

