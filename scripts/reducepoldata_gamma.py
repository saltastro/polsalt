import os, sys, glob, shutil
poldir = '/usr/users/khn/src/salt/polsaltcurrent/'

reddir=poldir+'polsalt/'
scrdir=poldir+'scripts/'
sys.path.extend((reddir,scrdir,poldir))

print sys.path

datadir = reddir+'data/'
import numpy as np
from astropy.io import fits as pyfits
from specpolview import printstokes
from imred import imred
from specpolwavmap import specpolwavmap
from specpolextract_sc import specpolextract_sc, specpolcorrect_sc, specpolspectrum_sc
from specpolrawstokes import specpolrawstokes
from polfinalstokes import polfinalstokes 

print sys.argv
obsdate = sys.argv[1]
print obsdate
os.chdir(obsdate)
if not os.path.isdir('sci_gamma'): os.mkdir('sci_gamma')
os.chdir('sci_gamma')

# basic image reductions
infilelist = sorted(glob.glob('../raw/P*fits'))
imred(infilelist, './', datadir+'bpm_rss_11.fits', crthresh=False, cleanup=True)

# basic polarimetric reductions
logfile='specpol'+obsdate+'.log'

# wavelength map
infilelist = sorted(glob.glob('m*.fits'))
linelistlib=""
specpolwavmap(infilelist, linelistlib=linelistlib, logfile=logfile)

# background subtraction and extraction 
infilelist = sorted(glob.glob('wm*.fits'))
extract = 10.               # star +/-5, bkg= +/-(25-35)arcsec:  standard
locate = (-120.,120.)       # science target is brightest target in whole slit
specpolcorrect_sc(infilelist,logfile=logfile,locate=locate,extract=extract)

infilelist = sorted(glob.glob('cw*fits'))
specpolspectrum_sc(infilelist,logfile=logfile,locate=locate,extract=extract)

#raw stokes
infilelist = sorted(glob.glob('ecw*.fits'))
specpolrawstokes(infilelist, logfile=logfile)

#final stokes
infilelist = sorted(glob.glob('*_h*.fits'))
polfinalstokes(infilelist, logfile=logfile)

