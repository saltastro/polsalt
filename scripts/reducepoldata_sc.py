import os, sys, glob
reddir = '/d/carol/Synched/software/SALT/polsaltcopy/polsalt/'
scrdir = '/d/carol/Synched/software/SALT/polsaltcopy/scripts/'
poldir = '/d/carol/Synched/software/SALT/polsaltcopy/'
sys.path.extend((reddir,scrdir,poldir))

datadir = reddir+'data/'
import numpy as np
from astropy.io import fits as pyfits

# np.seterr(invalid='raise')

from imred import imred

from specpolwavmap import specpolwavmap
from specpolextract_sc import specpolextract_sc
from specpolrawstokes import specpolrawstokes
from specpolfinalstokes import specpolfinalstokes

obsdate = sys.argv[1]

os.chdir(obsdate)
if not os.path.isdir('sci'): os.mkdir('sci')
os.chdir('sci')

#basic image reductions
infilelist = glob.glob('../raw/P*fits')

#imred(infilelist, './', datadir+'bpm_rss_11.fits', cleanup=False)
imred(infilelist, './', datadir+'bpm_rss_11.fits', cleanup=True)

#basic polarimetric reductions
# debug=True
debug=False
logfile='specpol'+obsdate+'.log'

#wavelength map
infilelist = sorted(glob.glob('m*fits'))
linelistlib=""
specpolwavmap(infilelist, linelistlib=linelistlib, logfile=logfile)

#background subtraction and extraction
infilelist = sorted(glob.glob('wm*fits'))
dyasec = 5.     # star +/-5, bkg= +/-(25-35)arcsec:  2nd order is 9-20 arcsec away
yoffo = 0.      # optional offset of target (bins) from brightest in O (bottom) image
specpolextract_sc(infilelist,yoffo,dyasec, logfile=logfile)

#raw stokes
infilelist = sorted(glob.glob('e*fits'))
specpolrawstokes(infilelist, logfile=logfile)

#final stokes
infilelist = sorted(glob.glob('*_h*.fits'))
specpolfinalstokes(infilelist, logfile=logfile, debug=debug)
