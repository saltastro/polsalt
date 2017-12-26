import os, sys, glob, shutil
poldir = '/d/freyr/Dropbox/software/SALT/polsaltcurrent/'

reddir=poldir+'polsalt/'
scrdir=poldir+'scripts/'
sys.path.extend((reddir,scrdir,poldir))

datadir = reddir+'data/'
import numpy as np
from astropy.io import fits as pyfits
from specpolview import printstokes
from imred import imred
from specpolwavmap import specpolwavmap
from specpolextract_sc import specpolextract_sc
from specpolrawstokes import specpolrawstokes
from specpolfinalstokes import specpolfinalstokes

print sys.argv
obsdate = sys.argv[1]
print obsdate
os.chdir(obsdate)
if not os.path.isdir('sci'): os.mkdir('sci')
shutil.copy(scrdir+'script.py','sci')
os.chdir('sci')

#basic image reductions
infilelist = sorted(glob.glob('../raw/P*fits'))
imred(infilelist, './', datadir+'bpm_rss_11.fits', crthresh=False, cleanup=True)

#basic polarimetric reductions
logfile='specpol'+obsdate+'.log'

#wavelength map
infilelist = sorted(glob.glob('m*fits'))
linelistlib=""
specpolwavmap(infilelist, linelistlib=linelistlib, logfile=logfile)

#background subtraction and extraction
infilelist = sorted(glob.glob('wm*fits'))
extract = 10.   # star +/-5, bkg= +/-(25-35)arcsec:  2nd order is 9-20 arcsec away
locate = (-120.,120.)    # science target is brightest target in whole slit
#locate = (-20.,20.)

specpolextract_sc(infilelist,logfile=logfile,locate=locate,extract=extract)
#specpolextract_sc(infilelist,logfile=logfile,locate=locate,extract=extract,docomp=True,useoldc=True)

#raw stokes
infilelist = sorted(glob.glob('e*fits'))
specpolrawstokes(infilelist, logfile=logfile)

#final stokes
infilelist = sorted(glob.glob('*_h*.fits'))
specpolfinalstokes(infilelist, logfile=logfile)
