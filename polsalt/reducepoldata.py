import os, sys, glob
reddir = '/d/carol/Synched/software/SALT/polsaltcopy/polsalt/'

sys.path.append(reddir)
datadir = reddir+'data/'
import numpy as np
import pyfits

# np.seterr(invalid='raise')

from imred import imred

from specpolwavmap import specpolwavmap
from specpolextract import specpolextract
from specpolrawstokes import specpolrawstokes
from specpolfinalstokes import specpolfinalstokes

obsdate = sys.argv[1]

os.chdir(obsdate)
if not os.path.isdir('sci'): os.mkdir('sci')
os.chdir('sci')

#basic image reductions
infile_list = glob.glob('../raw/P*fits')

#imred(infile_list, './', datadir+'bpm_rss_11.fits', cleanup=False)
imred(infile_list, './', datadir+'bpm_rss_11.fits', cleanup=True)

#basic polarimetric reductions
# debug=True

debug=False
logfile='specpol'+obsdate+'.log'

#target and wavelength map
infile_list = sorted(glob.glob('m*fits'))
linelistlib=""
specpolwavmap(infile_list, linelistlib=linelistlib, inter=True, logfile=logfile)

#background subtraction and extraction
#infile_list = sorted(glob.glob('wm*099.fits')+glob.glob('wm*10[0-6].fits'))    # optional subselection

infile_list = sorted(glob.glob('wm*fits'))
specpolextract(infile_list, logfile=logfile, debug=debug)

#raw stokes
#infile_list = sorted(glob.glob('e*06[8-9].fits')+glob.glob('e*07[0-1].fits')+glob.glob('e*07[4-7].fits'))       # optional subselection

infile_list = sorted(glob.glob('e*fits'))
specpolrawstokes(infile_list, logfile=logfile)

#final stokes
#polcal = 'polcal0.txt'                                 # null calibration
#infile_list = sorted(glob.glob('*_h[0,2]*.fits'))      # optional subselection 

polcal = 'polcal.txt'
infile_list = sorted(glob.glob('*_h*.fits'))
specpolfinalstokes(infile_list, polcal=polcal, logfile=logfile, debug=debug)
