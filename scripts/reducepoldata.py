import os, sys, glob
import argparse
import numpy as np
import pyfits

# np.seterr(invalid='raise')

import polsalt
datadir = os.path.dirname(polsalt.__file__)+'/data/'
from polsalt.imred import imred

from polsalt.specpolwavmap import specpolwavmap
from polsalt.specpolextract import specpolextract
from polsalt.specpolrawstokes import specpolrawstokes
from polsalt.specpolfinalstokes import specpolfinalstokes

parser = argparse.ArgumentParser(description='Reduce SALT Lens Data')
parser.add_argument('ddir', help='Top level directory with SALT data')
parser.add_argument('-s', dest='basic_red', default=True, action='store_false',
                    help='Skip basic reduction')
parser.add_argument('-w', dest='basic_wave', default=True, action='store_false',
                    help='Skip wavelength calibration')

args = parser.parse_args()
obsdate = args.ddir

os.chdir(obsdate)
if not os.path.isdir('sci'): os.mkdir('sci')
os.chdir('sci')

#basic image reductions
infile_list = glob.glob('../raw/P*fits')
if args.basic_red:
    imred(infile_list, './', datadir+'bpm_rss_11.fits', cleanup=True)

#basic polarimetric reductions
logfile='specpol'+obsdate+'.log'

#target and wavelength map
infile_list = sorted(glob.glob('m*fits'))
linelistlib=""
if args.basic_wave:
    specpolwavmap(infile_list, linelistlib=linelistlib, logfile=logfile)

#background subtraction and extraction
#infile_list = sorted(glob.glob('wm*fits'))
#specpolextract(infile_list, logfile=logfile, debug=True)

#raw stokes
#infile_list = sorted(glob.glob('e*0[6-9].fits'))       # subselection
#infile_list = sorted(glob.glob('e*fits'))
#specpolrawstokes(infile_list, logfile=logfile)

#final stokes
#polcal = 'polcal0.txt'                                 # null calibration
#infile_list = sorted(glob.glob('*_h[0,2]*.fits'))      # subselection 
#polcal = 'polcal.txt'
#infile_list = sorted(glob.glob('*_h*.fits'))
#specpolfinalstokes(infile_list, polcal=polcal, logfile=logfile)
