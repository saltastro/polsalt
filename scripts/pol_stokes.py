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

logfile = 'temp.log'
#raw stokes
#infile_list = sorted(glob.glob('e*0[6-9].fits'))       # subselection
infile_list = sorted(glob.glob('e*fits'))
specpolrawstokes(infile_list, logfile=logfile)

#final stokes
#polcal = 'polcal0.txt'                                 # null calibration
#infile_list = sorted(glob.glob('*_h[0,2]*.fits'))      # subselection 
polcal = 'polcal.txt'
infile_list = sorted(glob.glob('*_h*.fits'))
specpolfinalstokes(infile_list, polcal=polcal, logfile=logfile)
