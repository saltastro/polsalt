
"""
poldisttest

print out distortion computation

"""

import os, sys, glob, shutil, inspect

import numpy as np

from pyraf import iraf
from iraf import pysalt
from saltobslog_kn import obslog
from saltsafelog import logging
from specpolutils import datedline, rssdtralign, rssmodelwave, configmap
from rssmaptools import boxsmooth1d, blksmooth2d, RSScolpolcam

datadir = os.path.dirname(__file__) + '/data/'
np.set_printoptions(threshold=np.nan)

import warnings 
# warnings.filterwarnings("error")

def poldisttest():
    wav_l = np.arange(3100.,10600.,100.)
    wavs = wav_l.shape[0]
    YX_di = np.array([np.linspace(-25.,25.,11),np.zeros(11)])
    ids = YX_di.shape[1]
    wav_s = np.tile(wav_l,ids)
    YX_ds = np.repeat(YX_di,wavs,axis=1)    
    yx_dps = RSScolpolcam(YX_ds,wav_s,7.5,7.5) 
    np.savetxt("y_wpY.txt",yx_dps[0].reshape((-1,wavs)).T,fmt="%9.5f")
    return

# ------------------------------------

if __name__=='__main__':   
    poldisttest()

