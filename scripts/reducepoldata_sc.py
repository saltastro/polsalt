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
from specpolextract import specpolextract
from specpolrawstokes import specpolrawstokes
from specpolfinalstokes import specpolfinalstokes
from correct_files import correct_files
from pol_extract_kn import extract,write_spectra 

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
specpolwavmap(infile_list, linelistlib=linelistlib, logfile=logfile)

#background subtraction and extraction
#specpolextract(infile_list, logfile=logfile, debug=debug)

log = open(logfile,'a')
infile_list = sorted(glob.glob('wm*fits'))

for img in infile_list:
    hdu=correct_files(pyfits.open(img))
    hdu.writeto('c' + img, clobber=True)
    print 'c' + img
    print >>log,'c' + img

infile_list = sorted(glob.glob('cw*fits'))

dyasec = 5.     # star +/-5, bkg= +/-(25-35)arcsec:  2nd order is 9-20 arcsec away
yoffo = 0.      # optional offset of target (bins) from brightest in O (bottom) image

for img in infile_list:
    hdu = pyfits.open(img)
    if hdu[0].header['LAMPID'].strip() != 'NONE': continue
    ybin = int(hdu[0].header['CCDSUM'].split(' ')[1])
    dy = int(dyasec*(8/ybin))
    yo_o = np.zeros(2,dtype=int)
    yo_o[0] = int(np.argmax(hdu[1].data[0].sum(axis=1)) + yoffo)   
    yo_o[1] = int(np.argmax(hdu[1].data[1].sum(axis=1)) + yoffo*1.0075)  # allow for magnification
    print img,"  yo,ye,dy: ",yo_o,dy  
    print >>log,img," yo,ye,dy: ",yo_o,dy
    log.flush

    # establish the wavelength range to be used
    ys, xs = hdu[1].data.shape[1:]                                       # khn
    wave_oyx = hdu[4].data                                               # khn
    wbin = wave_oyx[0,yo_o[0],xs/2]-wave_oyx[0,yo_o[0],xs/2-1]           # khn bin to nearest power of 2 angstroms
    wbin = 2.**(np.rint(np.log2(wbin)))                                  # khn
    wmin_ox = wave_oyx.max(axis=1)                                       # khn 
    wmin = wmin_ox[wmin_ox>0].reshape((2,-1)).min(axis=1).max()          # khn ignore wavmap 0 values
    wmax = wave_oyx[:,yo_o].reshape((2,-1)).max(axis=1).min()            # khn use wavs that are in both beams
    wmin = (np.ceil(wmin/wbin)+0.5)*wbin                                 # khn note: sc extract uses bin edges
    wmax = (np.floor(wmax/wbin)-0.5)*wbin                                # khn

    o = 0 
    y1 = yo_o[o] - dy
    y2 = yo_o[o] + dy
    data = hdu[1].data[o]
    var = hdu[2].data[o]
    mask = hdu[3].data[o]
    wave = wave_oyx[o]
    w, fo, vo, bo = extract(data, var, mask, wave, y1, y2, wmin, wmax, wbin)

    o = 1
    y1 = yo_o[o] - dy
    y2 = yo_o[o] + dy
    data = hdu[1].data[o]
    var = hdu[2].data[o]
    mask = hdu[3].data[o]
    wave = wave_oyx[o]
    w, fe, ve, be = extract(data, var, mask, wave, y1, y2, wmin, wmax, wbin)
    
    sci_list = [[fo], [fe]]
    var_list = [[vo], [ve]]
    bad_list = [[bo], [be]]
   
    write_spectra(w, sci_list, var_list, bad_list,  hdu[0].header, wbin, 'e' + img)

#raw stokes
infile_list = sorted(glob.glob('e*fits'))
specpolrawstokes(infile_list, logfile=logfile)

#final stokes
#debug=True
infile_list = sorted(glob.glob('*_h*.fits'))
specpolfinalstokes(infile_list, logfile=logfile, debug=debug)
