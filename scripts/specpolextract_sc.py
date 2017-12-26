
"""

extract spectropolarimetry from image or images

"""

import os, sys, glob, shutil

import numpy as np
from astropy.io import fits as pyfits
from scipy.ndimage.filters import median_filter
from scipy.interpolate import interp1d
from correct_files import correct_files

from PySpectrograph.Spectra import findobj, Spectrum

import pylab as pl

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

def specpolextract_sc(infilelist,*posargs,**kwargs):  
    print '\nspecpolextract_sc version: 20171226'
    if len(posargs):
        print "Incorrect call of specpolextract_sc.  Likely need to update reducepoldata_sc.py from:"
        print polsaltdir+'/scripts/reducepoldata_sc.py'
        exit()
    logfile = kwargs.pop('logfile','salt.log')
    locatewindow = kwargs.pop('locate',(-120.,120.))
    xtrwindow = kwargs.pop('extract',10.)
    docomp = kwargs.pop('docomp',False)
    useoldc = kwargs.pop('useoldc',False)

    log = open(logfile,'a')
    print >> log, '\nspecpolextract_sc version: 20171226'  

  # first estimate image tilt on brightest spectrum
    tilt_i = np.full(len(infilelist),np.nan)
    filestocorrect = 0
    wollaston_file=datadir+"wollaston.txt"
    lam_m = np.loadtxt(wollaston_file,dtype=float,usecols=(0,))
    rpix_om = np.loadtxt(wollaston_file,dtype=float,unpack=True,usecols=(1,2))

    infilelist = sorted(infilelist)
    for i,img in enumerate(infilelist):
        if (useoldc & os.path.exists('c' + img)): continue
        hdul = pyfits.open(img)
        if hdul[0].header['CCDTYPE'] == 'ARC': continue
        filestocorrect += 1
        sci_orc = hdul['SCI'].data
        rows, cols = sci_orc[0].shape
        cbin, rbin = [int(x) for x in hdul[0].header['CCDSUM'].split(" ")]

      # col_dC = columns C in each sample d=(0,1) centered at c_d (0,1)
        if hdul[0].header['GRATING'].strip() == 'PG0300': 
            c_d = (cols*np.array([.25,.5])).astype(int)                        # avoid 2nd order
        else: c_d = (cols*np.array([.25,.75])).astype(int)
        dc = c_d[1]-c_d[0]
        col_dC = (np.arange(c_d[0]-.05*cols,c_d[0]+.05*cols) + dc*np.arange(2)[:,None]).astype(int)

      # row_odR = rows R to sample centered on untilted polarimetric spectra (brightest in O)
      #   as small as possible to avoid picking up more than one spectrum
        row_o = np.zeros(2,dtype=int)
        row_o[0] = np.argmax(np.median(sci_orc[0,:,(2*cols/5):(3*cols/5)],axis=1)).astype(int)
        lam_c = hdul['WAV'].data[0,row_o[0]]
        rpix_oc = interp1d(lam_m,rpix_om,kind='cubic',bounds_error=False)(lam_c)
        rcenter_o = (np.array([rows,0]) +   \
            np.array([-1.,1.])*0.5*(rpix_oc[1,cols/2] - rpix_oc[0,cols/2])/rbin).astype(int)
        row_o[1] = rcenter_o[1] + row_o[0] - rcenter_o[0]
        drow = int(round(4*xtrwindow+0.01*dc*cbin)/rbin)   #  1% allowed tilt 

        row_odR = (row_o[:,None,None] +  \
            (rpix_oc[:,c_d]-rpix_oc[:,cols/2][:,None])[:,:,None]/rbin + np.arange(-drow,drow)).astype(int)

        row_od = np.zeros((2,2))
        for o,d in np.ndindex(2,2):
            sci_Rc = sci_orc[o][row_odR[o,d]] 
            row_od[o,d] = np.median(sci_Rc[:,col_dC[d]].sum(axis=1).argsort()[-17:])

        yoff_o = rbin*(row_o - rcenter_o)/8. 
        tilt_i[i] = (row_od.mean(axis=0)[0]-row_od.mean(axis=0)[1])*float(cols)/float(dc)
        print img,(": brtest object at %6.1f arcsec, tilt: %3i" % (yoff_o[0],tilt_i[i]))
        print >>log,img,(": brtest object at %6.1f arcsec, tilt: %3i" % (yoff_o[0],tilt_i[i]))

    if filestocorrect:
        if ((yoff_o.min() < locatewindow[0]) | (yoff_o.max() > locatewindow[1])):
            print "\nNOTE: science locate window may not contain brightest object\n"
            print >>log,"\nNOTE: science locate window may not contain brightest object\n"
        tilt = np.median(tilt_i[~np.isnan(tilt_i)])
        print "Median image tilt (cw,total,bins): ",tilt
        print >>log,"Median image tilt (cw,total,bins): ",tilt
        log.flush
        
    # straighten out the spectra
        for img in infilelist:
            if (useoldc & os.path.exists('c' + img)): continue
            hdu=correct_files(pyfits.open(img),tilt=tilt)
            hdu.writeto('c' + img, overwrite=True)
            print 'c' + img
            print >>log,'c' + img

    infilelist = [ 'c'+file for file in infilelist ]

    for img in infilelist:
        hdul = pyfits.open(img)
        if hdul[0].header['LAMPID'].strip() != 'NONE': continue
        sci_orc = hdul['SCI'].data
        rows, cols = sci_orc[0].shape
        cbin, rbin = [int(x) for x in hdul[0].header['CCDSUM'].split(" ")]

      # find row center of O and E
        hduw = pyfits.open(img[1:])                                         # use wavmap from wm (cwm messed up)                                      
        wave_orc = hduw['WAV'].data           
        lam_c = wave_orc[0,rows/2]
        rpix_oc = interp1d(lam_m,rpix_om,kind='cubic',bounds_error=False)(lam_c)
        rcenter_o = (np.array([rows,0]) +   \
            np.array([-1.,1.])*0.5*(rpix_oc[1,cols/2] - rpix_oc[0,cols/2])/rbin).astype(int)
        drow_d = (np.round(8*np.array(locatewindow)/rbin)).astype(int)
        sci_or = sci_orc.mean(axis=2)
        droff = 0.
                                    
        if docomp:                         # for comp, offset by row offset of brtest O outside locate window
            userow_r =  np.in1d(np.arange(rows),(rcenter_o[0] + np.arange(*drow_d)),invert=True)
            rlist = np.where(userow_r)[0]
            droff = rlist[np.argmax(sci_or[0,userow_r])] - rcenter_o[0]

        row_o = np.zeros(2,dtype=int)
        for o in (0,1):
            userow_r =  np.in1d(np.arange(rows),(rcenter_o[o] + np.arange(*drow_d) + droff))
            rlist = np.where(userow_r)[0]
            row_o[o] = rlist[0] + np.median(sci_or[o][rlist].argsort()[-17:])

      # establish the wavelength range to be used
        wbin = wave_orc[0,row_o[0],cols/2]-wave_orc[0,row_o[0],(cols/2-1)]   # bin to nearest power of 2 angstroms          
        wbin = 2.**(np.rint(np.log2(wbin)))                                 
        wmin_oc = wave_orc.max(axis=1)                                     
        wmin = wmin_oc[wmin_oc>0].reshape((2,-1)).min(axis=1).max()          # ignore wavmap 0 values        
        wmax = wave_orc[:,row_o].reshape((2,-1)).max(axis=1).min()           # use wavs that are in both beams          
        wmin = (np.ceil(wmin/wbin)+1)*wbin                       
        wmax = (np.floor(wmax/wbin)-1)*wbin 
        drow = int(4*xtrwindow/rbin)
        yoff_o = rbin*(row_o - rcenter_o)/8.                     

        o = 0 
        row1 = row_o[o] - drow
        row2 = row_o[o] + drow 
        data = hdul['SCI'].data[o]
        var = hdul['VAR'].data[o]
        bpm = hdul['BPM'].data[o]
        wave = wave_orc[o]

        w, fo, vo, co, bo = extract(data, var, bpm, wave, row1, row2, wmin, wmax, wbin)

        o = 1
        row1 = row_o[o] - drow
        row2 = row_o[o] + drow   
        data = hdul['SCI'].data[o]
        var = hdul['VAR'].data[o]
        bpm = hdul['BPM'].data[o]
        wave = wave_orc[o]

        w, fe, ve, ce, be = extract(data, var, bpm, wave, row1, row2, wmin, wmax, wbin)
    
        sci_list = [[fo], [fe]]
        var_list = [[vo], [ve]]
        covar_list = [[co], [ce]]
        bad_list = [[bo], [be]]

        outfile = 'e'+img
        if docomp:
            outfile = outfile.split('P')[0]+'P_comp_'+outfile.split('P')[1]
            hdul[0].header['OBJECT'] = 'COMP_'+hdul[0].header['OBJECT']
        write_spectra(w, sci_list, var_list, covar_list, bad_list,  hdul[0].header, wbin, outfile)

        print outfile,("  dyo,dye: %6.1f %6.1f arcsec" % tuple(yoff_o))
        print >>log, (outfile," dyo,dye: %6.1f %6.1f arcsec" % tuple(yoff_o))
        log.flush

    return

def extract(data, var, bpm, wave, row1, row2, wmin, wmax, wbin):
    """Extract the spectra

    Parameters
    ----------
    data: _rc numpy.ndarray
        Flux data for spectra

    var: _rc numpy.ndarray
        variance data for spectra

    mask: _rc numpy.ndarray
        mask data for spectra

    wave: _rc numpy.ndarray
        wavelength map for spectra

   Returns
   -------
    
    """
    wave_W = np.arange(wmin, wmax, wbin)                    # _W = resampled wavelength bin edge
    Waves = wave_W.shape[0]
    row0 = int((row1+row2)/2)
    wmask_c = ((wave[row0] > wmin) & (wave[row0] < (wmax+wbin)))   
    wave_C = wave[row0,wmask_c]                             # _C = original bin centers within wavelength limits
    cmin = np.where(wmask_c)[0][0]                          # column c of C=0
    dwave_C = wave_C[1:]-wave_C[:-1]
    dwave_C = np.append(dwave_C,dwave_C[-1])
    dwavpoly = np.polyfit(wave_C-wave_C.mean(),dwave_C-dwave_C.mean(),3)
    binrat_W = (np.polyval(dwavpoly,wave_W-wave_C.mean()) + dwave_C.mean())/wbin   # old/new bin widths 

    C_W = np.zeros(Waves).astype(int)                       # closest column for each wavelength bin
    for W in range(Waves): C_W[W] = np.where(wave_C > (wave_W[W]))[0][0] -1

    binoff_W = (wave_W - wave_C[C_W])/(wbin*binrat_W)       # offset in columns of closest wavelength bin centers  
    binfrac_dW = np.zeros((3,Waves))
    for d in (-1,0,1):                                      # contribution of nearest old bins to new one
        binfrac_dW[d+1][1:-1] = (np.minimum(wave_W[1:-1]+wbin/2.,wave_C[C_W+d][1:-1]+dwave_C[C_W+d][1:-1]/2.) -    \
            np.maximum(wave_W[1:-1]-wbin/2.,wave_C[C_W+d][1:-1]-dwave_C[C_W+d][1:-1]/2.)) / dwave_C[C_W+d][1:-1]
    binfrac_dW[binfrac_dW < 0.] = 0.

    #estimate the sky
    sky_W = np.zeros_like(wave_W)
    count = 0
    drow = row2-row1
    Rows = 2*drow
    sky_RW = np.zeros((Rows,Waves))
    R = -1
    for r in range(row1-3*drow, row1-2*drow) + range(row2+2*drow, row2+3*drow):
        xmask_c = (wmask_c & (bpm[r]==0))
        R += 1
        sky_RW[R] = np.interp(wave_W, wave[r, xmask_c], data[r, xmask_c])
        sky_W += np.interp(wave_W, wave[r, xmask_c], data[r, xmask_c])
        count += 1
    sky_W = sky_W / count

    # extract the spectra
    f_W = np.zeros_like(wave_W)
    v_W = np.zeros_like(wave_W)
    cov_W = np.zeros_like(wave_W)
    b_W = np.zeros_like(wave_W).astype(int)

    for r in range(row1, row2):
        xmask_c = (wmask_c & (bpm[r]==0))
        f_W += np.interp(wave_W, wave[r, xmask_c], data[r, xmask_c]) - sky_W
        dv_W = (binfrac_dW**2*var[r,C_W+cmin][None,:]).sum(axis=0)        
        v_W += dv_W
        cov_W[:-1] += dv_W[:-1]*binfrac_dW[1,:-1]*binfrac_dW[2,1:]
        b_W += (np.interp(wave_W, wave[r, wmask_c], xmask_c[wmask_c].astype(float)) < 0.5).astype(int)

    f_W /= binrat_W
    b_W = ((b_W > 0.25*(row2-row1+1)) | (f_W == 0.) | (v_W == 0.)).astype('uint8')    # bad if 0 or > 25% spectrum bad

    return wave_W, f_W, v_W, cov_W, b_W

def write_spectra(wave, sci_ow, var_ow, covar_ow, badbin_ow, header, wbin, outfile):
    """Write out the spectra in the correct format

    """
    header['VAREXT'] = 2
    header['COVEXT'] = 3
    header['BPMEXT'] = 4
    header['CRVAL1'] = wave[0]
    header['CRVAL2'] = 0
    header['CDELT1'] = wbin             
    header['CTYPE1'] = 'Angstroms'
    hduout = pyfits.PrimaryHDU(header=header)
    hduout = pyfits.HDUList(hduout)

    #what's the initial shape? 
    hduout.append(pyfits.ImageHDU(data=sci_ow, header=header, name='SCI'))
    header.set('SCIEXT',1,'Extension for Science Frame',before='VAREXT')
    hduout.append(pyfits.ImageHDU(data=var_ow, header=header, name='VAR'))
    hduout.append(pyfits.ImageHDU(data=covar_ow, header=header, name='COV'))
    hduout.append(pyfits.ImageHDU(data=badbin_ow, header=header, name='BPM'))

    hduout.writeto(outfile,overwrite=True,output_verify='warn')

