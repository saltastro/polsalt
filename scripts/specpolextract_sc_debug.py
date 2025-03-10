
"""

extract spectropolarimetry from image or images
split into correct and spectrum-extract

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

from rssmaptools import ccdcenter
from specpolutils import datedfile, configmap

def specpolcorrect_sc(infileList,**kwargs):
    print '\nspecpolcorrect_sc version: 20191123'
    logfile = kwargs.pop('logfile','salt.log')
    locatewindow = kwargs.pop('locate',(-120.,120.))
    xtrwindow = kwargs.pop('extract',10.)
    useoldc = kwargs.pop('useoldc',False)

    log = open(logfile,'a')
    print >> log, '\nspecpolcorrect_sc version: 20191210'  

  # group the files together
    confitemlist = ['GRATING','GR-ANGLE','CAMANG','BVISITID']
    obs_i,config_i,obstab,configtab = configmap(infileList, confitemlist)    
    obss = len(obstab)
    configs = len(configtab)
    infiles = len(infileList)
    wollaston_file=datadir+"wollaston.txt"
    lam_m = np.loadtxt(wollaston_file,dtype=float,usecols=(0,))
    rpix_om = np.loadtxt(wollaston_file,dtype=float,unpack=True,usecols=(1,2))     

    for o in range(obss):
        grating, grangle, camang, bvisitid = configtab[obstab[o]['config']]
        objectname = obstab[o]['object']
        fileListj = [infileList[i] for i in range(infiles) if obs_i[i]==o]
      # first estimate image tilt on brightest spectrum
        tilt_i = np.full(len(fileListj),np.nan)
        filestocorrect = 0
        infilelist = sorted(fileListj)
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
            
            raxis_o = (np.array([rows,0]) +   \
                np.array([-1.,1.])*0.5*(rpix_oc[1,cols/2] - rpix_oc[0,cols/2])/rbin).astype(int)
            row_o[1] = raxis_o[1] + row_o[0] - raxis_o[0]
            drow = int(round(4*xtrwindow+0.01*dc*cbin)/rbin)   #  1% allowed tilt 

            row_odR = (row_o[:,None,None] +  \
                (rpix_oc[:,c_d]-rpix_oc[:,cols/2][:,None])[:,:,None]/rbin + np.arange(-drow,drow)).astype(int)

            print row_o[0]
            print c_d
            print lam_c[c_d]
            np.savetxt('lam_c.txt',lam_c.T,fmt = "%8.2f")
            exit()

            row_od = np.zeros((2,2))
            for o,d in np.ndindex(2,2):
                sci_Rc = sci_orc[o][row_odR[o,d]] 
                row_od[o,d] = np.median(sci_Rc[:,col_dC[d]].sum(axis=1).argsort()[-17:])

            yoff_o = rbin*(row_o - raxis_o)/8. 
            tilt_i[i] = (row_od.mean(axis=0)[0]-row_od.mean(axis=0)[1])*float(cols)/float(dc)
            print img,(": brtest O object at %6.1f arcsec, row: %4i, tilt: %4.0f" % (yoff_o[0],row_o[0],tilt_i[i]))
            print >>log,img,(": brtest O object at %6.1f arcsec, row: %4i, tilt: %4.0f" % (yoff_o[0],row_o[0],tilt_i[i]))

        if filestocorrect:
            if ((yoff_o.min() < locatewindow[0]) | (yoff_o.max() > locatewindow[1])):
                print "\nNOTE: science locate window may not contain brightest object\n"
                print >>log,"\nNOTE: science locate window may not contain brightest object\n"
            tilt = np.median(tilt_i[~np.isnan(tilt_i)])
            print ("Median image tilt (cw,total,bins): %6.1f" % tilt)
            print >>log,("Median image tilt (cw,total,bins): %6.1f" % tilt)
            log.flush
        
          # straighten out the spectra
            for img in infilelist:
                if (useoldc & os.path.exists('c' + img)): continue
                hdu=correct_files(pyfits.open(img),tilt=tilt)
                hdu.writeto('c' + img, overwrite=True)
                print 'c' + img
                print >>log,'c' + img

    return

def specpolspectrum_sc(infileList,**kwargs):
    print '\nspecpolspectrum_sc version: 20210302'
    logfile = kwargs.pop('logfile','salt.log')
    locatewindow = kwargs.pop('locate',(-120.,120.))    # arcsec relative to optical axis
    xtrwindow = kwargs.pop('extract',10.)               # arcsec relative to located spectrum               
                                                        # optional list t1,t2,lb1,lb2,rb1,rb2; default t2-t1
    docomp = kwargs.pop('docomp',False)                 # comp locate window is inverse of locate window
    dolamp = kwargs.pop('dolamp',False)                 # reduce lamp data   

    log = open(logfile,'a')
    print >> log, '\nspecpolspectrum_sc version: 20210302' 

  # group the files together
    confitemlist = ['GRATING','GR-ANGLE','CAMANG','BVISITID']
    obs_i,config_i,obstab,configtab = configmap(infileList, confitemlist)    
    obss = len(obstab)
    configs = len(configtab)
    infiles = len(infileList)
    wollaston_file=datadir+"wollaston.txt"
    lam_m = np.loadtxt(wollaston_file,dtype=float,usecols=(0,))
    rpix_om = np.loadtxt(wollaston_file,dtype=float,unpack=True,usecols=(1,2))

    for o in range(obss):
        grating, grangle, camang, bvisitid = configtab[obstab[o]['config']]
        objectname = obstab[o]['object']
        fileListj = [infileList[i] for i in range(infiles) if obs_i[i]==o]
      # get gain correction data
        hdul0 = pyfits.open(fileListj[0])
        sci_orc = hdul0['SCI'].data
        rows, cols = sci_orc[0].shape
        cbin, rbin = [int(x) for x in hdul0[0].header['CCDSUM'].split(" ")]
        dateobs =  hdul0[0].header['DATE-OBS'].replace('-','')
        utchour = int(hdul0[0].header['UTC-OBS'].split(':')[0])
        mjdateobs = str(int(dateobs) - int(utchour < 10))
        gain = hdul0[0].header['GAINSET']
        speed = hdul0[0].header['ROSPEED']
        rccenter_d, cgap_c = ccdcenter(hdul0['SCI'].data[0])
        c_a = np.array([0, cgap_c[0]-1024/cbin+1, cgap_c[[0,1]].mean(),   \
            cgap_c[1]+1024/cbin, cgap_c[[2,3]].mean(), cgap_c[3]+1024/cbin, cols])  

        GainCorrectionFile = datedfile(datadir+"RSS_Gain_Correction_yyyymmdd_vnn.txt",mjdateobs)
        mode_ld = np.loadtxt(GainCorrectionFile, dtype='string', usecols=(0,1))
        lmode = np.where((mode_ld[:,0]==gain) & (mode_ld[:,1] == speed))[0][0]
        gaincor_a = np.loadtxt(GainCorrectionFile, usecols=range(2,8), unpack=True).T[lmode]
        gaincor_c = np.ones(cols)
        for a in range(6):
            if (gaincor_a[a] == 1.): continue
            isa_c = ((np.arange(cols) >= c_a[a]) & (np.arange(cols) < c_a[a+1]))
            gaincor_c[isa_c] = gaincor_a[a]

      # define locate and extraction windows
        if (type(xtrwindow) == list):
            if len(xtrwindow) != 6:
                print ("Extract aperture list must have 6 elements: " & xtrwindow)
                exit()
            xtr_sd = np.array(xtrwindow).reshape((3,2))
            docomp = False                  # these are relative to main target in locate window
        else:
            xtr_sd = np.array([[-xtrwindow/2.,xtrwindow/2.],    \
                [-3.5*xtrwindow,-2.5*xtrwindow],[2.5*xtrwindow,3.5*xtrwindow]])
        
        print ("Extraction aperture (arcsecs), target: %6.1f %6.1f,  bkg1: %6.1f %6.1f,  bkg2: %6.1f %6.1f" % \
            tuple(xtr_sd.flatten()))
        print >>log, ("Extraction aperture (arcsecs), target: %6.1f %6.1f,  bkg1: %6.1f %6.1f,  bkg2: %6.1f %6.1f" % \
            tuple(xtr_sd.flatten()))

      # process images
        for i,img in enumerate(fileListj):
            hdul = pyfits.open(img)
            if ((not dolamp) & (hdul[0].header['LAMPID'].strip() != 'NONE')): continue
            sci_orc = hdul['SCI'].data
            rows, cols = sci_orc[0].shape
            cbin, rbin = [int(x) for x in hdul[0].header['CCDSUM'].split(" ")]

          # find row of O and E spectrum, center ccd only
            hduw = pyfits.open(img[1:])         # compute axis row using wavmap from wm (ok at col center)                                      
            wave_orc = hduw['WAV'].data           
            lam_c = wave_orc[0,rows/2]
            rpix_oc = interp1d(lam_m,rpix_om,kind='cubic',bounds_error=False)(lam_c)
            raxis_o = (np.array([rows,0]) +   \
                np.array([-1.,1.])*0.5*(rpix_oc[1,cols/2] - rpix_oc[0,cols/2])/rbin).astype(int)
            drowlocate_d = (np.round(8*np.array(locatewindow)/rbin)).astype(int)    # locate window relative to axis in rows
            sci_or = np.median(sci_orc[:,:,(cols/3):(2*cols/3)],axis=2)             # vertical profile, center ccd

          # fine locate window is twice the nominal extraction aper (20 arcsec), 
          #   offset by row offset of brtest O in locate window (dcomp, outside locate window)                                   
            invert=docomp                         
            userow_r =  np.in1d(np.arange(rows),(raxis_o[0] + np.arange(*drowlocate_d)),invert=invert)
            rlist = np.where(userow_r)[0]
            droff = rlist[np.argmax(sci_or[0,userow_r])] - raxis_o[0]
            drowfinelocate_d = droff + np.array([-1,1])*int(8*10./rbin)
            
            row_o = np.zeros(2,dtype=int)
            for o in (0,1):
                userow_r =  np.in1d(np.arange(rows),(raxis_o[o] + np.arange(*drowfinelocate_d)))
                rlist = np.where(userow_r)[0]
                row_o[o] = rlist[0] + np.median(sci_or[o][rlist].argsort()[-17:])
            xtrrow_osd = (row_o[:,None,None] + 8.*xtr_sd[None,:,:]/rbin).astype(int)

          # establish the wavelength range to be used, based on first img, so extractions match
            if (i==0):
                wbin = wave_orc[0,row_o[0],cols/2]-wave_orc[0,row_o[0],(cols/2-1)]   # bin to nearest power of 2 angstroms          
                wbin = 2.**(np.rint(np.log2(wbin)))                                 
                wmin_oc = wave_orc.max(axis=1)                                     
                wmin = max((wmin_oc[0][wmin_oc[0]>0]).min(), (wmin_oc[1][wmin_oc[1]>0]).min())  # ignore wavmap 0 values        
                wmax = wave_orc[:,row_o].reshape((2,-1)).max(axis=1).min()           # use wavs that are in both beams          
                wmin = (np.ceil(wmin/wbin)+1)*wbin                       
                wmax = (np.floor(wmax/wbin)-1)*wbin                      

            o = 0
            xtrrow_sd = xtrrow_osd[o]
            data = hdul['SCI'].data[o]/gaincor_c[None,:]
#            var = hdul['VAR'].data[o]/gaincor_c[None,:]**2 before 20210302
            var = hdul['VAR'].data[o]/gaincor_c[None,:]
            bpm = hdul['BPM'].data[o]
            wave = wave_orc[o]

            w, fo, vo, co, bo = extract(data, var, bpm, wave, xtrrow_sd, wmin, wmax, wbin)

            o = 1
            xtrrow_sd = xtrrow_osd[o]
            data = hdul['SCI'].data[o]/gaincor_c[None,:]
#            var = hdul['VAR'].data[o]/gaincor_c[None,:]**2 before 20210302
            var = hdul['VAR'].data[o]/gaincor_c[None,:]            
            bpm = hdul['BPM'].data[o]
            wave = wave_orc[o]

            w, fe, ve, ce, be = extract(data, var, bpm, wave, xtrrow_sd, wmin, wmax, wbin)
    
            sci_list = [[fo], [fe]]
            var_list = [[vo], [ve]]
            covar_list = [[co], [ce]]
            bad_list = [[bo], [be]]

            outfile = 'e'+img
            datecor = GainCorrectionFile.split('_')[-2]
            hdul[0].header.add_history('GainCorrection: '+datecor+6*' %6.4f' % tuple(gaincor_a))
            if docomp:
                outfile = outfile.split('P')[0]+'P_comp_'+outfile.split('P')[1]
                hdul[0].header['OBJECT'] = 'COMP_'+hdul[0].header['OBJECT']
            write_spectra(w, sci_list, var_list, covar_list, bad_list,  hdul[0].header, wbin, outfile)

            print outfile,(("  ro "+3*"%4i %4i, "+"  re "+3*"%4i %4i, ") % tuple(xtrrow_osd.flatten()))
            print >>log, outfile,("  ro "+3*"%4i %4i, "+"  re "+3*"%4i %4i, ") % tuple(xtrrow_osd.flatten())
            log.flush

    return

def specpolextract_sc(infilelist,**kwargs):
    specpolcorrect_sc(infilelist,**kwargs)
    infilelist = [ 'c'+file for file in infilelist ]    
    specpolspectrum_sc(infilelist,**kwargs)
    return
  
def extract(data, var, bpm, wave, xtrrow_sd, wmin, wmax, wbin):
    """Extract the spectra

    Parameters
    ----------
    data: _rc numpy.ndarray
        Flux data for spectra

    var: _rc numpy.ndarray
        variance data for spectra

    bpm: _rc numpy.ndarray
        bpm data for spectra

    wave: _rc numpy.ndarray
        wavelength map for spectra

    xtrrow_sd: 3x2 int ndarray
        extraction apertures (row1,row2) (inclusive) for target, bkg1, bkg2

   Returns
   -------
    
    """

    rccenter_d, cgap_c = ccdcenter(data)
    cbin = 2048/(cgap_c[2]-cgap_c[1]+1)                       # _A central amps in each ccd, to be marked bad
    c_A = np.array([cgap_c[0]-1024/cbin, cgap_c[1]+1024/cbin -1, cgap_c[3]+1024/cbin -1])

    wave_W = np.arange(wmin, wmax, wbin)                    # _W = resampled wavelength bin edge
    Waves = wave_W.shape[0]
    row0 = int((xtrrow_sd[0].sum())/2)
    wmask_c = ((wave[row0] > wmin) & (wave[row0] < (wmax+wbin)))   
    wave_C = wave[row0,wmask_c]                             # _C = original bin centers within wavelength limits
    cmin = np.where(wmask_c)[0][0]                          # column c of C=0
    wave_A = 0.5*(wave[row0,c_A] + wave[row0,c_A+1])
    W_A = ((wave_A - wmin)/wbin).astype(int)                # Wave index of central amps

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
    Rows_s = np.diff(xtrrow_sd,axis=1)[:,0] + 1
    sky_W = np.zeros_like(wave_W)
    count = 0

    sky_RW = np.zeros((Rows_s[1:].sum(),Waves))
    R = -1
    for r in range(xtrrow_sd[1,0],xtrrow_sd[1,1]+1) + range(xtrrow_sd[2,0],xtrrow_sd[2,1]+1):
        xmask_c = (wmask_c & (bpm[r]==0))
        xmask_c[1:-1] &= (xmask_c[:-2] & xmask_c[2:])           # more conservative CR
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

    for r in range(xtrrow_sd[0,0],xtrrow_sd[0,1]+1):
        xmask_c = (wmask_c & (bpm[r]==0))
        xmask_c[1:-1] &= (xmask_c[:-2] & xmask_c[2:])           # more conservative CR
        f_W += np.interp(wave_W, wave[r, xmask_c], data[r, xmask_c]) - sky_W
        dv_W = (binfrac_dW**2*var[r,C_W+cmin][None,:]).sum(axis=0)        
        v_W += dv_W
        cov_W[:-1] += dv_W[:-1]*binfrac_dW[1,:-1]*binfrac_dW[2,1:]
        b_W += (np.interp(wave_W, wave[r, wmask_c], xmask_c[wmask_c].astype(float)) < 0.5).astype(int)

    f_W /= binrat_W                                             # bad if 0 or > 25% spectrum bad:
    b_W = ((b_W > 0.25*Rows_s[0]) | (f_W == 0.) | (v_W == 0.)).astype('uint8') 
    b_W[W_A] = 2                                            # special badpix marker for central amps

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

