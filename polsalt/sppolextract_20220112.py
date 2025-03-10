
"""
sppolextract

Optimal extraction for grating polarimetry data
Write out extracted data fits (etm*) dimensions wavelength,target #

sppolextract
splinemapextract
"""

import os, sys, glob, shutil, inspect, pprint

import numpy as np
from scipy.interpolate import griddata, interp1d, UnivariateSpline, LSQUnivariateSpline
from scipy.ndimage.interpolation import shift
from scipy.signal import convolve2d
from scipy.linalg import lstsq
from scipy.optimize import fmin
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table

# this is pysalt-free

import rsslog
from obslog import create_obslog
from scrunch1d import scrunch1d, scrunchvar1d
from polutils import datedline, rssdtralign, fence
from polmaptools import ccdcenter,boxsmooth1d,impolguide,rotate2d,Tableinterp,fracmax
from rssoptics import RSScolpolcam,RSSpolgeom

datadir = os.path.dirname(__file__) + '/data/'
keywordfile = datadir+"obslog_config.json"
np.set_printoptions(threshold=np.nan)

import warnings
# warnings.filterwarnings("error")

# ----------------------------------------------------------
def sppolextract(fileList,name,logfile='salt.log',debug=False,fpidebug=(-1,-1,-1)):
    """derive extracted target data vs target and wavelength for grating spectropolarimetry

    Parameters
    ----------
    fileList: list of strings

    """
    """
    _d dimension index r,c = 0,1
    _f file index
    _i mask slits (entries in xml and tgtTab)
    _p pol beam = 0,1 for O,E
    _R, _C bin coordinate within target box
    _a amplifier = 0,1,2,3,4,5

    """

  # data in common, from first file
  
    pixmm = 0.015
    files = len(fileList)
    imgno_f = np.array([f.split('.')[0][-4:] for f in fileList]).astype(int)
    hdul0 = pyfits.open(fileList[0])
    hdr0 = hdul0[0].header
    exptime = hdr0['EXPTIME']
    lampid = hdr0['LAMPID'].strip().upper()
    rcbin_d = np.array(hdr0['CCDSUM'].split(' ')).astype(int)[::-1]
    prows, cols = hdul0[1].data.shape[1:]
    camtem = hdr0['CAMTEM']
    coltem = hdr0['COLTEM']
    calimg = hdr0['CALIMG'].strip()
    filter = hdr0['FILTER']
    trkrho = hdr0['TRKRHO']    
    dateobs =  hdr0['DATE-OBS'].replace('-','')
    diffuse = ('DIFFUSE' in hdr0)

    tgtTab = Table.read(hdul0['TGT'])
    entries = len(tgtTab['CATID'])
    tgtname_i = np.array(tgtTab['CATID'])
    oktgt_i = (tgtTab['CULL']=="")
    oktargs = oktgt_i.sum()
    YX_di = np.array([tgtTab['YCE'],tgtTab['XCE']])
    slit_prc = hdul0['TMAP'].data
    bkg_prc = hdul0['BMAP'].data
    wav_prc = hdul0['WMAP'].data    
    isslit_pirc = np.zeros((2,entries,prows,cols),dtype=bool)
    isbkg_pirc = np.zeros_like(isslit_pirc)
    for p,i in np.ndindex(2,entries): 
        isslit_pirc[p,i] = (slit_prc[p] == i+1)
        isbkg_pirc[p,i] = (bkg_prc[p] == i+1)                   

  # data from cal file
    hdulcal = pyfits.open(glob.glob('tm*'+calimg+".fits")[0])
    werr_pic = hdulcal['WERR'].data                
    calimgno = int(calimg)
    wav0 = hdulcal[0].header['REFWAV']    
    yx0nom_dp,rshift,yxp0nom_dp,dum = RSSpolgeom(hdulcal,wav0)    # nominal dtr geom from arc image
    yxp0ref_dp = np.array([map(float,hdr0['YXAXISO'].split()),map(float,hdr0['YXAXISE'].split())]).T

    ur0cal,uc0cal,saltfps = rssdtralign(dateobs,trkrho)
    yx0cal_d = -0.015*np.array([ur0cal,uc0cal])
    dtrfps = saltfps*np.diff(RSScolpolcam(np.array([[0.,0.],[0.,1.]]),wav0,coltem,camtem)[1,0])[0]

  # fixed geometry info.
    rccenter_d, cgap_c = ccdcenter(hdul0[1].data[0])
    rccenter_d[0] = prows                                       # gets center of unsplit data
    c0_a = (cgap_c[0]-2048/rcbin_d[1])*np.ones(6,dtype=int)
    c0_a[[2,4]] = cgap_c[[1,3]]
    c0_a[[1,3,5]] = c0_a[[0,2,4]]+1024/rcbin_d[1]               # gets first column in each amplifier

  # assemble target data from all files, compute background
    rsslog.message ("Assembling target data, computing background", logfile)
    image_fprc = np.zeros((files,2,prows,cols))
    var_fprc = np.zeros_like(image_fprc) 
    okbin_fprc =  np.zeros_like(image_fprc).astype(bool)
    bkg_fprc = np.zeros((files,2,prows,cols))
    hsta_f = np.zeros(files,dtype=int) 
    qsta_f = np.zeros(files,dtype=int)
    hasslit_pir = isslit_pirc.any(axis=3)     
    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)
        image_fprc[f] = hdul['SCI'].data
        var_fprc[f] = hdul['VAR'].data
        okbin_fprc[f] = (hdul['BPM'].data==0)
        hsta_f[f] = int(round(float(hdul[0].header['HWP-ANG'])/11.25))
        qsta_f[f] = int(round(float(hdul[0].header['QWP-ANG'])/45.))          
      # compute background
        if diffuse:
          # for diffuse, use any bkg (bkg=255) within one slit
            for p,i in np.ndindex(2,entries):
                if (~oktgt_i[i]): continue
                slitrows = isslit_pirc[p,i].sum(axis=0).max()
                isbkgarea_rc = ((shift(isslit_pirc[p,i],(-slitrows,0),order=0) | \
                          shift(isslit_pirc[p,i],(+slitrows,0),order=0)) & (bkg_prc[p]==255))
                bkg_c = np.ma.median(np.ma.masked_array(image_fprc[f,p,hasslit_pir[p,i],:],    \
                    mask=~isbkgarea_rc[hasslit_pir[p,i],:]),axis=0).data                
                bkg_fprc[f,p,isslit_pirc[p,i]] = np.repeat(bkg_c[None,:],prows,axis=0)[isslit_pirc[p,i]]
                
        rsslog.message("  "+file,logfile)

  # compute profiles for CRs and extraction from mean data, binned by profbin in column direction
    rsslog.message("\nCompute spectral profiles from mean data", logfile)
    rsslog.message("  mean width (bins)/ wavelength samples:", logfile)    
    rsslog.message((" O/E"+oktargs*"%4i ") % tuple(np.arange(entries)[oktgt_i]), logfile)

    legx_c = 2*np.arange(cols,dtype=float)/cols-1.  
    okmean_prc = okbin_fprc.all(axis=0)
    signalmean_prc = okmean_prc*(image_fprc - bkg_fprc).sum(axis=0)/files
    varmean_prc = okmean_prc*var_fprc.sum(axis=0)/files**2
    
    profbin = 8
    snlim = 8.
    sn_prc = np.zeros((2,prows,cols))
    sn_prc[varmean_prc>0.] = signalmean_prc[varmean_prc>0.]/np.sqrt(varmean_prc[varmean_prc>0.])
    if debug:
        hdusn = pyfits.PrimaryHDU(sn_prc.astype('float32'))
        hdulsn = pyfits.HDUList([hdusn])
        hdulsn.writeto("sn_prc.fits",overwrite=True) 
        
  # fit 2d profile to spline(row), with row offset and scale fit by legendre
    sigprof_pirc = np.zeros((2,entries,prows,cols))
    okprof_pirc = np.zeros((2,entries,prows,cols),dtype=bool)    
    rprof_pic = np.zeros((2,entries,cols))  # row position (float) of center of fitted prof, for scrunch
    fiterr_dpi = np.zeros((3,2,entries))    # start and end fiterr, for debug     
    drcenter_pic = np.zeros((2,entries,cols))   # prof center distance from center of slit, for debug
    rbottomclr_pic = np.zeros((2,entries,cols)) # prof minimum bottom,top clearance, for debug 
    rtopclr_pic = np.zeros((2,entries,cols))
    rbottomslit_pic = np.zeros((2,entries,cols),dtype=int) # slit bottom,top for debug 
    rtopslit_pic = np.zeros((2,entries,cols),dtype=int)                    
    meanwidth_pi = np.zeros((2,entries))
    profsamples_pi = np.zeros((2,entries),dtype=int)
    snmedian_pi = np.zeros((2,entries))
    snwavlim_dpi = np.zeros((2,2,entries))
    legx_c = 2*np.arange(cols,dtype=float)/cols -1. 
        
    ff,pp,ii = fpidebug                     # for debug printout    

    for p in (0,1):
        for i in range(entries):
            label = str(p)+("_%02i" % i)            
            if (~oktgt_i[i]): continue        
            if ((p==pp) & (i==ii)):
                debugname = label
            else:
                debugname=None            
            usesig_Rc = (okmean_prc[p] & isslit_pirc[p,i])[hasslit_pir[p,i],:]                                    
            signal_Rc = signalmean_prc[p][hasslit_pir[p,i],:]
            var_Rc = varmean_prc[p][hasslit_pir[p,i],:]
            sn_Rc = sn_prc[p][hasslit_pir[p,i],:]                          
            snmedian_pi[p,i] = np.median((usesig_Rc*sn_Rc).max(axis=0))
            oksn_c = (sn_Rc.max(axis=0) > snlim/np.sqrt(profbin))
            rslitcenter_c = np.argmax(isslit_pirc[p,i],axis=0) +  \
                (isslit_pirc[p,i].sum(axis=0)).astype(float)/2.
            cspecend_d = np.where(oksn_c & (wav_prc[p,rslitcenter_c.astype(int),range(cols)]>0.))[0][[0,-1]]                                                
            snwavlim_dpi[:,p,i] = wav_prc[p,rslitcenter_c[cspecend_d].astype(int),cspecend_d]
            
            fitprof_cR, okprof_cR, rfit_d, rrfit_d, profsamples_pi[p,i],fiterr0,fiterr,iters =   \
                profile2dfit(signal_Rc.T,var_Rc.T,usesig_Rc.T,label,stackbin=profbin,snlim=snlim,debugname=debugname)

            sigprof_pirc[p,i,hasslit_pir[p,i],:] = fitprof_cR.T
            okprof_pirc[p,i,hasslit_pir[p,i],:] = okprof_cR.T               
            usesig_Rc &= okprof_cR.T                                                    
            usesig_c = usesig_Rc.any(axis=0)                       
            rprof_pic[p,i,usesig_c] = np.polynomial.legendre.legval(legx_c[usesig_c],rfit_d) +   \
                np.where(hasslit_pir[p,i])[0][0]                                        
            drcenter_pic[p,i,usesig_c] = (isslit_pirc[p,i].any(axis=0)* \
                (rprof_pic[p,i] - rslitcenter_c))[usesig_c]                
                
            print p,i,np.argmax(fitprof_cR[cols/2]),drcenter_pic[p,i,cols/2],fiterr0,fiterr,iters                   
                
            rbottomclr_pic[p,i,usesig_c] = (rprof_pic[p,i] - np.argmax(isslit_pirc[p,i],axis=0))[usesig_c]
            rtopclr_pic[p,i,usesig_c] = ((prows - np.argmax(isslit_pirc[p,i,::-1],axis=0)) - rprof_pic[p,i])[usesig_c]
            rbottomslit_pic[p,i,usesig_c] = np.argmax(isslit_pirc[p,i],axis=0)[usesig_c]
            rtopslit_pic[p,i,usesig_c] = prows - (np.argmax(isslit_pirc[p,i,::-1],axis=0))[usesig_c]            
            meanwidth_pi[p,i] = fitprof_cR[usesig_Rc.T].sum()/usesig_c.sum()        
            fiterr_dpi[:,p,i] = (fiterr0,fiterr,iters)
                 
        rsslog.message(((" %2i "+oktargs*"%4.2f ") % ((p,)+tuple(meanwidth_pi[p,oktgt_i]))),logfile) 
        rsslog.message((("    "+oktargs*"%4i ") % (tuple(profsamples_pi[p,oktgt_i]))),logfile)         
    sigprof_pirc[sigprof_pirc < 0.] = 0.

    exit()
    
    if debug:
        np.savetxt("proffitstats.txt",np.vstack((np.indices((2,entries)).reshape((2,-1)),   \
            fiterr_dpi.reshape((3,-1)))).T,fmt="%2i %3i %8.3f %8.3f %3i")
        np.savetxt("rprof_pic.txt",np.vstack((range(cols),    \
            rprof_pic.reshape((-1,cols)))).T,fmt="%4i "+2*entries*"%8.2f ")            
        np.savetxt("drcenter_pic.txt",np.vstack((range(cols), drcenter_pic.reshape((-1,cols)),   \
            rbottomclr_pic.reshape((-1,cols)),rtopclr_pic.reshape((-1,cols)))).T,    \
            fmt="%4i "+6*entries*"%8.2f ")
        np.savetxt("rslitedge_pic.txt",np.vstack((range(cols),   \
            rbottomslit_pic.reshape((-1,cols)),rtopslit_pic.reshape((-1,cols)))).T,    \
            fmt="%4i "+4*entries*"%4i ")            
        np.savetxt("snstats_dpi.txt",np.vstack((np.indices((2,entries)).reshape((2,-1)),    \
            snmedian_pi.flatten(),snwavlim_dpi.reshape((2,-1)))).T,fmt="%2i %3i "+3*"%8.2f ")
        hdubkg = pyfits.PrimaryHDU(bkg_fprc.astype('float32'))
        hdulbkg = pyfits.HDUList([hdubkg])
        hdulbkg.writeto("bkg_fprc.fits",overwrite=True)            
        hduprof = pyfits.PrimaryHDU(sigprof_pirc.sum(axis=1).astype('float32'))
        hdulprof = pyfits.HDUList([hduprof])
        hdulprof.writeto("sigprof_prc.fits",overwrite=True)       
            
  # find cr's in each image using fit profiles
    rsslog.message("\nFlag cosmic rays, do extraction", logfile)
    compsigma = 12.
    compslope = 5.
    photerr_prc = np.zeros((2,prows,cols))  
    crs_f = np.zeros(files)
    oksig_fprc = np.zeros((files,2,prows,cols),dtype=bool)
    end_c = np.zeros(cols)
    signal_fprc = okbin_fprc*(image_fprc - bkg_fprc)
    sigerr_fprc = np.zeros_like(signal_fprc)
    iscr_fprc = np.zeros((files,2,prows,cols),dtype=bool) 
    crphotsigma_fprc = np.zeros((files,2,prows,cols)) 
    crsloperat_fprc = np.zeros((files,2,prows,cols))
                    
    for f,file in enumerate(fileList):
        photerr_prc[okbin_fprc[f]] = np.sqrt(var_fprc[f,okbin_fprc[f]])    
        for p,i in np.ndindex(2,entries):
            if (~oktgt_i[i]): continue                
            Rows = hasslit_pir[p,i].sum()
            fnorm_c = np.zeros(cols)
            photsigma_Rc = np.zeros((Rows,cols))
            issig_rc = (isslit_pirc[p,i] & okbin_fprc[f,p] & okprof_pirc[p,i])                                         
            issig_Rc = issig_rc[hasslit_pir[p,i],:]
            sigprof_Rc = sigprof_pirc[p,i,hasslit_pir[p,i],:]            
            usenorm_Rc = issig_Rc & (sigprof_Rc > 0.2)  # only use profile peak to normalize it             
            issig_c = usenorm_Rc.any(axis=0)            # if none, not a usable signal                              
            signal_Rc = (issig_rc*signal_fprc[f,p])[hasslit_pir[p,i],:]

            for c in np.where(issig_c)[0]:
                fnorm_c[c] = np.median(signal_Rc[usenorm_Rc[:,c],c]/sigprof_Rc[usenorm_Rc[:,c],c])
            sigprof_Rc = fnorm_c[None,:]*sigprof_Rc     # normalize profile intensity
            
            rslope_Rc = np.diff(sigprof_Rc,axis=0)                              
            rslope_Rc = np.abs((rslope_Rc[:-1] + rslope_Rc[1:])/2.)
            rslope_Rc = np.vstack((rslope_Rc[0],rslope_Rc,rslope_Rc[-1]))
            rslope_Rc = np.clip(rslope_Rc,0.01*fnorm_c[None,:],fnorm_c[None,:])
            
            sigerr_Rc = signal_Rc - sigprof_Rc
            photsigma_Rc[issig_Rc] = sigerr_Rc[issig_Rc] /    \
                photerr_prc[p,hasslit_pir[p,i]][issig_Rc]
                                  
            iscr_Rc = (issig_Rc &     \
                (photsigma_Rc > compsigma) & (sigerr_Rc > compslope*rslope_Rc))
            sigerr_fprc[f,p,issig_rc] = (signal_Rc - sigprof_Rc)[issig_Rc]                
            iscr_fprc[f,p,issig_rc] = iscr_Rc[issig_Rc]           
            
            if (iscr_Rc.sum() > 0):
                iscr_rc = (issig_rc & iscr_fprc[f,p])
                crphotsigma_fprc[f,p,iscr_rc] = photsigma_Rc[iscr_Rc] 
                crsloperat_fprc[f,p,iscr_rc] = (sigerr_Rc[iscr_Rc] / rslope_Rc[iscr_Rc])                         
                
        crs_f[f] = iscr_fprc[f].sum()
        oksig_fprc[f] = (okbin_fprc[f] & ~iscr_fprc[f])

    if debug:
        hduerr = pyfits.PrimaryHDU(sigerr_fprc.astype('float32'))
        hdulerr = pyfits.HDUList([hduerr])
        hdulerr.writeto("sigerr_fprc.fits",overwrite=True) 
        hducr = pyfits.PrimaryHDU(iscr_fprc.astype('uint8'))
        hdulcr = pyfits.HDUList([hducr])
        hdulcr.writeto("iscr_fprc.fits",overwrite=True)
        hdulcrsig = pyfits.PrimaryHDU(crphotsigma_fprc.astype('float32'))
        hdulcrsig = pyfits.HDUList([hdulcrsig])
        hdulcrsig.writeto("crphotsigma_fprc.fits",overwrite=True)                  
        hduslrat = pyfits.PrimaryHDU(crsloperat_fprc.astype('float32'))
        hdulslrat = pyfits.HDUList([hduslrat])
        hdulslrat.writeto("crsloperat_fprc.fits",overwrite=True)          
        np.savetxt("crstats.txt",np.vstack((np.where(iscr_fprc),crphotsigma_fprc[iscr_fprc],    \
            crsloperat_fprc[iscr_fprc])).T,fmt=4*"%4i "+2*"%10.3f ")                   

  # do extraction in row direction using fit profiles as weights
  
    fopt_fpic = np.zeros((files,2,entries,cols))
    vopt_fpic = np.zeros_like(fopt_fpic)
    fstd_fpic = np.zeros_like(fopt_fpic)
    vstd_fpic = np.zeros_like(fopt_fpic)
    ok_fpic = np.zeros((files,2,entries,cols),dtype=bool)     

    for f in range(files):
        for p,i in np.ndindex(2,entries):
            if (~oktgt_i[i]): continue            
            Rows = hasslit_pir[p,i].sum()
            issig_rc = (isslit_pirc[p,i] & oksig_fprc[f,p] & okprof_pirc[p,i])
            issig_Rc = issig_rc[hasslit_pir[p,i],:]
            issig_c = issig_Rc.any(axis=0)                
            signal_Rc = issig_Rc*signal_fprc[f,p,hasslit_pir[p,i]]
            var_Rc = issig_Rc*var_fprc[f,p,hasslit_pir[p,i]]
            bkg_Rc = issig_Rc*bkg_fprc[f,p,hasslit_pir[p,i]]         
            xtrnorm_c = sigprof_pirc[p,i,hasslit_pir[p,i]].sum(axis=0)
            xtrwt_Rc = np.zeros((Rows,cols))
            xtrwt_Rc[:,issig_c] = (issig_Rc*sigprof_pirc[p,i,hasslit_pir[p,i]])[:,issig_c]/   \
                np.tile(xtrnorm_c[None,issig_c],(Rows,1))
                                  
            fstd_fpic[f,p,i,issig_c] = (issig_Rc*signal_Rc).sum(axis=0)[issig_c]/    \
                (issig_Rc*xtrwt_Rc).sum(axis=0)[issig_c]
            vstd_fpic[f,p,i,issig_c] = (issig_Rc*var_Rc).sum(axis=0)[issig_c]/       \
                ((issig_Rc*xtrwt_Rc).sum(axis=0)[issig_c])**2
                                      
            vopt_Rc = fstd_fpic[f,p,i,None,:]*xtrwt_Rc + bkg_Rc     # first, vopt           
            issig_Rc = (issig_Rc & (vopt_Rc > 0.))                  # for opt, get rid of bad bins 
            
            lostprof_c = 1. - (issig_Rc*xtrwt_Rc).sum(axis=0)       # first, flag profs with >25% lost
            issig_c = (issig_Rc.any(axis=0) & (lostprof_c < 0.25))      
              
            norm_Rc = np.zeros((Rows,cols))                         # finally, fopt
            fopt_Rc = np.zeros((Rows,cols))                         
            norm_Rc[issig_Rc] = xtrwt_Rc[issig_Rc]**2/vopt_Rc[issig_Rc]
            norm_c = norm_Rc.sum(axis=0)
            fopt_Rc[issig_Rc] = (xtrwt_Rc*signal_Rc)[issig_Rc]/vopt_Rc[issig_Rc]            
            fopt_fpic[f,p,i,issig_c] = fopt_Rc.sum(axis=0)[issig_c]/norm_c[issig_c]
            vopt_fpic[f,p,i,issig_c] = xtrwt_Rc.sum(axis=0)[issig_c]/norm_c[issig_c]        
            ok_fpic[f,p,i] = issig_c

        if debug:
            np.savetxt("fstd_pic_"+str(f)+".txt",np.vstack(( \
                fstd_fpic[f].reshape((-1,cols)))).T,fmt="%8.0f ")
            np.savetxt("vstd_pic_"+str(f)+".txt",np.vstack(( \
                vstd_fpic[f].reshape((-1,cols)))).T,fmt="%8.0f ")
            np.savetxt("fopt_pic_"+str(f)+".txt",np.vstack(( \
                fopt_fpic[f].reshape((-1,cols)))).T,fmt="%8.0f ")
            np.savetxt("vopt_pic_"+str(f)+".txt",np.vstack(( \
                vopt_fpic[f].reshape((-1,cols)))).T,fmt="%8.0f ")                                                

  # resample onto wavelength grid
  # Note: for now, wavelength of raw extracted bin is center of profile, ignoring change across profile
  #  eventually, will want 2d scrunch to improve spectral resolution off axis at high resolution

  # compute raw bin wavelength array of extraction, using profile center
    wav_pic = np.zeros((2,entries,cols))
    escapedprofList = []     
    for p,i in np.ndindex(2,entries):    
        if (~oktgt_i[i]): continue    
        for c in np.where((rprof_pic[p,i]>0) & isslit_pirc[p,i].any(axis=0))[0]:
            rowList = list(np.where(isslit_pirc[p,i,:,c])[0])            
            if ((rprof_pic[p,i,c] < rowList[0]) | (rprof_pic[p,i,c] > rowList[-1])):
                escapedprofList.append(np.array([p,i,c]))
                
                print p,i,c,rprof_pic[p,i,c],rowList
                exit()
                continue   
            wav_pic[p,i,c] = interp1d(rowList,wav_prc[p,rowList,c])(rprof_pic[p,i,c])

    if len(escapedprofList):
        img0 = hdul0.filename().split('.')[0][-4:]   
        np.savetxt("escapedprof_"+img0+".txt",np.array(escapedprofList),fmt=3*"%4i ")
        rsslog.message("\n Warning: %3i escaped profiles" % len(escapedprofList), logfile)
    
  # establish the wavelength range to be used
    wbin = np.diff(wav_pic,axis=2)[:,:,cols/2].mean()   # mean bin size at center          
    wbin = 2.**(np.rint(np.log2(wbin)))                 # bin to nearest power of 2 angstroms                                   
    wmin = wav_prc[wav_prc > 0.].min()
    wmax = wav_prc.max()
    wmin = (np.ceil(wmin/wbin)+1)*wbin
    wmax = (np.floor(wmax/wbin)-1)*wbin
    wavs = int((wmax - wmin)/wbin + 1)
    wav_w = np.arange(wmin,wmax+wbin,wbin)
    wedge_w = np.arange(wmin,wmax+wbin,wbin) - wbin/2.
    
    fopt_fpiw = np.zeros((files,2,entries,wavs))
    vopt_fpiw = np.zeros_like(fopt_fpiw) 
    covar_fpiw = np.zeros_like(fopt_fpiw) 
    ok_fpiw = np.zeros_like(fopt_fpiw,dtype=bool)
    werr_piw = np.zeros((2,entries,wavs))

    for f in range(files):    
        for p,i in np.ndindex(2,entries):
            if (~oktgt_i[i]): continue    
            colArray = np.where(wav_pic[p,i] > 0)[0]
            wedgeArray = np.where((wedge_w >= wav_pic[p,i,colArray].min()) &    \
                                (wedge_w < wav_pic[p,i,colArray].max()))[0]
            wavout = wedgeArray[:-1]          # scrunch has input bins starting at bin, not bin-1/2:         
            cbinedge_w = interp1d(wav_pic[p,i,colArray],(colArray+0.5))(wedge_w[wedgeArray])        
            okpi_W = (scrunch1d(ok_fpic[f,p,i].astype(int),cbinedge_w)   \
                == (cbinedge_w[1:] - cbinedge_w[:-1]))
            ok_fpiw[f,p,i,wavout] = okpi_W                     
            fopt_fpiw[f,p,i,wavout] = scrunch1d(fopt_fpic[f,p,i],cbinedge_w)*okpi_W
            vopt_W,covar_W = scrunchvar1d(vopt_fpic[f,p,i],cbinedge_w)
            vopt_fpiw[f,p,i,wavout] = vopt_W*okpi_W
            covar_fpiw[f,p,i,wavout] = covar_W*okpi_W
            if (f==0): 
                werr_piw[p,i,wavout] =  \
                    scrunch1d(werr_pic[p,i],cbinedge_w)/(cbinedge_w[1:] - cbinedge_w[:-1])
            if (debug & (f==ff) & (p==pp) & (i==ii)): 
                np.savetxt("cbinedge_"+str(f)+"_"+str(p)+"_"+str(i)+".txt", \
                    np.vstack((wav_w[wedgeArray], cbinedge_w)).T,fmt="%8.1f %9.4f ")
                np.savetxt("collArray_"+str(f)+"_"+str(p)+"_"+str(i)+".txt", \
                    np.vstack((colArray, wav_pic[p,i,colArray])).T,fmt="%8.1f %9.2f ")                    

        if debug:
            np.savetxt("fopt_piw_"+str(f)+".txt",np.vstack((wav_w,  \
                fopt_fpiw[f].reshape((-1,wavs)))).T,fmt="%6.0f "+2*entries*"%8.0f ")
            np.savetxt("vopt_piw_"+str(f)+".txt",np.vstack((wav_w,  \
                vopt_fpiw[f].reshape((-1,wavs)))).T,fmt="%6.0f "+2*entries*"%8.0f ")

    rsslog.message("\n      Output file                crs", logfile)
            
  # save the result

    for f,file in enumerate(fileList):
        hdul = pyfits.open(file)
        outfile = 'e'+file
        del(hdul['TMAP'])
        del(hdul['BMAP'])          
        del(hdul['WMAP'])
        hdul['SCI'].data = fopt_fpiw[f][:,oktgt_i].astype('float32')                
        hdr1 = hdul['SCI'].header
        hdul['VAR'].data = vopt_fpiw[f][:,oktgt_i].astype('float32')
        hdul.append(pyfits.ImageHDU(data=covar_fpiw[f][:,oktgt_i].astype('float32'),header=hdr1, name='COV'))
        hdul['BPM'].data = (~ok_fpiw[f][:,oktgt_i]).astype('uint8')
        hdul['WERR'].data = werr_piw[:,oktgt_i].astype('float32')
        for ext in [0,'SCI','VAR','COV','BPM','WERR']:     # specpolrawstokes wants them in both ext's
            hdul[ext].header['CRVAL1'] = wmin
            hdul[ext].header['CDELT1'] = wbin 
            hdul[ext].header['CTYPE1'] = 'Angstroms'
            hdul[ext].header['CRVAL2'] = 0
            hdul[ext].header['CDELT2'] = 1 
            hdul[ext].header['CTYPE2'] = 'Target'           
        hdul['TGT'] = pyfits.table_to_hdu(tgtTab)     
        hdul.writeto(outfile,overwrite=True)

        outfile = 'e'+file
        rsslog.message ("%30s  %4i" % (outfile,crs_f[f]), logfile)
                
    return

# ----------------------------------------------------------
def splinemapextract(fileList,name,logfile='salt.log',debug=False):
    return

# ----------------------------------------------------------
def profile2dfit(prof_ab,var_ab,ok_ab,label,stackbin=8,snlim=8.,debugname=None):
    """fit supersampled spline profile to stack of 1D profiles
    Parameters 
    ----------
    prof_ab: 1d float profile in b, stacked in a 
    ok_ab: boolean array mask

    Returns: fitprof_ab, ok_ab, rfit_d, rrfit_d, profsamples,fiterr0,fiterr,iters    
    ----------
    fitprof_ab = 1dSpline((b - b_a)*bb_a - b_a)
    _i: stackbinned _a    

    """
    profs,bins = prof_ab.shape
            
  # bin in the a (stacking) direction, initial cull based on sn of max deviation from linear fit
    stacks = int(np.ceil(profs/float(stackbin)))              # last bin may be partial
    i_a = np.digitize(range(profs),np.arange(stacks)*stackbin) - 1
    ok_iab = ((np.arange(stacks)[:,None,None]==i_a[None,:,None]) & ok_ab[None,:,:])    
    count_ib = ok_iab.sum(axis=1)
    ok_ib = (count_ib>0)
    prof_ib = np.zeros((stacks,bins)) 
    var_ib = np.zeros_like(prof_ib)        
    prof_ib[ok_ib] = (prof_ab[None,:]*ok_iab).sum(axis=1)[ok_ib]/count_ib[ok_ib]
    var_ib[ok_ib] = (var_ab[None,:]*ok_iab).sum(axis=1)[ok_ib]/count_ib[ok_ib]**2
    snmax_i = np.zeros(stacks)
    okprof_i = np.zeros(stacks,dtype=bool)    
    for i in np.where(ok_ib.sum(axis=1) > 2)[0]:
        linfit_d = np.polyfit(np.where(ok_ib[i])[0],prof_ib[i,ok_ib[i]],1) 
        dev_b = ok_ib[i]*(prof_ib[i] - np.polyval(linfit_d,np.arange(bins)))
        bsnmax = np.argmax(dev_b)  
        snmax_i[i] = dev_b[bsnmax]/np.sqrt(var_ib[i,bsnmax])
        okprof_i[i] = ((snmax_i[i] > snlim) &     \
            (prof_ib[i,bsnmax] >= np.clip(prof_ib[i,ok_ib[i]].max(),0,None))) 
    samples_i = (okprof_i[:,None] & ok_iab.any(axis=2)).sum(axis=1) 
    ok_ib = (ok_ib & okprof_i[:,None])
    a_a = np.arange(profs)
    a_i = np.zeros(stacks)
    ok_ia = ok_iab.any(axis=2)                
    a_i[okprof_i] = (a_a[None,:]*ok_ia).sum(axis=1)[okprof_i]/ok_ia.sum(axis=1)[okprof_i]

  # for fnorm, b0 first guess, use 3 highest ok neighbor bins, fit quadratic                  
    fnorm_i = np.ma.max(np.ma.masked_array(prof_ib,mask=~ok_ib),axis=1).data       
    idx0_i = np.ma.argmax(np.ma.masked_array(prof_ib,mask=~ok_ib),axis=1).clip(1,bins-2)
    i_i = range(stacks)           
    okmax_i = (okprof_i & (ok_ib[i_i,idx0_i-1] & ok_ib[i_i,idx0_i+1]))
    fnorm_i *= okmax_i.astype(float)
    fiterr_ib = np.zeros((stacks,bins))         
    pcof0_i = (prof_ib[i_i,idx0_i-1] -2.*prof_ib[i_i,idx0_i] + prof_ib[i_i,idx0_i+1])/2.
    pcof1_i = (prof_ib[i_i,idx0_i] - prof_ib[i_i,idx0_i-1]) - pcof0_i*(2.*idx0_i - 1.)
    pcof2_i = prof_ib[i_i,idx0_i] - pcof0_i*idx0_i**2 - pcof1_i*idx0_i      
    b0_i = idx0_i.astype(float)                     # default
    b0_i[okprof_i] = -pcof1_i[okprof_i]/(2.*pcof0_i[okprof_i])
    okprof_i &= ((b0_i > 1.) & (b0_i < bins-1))

  # for width first guess use prof summed flux / fnorm
    width_i = np.zeros(stacks)
    width_i[okmax_i] = (ok_ib*prof_ib)[okmax_i].sum(axis=1)/fnorm_i[okmax_i]
    
  # cull out width outliers
    dum, widthlo, widthhi, dum = fence(width_i[okmax_i])
    okprof_i &= ((width_i > widthlo) & (width_i < widthhi))
    okmax_i = (okmax_i & okprof_i)
    fnorm_i[okmax_i] = pcof2_i[okmax_i] - pcof1_i[okmax_i]**2/(4.*pcof0_i[okmax_i])        
    ok_ib = (ok_ib & okprof_i[:,None])

  # fit to Legendre polynomials.  Width is relative to center
    prof_ib[okprof_i] /= fnorm_i[okprof_i][:,None]
    legx_i = 2*a_i/profs -1.
    blegcofs = 7
    bfit_d = np.polynomial.legendre.legfit(legx_i[okprof_i],b0_i[okprof_i],blegcofs-1)

    bbfit_d = np.polynomial.legendre.legfit(legx_i[okprof_i],width_i[okprof_i],2)
    width0 = np.polynomial.legendre.legval(0.,bbfit_d)
    bbfit_d /= width0        

    if debugname:
        b0 = np.polynomial.legendre.legval(0.,bfit_d)    
        b0err_i = b0_i - np.polynomial.legendre.legval(legx_i,bfit_d)
        widtherr_i = width_i - width0*np.polynomial.legendre.legval(legx_i,bbfit_d)
        np.savetxt("b0width_i_"+debugname+".txt",np.vstack((ok_ib.sum(axis=1),okmax_i.astype(int),    \
            okprof_i.astype(int),fnorm_i,snmax_i,legx_i,b0_i,width_i,b0err_i,widtherr_i)).T,    \
            fmt=3*"%2i "+"%9.1f "+6*"%8.3f ")

    x0 = np.hstack((bfit_d,bbfit_d))
    fiterr0 = profilefiterr(x0,legx_i,prof_ib,ok_ib,label)
        
    xopt_d,fiterr,iters,funcalls,warnflag = fmin(profilefiterr,x0,xtol=1.,ftol=0.01,maxiter=100,  \
        args=(legx_i,prof_ib,ok_ib,label),full_output=True,disp=False)
    bfit_d = xopt_d[:blegcofs]
    bbfit_d = xopt_d[blegcofs:]
    profileSpline, B_ib, Bmax, Bmin = profilefit(xopt_d,legx_i,prof_ib,ok_ib,label)[1:5]
        
  # evaluate spline at unbinned stack
    legx_a = 2*np.arange(profs,dtype=float)/profs -1.                    
    b_a = np.polynomial.legendre.legval(legx_a,bfit_d)
    bb_a = np.polynomial.legendre.legval(legx_a,bbfit_d)
    B_ab = (np.arange(bins)[None,:] - b_a[:,None])/bb_a[:,None]
  # output profile is interpolated across input fit problems      
    okfitprof_a = (ok_ia & okprof_i[:,None]).any(axis=0)    
    okfitprof_ab = (ok_ab & ((B_ab >= Bmin) & (B_ab <= Bmax)) & okfitprof_a[:,None])
    amin,amax = np.where(okfitprof_a)[0][[0,-1]] 
    okoutprof_ab = ((B_ab >= Bmin) & (B_ab <= Bmax) &   \
                    ((np.arange(profs) >= amin) & (np.arange(profs) <= amax))[:,None])           
    outprof_ab = okoutprof_ab * profileSpline(B_ab)         

    if debugname:
        debugfile = open("debug"+debugname+".txt",'a')
        print >> debugfile, ("Bmin, Bmax: %8.3f %8.3f" % (Bmin,Bmax))
        print >> debugfile, ("Spline(Bmin), (Bmax): %8.5f %8.5f" % (profileSpline(Bmin), profileSpline(Bmax)))
        b_i = np.polynomial.legendre.legval(legx_i,bfit_d)
        width_i = width0*np.polynomial.legendre.legval(legx_i,bbfit_d)
        np.savetxt("b1width_i_"+debugname+".txt",np.vstack((ok_ib.sum(axis=1),okmax_i.astype(int),    \
            okprof_i.astype(int),a_i,fnorm_i,legx_i,b_i,width_i)).T,    \
            fmt=3*"%2i "+"%4i "+"%9.1f "+3*"%8.3f ")
        np.savetxt("2dproffit_i_"+debugname+".txt",np.vstack((np.where(ok_ib),B_ib[ok_ib],  \
            profileSpline(B_ib[ok_ib]),prof_ib[ok_ib])).T,fmt="%4i %2i "+3*"%8.3f ")
        width_a = width0*bb_a
        np.savetxt("b1width_a_"+debugname+".txt",np.vstack((okfitprof_ab.sum(axis=1),legx_a,b_a,width_a)).T,    \
            fmt="%2i "+3*"%8.3f ")
        np.savetxt("2dproffit_a_"+debugname+".txt",np.vstack((np.where(okfitprof_ab),B_ab[okfitprof_ab],  \
            profileSpline(B_ab[okfitprof_ab]),prof_ab[okfitprof_ab])).T,fmt="%4i %2i "+3*"%8.3f ")            
        
    return outprof_ab, okoutprof_ab, bfit_d, bbfit_d, samples_i.sum(), fiterr0, fiterr, iters

# ----------------------------------------------------------
def profilefiterr(xfit_d,legx_i,prof_ib,ok_ib,label):
    return profilefit(xfit_d,legx_i,prof_ib,ok_ib,label)[0]
    
# ----------------------------------------------------------
def profilefit(xfit_d,legx_i,prof_ib,ok_ib,label):
    """compute rms error for current profile fit for fmin
    Parameters 
    ----------
    xfit_d: parameters being varied

    Returns: rms of fit, profileSpline, B_ib 
    """

    cofs = xfit_d.shape[0]    
    bins = ok_ib.shape[1]  
    bfit_d = xfit_d[:(cofs-3)]
    bbfit_d = xfit_d[(cofs-3):]  
    b_i = np.polynomial.legendre.legval(legx_i,bfit_d)
    bb_i = np.polynomial.legendre.legval(legx_i,bbfit_d)
    B_ib = (np.arange(bins)[None,:] - b_i[:,None])/bb_i[:,None]      

    idxsort = np.argsort(B_ib[ok_ib])
    Bsorted = B_ib[ok_ib][idxsort]
    profsorted = prof_ib[ok_ib][idxsort]
    while ((Bsorted[2:4].mean()- Bsorted[0]) > 0.2):     # avoid outlier causing very wide outer knots
        Bsorted = np.delete(Bsorted,0)
        profsorted = np.delete(profsorted,0)
    while ((Bsorted[-1] - Bsorted[-4:-2].mean()) > 0.2):        
        Bsorted = np.delete(Bsorted,-1)
        profsorted = np.delete(profsorted,-1)
                                  
    Bmin,Bmax = Bsorted[[0,-1]]

    dbmax = ok_ib.sum(axis=1).max()                      # avoid crazy b_i guesses     
    if ((Bmin < -2*dbmax) | (Bmin > 0.) | (Bmax < 0.) | (Bmax > 2*dbmax)): 
        fiterr = 1.e9
        profileSpline = 0
        return fiterr, profileSpline, B_ib, Bmax, Bmin       
        
    Bmin2,Bmax2 = Bsorted[2:4].mean(), Bsorted[-4:-2].mean()    # avoid data on interior knot    
    Bknot_k = np.linspace(Bmin2,Bmax2,num=int(8*(Bmax2-Bmin2)),dtype=float)
    Bcount_k = np.histogram(Bsorted,Bknot_k)[0]        
    SWcount_k = np.convolve(Bcount_k,np.ones(3),mode='same')
    okknot_k = np.ones(len(Bknot_k),dtype=bool)       
    okknot_k[2:-3] = (SWcount_k[2:-2] > 1)                      # mask out knots to obey SW
    badknots = okknot_k.shape[0] - okknot_k.sum()
#    if (badknots):
#        badknot_k = np.logical_not(okknot_k) 
#        print np.where(badknot_k)[0], Bknot_k[badknot_k]
        
    try:       
        profileSpline = LSQUnivariateSpline(Bsorted,   \
            profsorted, Bknot_k[okknot_k], bbox = [Bmin, Bmax], k=3)
    except ValueError:
        print "\nfmin Schoenberg-Whitney exception, label= : ",label
        print "Bmin, Bmax, Bmin2, Bmax2: ", Bmin, Bmax, Bmin2,Bmax2
        np.savetxt("dbbb_i.txt",np.vstack((legx_i,b_i,bb_i)).T,fmt="%8.3f ")
        np.savetxt("B_ib_sorted.txt",Bsorted.T,fmt="%8.3f ")        
        Bcount_k = np.append(Bcount_k,0.)
        SWcount_k = np.append(SWcount_k,0.)        
        np.savetxt("Bknot_k.txt",np.vstack((okknot_k,Bcount_k,SWcount_k,Bknot_k)).T,   \
            fmt=3*" %3i"+" %8.3f")
        Bcount_K = np.histogram(B_ib[ok_ib],Bknot_k[okknot_k])[0]    
        SWcount_K = np.convolve(Bcount_K,np.ones(3),mode='same')           
        Bcount_K = np.append(Bcount_K,0.)
        SWcount_K = np.append(SWcount_K,0.)   
        np.savetxt("Bknot_K.txt",np.vstack((Bcount_K,SWcount_K,Bknot_k[okknot_k])).T, \
            fmt=2*" %3i"+" %8.3f")
        fiterr = 1.e9
        profileSpline = 0
    else:        
        fiterr = np.std(profileSpline(B_ib[ok_ib]) - prof_ib[ok_ib])

#    print 9*"%8.4f " % ((fiterr,)+tuple(xfit_V))

    return fiterr, profileSpline, B_ib, Bmax, Bmin

# ------------------------------------

if __name__=='__main__':
    infileList=[x for x in sys.argv[1:] if x.count('.fits')]
    kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.fits')==0)   
    sppolextract(infileList,**kwargs)

