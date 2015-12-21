
"""
specpollampextract

Extract spectropolarimetric lamp spectrum data.  

"""

import os, sys, glob, shutil, inspect

import numpy as np
import pyfits
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

from scrunch1d import scrunch1d
from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging

# np.seterr(invalid='raise')
np.set_printoptions(threshold=np.nan)
debug = True

# ---------------------------------------------------------------------------------
def specpollampextract(infilelist, logfile='salt.log'):

    obsdate=os.path.basename(infilelist[0])[8:16]

    with logging(logfile, debug) as log:
        log.message('Extraction of Lamp Images' , with_header=False)
        obsdict=obslog(infilelist)

        hdu0 =  pyfits.open(infilelist[0])       
        rows,cols = hdu0['SCI'].data.shape[1:3]
        cbin,rbin = np.array(obsdict["CCDSUM"][0].split(" ")).astype(int)
        slitid = obsdict["MASKID"][0]
        lampid = obsdict["LAMPID"][0].strip().upper()
        lam_c = hdu0['WAV'].data[0,rows/2]
        files = len(infilelist)
        outfilelist = infilelist

# sum spectra to find target
        count = 0
        for i in range(files):
            badbin_orc = pyfits.open(outfilelist[i])['BPM'].data.astype(bool)
            if count == 0: 
                count_orc = (~badbin_orc).astype(int)
                image_orc = pyfits.open(outfilelist[i])['SCI'].data*count_orc
                var_orc = pyfits.open(outfilelist[i])['VAR'].data
            else:
                count_orc += (~badbin_orc).astype(int)
                image_orc += pyfits.open(outfilelist[i])['SCI'].data*(~badbin_orc)
                var_orc += pyfits.open(outfilelist[i])['VAR'].data
            count += 1
        if count ==0:
            print 'No valid images'
            exit()

        image_orc[count_orc>0] /= count_orc[count_orc>0]
        badbin_orc = (count_orc==0) | (image_orc==0)
        okbinpol_orc = (count_orc == count) & (image_orc != 0)    # conservative bpm for pol extraction
        var_orc[count_orc>0] /= count_orc[count_orc>0]**2
        wav_orc = pyfits.open(outfilelist[0])['WAV'].data
#        pyfits.PrimaryHDU(image_orc.astype('float32')).writeto('lampsum_orc.fits',clobber=True)            

        lam_m = np.loadtxt(datadir+"wollaston.txt",dtype=float,usecols=(0,))
        rpix_om = np.loadtxt(datadir+"wollaston.txt",dtype=float,unpack=True,usecols=(1,2))

    # trace spectrum, compute spatial profile 
        profile_orc = np.zeros_like(image_orc)
        drow_oc = np.zeros((2,cols))
        expectrow_oc = np.zeros((2,cols),dtype='float32')
        maxrow_oc = np.zeros((2,cols),dtype=int)
        maxval_oc = np.zeros((2,cols),dtype='float32')
        cross_orC = np.zeros((2,rows,2))
        col_cr,row_cr = np.indices(image_orc[0].T.shape)
    # sample cross-dispersion at center and on right (_C) to get offset and tilt
        Collist = [cols/2,0.8*cols]
        for C in (0,1): cross_orC[:,:,C] = np.sum(image_orc[:,:,Collist[C]-cols/16:Collist[C]+cols/16],axis=2)

        drow_oC = np.zeros((2,2))
        trow_o = np.zeros((2),dtype='int')
        okprof_oc = np.zeros((2,cols),dtype='bool')       
        okprof_orc = np.zeros((2,rows,cols),dtype='bool')
        norm_orc = np.zeros((2,rows,cols))
        sig_c = np.zeros((cols))
        sigmin = 20.; drowmax = 8.

        # find spectrum offset and tilt roughly from max of two cross-dispersion samples
        for o in (0,1):
            expectrow_oc[o] = (1-o)*rows + interp1d(lam_m,rpix_om[o],kind='cubic')(lam_c)/rbin
            for C in (0,1):    
                crossmaxval = np.max(cross_orC[o,   \
                            expectrow_oc[o,Collist[C]]-100/rbin:expectrow_oc[o,Collist[C]]+100/rbin,C])
                drow_oC[o,C] = np.where(cross_orC[o,:,C]==crossmaxval)[0][0] - expectrow_oc[o,Collist[C]]
        drow_o = drow_oC[:,0]
        rowtilt = (drow_oC[:,1]-drow_oC[:,0]).mean()/(Collist[1]-Collist[0])
        expectrow_oc += drow_o[:,None] + rowtilt*np.arange(-cols/2,cols/2)

        # get trace by finding max in narrow curved aperture and smoothing it
        for o in (0,1):
            row_c = expectrow_oc[o].astype(int)
            aperture_cr = ((row_cr-row_c[:,None])>=-20/rbin) & ((row_cr-row_c[:,None])<=20/rbin)            
            maxrow_oc[o] = np.argmax(image_orc[o].T[aperture_cr].reshape((cols,-1)),axis=1) + row_c - 20/rbin
            maxval_oc[o] = image_orc[o,maxrow_oc[o]].diagonal()
            trow_o[o] = maxrow_oc[o,cols/2]

        # mark as bad where signal drops too low or position is off
            median_c = np.median(image_orc[o].T[aperture_cr].reshape((cols,-1)),axis=1)
            var_c = np.mean(var_orc[o].T[aperture_cr].reshape((cols,-1)),axis=1)
            sig_c[var_c>0] = (maxval_oc[o] - median_c)[var_c>0]/np.sqrt(var_c[var_c>0])
            drow1_c = maxrow_oc[o] -expectrow_oc[o]
            okprof_oc[o] = (sig_c > sigmin) & (abs(drow1_c - np.median(drow1_c)) < drowmax)

        # divide out spectrum (allowing for spectral curvature and tilt) to make spatial profile
            drow2_c = np.polyval(np.polyfit(np.where(okprof_oc[o])[0],drow1_c[okprof_oc[o]],3),(range(cols)))
            okprof_orc[o] = (np.abs(drow2_c - drow1_c) < 3) & okprof_oc[o][None,:]
            drow_oc[o] = -(expectrow_oc[o] - expectrow_oc[o,cols/2] + drow2_c -drow2_c[cols/2])
            for r in range(rows):
                norm_orc[o,r] = interp1d(wav_orc[o,trow_o[o],okprof_oc[o]],maxval_oc[o,okprof_oc[o]], \
                    bounds_error = False, fill_value=0.)(wav_orc[o,r])

        log.message('Image tilt: %8.1f arcmin' % (60.*np.degrees(rowtilt*rbin/cbin)), with_header=False)        
        log.message('Target offset:     O    %4i     E    %4i' % tuple(drow_o), with_header=False)
        log.message('Target center row: O    %4i     E    %4i' % tuple(trow_o), with_header=False)

        okprof_orc &= (norm_orc != 0.)
        profile_orc[okprof_orc] = image_orc[okprof_orc]/norm_orc[okprof_orc]
        var_orc[okprof_orc] = var_orc[okprof_orc]/norm_orc[okprof_orc]**2
#        pyfits.PrimaryHDU(norm_rc.astype('float32')).writeto('norm_rc.fits',clobber=True)     
#        pyfits.PrimaryHDU(okprof_oc.astype('uint8')).writeto('okprof_oc.fits',clobber=True) 
        okprof_c = okprof_oc.all(axis=0)

    # Sample the normalized row profile at 5 places (_C)
        Cols = 5
        dcols = 64/cbin
        Collist = [np.argmax(okprof_c)+dcols, 0, cols/2, 0, cols-np.argmax(okprof_c[::-1])-dcols]
        for C in (1,3): Collist[C] = 0.5*(Collist[C-1] + Collist[C+1])
        Collist = map(int,Collist)
        profile_Cor = np.zeros((Cols,2,rows))

    # Using profile at center, find, mask off fov edge, including possible beam overlap
        edgerow_do = np.zeros((2,2),dtype=int)
        badrow_or = np.zeros((2,rows),dtype=bool)
        axisrow_o = np.zeros(2)
        maxoverlaprows = 34/rbin
        profile_Cor[Cols/2] = np.median(profile_orc[:,:,cols/2-dcols:cols/2+dcols],axis=2)                       
        for d,o in np.ndindex(2,2):                         # _d = (0,1) = (bottom,top)
            row_y = np.where((d==1) ^ (np.arange(rows) < trow_o[o]))[0][::2*d-1]
            edgeval = np.median(profile_Cor[Cols/2,o,row_y],axis=-1)        
            hist,bin = np.histogram(profile_Cor[Cols/2,o,row_y],bins=32,range=(0,edgeval))
            histarg = 32 - np.argmax(hist[::-1]<3)      # edge: <3 in hist in decreasing dirn
            edgeval = bin[histarg]
            edgerow_do[d,o] = trow_o[o] + (2*d-1)*(np.argmax(profile_Cor[Cols/2,o,row_y] <= edgeval))
            axisrow_o[o] += edgerow_do[d,o]
            edgerow_do[d,o] = np.clip(edgerow_do[d,o],maxoverlaprows,rows-maxoverlaprows)
            badrow_or[o] |= ((d==1) ^ (np.arange(rows) < (edgerow_do[d,o]+d)))       
        axisrow_o /= 2.

    # Row profile sample, now background subtracted
        profile_orc[okprof_orc] = ((image_orc-np.median(image_orc,axis=1)[:,None,:])[okprof_orc]) \
                                    /(norm_orc-np.median(image_orc,axis=1)[:,None,:])[okprof_orc]
#        pyfits.PrimaryHDU(profile_orc.astype('float32')).writeto('profile_orc.fits',clobber=True)
        for C in range(Cols): 
            okcol_c = (profile_orc.sum(axis=0).sum(axis=0)>0) & \
                    (np.abs(np.arange(cols)-Collist[C])<dcols)
            Collist[C] = np.where(okcol_c)[0].mean()
            profile_Cor[C] = np.median(profile_orc[:,:,okcol_c],axis=2)
#        print 5*"%7.1f " % tuple(Collist)
#        pyfits.PrimaryHDU(okprof_orc.astype('uint8')).writeto('okprof_orc.fits',clobber=True) 
        np.savetxt("profile_oCr.txt",profile_Cor.transpose((1,0,2)).reshape((2*Cols,-1)).T,fmt="%10.6f")

    # find edge of target slit, and neighboring slits, if multiple slits
    # background picked small enough to miss neighbors in all samples, but matched E and O
        isneighbor_d = np.zeros((2),dtype='bool')
        edgeoff_doC = np.zeros((2,2,Cols))
        for o in (0,1):
            plim = 0.35                         # slit finder
            bkgsafe = 0.90                      # avoiding next slit
            for C in range(Cols):
                leftrow_s = np.flatnonzero((profile_Cor[C,o,:-1] < plim) & (profile_Cor[C,o,1:] > plim))
                rightrow_s = np.flatnonzero((profile_Cor[C,o,leftrow_s[0]:-1] > plim) \
                            & (profile_Cor[C,o,leftrow_s[0]+1:] < plim)) + leftrow_s[0]
                slits = rightrow_s.shape[0]     # eliminate spikes:
                slitrow_s = 0.5*(rightrow_s + leftrow_s[:slits])[(rightrow_s-leftrow_s[:slits]) > 2]
                slits = slitrow_s.shape[0]
                targetslit = np.where(abs(maxrow_oc[o,Collist[C]] - slitrow_s) < 6)[0][0]
                if targetslit > 0:
                    edgeoff_doC[0,o,C] = maxrow_oc[o,Collist[C]] - slitrow_s[targetslit-1:targetslit+1].mean()
                    isneighbor_d[0] |= True
                if targetslit < slits-1:
                    edgeoff_doC[1,o,C] = slitrow_s[targetslit:targetslit+2].mean() - maxrow_oc[o,Collist[C]]
                    isneighbor_d[1] |= True

        for d in (0,1):
            if isneighbor_d[d]: edgerow_do[d] = trow_o + bkgsafe*(2*d-1)*edgeoff_doC[d].min()

        edgerow_doc = (edgerow_do[:,:,None] - drow_oc[None,:,:]).astype(int)
        bkgrows_do = ((trow_o - edgerow_do)/2.).astype(int)
        bkgrow_doc = edgerow_doc + bkgrows_do[:,:,None]/2
        isbkg_dorc = (((np.arange(rows)[:,None] - edgerow_doc[:,:,None,:]) * \
              (np.arange(rows)[:,None] - edgerow_doc[:,:,None,:] - bkgrows_do[:,:,None,None])) < 0)
        istarg_orc = ((np.arange(rows)[:,None] - edgerow_doc[:,:,None,:]).prod(axis=0) < 0)
        istarg_orc &= ~isbkg_dorc.any(axis=0)
        okbinpol_orc &= okprof_oc[:,None,:]

#        pyfits.PrimaryHDU(image_orc*(isbkg_dorc.sum(axis=0)).astype('float32')).writeto('lampbkg_orc.fits',clobber=True)  
#        pyfits.PrimaryHDU(istarg_orc.astype('uint8')).writeto('istarg_orc.fits',clobber=True)

        log.message('Bottom, top row:   O %4i %4i   E %4i %4i \n' \
                % tuple(edgerow_do.T.flatten()), with_header=False)

    # background-subtract and extract spectra

    # set up scrunch table and badpixels in wavelength space
        wbin = wav_orc[0,rows/2,cols/2]-wav_orc[0,rows/2,cols/2-1] 
        wbin = float(int(wbin/0.75))
        wmin,wmax = wav_orc.min(axis=2).max(),wav_orc.max(axis=2).min()
        wedgemin = wbin*int(wmin/wbin+0.5) + wbin/2.
        wedgemax = wbin*int(wmax/wbin-0.5) + wbin/2.
        wedge_w = np.arange(wedgemin,wedgemax+wbin,wbin)
        wavs = wedge_w.shape[0] - 1
        badbin_orc = ~okbinpol_orc 
        binedge_orw = np.zeros((2,rows,wavs+1))
        badbin_orw = np.ones((2,rows,wavs),dtype=bool); nottarg_orw = np.ones_like(badbin_orw)
        for o in (0,1):
            for r in range(edgerow_doc[0,o].min(),edgerow_doc[1,o].max()):
                binedge_orw[o,r] = interp1d(wav_orc[o,r],np.arange(cols))(wedge_w)
                badbin_orw[o,r] = (scrunch1d(badbin_orc[o,r].astype(int),binedge_orw[o,r]) > 0.)
                nottarg_orw[o,r] = (scrunch1d((~istarg_orc[o,r]).astype(int),binedge_orw[o,r]) > 0.)                
        okbin_orw = ~badbin_orw
        istarg_orw = ~nottarg_orw

    # wavelengths with bad pixels in targ area are flagged as bad
        badcol_ow = (istarg_orw & ~okbin_orw).any(axis=1)            
        for o in (0,1): okbin_orw[o] &= ~badcol_ow[o]

        for i in range(files):
            imageno = int(os.path.basename(outfilelist[i]).split('.')[0][-4:])
            hdulist = pyfits.open(outfilelist[i])
            sci_orc = hdulist['sci'].data
            var_orc = hdulist['var'].data

        # make background continuum image, linearly interpolated in row
            bkg_doc = np.zeros((2,2,cols))
            for d,o in np.ndindex(2,2):
                bkg_doc[d,o] = np.median(sci_orc[o].T[isbkg_dorc[d,o].T].reshape((cols,-1)),axis=1)         
            bkgslp_oc = (bkg_doc[1] - bkg_doc[0])/(bkgrow_doc[1] - bkgrow_doc[0])
            bkgbase_oc = (bkg_doc[1] + bkg_doc[0])/2. - bkgslp_oc*(bkgrow_doc[1] + bkgrow_doc[0])/2.
            bkg_orc = bkgbase_oc[:,None,:] + bkgslp_oc[:,None,:]*np.arange(rows)[:,None]
            target_orc = sci_orc-bkg_orc             
#            np.savetxt('bkg.txt',np.vstack((bkg_doc.reshape((4,-1)),bkgslp_oc,bkgbase_oc)).T,fmt="%11.4f")
#            pyfits.PrimaryHDU(bkg_orc.astype('float32')).writeto('bkg_orc_'+str(imageno)+'.fits',clobber=True)   
#            pyfits.PrimaryHDU(target_orc.astype('float32')).writeto('target_orc_'+str(imageno)+'.fits',clobber=True)

        # extract spectrum 
            target_orw = np.zeros((2,rows,wavs));   var_orw = np.zeros_like(target_orw)
            for o in (0,1):
                for r in range(edgerow_doc[0,o].min(),edgerow_doc[1,o].max()):
                    target_orw[o,r] = scrunch1d(target_orc[o,r],binedge_orw[o,r])
                    var_orw[o,r] = scrunch1d(var_orc[o,r],binedge_orw[o,r])
  
        # columns with negative extracted intensity are marked as bad
            sci_ow = (target_orw*okbin_orw).sum(axis=1)
#            pyfits.PrimaryHDU((target_orw*okbin_orw).astype('float32')).writeto('sci_orw.fits',clobber=True)
            var_ow = (var_orw*okbin_orw).sum(axis=1)    
            okbin_ow = (okbin_orw.any(axis=1) & (sci_ow > 0.))
            bpm_ow = (~okbin_ow).astype('uint8')

        # write O,E spectrum, prefix "s". VAR, BPM for each spectrum. y dim is virtual (length 1)
        # for consistency with other modes
            hduout = pyfits.PrimaryHDU(header=hdulist[0].header)    
            hduout = pyfits.HDUList(hduout)
            hduout[0].header.update('OBJECT',lampid)
            header=hdulist['SCI'].header.copy()
            header.update('VAREXT',2)
            header.update('BPMEXT',3)
            header.update('CRVAL1',wedge_w[0]+wbin/2.)
            header.update('CRVAL2',0)
            header.update('CDELT1',wbin)
            header.update('CTYPE1','Angstroms')
            
            hduout.append(pyfits.ImageHDU(data=sci_ow.reshape((2,1,wavs)).astype('float32'), header=header, name='SCI'))
            header.update('SCIEXT',1,'Extension for Science Frame',before='VAREXT')
            hduout.append(pyfits.ImageHDU(data=var_ow.reshape((2,1,wavs)).astype('float32'), header=header, name='VAR'))
            hduout.append(pyfits.ImageHDU(data=bpm_ow.reshape((2,1,wavs)), header=header, name='BPM'))            
            
            hduout.writeto('e'+outfilelist[i],clobber=True,output_verify='warn')
            log.message('Output file '+'e'+outfilelist[i] , with_header=False)
      
    return

