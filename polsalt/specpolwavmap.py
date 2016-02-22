
"""
specpolwavmap

Split O and E and produce wavelength map for spectropolarimetric data

"""

import os, sys, glob, shutil, inspect

import numpy as np
import pyfits
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
from scipy import linalg as la

from pyraf import iraf
from iraf import pysalt

from saltobslog import obslog

from specidentify import specidentify
from saltsafelog import logging

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"
np.set_printoptions(threshold=np.nan)
debug = True
#----------------------------------------------------------------------
def configmap(infilelist):
    obs_dict=obslog(infilelist)
    infiles = len(infilelist)
    grating_i = [obs_dict['GRATING'][i].strip() for i in range(infiles)]
    grang_i = np.array(map(float,obs_dict['GR-ANGLE']))
    artic_i = np.array(map(float,obs_dict['CAMANG']))
    configdat_i = [tuple((grating_i[i],grang_i[i],artic_i[i])) for i in range(infiles)]
    confdatlist = list(set(configdat_i))          # list tuples of the unique configurations _c
    confno_i = np.array([confdatlist.index(configdat_i[i]) for i in range(infiles)],dtype=int)
    return confno_i,confdatlist

#----------------------------------------------------------------------
def specpolwavmap(infilelist, linelistlib="", automethod='Matchlines',logfile='salt.log',debug=False):
    obsdate=os.path.basename(infilelist[0])[7:15]

    with logging(logfile, debug) as log:
    # create the observation log
        obs_dict=obslog(infilelist)
        log.message('Pysalt Version: '+pysalt.verno, with_header=False)
      
        
    # eliminate inapplicable images
        for i in reversed(range(len(infilelist))):
            if int(obs_dict['BS-STATE'][i][1])!=2: del infilelist[i]
        obs_dict=obslog(infilelist)

    # Map out which arc goes with which image.  Use arc in closest wavcal block of the config.
    # wavcal block: neither spectrograph config nor track changes, and no gap in data files
        infiles = len(infilelist)
        newtrk = 5.                                     # new track when rotator changes by more (deg)
        trkrho_i = np.array(map(float,obs_dict['TRKRHO']))
        trkno_i = np.zeros((infiles),dtype=int)
        trkno_i[1:] = ((np.abs(trkrho_i[1:]-trkrho_i[:-1]))>newtrk).cumsum()

        confno_i,confdatlist = configmap(infilelist)
        configs = len(confdatlist)

        imageno_i = np.array([int(os.path.basename(infilelist[i]).split('.')[0][-4:]) \
                for i in range(infiles)])
        filegrp_i = np.zeros((infiles),dtype=int)
        filegrp_i[1:] = ((imageno_i[1:]-imageno_i[:-1])>1).cumsum()
        isarc_i = np.array([(obs_dict['OBJECT'][i].upper().strip()=='ARC') for i in range(infiles)])

        wavblk_i = np.zeros((infiles),dtype=int)
        wavblk_i[1:] = ((filegrp_i[1:] != filegrp_i[:-1]) \
                    | (trkno_i[1:] != trkno_i[:-1]) \
                    | (confno_i[1:] != confno_i[:-1])).cumsum()
        wavblks = wavblk_i.max() +1

        arcs_c = (isarc_i[:,None] & (confno_i[:,None]==range(configs))).sum(axis=0)
        np.savetxt("wavblktbl.txt",np.vstack((trkrho_i,imageno_i,filegrp_i,trkno_i, \
                confno_i,wavblk_i,isarc_i)).T,fmt="%7.2f "+6*"%3i ",header=" rho img grp trk conf wblk arc")
        for c in range(configs):                               # worst: no arc for config, remove images
            if arcs_c[c] == 0:
                lostimages = imageno_i[confno_i==c]
                log.message('No Arc for this configuration: ' \
                    +("Grating %s Grang %6.2f Artic %6.2f" % confdatlist[c])  \
                    +("\n Images: "+lostimages.shape[0]*"%i " % tuple(lostimages)), with_header=False)
                wavblk_i[confno_i==c] = -1
            if arcs_c.sum() ==0: 
                log.message("Cannot calibrate any images", with_header=False)
                exit()
        iarc_i = -np.zeros((infiles),dtype=int)
        for w in range(wavblks):
            blkimages =  imageno_i[wavblk_i==w]
            if blkimages.shape[0]==0: continue 
            iarc_I = np.where((wavblk_i==w) & (isarc_i))[0]
            if iarc_I.shape[0] >0: 
                iarc = iarc_I[0]                        # best: arc is in wavblk, take first
            else:
                conf = confno_i[wavblk_i==w][0]       # fallback: take closest arc of this config
                iarc_I = np.where((confno_i==conf) & (isarc_i))[0]
                blkimagepos = blkimages.mean()
                iarc = iarc_I[np.argmin(imageno_i[iarc_I] - blkimagepos)]
            iarc_i[wavblk_i==w] = iarc                    
            log.message(("\nFor images: "+blkimages.shape[0]*"%i " % tuple(blkimages)) \
                +("\n  Use Arc %5i" % imageno_i[iarc]), with_header=False)               

        iarc_a = np.unique(iarc_i[iarc_i != -1])
        arcs = iarc_a.shape[0]
        lam_m = np.loadtxt(datadir+"wollaston.txt",dtype=float,usecols=(0,))
        rpix_om = np.loadtxt(datadir+"wollaston.txt",dtype=float,unpack=True,usecols=(1,2))

        for a in range(arcs):
            iarc = iarc_a[a]
            conf = confno_i[iarc]
            grating,grang,artic = confdatlist[confno_i[iarc]]
            
            if len(linelistlib) ==0: 
                linelistlib=datadir+"linelistlib.txt"   
                if grating=="PG0300": 
                    linelistlib=datadir+"linelistlib_300.txt"
            with open(linelistlib) as fd:
                linelistdict = dict(line.strip().split(None, 1) for line in fd)  

        # use arc to make first-guess wavecal from model
            cbin,rbin = np.array(obs_dict["CCDSUM"][iarc].split(" ")).astype(int)

            hduarc = pyfits.open(infilelist[iarc])
            arc_rc = hduarc['SCI'].data
            rows,cols = arc_rc.shape
            lam_c = rssmodelwave(grating,grang,artic,cbin,cols)
            arc_r = arc_rc.sum(axis=1)
#            if debug: np.savetxt("arc_r_"+str(imageno_i[iarc]+".txt"),arc_r,fmt="%8.2f")
        # locate beamsplitter split point
            axisrow_o = ((2052 + interp1d(lam_m,rpix_om,kind='cubic',bounds_error=False) \
                        (lam_c[cols/2]))/rbin).astype(int)
            top = axisrow_o[1] + np.argmax(arc_r[axisrow_o[1]:] <  0.5*arc_r[axisrow_o[1]])
            bot = axisrow_o[0] - np.argmax(arc_r[axisrow_o[0]::-1] <  0.5*arc_r[axisrow_o[0]])
            splitrow = 0.5*(bot + top)
            offset = int(splitrow - rows/2)                 # how far split is from center of detector

        # split arc into o/e images
            padbins = (np.indices((rows,cols))[0]<offset) | (np.indices((rows,cols))[0]>rows+offset)
            arc_rc = np.roll(arc_rc,-offset,axis=0)
            arc_rc[padbins] = 0.
            arc_orc =  arc_rc.reshape((2,rows/2,cols))
    
        # for O,E arc straighten spectrum, find fov, identify for each, form (unstraightened) wavelength map   
            lamp=obs_dict['LAMPID'][iarc].strip().replace(' ', '')
            if lamp == 'NONE': lamp='CuAr'
            hduarc[0].header.update('MASKTYP','LONGSLIT')
            del hduarc['VAR']
            del hduarc['BPM']
            lampfile=iraf.osfn("pysalt$data/linelists/"+linelistdict[lamp])    
            rpix_oc = interp1d(lam_m, rpix_om,kind ='cubic',bounds_error=False,fill_value=0.)(lam_c)
            drow_oc = (rpix_oc-rpix_oc[:,cols/2][:,None])/rbin

            log.message('\nARC: image '+str(imageno_i[iarc])+' GRATING '+grating\
                +' GRANG '+("%8.3f" % grang)+' ARTIC '+("%8.3f" % artic)+' LAMP '+lamp, with_header=False)
            log.message('  Split Row: '+("%4i " % splitrow), with_header=False)

            wavmap_orc = np.zeros((2,rows/2,cols))
            edgerow_od = np.zeros((2,2))
            cofrows_o = np.zeros(2)
            legy_od = np.zeros((2,2))
            guessfile=None
            for o in (0,1):
                axisrow_o[o] += -offset - o*rows/2
                arc_yc = np.zeros((rows/2,cols),dtype='float32')
                for c in range(cols): 
                    shift(arc_orc[o,:,c],-drow_oc[o,c],arc_yc[:,c])

                maxoverlaprows = 34/rbin                        # beam overlap for 4' longslit in NIR
                arc_yc[(0,rows/2-1)] = 0.
                arc_y = arc_yc.sum(axis=1)
                edgerow_od[o,0] = axisrow_o[o] - np.argmax(arc_y[axisrow_o[o]::-1] <  0.5*arc_y[axisrow_o[o]])
                edgerow_od[o,1] = axisrow_o[o] + np.argmax(arc_y[axisrow_o[o]:] <  0.5*arc_y[axisrow_o[o]])
                axisrow_o[o] = edgerow_od[o].mean()
                if np.abs(edgerow_od[o] - np.array([0,rows/2-1])).min() < maxoverlaprows:
                    edgerow_od[o] += maxoverlaprows*np.array([+1,-1])
                hduarc['SCI'].data = arc_yc
                order = 3
                arcimage = "arc_"+str(imageno_i[iarc])+"_"+str(o)+".fits"
                dbfilename = "arcdb_"+str(imageno_i[iarc])+"_"+str(o)+".txt"

                if (not os.path.exists(dbfilename)):
                    if guessfile is not None:
                        guesstype = 'file'
                    else:
                        guessfile=dbfilename
                        guesstype = 'rss'
                    hduarc.writeto(arcimage,clobber=True)
                    ystart = axisrow_o[o]
                    specidentify(arcimage, lampfile, dbfilename, guesstype=guesstype,
                        guessfile=guessfile, automethod=automethod,  function='legendre',  order=order,
                        rstep=20, rstart=ystart, mdiff=20, thresh=3, niter=5, smooth=3,
                        inter=True, clobber=True, logfile=logfile, verbose=True)
                    if (not debug): os.remove(arcimage)

            # process dbfile legendre coefs within FOV into wavmap (_Y = line in dbfile)
                legy_Y = np.loadtxt(dbfilename,dtype=float,usecols=(0,),ndmin=1)
                dblegcof_lY = np.loadtxt(dbfilename,unpack=True,dtype=float,usecols=range(1,order+2),ndmin=2)

            # first convert to centered legendre coefficients to remove crosscoupling
                xcenter = cols/2.
                legcof_lY = np.zeros_like(dblegcof_lY)
                legcof_lY[2] = dblegcof_lY[2] + 5.*dblegcof_lY[3]*xcenter
                legcof_lY[3] = dblegcof_lY[3]
                legcof_lY[0] = 0.5*legcof_lY[2] + (dblegcof_lY[0]-dblegcof_lY[2]) + \
                    (dblegcof_lY[1]-1.5*dblegcof_lY[3])*xcenter + 1.5*dblegcof_lY[2]*xcenter**2 + \
                    2.5*dblegcof_lY[3]*xcenter**3
                legcof_lY[1] = 1.5*legcof_lY[3] + (dblegcof_lY[1]-1.5*dblegcof_lY[3]) + \
                    3.*dblegcof_lY[2]*xcenter + 7.5*dblegcof_lY[3]*xcenter**2

            # remove rows outside slit and outlier fits
                argYbad = np.where((legy_Y<edgerow_od[o,0]) | (legy_Y>edgerow_od[o,1]))[0]
                legy_Y = np.delete(legy_Y, argYbad,axis=0)
                legcof_lY = np.delete(legcof_lY, argYbad,axis=1)
                mediancof_l = np.median(legcof_lY,axis=1)
                rms_l = np.sqrt(np.median((legcof_lY - mediancof_l[:,None])**2,axis=1))
                sigma_lY = np.abs((legcof_lY - mediancof_l[:,None]))/rms_l[:,None]
                argYbad = np.where((sigma_lY>4).any(axis=0))[0]
                legy_Y = np.delete(legy_Y, argYbad,axis=0)
                legcof_lY = np.delete(legcof_lY, argYbad,axis=1)
                cofrows_o[o] = legy_Y.shape[0]
                legy_od[o] = legy_Y.min(),legy_Y.max()

                if cofrows_o[o] < 5:
            # for future: if few lines in db, use model shifted to agree, for now just use mean
                    legcof_l = legcof_lY.mean(axis=1)
                    wavmap_yc = np.polynomial.legendre.legval(np.arange(cols),legcof_l)
                else:
            # smooth wavmap along rows by fitting L_0 to quadratic, others to linear fn of row
                    ycenter = rows/4.
                    Y_y = np.arange(-ycenter,ycenter)
                    aa = np.vstack(((legy_Y-ycenter)**2,(legy_Y-ycenter),np.ones(cofrows_o[o]))).T
                    polycofs = la.lstsq(aa,legcof_lY[0])[0]
                    legcof_ly = np.zeros((order+1,rows/2))
                    legcof_ly[0] = np.polyval(polycofs,Y_y)
                    for l in range(1,order+1):
                        polycofs = la.lstsq(aa[:,1:],legcof_lY[l])[0]
                        legcof_ly[l] = np.polyval(polycofs,Y_y)
                    wavmap_yc = np.zeros((rows/2,cols))
                    for y in range(rows/2):
                        wavmap_yc[y] = np.polynomial.legendre.legval(np.arange(-cols/2,cols/2),legcof_ly[:,y])

            # put curvature back in, zero out areas beyond slit and wavelength range (will be flagged in bpm)
                if debug: np.savetxt("drow_wmap_oc.txt",drow_oc.T,fmt="%8.3f %8.3f")
                for c in range(cols): 
                    shift(wavmap_yc[:,c],drow_oc[o,c],wavmap_orc[o,:,c],order=1)

                isoffslit_rc = ((np.arange(rows/2)[:,None] < (edgerow_od[o,0]+(rpix_oc[o]-rpix_oc[o,cols/2])/rbin)[None,:]) \
                           | (np.arange(rows/2)[:,None] > (edgerow_od[o,1]+(rpix_oc[o]-rpix_oc[o,cols/2])/rbin)[None,:]))
                notwav_rc = (rpix_oc[o]==0.)[None,:]
                wavmap_orc[o,(isoffslit_rc | notwav_rc)] = 0.

            log.message('\n  Wavl coeff rows:  O    %4i     E    %4i' % tuple(cofrows_o), with_header=False)
            log.message('  Bottom, top row:  O %4i %4i   E %4i %4i' \
                % tuple(legy_od.flatten()), with_header=False)
            log.message('\n  Slit axis row:    O    %4i     E    %4i' % tuple(axisrow_o), with_header=False)
            log.message('  Bottom, top row:  O %4i %4i   E %4i %4i \n' \
                % tuple(edgerow_od.flatten()), with_header=False)

        # for images using this arc,save split data along third fits axis, 
        # add wavmap extension, save as 'w' file
            hduwav = pyfits.ImageHDU(data=wavmap_orc.astype('float32'), header=hduarc['SCI'].header, name='WAV')                 
            for i in np.where(iarc_i==iarc_a[a])[0]:  
                hdu = pyfits.open(infilelist[i])
                image_rc = np.roll(hdu['SCI'].data,-offset,axis=0)
                image_rc[padbins] = 0.
                hdu['SCI'].data = image_rc.reshape((2,rows/2,cols))
                var_rc = np.roll(hdu['VAR'].data,-offset,axis=0)
                var_rc[padbins] = 0.
                hdu['VAR'].data = var_rc.reshape((2,rows/2,cols))
                bpm_rc = np.roll(hdu['BPM'].data,-offset,axis=0)
                bpm_rc[padbins] = 1
                bpm_orc = bpm_rc.reshape((2,rows/2,cols))
                bpm_orc[wavmap_orc==0.] = 1
                hdu['BPM'].data = bpm_orc
                hdu.append(hduwav)
                for f in ('SCI','VAR','BPM','WAV'): hdu[f].header.update('CTYPE3','O,E')
                hdu.writeto('w'+infilelist[i],clobber='True')
                log.message('Output file '+'w'+infilelist[i] , with_header=False)

    return
 
def rssmodelwave(grating,grang,artic,cbin,cols):
#   compute wavelengths from model (this can probably be done using pyraf spectrograph model)
    spec=np.loadtxt(datadir+"spec.txt",usecols=(1,))
    Grat0,Home0,ArtErr,T2Con,T3Con=spec[0:5]
    FCampoly=spec[5:11]
    grname=np.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=np.loadtxt(datadir+"gratings.txt",usecols=(1,2),unpack=True)

    grnum = np.where(grname==grating)[0][0]
    lmm = grlmm[grnum]
    alpha_r = np.radians(grang+Grat0)
    beta0_r = np.radians(artic*(1+ArtErr)+Home0)-alpha_r
    gam0_r = np.radians(grgam0[grnum])
    lam0 = 1e7*np.cos(gam0_r)*(np.sin(alpha_r) + np.sin(beta0_r))/lmm
    ww = lam0/1000. - 4.
    fcam = np.polyval(FCampoly,ww)
    disp = (1e7*np.cos(gam0_r)*np.cos(beta0_r)/lmm)/(fcam/.015)
    dfcam = 3.162*disp*np.polyval([FCampoly[x]*(5-x) for x in range(5)],ww)
    T2 = -0.25*(1e7*np.cos(gam0_r)*np.sin(beta0_r)/lmm)/(fcam/47.43)**2 + T2Con*disp*dfcam
    T3 = (-1./24.)*3162.*disp/(fcam/47.43)**2 + T3Con*disp
    T0 = lam0 + T2 
    T1 = 3162.*disp + 3*T3
    X = (np.array(range(cols))+1-cols/2)*cbin/3162.
    lam_X = T0+T1*X+T2*(2*X**2-1)+T3*(4*X**3-3*X)
    return lam_X
    


