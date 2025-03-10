def bkgsub(hdulist,badbinbkg_orc,isbkgcont_orc,skyflat_orc,maprow_ocd,tnum):
    """
    Remove remaining detector bias and sky continuum
    Use Kotulla 2D sky subtract to remove sky lines

    Parameters
    ----------
    hdu: fits.HDUList
       polarimetric image cube for processing
    badbinbkg_orc: np 3D boolean array
       bins to be avoided in background estimation
    isbkgcont_orc: np 3D boolean array
       bins used for sky continuum background estimation (avoids target, sky lines)
    skyflat_orc: np 3D float array
       skyflat estimate, divided out before sky line removal
    maprow_ocd: np 3D float array
       row number of edge of fov and target, top and bottom, vs column for O,E 
    tnum: int
       image number (to label debug fits files)

    Output
    ------
    target_orc: np 3D float array 
        background-subtracted image

    """

    sci_orc = np.copy(hdulist['SCI'].data)
    wav_orc = hdulist['WAV'].data

    rows,cols = hdulist['SCI'].data.shape[1:3]
    cbin,rbin = np.array(hdulist[0].header["CCDSUM"].split(" ")).astype(int)
    slitid = hdulist[0].header["MASKID"]
    if slitid[0] =="P": slitwidth = float(slitid[2:5])/10.
    else: slitwidth = float(slitid)         # in arcsec

  # remove remaining detector bias, independently for each amplifier
  # use median of rows outside of fov at bottom and top of full image (bottom of O, top of E)
    dtrbkg_dc = np.zeros((2,cols))
    dtrbkg_dc[0] = np.median(sci_orc[0,2:(maprow_ocd[0,:,0].min()-2)],axis=0)
    dtrbkg_dc[1] = np.median(sci_orc[1,(maprow_ocd[1,:,3].max()+2):(rows-2)],axis=0)
    dtrbkg_c = np.mean(dtrbkg_dc,axis=0)
    dtrbkgslope_c = (dtrbkg_dc[1] - dtrbkg_dc[0])/(rows+maprow_ocd[1,cols/2,3]-maprow_ocd[0,cols/2,0])
  # first find amplifier edges for each CCD
    a_c = -np.ones(cols)
    cstart = 0
    for CCD in (1,2,3):
        cmin = np.argmax((dtrbkg_c[cstart: ] != 0)&(dtrbkg_c[cstart: ] != -1)) + cstart
        cmax = np.argmax((dtrbkg_c[cmin: ] == 0)|(dtrbkg_c[cmin: ] == -1)) + cmin-1
        if cmax == cmin-1: cmax = cols-1
        a_c[cmin:(cmin+cmax)/2] = 2*CCD-2
        a_c[(cmin+cmax)/2:cmax] = 2*CCD-1
        cstart = cmax+1
    bias_Rc = np.zeros((2*rows,cols))
    for a in range(6):
        slope = np.median(dtrbkgslope_c[a_c==a])
        p = np.poly1d(np.polyfit(np.where(a_c==a)[0],dtrbkg_c[a_c==a],2))
        bias_Rc[:,(a_c==a)] = p(np.arange(cols)[a_c==a]) + slope*np.arange(-rows,rows)[:,None]
    bias_orc = bias_Rc.reshape((2,rows,cols))
    target_orc = sci_orc - bias_orc
    pyfits.PrimaryHDU(bias_orc.astype('float32')).writeto('bias_'+tnum+'_orc.fits',clobber=True)   
    pyfits.PrimaryHDU(target_orc.astype('float32')).writeto('target-bias_'+tnum+'_orc.fits',clobber=True)

  # make background continuum image, smoothed over resolution element
    rblk,cblk = int(1.5*8./rbin), int(slitwidth*8./cbin)
    isedge_orc = (np.arange(rows)[:,None] < maprow_ocd[:,None,:,0]) | \
          (np.arange(rows)[:,None] > maprow_ocd[:,None,:,3])
    isskycont_orc = (((np.arange(rows)[:,None] < maprow_ocd[:,None,:,0]+rows/16) |  \
          (np.arange(rows)[:,None] > maprow_ocd[:,None,:,3]-rows/16)) & ~isedge_orc)  

    for o in (0,1):
        bkgcont_rc = blksmooth2d(target_orc[o],isbkgcont_orc[o],rblk,cblk,0.25,mode="mean")
           
    # remove sky continuum: ends of bkg continuum * skyflat
        skycont_c = (bkgcont_rc.T[isskycont_orc[o].T]/skyflat_orc[o].T[isskycont_orc[o].T])  \
           .reshape((cols,-1)).mean(axis=1)
        skycont_rc = skycont_c*skyflat_orc[o]
        
    # remove sky lines: image - bkgcont run through 2d sky averaging
        objdata_rc = ((target_orc[o] - bkgcont_rc)/skyflat_orc)[o]
        objdata_rc[badbinbkg_orc[o]] = np.nan

#        pyfits.PrimaryHDU(objdata_rc.astype('float32')).writeto('objdata_'+tnum+'_'+str(o)+'.fits',clobber=True)

        skylines_rc = make_2d_skyspectrum(objdata_rc,wav_orc[o],np.array([[0,rows],]))*skyflat_orc[o]
        target_orc[o] -= skycont_rc + skylines_rc

#        pyfits.PrimaryHDU(skylines_rc.astype('float32')).writeto('skylines_rc_'+tnum+'_'+str(o)+'.fits',clobber=True)
#        pyfits.PrimaryHDU(skycont_rc.astype('float32')).writeto('skycont_rc_'+tnum+'_'+str(o)+'.fits',clobber=True)

    return target_orc,dtrbkg_dc
