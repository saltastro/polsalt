
"""
sppolmasktool

compute predicted spectral image for MOS spectropolarimetry

"""

import os, sys, glob, datetime
polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table

from rssoptics import RSScolgratpolcam,RSSpolgeom
from polmaptools import readmaskxml, readmaskgcode
from sppolmap import slitpredict

np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.inf)

# warnings.simplefilter("error")
debug = False

# ------------------------------------
def sppolmasktool(slitfile,filter,grating,grang,artic,debug=False):
    """return text and fits image of predicted spectrum  

    Parameters 
    ----------
    mask: text name of mask definition file, either xml or gcode
    filter: text name of filter
    grating: text name of grating
    grang: float (deg) of grang
    artic: float (deg of artic

    """
    """
    _r, _c binned pixel cordinate (assume 2x2 binning)
    _i mask slits (entries in xml)
    """
    rows,cols = (2056,3200)
    ccd_rc = np.ones((rows,cols),dtype=int)
    ccd_rc[:,[1000,2200]] = 0
    bpm_rc = np.zeros((rows,cols),dtype=int)    
    hdul = pyfits.HDUList(pyfits.PrimaryHDU())
    hdul.append(pyfits.ImageHDU(data=ccd_rc.astype('uint8'),name='SCI'))
    hdul.append(pyfits.ImageHDU(data=bpm_rc.astype('uint8'),name='BPM'))            
    hdr = hdul[0].header
    cbin, rbin = (2,2)
    rcbin_d = np.array([rbin,cbin])
    hdr['FILTER'] = filter   
    hdr['GRATING'] = grating 
    hdr['GR-ANGLE'] = grang
    hdr['CAMANG'] = artic
    hdr['CAMTEM'] = 5.
    hdr['COLTEM'] = 5.
    hdr['DATE-OBS'] = str(datetime.date.today())
    hdr['TRKRHO'] = 0.
    hdr['CCDSUM'] = '2 2'
    
    wav_w = np.arange(float(filter[3:])-50.,10500.,100.)
    maskid,slittype = slitfile.split('.')                 
    if (not (slittype in ('xml','gcode'))):
        print 'no MOS xml or gcode file found for ',maskid
        exit()
    if slittype.count('xml'):                       
        slitTab, PAd, RAd, DECd = readmaskxml(maskid+'.xml')
        YX_dt = YXcalc(slitTab['RACE'],slitTab['DECCE'],RAd,DECd,PAd,fps)
        slitTab['YCE'],slitTab['XCE'] = tuple(YX_dt)
    else:
        fps = 226.143                                   # fps =micr/arcsec after 20110409   
        slitTab = readmaskgcode(maskid+'.gcode')
        slitTab['WIDTH'] = 1000.*slitTab['XLEN']/fps
        slitTab['LENGTH'] = 1000.*slitTab['YLEN']/fps                        
    if debug:
        filter = hdul0[0].header['FILTER']
        for c,col in enumerate(slitTab.colnames[2:]): 
            slitTab[col].format=(2*['%.6f']+4*['%7.3f'])[c]
            slitTab.write(objectname+"_"+filter+"_slitTab.txt",format='ascii.fixed_width',   \
                bookend=False, delimiter=None, overwrite=True)           
    YX_di = np.array([slitTab['YCE'],slitTab['XCE']])
    entries = YX_di.shape[1]

    img_rc = np.zeros((rows,cols),dtype=int)
    wav_rc = np.zeros((rows,cols))    
    fitsimg_rc = np.zeros((rows,cols),dtype=int)          
    orders = [1,2][grating=='PG0300']
    for order in np.arange(1,orders+1):  
        wav_pic,rce_pic,ok_pic = slitpredict(hdul,YX_di,wav_w,order=order,debug=False)
        rce_pic[rce_pic > rows-2] = 0         
        rcidx_pic = np.round(ok_pic*rce_pic).astype(int)
        rcidxc_pic = np.ceil(ok_pic*rce_pic).astype(int)
        rcidxf_pic = np.floor(ok_pic*rce_pic).astype(int)                   
        for p,i in np.ndindex(2,entries):
            isspec_c = (rcidx_pic[p,i] != 0)
            
            if (order==2):
                ioverlap = i +[-5,5][p]
                c4600 = np.where(isspec_c)[0][0]           
                print c4600,wav_pic[p,i,c4600],
                if ((ioverlap >= 0) & (ioverlap <= 24)):
                    print wav1st_pic[p,ioverlap,c4600]
                else:
                    print 
            
            isspec_rc = np.zeros((rows,cols),dtype=bool)
            isspecc_rc = np.zeros((rows,cols),dtype=bool)
            isspecf_rc = np.zeros((rows,cols),dtype=bool)                        
            isspec_rc[rcidx_pic[p,i],range(cols)] = isspec_c
            img_rc[isspec_rc] = i+1
            wav_rc[isspec_rc] = wav_pic[p,i,isspec_c]            
            isspecc_rc[rcidxc_pic[p,i],range(cols)] = isspec_c            
            isspecf_rc[rcidxf_pic[p,i],range(cols)] = isspec_c                        
            fitsimg_rc[isspecc_rc | isspecf_rc] = i+1
        wav1st_pic = np.copy(wav_pic)

    hdul['SCI'].data = fitsimg_rc.astype('uint8')
    
    return img_rc,wav_rc,hdul        
# ------------------------------------
 
if __name__=='__main__':
    slitfile,filter,grating = sys.argv[1:4]
    img_rc,wav_rc,hdul = sppolmasktool(slitfile,filter,grating, float(sys.argv[4]), float(sys.argv[5]))
    np.savetxt("imgrc.txt",img_rc,fmt="%3i ")
    np.savetxt("wavrc.txt",wav_rc,fmt="%8.2f ")    
    hdul.writeto("sppolmasktool.fits",overwrite=True)    
    
# debug
# sppolmasktool.py P000000P17.gcode PC03850 PG0300 5.38 10.68

