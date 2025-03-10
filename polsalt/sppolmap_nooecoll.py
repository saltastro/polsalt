
"""
sppolmap

reducepoldata grating spectropolarimetry, longslit and mos
longslit is treated as single-target mos
Locate targets in arc image, save with 2D target map 
Compute 1D extraction map, 1D wavelength map for grating spectropolarimetric data

sparcpolmap
spmospolmap

"""

import os, sys, glob, shutil, inspect, warnings

import numpy as np
from scipy import linalg as la
from scipy.special import legendre as legFn
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage.interpolation import shift
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import Latitude,Longitude,Angle
from astropy.table import Table

# this is pysalt-free

import rsslog
from obslog import create_obslog
from rssoptics import RSScolpolcam,RSScolpolcam,RSScolgratpolcam,RSSpolgeom
from polutils import rssdtralign,rssmodelwave,fargmax,fence,legfit_cull
from polmaptools import sextract,catid,ccdcenter,gaincor,   \
    YXcalc,impolguide,rotate2d,Tableinterp
from immospolextract import moffat,moffat1dfit    

datadir = os.path.dirname(__file__) + '/data/'
keywordfile = datadir+"obslog_config.json"
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.inf)

# warnings.simplefilter("error")
debug = False

# ------------------------------------
def sptiltcalc(dataList,slitTab,**kwargs):
    """return mean spectral tilt (deg), and spectral row offset (bins) from sum image  

    Parameters 
    ----------
    dataList: list of hdul' of data
    slitTab: table from xml with slit positions

    """
    """
    _r, _c binned pixel cordinate
    _i mask slits (entries in xml)
    """

    cutwavoverride = float(kwargs.pop('cutwavoverride',0.)) 
    debug = kwargs.pop('debug',False)

    files = len(dataList)
    hdul0 = dataList[0]
    hdr = hdul0[0].header
    rows, cols = hdul0['SCI'].data.shape
    cbin, rbin = [int(x) for x in hdr['CCDSUM'].split(" ")]
    rcbin_d = np.array([rbin,cbin])
    filter = hdr['FILTER']    
    
    imgsum_rc = np.zeros((rows,cols))
    varsum_rc = np.zeros((rows,cols))    
    okimg_rc = np.ones((rows,cols),dtype=bool)
    for hdul in dataList:
        imgsum_rc += hdul['SCI'].data
        varsum_rc += hdul['VAR'].data        
        okimg_rc &= (hdul['BPM'].data != 1)
         
    if debug:
        img0 = hdul0.filename().split('.')[0][-4:]
        debugfile = open("sptilt_"+img0+".txt",'w')        
        debugfile = open("sptilt_"+img0+".txt",'a')
            
  # get nominal spectrum location
    wmin = max(float(filter[3:])-50.,cutwavoverride)
    wav_w = np.arange(3000.,10000.,100.)
    wav_w = np.insert(wav_w[wav_w > wmin],0,wmin)
    YX_di = np.array([slitTab['YCE'],slitTab['XCE']])
    entries = YX_di.shape[1]  
        
  # get mean row offset droff and tilt sptilt from row profile maximum left and right.  Iterate once.  
  #   Only use spectra within 16 rows of nominal
    droff = 0.
    tiltoff = 0.
    sptilt = 0.
    for iter in (1,2):
        wav_pic,rce_pic,ok_pic = slitpredict(hdul,YX_di,wav_w,droff=droff,sptilt=sptilt,debug=False)    
        rcidx_pic = np.round(rce_pic).astype(int)
        tiltoff_pi = np.zeros((2,entries))
        oksig_pi = np.zeros((2,entries),dtype=bool)
        drmax_pi = np.zeros((2,entries))
           
        for p,i in np.ndindex(2,entries):
            c1 = np.where(wav_pic[p,i] > wav_w[0])[0][0]
            c2 = np.where((wav_pic[p,i] < wav_w[-1]) & (wav_pic[p,i] > 0.))[0][-1]
            cmid = (c1+c2)/2        
            ridx_Rc = rcidx_pic[p,i][None,:] + np.arange(-15,16)[:,None]   
            imgleft_R = imgsum_rc[ridx_Rc,range(cols)][:,c1:cmid].sum(axis=1)
            varleft_R = varsum_rc[ridx_Rc,range(cols)][:,c1:cmid].sum(axis=1)        
            cleft = (np.arange(cols)[None,:]*imgsum_rc[ridx_Rc,range(cols)])[:,c1:cmid].sum()/  \
                imgsum_rc[ridx_Rc,range(cols)][:,c1:cmid].sum()
            Rmaxleft = np.argmax(imgleft_R)
            imgleft = (imgleft_R.max() - (imgleft_R[0]+imgleft_R[-1])/2.)/imgleft_R.shape[0]
            varleft = varleft_R.max()/imgleft_R.shape[0]**2
            oksigleft = ((Rmaxleft>0) & (Rmaxleft<(imgleft_R.shape[0]-1)) & (imgleft/np.sqrt(varleft) > 10.)) 
             
            imgright_R = imgsum_rc[ridx_Rc,range(cols)][:,cmid:c2].sum(axis=1)
            varright_R = varsum_rc[ridx_Rc,range(cols)][:,cmid:c2].sum(axis=1)        
            cright = (np.arange(cols)[None,:]*imgsum_rc[ridx_Rc,range(cols)])[:,cmid:c2].sum()/    \
                imgsum_rc[ridx_Rc,range(cols)][:,cmid:c2].sum()
            Rmaxright = np.argmax(imgright_R)        
            imgright = (imgright_R.max() - (imgleft_R[0]+imgleft_R[-1])/2.)/imgright_R.shape[0]
            varright = varright_R.max()/imgright_R.shape[0]**2
            oksigright = ((Rmaxright>0) & (Rmaxright<(imgright_R.shape[0]-1)) & (imgright/np.sqrt(varright) > 10.))                                     
            drmax_pi[p,i] = droff + (fargmax(imgright_R) + fargmax(imgleft_R))/2. - 15.               
            tiltoff_pi[p,i] = tiltoff + (fargmax(imgright_R) - fargmax(imgleft_R))/(cright-cleft)
            oksig_pi[p,i] = (oksigleft & oksigright)
        
            if debug:                                 
                print >> debugfile,("%2i %3i %4i %4i %4i %7.1f %7.1f %8.0f %8.0f  %8.0f %8.0f %2i %8.4f %8.4f") %    \
                    (p,i,c1,c2,cmid,cleft,cright,imgleft,np.sqrt(varleft),imgright,np.sqrt(varright),oksig_pi[p,i],  \
                    drmax_pi[p,i,],np.degrees(tiltoff_pi[p,i]*rbin/cbin))

        oksig_i = oksig_pi.all(axis=0)        
        droff = np.median(drmax_pi[:,oksig_i])
        tiltoff = np.median(tiltoff_pi[:,oksig_i])             
        sptilt = np.degrees(tiltoff*rbin/cbin)     
     
    return sptilt, drmax_pi, oksig_i

# ------------------------------------

def sparcpolmap(hdul,slitTab,objectname,sptilt,droff_pi,oksig_i,**kwargs):
    """predict E and O target positions, for arclines, update based on arc for TGTmap  

    Parameters 
    ----------
    hdul: hdul of arc
    slitTab: table from xml with slit positions
    objectname: hdr OBJECT for sci images this arc is for
    sptilt: spectral tilt for sci images
    droff_pi: row offset for sci spectra
    isdiffuse: boolean: object fills slit

    """
    """
    _f file index in list
    _d dimension index y,x = 0,1
    _y, _x unbinned pixel coordinate relative to optic axis
    _r, _c binned pixel cordinate
    _i mask slits (entries in xml)
    _t culled MOS target index for extract
    _m index of line in lamp linelist
    _l index of line located in spectrum
    _b index of lineid output

    YX_d image coords (mm) at SALT focal plane
    yx_d: image coords (mm) at detector (relative to imaging optic axis)
    yx0_d: position (mm) of center of CCD image (relative to imaging optic axis)  
    yx_dp image coords (relative to imaging optic axis), separated by beam, polarimetric mode
    yx0_dp position (mm) of O,E optic axes at this wavelength (relative to imaging optic axis)
    yxp_dp image coords of O,E images (relative to O,E optic axes at this wavelength)
    yxp0_dp: position (mm) of center of split O,E images (relative to O,E optic axes at this wavelength) 
    """

    cutwavoverride = float(kwargs.pop('cutwavoverride',0.))
    isdiffuse = kwargs.pop('isdiffuse',False)    
    pidebug = tuple(kwargs.pop('pidebug',(-1,-1)))
    logfile= kwargs.pop('logfile','poltarget.log')    
    debug = kwargs.pop('debug',False)
      
  # get data
    infile = hdul.filename()
    arcname = infile.split('.')[0]
    arcimg = arcname[-4:]
    calhdr = hdul[0].header
    img_rc = hdul['SCI'].data
    var_rc = hdul['VAR'].data    
    okimg_rc = (hdul['BPM'].data == 0)
    okimg_rc &= (var_rc > 0.)
    rows, cols = img_rc.shape
    cbin, rbin = [int(x) for x in calhdr['CCDSUM'].split(" ")]
    rcbin_d = np.array([rbin,cbin])
    rccenter_d, cgapedge_d = ccdcenter(img_rc)    
    pixmm = 0.015
    pm1_p = np.array([1.,-1.])
    pp,ii = pidebug

    filter = calhdr['FILTER']
    camtem = calhdr['CAMTEM']
    coltem = calhdr['COLTEM']
    dateobs =  calhdr['DATE-OBS'].replace('-','')
    trkrho = calhdr['TRKRHO']
    grating = calhdr['GRATING'].strip()
    grang = calhdr['GR-ANGLE']
    artic = calhdr['CAMANG'] 
    lampid = calhdr['LAMPID']        
    RAd = Longitude(calhdr['RA']+' hours').degree
    DECd = Latitude(calhdr['DEC']+' degrees').degree
    PAd = calhdr['TELPA']  
    name = objectname+"_"+filter+"_maparc_"
    prows = rows/2
    pm1_p = np.array([1.,-1.])
    ur0,uc0,saltfps = rssdtralign(dateobs,trkrho)           # ur, uc =unbinned pixels, saltfps =micr/arcsec    
    yx0_d = -pixmm*np.array([ur0,uc0])                      # optical axis in mm
    dtrfps = saltfps*np.diff(RSScolpolcam(np.array([[0.,0.],[0.,1.]]),5500.,coltem,camtem)[1,0])[0] 
    imgoptfile = datadir+'RSSimgopt.txt'
    distTab = ascii.read(imgoptfile,data_start=1,   \
            names=['Wavel','Fcoll','Acoll','Bcoll','ydcoll','xdcoll','Fcam','acam','alfydcam','alfxdcam'])
    FColl6000 = distTab['Fcoll'][list(distTab['Wavel']).index(6000.)]
    FCam6000 = distTab['Fcam'][list(distTab['Wavel']).index(6000.)]
                
  # get lamp linelist
    lamplistdir = os.path.dirname(datadir[:-1])+"/linelists/"
    lampfileList = [os.path.basename(f) for f in glob.glob(lamplistdir+lampid+"*.txt")]
    lampfile = lamplistdir+lampid+".txt"    
    if lampfileList.count(lampid+"_"+grating[2:]+".txt"):
        lampfile = lamplistdir+lampid+"_"+grating[2:]+".txt"     
        
    oklamp_m = np.zeros(0,dtype=bool)        
    lampwav_m = np.zeros(0,dtype=float)
    lampint_m = np.zeros(0,dtype=float)
    for line in open(lampfile):
        rejectline = (line[0]=="#")        
        if (rejectline & (line[1:].split()[0].lower().islower())): continue     # skip comment lines
        oklamp_m = np.append(oklamp_m,(not rejectline))
        lampwav_m = np.append(lampwav_m,(float(line[int(rejectline):].split()[0])))
        lampint_m = np.append(lampint_m,(float(line[int(rejectline):].split()[1])))                        
    lamplines = lampwav_m.shape[0]

    if debug:                                                                   # lamp lines for spreadsheet
        np.savetxt("lampline_m.txt",np.vstack((range(lamplines),oklamp_m,lampwav_m,lampint_m)).T,   \
            fmt=" %3i %2i %10.4f %6.0f")

  # predict spectra
    wmin = max(float(filter[3:])-50.,cutwavoverride)
    wav_w = np.arange(3000.,10000.,100.)
    wav_w = np.insert(wav_w[wav_w > wmin],0,wmin)   
    wavs = wav_w.shape[0]
    YX00_di = np.array([[0.,],[0.,]])
    wav00_pic,rce00_pic,ok00_pic = slitpredict(hdul,YX00_di,wav_w,debug=False)
    col0 = cols/2 + yx0_d[1]/(cbin*pixmm)    
    usewav_c = (wav00_pic[0,0] > 0.)                              
    wav0 = np.interp(col0,np.where(usewav_c)[0],wav00_pic[0,0,usewav_c])    

    YX_di = np.array([slitTab['YCE'],slitTab['XCE']])
    entries = YX_di.shape[1]
    icenter = np.argmin((YX_di**2).sum(axis=0))    
    slitTab['CULL'] = np.array(entries*"",dtype='|S5')
    slitTab['CULL'][~oksig_i] = "NOSIG"    
         
    wav_pic,rce_pic,ok_pic = slitpredict(hdul,YX_di,wav_w,droff=droff_pi,sptilt=sptilt,debug=False)
        
    usewav_c = (wav_pic[0,icenter] > 0.)    
    disp0 = np.polyfit((np.arange(cols)-cols/2)[usewav_c],wav_pic[0,icenter,usewav_c],3)[2]                     
    arc_pic = np.zeros((2,entries,cols))
    var_pic = np.zeros((2,entries,cols))    

  # find center of short slits, r flexure first estimated from most central slit
  # for diffuse, assume slits centered on found spectra

    isshortslit_i = ((slitTab['LENGTH'] < 20.) & (slitTab['CULL'] == ""))
    if (isshortslit_i.sum()==0): exit()           # for now, punt if all longslit
    ishortArray = np.where(isshortslit_i)[0]      # _I is index in ishortArray    
    Icentershort = np.argmin((YX_di[:,isshortslit_i]**2).sum(axis=0)) 
    icentershort = ishortArray[Icentershort]
    Rows = int((slitTab['LENGTH'][icentershort]+3.)*dtrfps/(1000.*pixmm*rbin))
    Rows_pi = np.zeros((2,entries),dtype=int)    
    rflex_pi = np.zeros((2,entries)) 
            
    if isdiffuse:
        Rows_pi[:,:] = Rows
    else:
        startrflex = (arcflex(img_rc,rce_pic[0,icentershort],Rows)[0] +  \
                  arcflex(img_rc,rce_pic[1,icentershort],Rows)[0])/2
        rce_pic += startrflex
                        
        if debug:
            rflexfile = open('rflex.txt','w') 
            print >>rflexfile, "start rflex: ", startrflex     
            print >>rflexfile, "\n  p  i rflex rwidth   rmax crs cwidth "  

        rprofmax_pi = np.zeros((2,entries))             
        for p in (0,1):
            for i in ishortArray:
                Rows = int((slitTab['LENGTH'][i]+3.)*dtrfps/(1000.*pixmm*rbin))            
                rflex_pi[p,i], Rows_pi[p,i], rprofmax_pi[p,i] = arcflex(img_rc,rce_pic[p,i],Rows)
                rce_pic[p,i] += rflex_pi[p,i]                

  # for slits of same length, use same Rows_pi.  Add two bins on each side for safety
    lengthArray,idlength_I = np.unique(slitTab['LENGTH'][isshortslit_i],return_inverse=True)    
    for idlength in range(len(lengthArray)):
        iidArray = np.where(idlength_I == idlength)[0]
        Rows_pi[:,iidArray] = np.ceil(Rows_pi[:,iidArray].mean()).astype(int) + 4

  # form initial targetmap for arc extraction.  Collisions ok here, due to id process
    isslit_pirc = np.zeros((2,entries,rows,cols),dtype=bool)  
    targetmap_prc = np.zeros((2,rows,cols),dtype='uint8')   # slitmap with dropouts for badpix,ccd gaps,collisions
    r12_dpic = np.zeros((2,2,entries,cols))
    okwav_pic = (wav_pic > 0.)     
    mp_d = np.array([-1.,1.])
    r12_dpic[:,okwav_pic] = rce_pic[None,okwav_pic] +     \
        mp_d[:,None]*np.repeat(Rows_pi[:,:,None],cols,axis=2)[None,okwav_pic]/2.
    for p,i in np.ndindex(2,entries):
        rint12_dc = np.clip(np.round(r12_dpic[:,p,i]).astype(int),0,rows-1)                            
        for c in range(cols):
            isslit_pirc[p,i,rint12_dc[0,c]:(rint12_dc[1,c]+1),c] = True        
            if (not ok_pic[p,i,c]): continue
            targetmap_prc[p,rint12_dc[0,c]:(rint12_dc[1,c]+1),c] = i+1 
                       
  # find up to 100 strongest lines
    obscol_pil = np.zeros((2,entries,100))  
    obswav_pil = np.zeros((2,entries,100))
    colerr_pil = np.zeros((2,entries,100))      
    colerr1_pil = np.zeros((2,entries,100)) 
    relwidth_pil = np.zeros((2,entries,100))
    fmax_pil = np.zeros((2,entries,100))
    img_rc *= (targetmap_prc > 0).any(axis=0).astype(float) 
    var_rc *= (targetmap_prc > 0).any(axis=0).astype(float)
    iscr_pic = np.zeros((2,entries,cols),dtype=bool) 
                                  
    for p in (0,1):
        for i in ishortArray:                          
            rint_c = np.clip(np.round(rce_pic[p,i]).astype(int),0,rows-1)
                        
          # extract arc spectra
            for R in range(Rows_pi[p,i]):                            
                arc_c = img_rc[np.clip(rint_c+R-Rows_pi[p,i]/2,0,rows-1),range(cols)]
                ok_c = okimg_rc[np.clip(rint_c+R-Rows_pi[p,i]/2,0,rows-1),range(cols)]
                ok_pic[p,i] &= ok_c                                          
                arc_pic[p,i] += ok_c*arc_c
                var_pic[p,i] += ok_c*var_rc[np.clip(rint_c+R-Rows_pi[p,i]/2,0,rows-1),range(cols)]
              
          # remove background from arc, quadratic fit over mean in each ccd conaining a signal
            cedge_dC = np.array([[0,cgapedge_d[0]],list(cgapedge_d[1:3]),[cgapedge_d[3],cols]]).T
            ccenter_C = cedge_dC.mean(axis=0)
            bkg_C = np.zeros(3)
                
            for C in [0,1,2]:
                use_c = (ok_pic[p,i] & (range(cols)>cedge_dC[0,C]) & (range(cols)<cedge_dC[1,C]))             
                if (use_c.sum() == 0): continue
                bkg_C[C] = np.sort(arc_pic[p,i,use_c])[:(use_c.sum()/2)].mean()   # bkg is mean of lowest 50%
            arc_pic[p,i] -= np.polyval(np.polyfit(ccenter_C,bkg_C,2),range(cols))

          # flag cr's, by looking for 10x steps in row direction
            rint_Rc = rint_c[None,:] + np.arange(Rows_pi[p,i])[:,None] - int(Rows_pi[p,i]/2)
            rint_Rc = np.clip(rint_Rc,0,rows-1)
            img_Rc = np.clip(img_rc[rint_Rc,range(cols)],1.,1.e9)          
            ratimg_xc = img_Rc[1:]/img_Rc[:-1]
            iscrleft_c = ok_pic[p,i] & ((ratimg_xc > 10.) &   \
                (img_Rc[1:] > 20.*np.sqrt(var_rc[rint_Rc,range(cols)][1:]))).any(axis=0)
            iscrright_c = ok_pic[p,i] & ((1./ratimg_xc > 10.) &   \
                (img_Rc[:-1] > 20.*np.sqrt(var_rc[rint_Rc,range(cols)][:-1]))).any(axis=0)                           
            iscr_pic[p,i] = (iscrleft_c | iscrright_c)
            
            ok_pic[p,i] &= np.logical_not(iscr_pic[p,i])
            usewav_c = (wav_pic[p,i] > 0.)
            for R in range(Rows_pi[p,i]):
                okimg_rc[np.clip(rint_c[usewav_c]+R-Rows_pi[p,i]/2,0,rows-1),usewav_c] &= ok_pic[p,i,usewav_c]

         # find lines
            label = ['',arcimg+'_'+str(p)+'_'+str(i)][(i==ii)]                       
            obscol_pil[p,i],colerr_pil[p,i],fitwidth,relwidth_pil[p,i],fmax_pil[p,i] =  \
                findlines(arc_pic[p,i],var_pic[p,i],ok_pic[p,i],label=label)
        
            obswav_pil[p,i] = np.interp(obscol_pil[p,i],np.where(usewav_c)[0],wav_pic[p,i,usewav_c])
            obswav_pil[p,i] *= (obscol_pil[p,i] > 0.)

    okline_pil = (fmax_pil > 0.)      

  # initial line identification 
  #   Bright lines only. Use 5/15 brightest lamp/obs lines left, 5/15 brightest right
  #   Compute FOV-average corrections

    brtfitdeg = 2  
    stdfit_pi = np.zeros((2,entries))
    brtfit_dpi = np.zeros((brtfitdeg+1,2,entries))
    oewaverr_i = np.zeros(entries)
    okbrtfit_pi = np.zeros((2,entries),dtype=bool)
    mbrtid_pil = -np.ones((2,entries,100),dtype=int)
       
    for i in ishortArray:
        for p in (0,1):
          # determine which lamp and obs lines to use
            wavmin = wav_pic[p,i,ok_pic[p,i]].min()         
            wavmax = min(9000.,wav_pic[p,i,ok_pic[p,i]].max())              # response way down beyond 9000
            isleft_m = oklamp_m & ((lampwav_m > wavmin) & (lampwav_m < (wavmin + wavmax)/2.))       
            isright_m = oklamp_m & ((lampwav_m > (wavmin + wavmax)/2.) & (lampwav_m < wavmax))
            usewav_c = (wav_pic[p,i] > 0.)                                
            col_m = np.round(np.interp(lampwav_m,wav_pic[p,i,usewav_c],np.where(usewav_c)[0])).astype(int)
            uselamp_m = oklamp_m & ok_pic[p,i,col_m]           
            lampintlimleft = np.sort((uselamp_m*lampint_m)[isleft_m])[-5]        
            lampintlimright = np.sort((uselamp_m*lampint_m)[isright_m])[-5]        
            oklampbrt_m = (uselamp_m & isleft_m &(lampint_m >= lampintlimleft)) |    \
                         (uselamp_m & isright_m &(lampint_m >= lampintlimright))     
            isleft_l = okline_pil[p,i] & ((obswav_pil[p,i] > wavmin) & (obswav_pil[p,i] < (wavmin + wavmax)/2.))
            isright_l = okline_pil[p,i] & ((obswav_pil[p,i] > (wavmin + wavmax)/2.) & (obswav_pil[p,i] < wavmax))
            okobsbrt_l = np.in1d(range(100),list(np.where(isleft_l)[0][:15])+list(np.where(isright_l)[0][:15]))
          # identify by finding most frequent waverr = obs - lamp                                  
            waverr_ML = (obswav_pil[p,i][okobsbrt_l][None,:] - lampwav_m[oklampbrt_m][:,None])
            laMplines,obsLines = oklampbrt_m.sum(),okobsbrt_l.sum()
            maxwaverr = 30.*disp0                  # first median waverr guess is highest peak of 2 staggered histograms
            bin_ad = np.arange(-maxwaverr,6.*maxwaverr/5.,maxwaverr/5.)[None,:] + np.array([0, maxwaverr/10.])[:,None]
            histcount_ad = np.array([np.histogram(waverr_ML[np.abs(waverr_ML) < maxwaverr],bins=bin_ad[a])[0] 
                for a in (0,1)])
            amax,dmax = np.unravel_index(np.argmax(histcount_ad),histcount_ad.shape)                       
            medwaverr0 = (np.argmax(histcount_ad[amax])- 4.5 + amax/2.)*maxwaverr/5.                             
            Lbest_M = np.argsort(np.abs(waverr_ML - medwaverr0),axis=1)[:,0] # tentative ID
            waverr_M = waverr_ML[range(laMplines),Lbest_M]
            use_M = np.ones(laMplines,dtype=bool)                
            Lsorted_q,q_M,Mcount_q  = np.unique(Lbest_M,return_inverse=True,return_counts=True)                                                 
            if (Mcount_q.max() > 1):                                        # for dup id's, pick closer line                  
                badqArray = np.where(Mcount_q > 1)[0]
                medwaverr = np.median(waverr_M[np.in1d(q_M,badqArray,invert=True)])
                for q in badqArray:
                    badMList = list(np.where(q_M==q)[0])
                    bestM = badMList[np.argsort(np.abs(waverr_M - medwaverr)[badMList])[0]]
                    badMList.remove(bestM)                    
                    use_M[badMList] = False
            medwaverr = np.median(waverr_M[use_M])           
            use_M &= (np.abs(waverr_M - medwaverr) < 4.*disp0)
            l_L = np.where(okobsbrt_l)[0]
            mbrtid_pil[p,i,l_L[Lbest_M[use_M]]] = np.where(oklampbrt_m)[0][use_M]

            if (i==ii):            
                np.savetxt("waverr_"+arcimg+"_"+str(p)+"_"+str(i)+"_M.txt",np.vstack((np.where(oklampbrt_m)[0], \
                    lampwav_m[oklampbrt_m],use_M,Lbest_M,l_L[Lbest_M],waverr_M)).T,fmt=" %3i %9.2f %2i %3i %3i %8.2f ")                
                np.savetxt("waverr_"+arcimg+"_"+str(p)+"_"+str(i)+"_ML.txt",waverr_ML,fmt="%9.2f ")
                np.savetxt("lampwav_"+arcimg+"_"+str(p)+"_"+str(i)+"_M.txt",lampwav_m[oklampbrt_m].T,fmt="%8.2f ")
                np.savetxt("obswav_"+arcimg+"_"+str(p)+"_"+str(i)+"_L.txt",np.vstack((obswav_pil[p,i][okobsbrt_l],
                    obscol_pil[p,i][okobsbrt_l],fmax_pil[p,i][okobsbrt_l],isright_l[okobsbrt_l].astype(int))).T,    \
                    fmt="%8.2f %8.2f %10.1f %2i")
          # linear fit of waverr vs wavelength, using Legendre -1,1 scaling                                             
            okbrtfit_pi[p,i] = (use_M.sum() > 7)           
            if (not okbrtfit_pi[p,i]): continue
            legx_M = (lampwav_m[oklampbrt_m]-wav0)/(disp0*cols/2)
            brtfit_dpi[:,p,i] = np.polyfit(legx_M[use_M],waverr_M[use_M],brtfitdeg)
            stdfit_pi[p,i] = np.std(np.polyval(brtfit_dpi[:,p,i],legx_M[use_M]) - waverr_M[use_M])
            
      # update the bright id's to only include those seen in both O,E
        mbrtidi_pl = mbrtid_pil[:,i,:]                  # this is a view
        countbrt_m = np.array([np.in1d(range(lamplines),mbrtidi_pl[p]) for p in (0,1)]).sum(axis=0)
        for m in np.where(countbrt_m != 2)[0]:
            mbrtidi_pl[mbrtidi_pl == m] = -1

    okbrtfit_i = okbrtfit_pi.all(axis=0)       
    stdfit_i = np.sqrt((stdfit_pi**2).mean(axis=0))
    oneokiArray = np.where(okbrtfit_pi.sum(axis=0)==1)[0]        # fallback for O or E bad brtfit: use other   
    if len(oneokiArray):                                                         
        pokiArray = np.where(okbrtfit_pi[:,oneokiArray])[0]                       
        brtfit_dpi[:,(1-pokiArray),oneokiArray] = brtfit_dpi[:,pokiArray,oneokiArray]      
        
    oewaverr_i = okbrtfit_i*np.diff(brtfit_dpi[brtfitdeg],axis=0)[0]

    wavoff = brtfit_dpi[brtfitdeg,okbrtfit_pi].mean()
    yxwavoff = oewaverr_i[okbrtfit_i].mean()
    brtfit_di = brtfit_dpi.mean(axis=1)
    if (okbrtfit_i.sum() >1):
        dwavoffdX, wavoff = np.polyfit(YX_di[1,okbrtfit_i],brtfit_di[brtfitdeg,okbrtfit_i],1)[0:2]
        yxwavoff = np.polyfit(YX_di[1,okbrtfit_i],oewaverr_i[okbrtfit_i],1)[1] 
        wavoff_i = wavoff + dwavoffdX*YX_di[1]                   # fallback for O and E bad brtfit: use X fit wavoff
        noneokiArray = np.where(isshortslit_i & (okbrtfit_pi.sum(axis=0)==0))[0]        
        if len(noneokiArray):                          
            brtfit_dpi[brtfitdeg,:,noneokiArray] = wavoff_i[noneokiArray]  
                                 
    articerr = np.degrees(cbin*pixmm*(wavoff/disp0)/FCam6000)
    yxOEoff_d = cbin*pixmm*np.array([0.,yxwavoff/disp0])   

    rsslog.message(("\narticulation error (deg): %8.4f" % articerr),logfile)
    rsslog.message(("beamsplitter rotation (mm)   : %8.4f" % yxOEoff_d[1]),logfile)    

  # re-id with obswav corrected with quadratic fit to bright line dwav vs wav
  # adjust for mean wav error    
    if debug:
        np.savetxt("ok_"+arcimg+"_pic.txt",np.vstack((range(cols),ok_pic.reshape((-1,cols)))).T,    \
            fmt=" %4i "+2*entries*"%2i ")
        np.savetxt("arccor_pic.txt",np.vstack((range(cols),wav_pic.reshape((-1,cols)), \
            arc_pic.reshape((-1,cols)))).T,fmt="%4i "+2*entries*"%10.3f "+2*entries*"%9.2f ")
        np.savetxt("obscol1_pil.txt",np.vstack((obscol_pil.reshape((-1,100)),  \
            colerr_pil.reshape((-1,100)),fmax_pil.reshape((-1,100)),obswav_pil.reshape((-1,100)))).T,   \
            fmt=(2*entries*"%8.2f "+2*entries*"%6.4f "+2*entries*"%8.1f "+2*entries*"%8.2f "))    
        np.savetxt("brtfit_"+arcimg+"_dpi.txt",np.vstack((np.indices((2,entries)).reshape((2,-1)),  \
            okbrtfit_pi.flatten(),brtfit_dpi.reshape((brtfitdeg+1,-1)),stdfit_pi.flatten())).T,    \
            fmt=" %2i %3i %2i "+(brtfitdeg+2)*"%8.4f ")
        np.savetxt("brtfit_"+arcimg+"_di.txt",np.vstack((range(entries),okbrtfit_i,brtfit_di,oewaverr_i,stdfit_i)).T,    \
            fmt=" %2i %2i "+(brtfitdeg+3)*"%8.4f ")         
        np.savetxt("isarccr_"+arcimg+"_pic.txt",np.vstack((range(cols),iscr_pic.reshape((2*entries,-1)))).T,    \
            fmt=" %4i "+2*entries*"%2i ")

    wavoff_piw = np.zeros((2,entries,wavs))
    for p,i in np.ndindex(2,entries):
        legx_w = (wav_w-wav0)/(disp0*cols/2)        
        wavoff_piw[p,i] = np.polyval(brtfit_dpi[:,p,i],legx_w)    
    
    wav_pic,rce1_pic,ok1_pic =    \
        slitpredict(hdul,YX_di,wav_w,droff=droff_pi,sptilt=sptilt,wavoff=wavoff_piw,debug=False)

    usewav_c = (wav_pic[0,icenter] > 0.)    
    disp0 = np.polyfit((np.arange(cols)-cols/2)[usewav_c],wav_pic[0,icenter,usewav_c],3)[2]  

  # get predicted spectrum dwav/dr for wavmap             
    YXoff_di = YX_di+np.array([0.1,0.])[:,None]
    wav2_pic,rce2_pic,ok2_pic =    \
        slitpredict(hdul,YXoff_di,wav_w,droff=droff_pi,sptilt=sptilt,wavoff=wavoff_piw,debug=False)    

    okwav_pic = ((wav_pic > 0.) & (wav2_pic > 0.))  # allow for endpoint rounding
    ok_pic &= okwav_pic
    wav_pic *= okwav_pic
    wav2_pic *= okwav_pic
    dwavdr_pic = (wav2_pic - wav_pic)/(rce2_pic - rce1_pic)
    
  # locate good id's, with cubic legendre fit wav vs col separately for O,E using full spectrum                       
    obswav_pil = np.zeros((2,entries,100))            
    stdfit_pi = np.zeros((2,entries))
    corfit_pid = np.zeros((2,entries,4))
    obscol_pim = np.zeros((2,entries,lamplines))
    docull_im = np.ones((entries,lamplines),dtype=bool)    
    colerr_pim = np.zeros_like(obscol_pim) 
    waverr_pim = np.zeros_like(obscol_pim)    
    fiterr_pim = np.zeros_like(obscol_pim)
    fiterr_pic = np.zeros((2,entries,cols))                   
    prederr_pim = np.zeros_like(obscol_pim)
    oklampreject_im = np.zeros((entries,lamplines),dtype=bool)      # for lamp linelist optimization
    cleg0_pi = np.zeros((2,entries))        # legfit center column is position of wav0
    cullmeanerrLog_pi = np.empty((2,entries),dtype=object)
    cullmaxerrLog_pi = np.empty((2,entries),dtype=object)
                           
    for i in ishortArray:
        debugname = ['',(arcimg+("_%02i" % ii))][int(i==ii)]
        m_b,obswav_pb,obscol_pb,l_pb,ok_b =     \
            lineid(calhdr,lampwav_m,oklamp_m,okline_pil[:,i],mbrtid_pil[:,i],obscol_pil[:,i],fmax_pil[:,i],    \
                wav_pic[:,i],logfile=logfile,debugname=debugname)        
         
        okfit_pb = np.zeros_like(obswav_pb,dtype=bool)
        daterr_pb = np.zeros_like(obswav_pb)        
        fiterr_pb = np.zeros_like(obswav_pb)
        disp_pb = np.zeros_like(obswav_pb)
        for p in (0,1):
            usewav_c = (wav_pic[p,i] > 0.)                              
            obswav_pil[p,i] = np.interp(obscol_pil[p,i],np.where(usewav_c)[0],wav_pic[p,i,usewav_c])
            cleg0_pi[p,i] = np.interp(wav0,wav_pic[p,i,usewav_c],np.where(usewav_c)[0])
            legx_b = (obscol_pil[p,i,l_pb[p]]-cleg0_pi[p,i])/(cols/2)
            xlim_d = (np.where(usewav_c)[0][[0,-1]]-cleg0_pi[p,i])/(cols/2)
            
            yerr_b = colerr_pil[p,i,l_pb[p]]
            docull_b = (mbrtid_pil[p,i,l_pb[p]] == -1)
            debugname = ['',("legfitdebug_"+str(p)+"_"+arcimg+("_%02i" % ii))][int(i==ii)]                                                                               
            corfit_pid[p,i],okfit_pb[p],daterr_pb[p],fiterr_pb[p],fiterr_X,cullLog =  \
                legfit_cull(legx_b,lampwav_m[m_b],ok_b,3,   \
                    xlim_d=xlim_d,yerr=yerr_b,docull=docull_b,IQcull=2.,maxerr=2.*disp0,debugname=debugname)
            cullmeanerrLog_pi[p,i] = cullLog[0]
            cullmaxerrLog_pi[p,i] = cullLog[1]                               
            c_X = (cols/2)*np.linspace(xlim_d[0],xlim_d[1]) + cleg0_pi[p,i]                     
            fiterr_pic[p,i,usewav_c] = np.interp(np.where(usewav_c)[0],c_X,fiterr_X)                                                                                
            disp_pb[p] = np.polynomial.legendre.legval(legx_b+2./cols,corfit_pid[p,i]) - \
                np.polynomial.legendre.legval(legx_b,corfit_pid[p,i])

        okfit_b = okfit_pb.all(axis=0)  # must be in both O and E
        oklampreject_im[i,m_b] = (np.logical_not(oklamp_m[m_b]) &   \
            (np.abs((obswav_pb-lampwav_m[None,m_b]).mean(axis=0)) < 2.*disp0))
        docull_im[i,m_b] = ((mbrtid_pil[0,i,l_pb[0]] == -1) | (mbrtid_pil[1,i,l_pb[1]] == -1))     
        obscol_pim[:,i,m_b[okfit_b]] = obscol_pb[:,okfit_b]
        colerr_pim[0,i,m_b[okfit_b]] = colerr_pil[0,i,l_pb[0,okfit_b]]     
        colerr_pim[1,i,m_b[okfit_b]] = colerr_pil[1,i,l_pb[1,okfit_b]]            
        waverr_pim[:,i,m_b[okfit_b]] = daterr_pb[:,okfit_b]  
        fiterr_pim[:,i,m_b] = fiterr_pb                       
        prederr_pim[:,i,m_b[okfit_b]] = disp_pb[0,okfit_b]*colerr_pim[:,i,m_b[okfit_b]]

    fiterrmean_pi = np.zeros((2,entries))
    fiterrmean_pi[:,ishortArray] =  \
        fiterr_pic[:,ishortArray].sum(axis=2)/(fiterr_pic[:,ishortArray]>0).sum(axis=2)
    fiterrmax_pi = fiterr_pic.max(axis=2)

  # flag as bad all id'd lines that give systematically bad fit errors
    isid_im = (waverr_pim[0] != 0.)

    idlines_m = isid_im.sum(axis=0)
    m1_M = np.where(idlines_m>0)[0]      # the arc lines id'd somewhere
            
    idlines = m1_M.shape[0]    
    sumerr_m = waverr_pim.sum(axis=(0,1))
    meanerr_M = sumerr_m[m1_M]/(2.*idlines_m[m1_M])
    sum2err_m = (waverr_pim**2).sum(axis=(0,1))    
    rmserr_M = np.sqrt(sum2err_m[m1_M]/(2.*idlines_m[m1_M]) - meanerr_M**2)

    Q1,Q3 = np.percentile(meanerr_M,(25.,75.))
    errfence_d = np.array([Q1,Q3]) - 2.*pm1_p*(Q3-Q1)
    isfenced_M = ((docull_im[:,m1_M]).all(axis=0) & ((meanerr_M < errfence_d[0]) | (meanerr_M > errfence_d[1])))
    rejectlineArray = lampwav_m[m1_M[np.where(isfenced_M)[0]]]
    allrejects = len(rejectlineArray)
    isid_im[:,m1_M[np.where(isfenced_M)[0]]] = False
    m2_M = np.where(isid_im.sum(axis=0)>0)[0]
    
  # flag as bad at most 2 lines in a target with oe difference large compared to predicted error
    oechi2_im = np.zeros((entries,lamplines))
    doewaverr_im = np.diff(waverr_pim,axis=0)[0]
    prederr_im = np.sqrt((prederr_pim**2).sum(axis=0))
    oechi2_im[isid_im] = (doewaverr_im[isid_im]/prederr_im[isid_im])**2
    dum,dum,dum,chihi = fence(oechi2_im[isid_im])    
    isoehigh_im = (docull_im & (oechi2_im > chihi))
    oecull_im = np.zeros_like(isid_im)
    
    for i in range(entries):
        if (isoehigh_im[i].sum()==0): continue        
        McullArray = np.argsort((isoehigh_im*oechi2_im)[i])[::-1]       
        McullArray = McullArray[:min(2,isoehigh_im[i].sum())] 
        oecull_im[i,McullArray] = True
    singlerejects = oecull_im.sum()
    
    rsslog.message((("\nrejected lines (all targets): "+allrejects*"%8.2f ") % tuple(rejectlineArray)),logfile)
    rsslog.message((("rejected lines (one target): %2i") % singlerejects),logfile)
    
    if debug:
        np.savetxt("arccor1_pic.txt",np.vstack((range(cols),wav_pic.reshape((-1,cols)), \
            arc_pic.reshape((-1,cols)),dwavdr_pic.reshape((-1,cols)))).T,   \
            fmt="%4i "+2*entries*"%10.3f "+2*entries*"%8.2f "+2*entries*"%8.5f ")
        np.savetxt("obscol2_pil.txt",np.vstack((obscol_pil.reshape((-1,100)),  \
            colerr_pil.reshape((-1,100)),fmax_pil.reshape((-1,100)),obswav_pil.reshape((-1,100)))).T,   \
            fmt=(2*entries*"%8.2f "+2*entries*"%6.4f "+2*entries*"%8.1f "+2*entries*"%8.2f "))
        np.savetxt("err1_M.txt",np.vstack((lampwav_m[m1_M],meanerr_M,rmserr_M,idlines_m[m1_M])).T,   \
            fmt="%8.2f %8.3f %8.3f %3i")                                        
        np.savetxt("waverr0p0_M.txt", np.vstack((m1_M,lampwav_m[m1_M],waverr_pim[0,:,m1_M].T, \
            prederr_pim[0,:,m1_M].T,fiterr_pim[0,:,m1_M].T)).T,fmt="%3i "+(3*entries+1)*"%8.3f ")
        np.savetxt("waverr0p1_M.txt", np.vstack((m1_M,lampwav_m[m1_M],waverr_pim[1,:,m1_M].T, \
            prederr_pim[1,:,m1_M].T,fiterr_pim[1,:,m1_M].T)).T,fmt="%3i "+(3*entries+1)*"%8.3f ")    
        np.savetxt("oechi2.txt",np.vstack((np.array(np.where(isid_im)),isoehigh_im[isid_im],oecull_im[isid_im],   \
            doewaverr_im[isid_im],prederr_im[isid_im],oechi2_im[isid_im])).T,fmt=4*"%3i "+3*"%8.3f ")
        np.savetxt("fiterr0_pic.txt",fiterr_pic.reshape((2*entries,-1)).T,fmt="%8.4f ")
        meanfile = open("cullmeanerrLog_"+arcimg+".txt",'w')
        maxfile = open("cullmaxerrLog_"+arcimg+".txt",'w')        
        for p,i in np.ndindex(2,entries):
            culls = len(cullmeanerrLog_pi[p,i]) 
            print >>meanfile, (" %2i %2i "+culls*"%8.3f ") % ((p,i,)+tuple(cullmeanerrLog_pi[p,i]))
            print >>maxfile, (" %2i %2i "+culls*"%8.3f ") % ((p,i,)+tuple(cullmaxerrLog_pi[p,i]))            
        shutil.copyfile(lampfile,"lamplines_"+arcimg+".txt")
  
    isid_im[oecull_im] = False 
    idlines_i = isid_im.sum(axis=1)    
    m3_M = np.where(isid_im.sum(axis=0)>0)[0]
    
  # cull out targets with too few line id'd in arc
    minarclines = 8
    icullarcArray = np.where(isshortslit_i & (idlines_i < minarclines))[0]
    if len(icullarcArray): rsslog.message("\n",logfile)
    for i in icullarcArray:    
        rsslog.message(("Target %2i culled, only %2i id'd arc lines" % (i,idlines_i[i])),logfile)
        slitTab['CULL'][i] = "arc"                       
        ishortArray = np.delete(ishortArray,np.where(ishortArray==i)[0][0])      

  # Final Legendre weighted polynomial fit to id'd lines for O and E, form wavmaps
  # use lstsq to get errors on Legendre coefficients
    coflegpred_pid = np.zeros((2,entries,4))      
    wavmap_prc = np.zeros((2,rows,cols))
    cofleg_pid = np.zeros((2,entries,4))
    wavchi2_pi = np.zeros((2,entries))
    legx_pic = (np.arange(cols,dtype=float)[None,None,:]-cleg0_pi[:,:,None])/(cols/2)
    fiterr_pic.fill(0.)    
    waverr_pim.fill(0.)  
    wavlim_dpi = np.zeros((2,2,entries))
    limitrange_i = np.zeros(entries,dtype=bool)
    
    badarc = 2.0
    
    for i in ishortArray:
        for p in (0,1):
            okpred_c = ok1_pic[p,i]                    
            a_Cd = np.vstack((np.ones(cols)[okpred_c], legx_pic[p,i,okpred_c],  \
                np.polyval(legFn(2),legx_pic[p,i])[okpred_c],  \
                np.polyval(legFn(3),legx_pic[p,i])[okpred_c])).T
            coflegpred_pid[p,i] = la.lstsq(a_Cd,wav_pic[p,i,okpred_c])[0]    # predicted leg cofs for obswav corrected

            use_m = isid_im[i]        
            legx_M = (obscol_pim[p,i,use_m] - cleg0_pi[p,i,None])/(cols/2)            
            usewav_c = (wav_pic[p,i] > 0.)              
            xlim_d = (np.where(usewav_c)[0][[0,-1]]-cleg0_pi[p,i])/(cols/2)
            yerr_M = prederr_pim[p,i,use_m]
            ok_M = np.ones_like(yerr_M,dtype=bool)
            debugname = ['',("legfinaldebug_"+str(p)+"_"+arcimg+("_%02i" % ii))][int(i==ii)]                                      
            cofleg_pid[p,i],okfit_M,daterr_M,fiterr_M,fiterr_X,dum =  \
                legfit_cull(legx_M,lampwav_m[use_m],ok_M,3,   \
                    xlim_d=xlim_d,yerr=yerr_M,docull=False)

            if debugname:
                np.savetxt(debugname+".txt",np.vstack((np.where(use_m),lampwav_m[use_m],legx_M,daterr_M,yerr_M,fiterr_M)).T,  \
                    fmt = " %3i %8.2f %8.4f "+3*"%8.3f ")
                
            c_X = (cols/2)*np.linspace(xlim_d[0],xlim_d[1]) + cleg0_pi[p,i]                     
            fiterr_pic[p,i,usewav_c] = np.interp(np.where(usewav_c)[0],c_X,fiterr_X)
            badcol_c = (fiterr_pic[p,i] > badarc*disp0)
            limitrange_i[i] |= badcol_c.any()
            wav_pic[p,i,badcol_c] = 0.
            wavlim_dpi[:,p,i] = (wav_pic[p,i,np.where(wav_pic[p,i] > 0)[0][0]], wav_pic[p,i].max())
            fiterr_pic[p,i,badcol_c] = 0.                                  
            disp_M = np.polynomial.legendre.legval(legx_M+2./cols,cofleg_pid[p,i]) -   \
                np.polynomial.legendre.legval(legx_M,cofleg_pid[p,i])
            waverr_pim[p,i,use_m] = daterr_M                 
            prederr_pim[p,i,use_m] = disp_M*colerr_pim[p,i,use_m]        
            wavchi2_pi[p,i] = ((daterr_M/prederr_pim[p,i,use_m])**2).mean()
            wav_rc = np.tile(np.polynomial.legendre.legval(legx_pic[p,i],cofleg_pid[p,i]),(rows,1))         
            isslit_rc = (isslit_pirc[p,i] & okwav_pic[None,p,i])

            istarget_rc = ((targetmap_prc[p]==i+1) & okwav_pic[None,p,i])            
            targetmap_prc[p,istarget_rc] = i+1
            wavmap_prc[p,isslit_rc] =  (wav_rc +   \
                (np.arange(rows)[:,None] - rce1_pic[p,i][None,:])*dwavdr_pic[p,i][None,:])[isslit_rc]                       

    fiterrmean_pi = np.zeros((2,entries))
    fiterrmean_pi[:,ishortArray] =  \
        fiterr_pic[:,ishortArray].sum(axis=2)/(fiterr_pic[:,ishortArray]>0).sum(axis=2)
    fiterrmax_pi = fiterr_pic.max(axis=2)
    wavlim_di = np.array([wavlim_dpi[0].max(axis=0),wavlim_dpi[1].min(axis=0)])
             
    rsslog.message("\nWavelength Fit errors(Ang), O, E: ",logfile)
    rsslog.message(("catidx "+2*"  rms   chi2   "+"lines  wavel range"),logfile)
    for i in sorted(list(ishortArray)+list(icullarcArray)):
        rsslog.message(("%4i  "+2*("%6.3f %6.1f  ")+"%4i  "+2*"%6.0f "+" %s") %     \
            ((i,)+tuple(np.vstack((fiterrmean_pi[:,i],wavchi2_pi[:,i])).T.flatten())+   \
            (idlines_i[i],)+tuple(wavlim_di[:,i])+(['','reduced range'][limitrange_i[i]],)),logfile)

    if debug:     
        np.savetxt("isid_"+arcimg+".txt",np.vstack((m3_M,lampwav_m[m3_M],isid_im[:,m3_M])).T,   \
            fmt=" %3i %8.3f "+entries*"%1i ")  
        isbrtid_im = np.array([np.in1d(range(lamplines),mbrtid_pil[0,i,:]) for i in range(entries)])
        mbrtArray = np.where(isbrtid_im.any(axis=0))[0]
        np.savetxt("isbrtid_"+arcimg+".txt",np.vstack((mbrtArray,lampwav_m[mbrtArray],isbrtid_im[:,mbrtArray])).T,    \
            fmt=" %3i %8.3f "+entries*"%1i ") 
        np.savetxt("wavleg_pi_"+arcimg+".txt",np.vstack((np.indices((2,entries)).reshape((2,-1)),  \
            cleg0_pi.flatten(),cofleg_pid.reshape((2*entries,-1)).T,  \
            coflegpred_pid.reshape((2*entries,-1)).T,wavchi2_pi.flatten(),
            fiterrmean_pi.flatten(),fiterrmax_pi.flatten())).T, fmt="%3i %3i %4i "+11*"%8.3f ")              
        np.savetxt("waverr1p0_M.txt", np.vstack((m3_M,lampwav_m[m3_M],waverr_pim[0,:,m3_M].T, \
            prederr_pim[0,:,m3_M].T,fiterr_pim[0,:,m3_M].T)).T,fmt="%3i "+(3*entries+1)*"%8.3f ")
        np.savetxt("waverr1p1_M.txt", np.vstack((m3_M,lampwav_m[m3_M],waverr_pim[1,:,m3_M].T, \
            prederr_pim[1,:,m3_M].T,fiterr_pim[1,:,m3_M].T)).T,fmt="%3i "+(3*entries+1)*"%8.3f ")      
        np.savetxt("fiterr1_pic.txt",fiterr_pic.reshape((2*entries,-1)).T,fmt="%8.4f ")

  # create extraction targetmap, zeroing mutual collisions      
    targetmap_prc, r12_dpic = targetmap(rce_pic,ok_pic,Rows_pi,rce_pic,ok_pic,Rows_pi,rows)

  # zero out targetmap immediately around second order, if necessary
    wavsec_pic,rcesec_pic,oksec_pic =     \
        slitpredict(hdul,YX_di,wav_w,order=2,droff=droff_pi,sptilt=sptilt,debug=False)      
        
    if oksec_pic.sum():
        rsslog.message("Second order possible in %i wavelengths" % oksec_pic.sum(),logfile) 
        targetmapsec_prc, r12sec_dpic =     \
            targetmap(rce_pic,ok_pic,Rows_pi,rcesec_pic,oksec_pic,Rows_pi/3,rows)        
        targetmap_prc *= (targetmapsec_prc > 0)
        r12_dpic[0] = np.maximum(r12_dpic[0],r12sec_dpic[0])
        r12_dpic[1] = np.minimum(r12_dpic[1],r12sec_dpic[1])
        issecond_prc = np.zeros((2,rows,cols),dtype=bool)       # for debug fits         
        for p,i in np.ndindex(2,entries):
            rint_c = np.clip(np.round(rcesec_pic[p,i]).astype(int),0,rows-1).astype(int)
            issecond_prc[p,rint_c[oksec_pic[p,i]],np.arange(cols)[oksec_pic[p,i]]] = True 
        if debug:
            hdusmap = pyfits.PrimaryHDU(issecond_prc.astype('uint8'))
            hdusmap = pyfits.HDUList([hdusmap])
            hdusmap.writeto("issecond_prc.fits",overwrite=True)            
                                
  # zero targetmap where fewer than 75% of Rows are now left
    isbadRows_pic = (np.diff(r12_dpic,axis=0)[0] < 0.75*Rows_pi[:,:,None])

    if debug:            
        hdut0map = pyfits.PrimaryHDU(targetmap_prc.astype('uint8'))
        hdut0map = pyfits.HDUList([hdut0map])
        hdut0map.writeto("targetmap0_prc.fits",overwrite=True)      
    for p,i in np.ndindex(2,entries):
        isbadRows_rc = (isbadRows_pic[None,p,i,:] & (targetmap_prc[p] == i+1))        
        targetmap_prc[p,isbadRows_rc] = 0
        ok_pic[p,i,isbadRows_rc.any(axis=0)] = False
    if debug:            
        hdut1map = pyfits.PrimaryHDU(targetmap_prc.astype('uint8'))
        hdut1map = pyfits.HDUList([hdut1map])
        hdut1map.writeto("targetmap1_prc.fits",overwrite=True)
              
  # split O and E, write out arc tm file  
    yx0_dp, rshift, yxp0_dp, isfov_rc = RSSpolgeom(hdul,wav0,yxOEoff_d=yxOEoff_d)
    yxp0_dp[0] += cbin*pixmm*np.median(droff_pi)
    rshift += int(np.round(np.median(droff_pi)))

  # allow for overlap of O and E beams
    rovlapO = max(0,np.where(targetmap_prc[0] > 0)[0][-1] - (prows + rshift) + 1)
    rovlapE = max(0,(prows + rshift) - np.where(targetmap_prc[1] > 0)[0][0] + 1)
    rshift_p = rshift + np.array([rovlapO,-rovlapE])
      
    rows = 2*prows                          # allow for input with odd rows 
    image_prc = np.zeros((2,prows,cols))
    var_prc = np.zeros_like(image_prc)
    gaincor_c = gaincor(hdul)        
    image_rc = hdul['SCI'].data/gaincor_c[None,:]    
    image_prc = shift(image_rc[:rows],(-rshift,0),order=0).reshape((2,prows,cols))
    var_rc = hdul['VAR'].data/gaincor_c[None,:]**2
    var_prc = shift(var_rc[:rows],(-rshift,0),order=0).reshape((2,prows,cols))
    okimg_prc = shift(okimg_rc[:rows],(-rshift,0),order=0).reshape((2,prows,cols))
    for p in (0,1):                         # maps should show no OE overlaps
        targetmap_prc[p,:rows] = shift(targetmap_prc[p,:rows],(-rshift_p[p],0),order=0)   
        wavmap_prc[p,:rows] = shift(wavmap_prc[p,:rows],(-rshift_p[p],0),order=0)
    targetmap_prc = targetmap_prc[:,:rows].reshape((4,prows,-1))[[0,3]]
    wavmap_prc = wavmap_prc[:,:rows].reshape((4,prows,-1))[[0,3]]    

    YXAXISO = ("%7.4f %7.4f" % tuple(yxp0_dp[:,0]))
    YXAXISE = ("%7.4f %7.4f" % tuple(yxp0_dp[:,1]))
    RSPLIT = ("%3i %3i" % tuple(rshift_p))
    CALIMG = infile.split('.')[0][-4:]
            
    hdrList = [YXAXISO,YXAXISE]       
    hdul['SCI'].data = image_prc.astype('float32')
    hdul['VAR'].data = var_prc.astype('float32')
    hdul['BPM'].data = (~okimg_prc).astype('uint8')     
    calhdr['YXAXISO'] = (YXAXISO,"O Optic Axis (mm)")
    calhdr['YXAXISE'] = (YXAXISE,"E Optic Axis (mm)")
    calhdr['RSPLIT'] = (RSPLIT,"row of O, E split")
    calhdr['CALIMG'] = (CALIMG,"cal image no(s)")
    calhdr['REFWAV'] = (wav0,"central wav (Ang)")              
    hdul.append(pyfits.ImageHDU(data=targetmap_prc.astype('uint8'),name='TMAP'))
    hdul.append(pyfits.ImageHDU(data=wavmap_prc.astype('float32'),name='WMAP'))
    hdul.append(pyfits.ImageHDU(data=fiterr_pic.astype('float32'),name='WERR'))        
    hdul.append(pyfits.table_to_hdu(slitTab))
    hdul[-1].header['EXTNAME'] = 'TGT'
    hdul.writeto("t"+infile,overwrite=True)
    rsslog.message(('\nOutput file '+'t'+infile ),logfile)  
                    
    return

# ------------------------------------
def spmospolmap(dataList, objectname, calHdul, **kwargs):
  # _i mask slits (entries in xml)
  # _f infileList index

    cutwavoverride = float(kwargs.pop('cutwavoverride',0.))
    isdiffuse = kwargs.pop('isdiffuse',False)
    if isinstance(isdiffuse, str): isdiffuse=(isdiffuse=="True")        
    pidebug = tuple(kwargs.pop('pidebug',(-1,-1)))
    if isinstance(pidebug, str): pidebug=tuple(pidebug.split(","))
    logfile= kwargs.pop('logfile','poltarget.log')    
    debug = kwargs.pop('debug',False)
    if isinstance(debug, str): debug=(debug=="True")

  # get configuration data from first image
    files = len(dataList)
    infileList = [hdul.filename() for hdul in dataList]  
    obsDictf = create_obslog(infileList,keywordfile)      

    hdul0 = dataList[0]    
    hdr = hdul0[0].header    
    trkrho = hdr['TRKRHO']    
    filter = hdr['FILTER']
    camtem = hdr['CAMTEM']
    coltem = hdr['COLTEM']
    dateobs =  hdr['DATE-OBS'].replace('-','')        
    lampid = hdr['LAMPID']     
    rows, cols = hdul0[1].data.shape
    cbin, rbin = [int(x) for x in hdul0[0].header['CCDSUM'].split(" ")]
    rcbin_d = np.array([rbin,cbin])
    pixmm = 0.015
    binmm_d = rcbin_d*pixmm
    prows = rows/2
    rows = 2*prows                           # allow for odd number of rows
    calhdr = calHdul[0].header     
    ur0,uc0,saltfps = rssdtralign(dateobs,trkrho)           # ur, uc =unbinned pixels, saltfps =micr/arcsec    
    yx0_d = -pixmm*np.array([ur0,uc0])                      # optical axis in mm
    
    wmin = max(float(filter[3:])-50.,cutwavoverride)
    wav_w = np.arange(3000.,10000.,100.)
    wav_w = np.insert(wav_w[wav_w > wmin],0,wmin)   
    YX00_di = np.zeros(2).reshape((2,1))
    wav00_pic,rce00_pic,ok00_pic = slitpredict(hdul0,YX00_di,wav_w,debug=False)
    col0 = cols/2 + yx0_d[1]/(cbin*pixmm)    
    usewav_c = (wav00_pic[0,0] > 0.)                              
    wav0 = np.interp(col0,np.where(usewav_c)[0],wav00_pic[0,0,usewav_c])     

    dtrfps = saltfps*np.diff(RSScolpolcam(np.array([[0.,0.],[0.,1.]]),5500.,coltem,camtem)[1,0])[0] 
        
    tgt_prc = calHdul['TMAP'].data
    tgtTab = Table.read(calHdul['TGT'])
    entries = len(tgtTab['CATID'])
    istarget_pirc = np.zeros((2,entries,prows,cols),dtype=bool)    
    for p,i in np.ndindex(2,entries):      
        istarget_pirc[p,i] = (tgt_prc[p] == i+1)        
           
  # input all data, split it into OE images       
    yxp0_dp = np.vstack((np.array(calhdr['YXAXISO'].split()), \
                        np.array(calhdr['YXAXISE'].split()))).astype(float).T
    rshift_p = np.array(calhdr['RSPLIT'].split()).astype(int)

    image_fprc = np.zeros((files,2,prows,cols))
    var_fprc = np.zeros_like(image_fprc)
    okbin_fprc = np.zeros((files,2,prows,cols),dtype=bool)            
    for f,hdul in enumerate(dataList):
        gaincor_c = gaincor(hdul) 
        image_rc = hdul['SCI'].data[:rows]/gaincor_c[None,:]
        var_rc = hdul['VAR'].data[:rows]/gaincor_c[None,:]**2
        okbin_rc = (hdul['BPM'].data[:rows]==0)
        for p in (0,1):
            image_fprc[f,p] = shift(image_rc,(-rshift_p[p],0),order=0)[(p*prows):((p+1)*prows)]
            var_fprc[f,p] = shift(var_rc,(-rshift_p[p],0),order=0)[(p*prows):((p+1)*prows)]       
            okbin_fprc[f,p] = shift(okbin_rc,(-rshift_p[p],0),order=0)[(p*prows):((p+1)*prows)] 

    bkg_prc = np.zeros_like(tgt_prc)
    if isdiffuse:
  #     for diffuse data , background is wherever there is no target
        bkg_prc = 255*(okbin_fprc[0] & (tgt_prc == 0))
                           
    else:    
  #     sum data and find spectra using moffat fits, corrected for row offset due to curvature
        image_prc = image_fprc.sum(axis=0)
        var_prc = var_fprc.sum(axis=0)
        okbin_prc = okbin_fprc.all(axis=0)
        
  #     moffat fit using min as bkg from summed column profile over all files
        oktgt_tpC = oktgt_ftpRC[0,:,:].any(axis=2)
        image_tpC = (image_ftpRC*oktgt_ftpRC).sum(axis=(0,3))/oktgt_ftpRC.sum(axis=(0,3))
        bkg_tp = np.zeros((targets,2))
        for t,p in np.ndindex(targets,2):     
            bkg_tp[t,p] = image_tpC[t,p,oktgt_tpC[t,p]].min()
        sigma_s, fmax_s, C0_s, fiterr_sb, okprof_s =    \
            moffat1dfit((image_tpC-bkg_tp[:,:,None]).reshape((-1,Cols)),oktgt_tpC,beta=2.5)     # _s = _tp
                
  #     flag bkg cols in TGT map    
        x_sC = np.arange(Cols)[None,:] - C0_s[:,None]    
        proffit_sC = moffat(np.ones(2*targets),sigma_s,x_sC,beta=2.5)
        fbkg = (1.+(0.375*oktgt_tpC.sum(axis=2).min()/sigma_s[okprof_s].max())**2)**(-2.5)  # > 25% columns are bkg
        isbkg_tpC = ((proffit_sC < fbkg).reshape((targets,2,-1)) & oktgt_ftpRC[0,:,:].any(axis=2))
        for t,p in np.ndindex(targets,2):
            if badsigma_t[t]: continue
            cbkgArray = np.where(isbkg_tpC[t,p])[0]+ci_tpC[t,p,0]
            rtgtArray = np.where((tgt_prc[p]==i_t[t]+1).any(axis=1))[0]
            for r,c in np.array(np.meshgrid(rtgtArray,cbkgArray)).T.reshape((-1,2)):
                tgt_prc[p,r,c] = 255

        if debug:            
            tgtTab.write(objectname+"_"+filter+"_mapTab.txt",format='ascii.fixed_width',   \
                bookend=False, delimiter=None, overwrite=True)   
            np.savetxt('image_tpC.txt',np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)),    \
                image_tpC.reshape((-1,Cols)).T)).T,fmt=" %2i %2i %2i "+Cols*"%9.2f ")
            np.savetxt('moffatfit_tp.txt',np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)),    \
                sigma_s, fmax_s, C0_s, okprof_s.astype(int))).T,fmt=" %2i %2i %2i %8.3f %9.2f %8.3f %2i")
            np.savetxt('isbkg_tpC.txt',np.vstack((i_t.repeat(2),np.indices((targets,2)).reshape((2,-1)),    \
                isbkg_tpC.astype(int).reshape((-1,Cols)).T)).T,fmt=" %2i %2i %2i  "+Cols*"%2i ")                 
                
  # output with calhdr data        
    for f,hdul in enumerate(dataList):   
        hdr = hdul[0].header        
        hdul['SCI'].data = image_fprc[f].astype('float32')
        hdul['VAR'].data = var_fprc[f].astype('float32') 
        hdul['BPM'].data = (~okbin_fprc[f]).astype('uint8')
        
        hdr['YXAXISO'] = calhdr['YXAXISO']
        hdr['YXAXISE'] = calhdr['YXAXISE']
        hdr['CALIMG'] = calhdr['CALIMG']
        if (isdiffuse): hdr['DIFFUSE'] = ("T","object fills slit")                          
        hdul.append(pyfits.ImageHDU(data=tgt_prc,name='TMAP'))
        hdul.append(pyfits.ImageHDU(data=bkg_prc.astype('uint8'),name='BMAP'))     
        hdul.append(pyfits.ImageHDU(data=calHdul['WMAP'].data,name='WMAP'))
        hdul.append(pyfits.ImageHDU(data=calHdul['WERR'].data,name='WERR'))                   
        hdul.append(pyfits.table_to_hdu(tgtTab))
        hdul[-1].header['EXTNAME'] = 'TGT'    
        hdr['object'] = objectname      # overrides with lamp name for lamp observations     
        hdul.writeto("t"+infileList[f],overwrite=True)
        rsslog.message(('Output file '+'t'+infileList[f]),logfile)

    return
    
#----------------------------------------------
def slitpredict(hdul,YX_di,wav_w,order=1,droff=0.,sptilt=0.,wavoff=0.,debug=False):
    hdr = hdul[0].header
    camtem = hdr['CAMTEM']
    coltem = hdr['COLTEM']
    dateobs =  hdr['DATE-OBS'].replace('-','')
    trkrho = hdr['TRKRHO']
    grating = hdr['GRATING'].strip()
    grang = hdr['GR-ANGLE']
    artic = hdr['CAMANG']
    okimg_rc = (hdul['BPM'].data == 0)    
    rows, cols = hdul['SCI'].data.shape
    cbin, rbin = [int(x) for x in hdr['CCDSUM'].split(" ")]
    rcbin_d = np.array([rbin,cbin])
    rccenter_d, cgapedge_d = ccdcenter(hdul['SCI'].data)
    pixmm = 0.015
    
    entries = YX_di.shape[1]
    wavs = wav_w.shape[0]
    droff_pi = [droff,droff*np.ones((2,entries))][np.isscalar(droff)]
    wavoff_piw = [wavoff,wavoff*np.ones((2,entries,wavs))][np.isscalar(wavoff)]       
        
    ur0,uc0,saltfps = rssdtralign(dateobs,trkrho)           # ur, uc =unbinned pixels, saltfps =micr/arcsec    
    yx0_d = -pixmm*np.array([ur0,uc0])                      # optical axis in mm    
    wav_s = np.repeat(wav_w,entries)
    YX_ds = np.tile(YX_di,(1,wavs))
        
    yx_dpwi = RSScolgratpolcam(YX_ds, wav_s, coltem, camtem, grating, grang, artic,       \
        dateobs, order=order, debug=debug).reshape((2,2,wavs,-1))
    rc_dpwi = (yx_dpwi - yx0_d[:,None,None,None])/(pixmm*rcbin_d[:,None,None,None]) +    \
        rccenter_d[:,None,None,None]        

    wav_pic = np.zeros((2,entries,cols))
    rce_pic = np.zeros((2,entries,cols))    
    ok_pic = np.zeros((2,entries,cols),dtype=bool)
    roff_pic = droff_pi[:,:,None] + np.radians(sptilt)*(cbin/rbin)*(np.arange(cols)-cols/2.)[None,None,:]

    for p,i in np.ndindex(2,entries):
        wavcor_w = wav_w - wavoff_piw[p,i]
        fcw = interp1d(rc_dpwi[1,p,:,i],wavcor_w,bounds_error=False,      \
            kind='cubic',fill_value=(wavcor_w[0],wavcor_w[-1]))
        fcr = interp1d(rc_dpwi[1,p,:,i],rc_dpwi[0,p,:,i],bounds_error=False,      \
            kind='cubic',fill_value=(rc_dpwi[0,p,:,i].min(),rc_dpwi[0,p,:,i].max()))
            
        wav_pic[p,i] = fcw(np.arange(cols))
        rce_pic[p,i] = fcr(np.arange(cols)) + roff_pic[p,i]
        rint_c = np.clip(np.round(rce_pic[p,i]).astype(int),0,rows-1)                               
        ok_pic[p,i] = (okimg_rc[rint_c,range(cols)] &   \
            (wav_pic[p,i] > wavcor_w[0]) & (wav_pic[p,i] < wavcor_w[-1]))
        wav_pic[p,i] *= ((wav_pic[p,i] > max(wav_w[0],wavcor_w[0])) &   \
                         (wav_pic[p,i] < min(wav_w[-1],wavcor_w[-1])))

    return wav_pic,rce_pic,ok_pic

#----------------------------------------------
def targetmap(rce_pic,ok_pic,Rows_pi,rcecoll_pIc,okcoll_pIc,Rowscoll_pI,rows):
  # compute target collisions
  # return targetmap, rowextent table r12_dpic (bottom,top) for each spectrum
  # do separately for O,E.  Assume same number of targets in each
  
    entries,cols = rce_pic.shape[1:]
    targetmap_prc = np.zeros((2,rows,cols),dtype='uint8')
    r12_dpic = np.zeros((2,2,entries,cols))
    r12coll_dpIc = np.zeros((2,2,entries,cols))
    isselfcoll = (rce_pic == rcecoll_pIc).all()      
    mp_d = np.array([-1.,1.])

    r12_dpic[:,ok_pic] = rce_pic[None,ok_pic] +     \
        mp_d[:,None]*np.repeat(Rows_pi[:,:,None],cols,axis=2)[None,ok_pic]/2.
        
    chkcoll_pIc = np.zeros_like(okcoll_pIc)         # count a collision with a spoiled spectrum
    for p,I in np.ndindex((2,entries)):
        if (okcoll_pIc[p,I].sum()==0): continue
        c1,c2 = np.where(okcoll_pIc[p,I])[0][[0,-1]]          
        chkcoll_pIc[p,I] = ((np.arange(cols) >= c1) & (np.arange(cols) <= c2))
    r12coll_dpIc[:,chkcoll_pIc] = rcecoll_pIc[None,chkcoll_pIc] +    \
        mp_d[:,None]*np.repeat(Rowscoll_pI[:,:,None],cols,axis=2)[None,chkcoll_pIc]/2.
            
    for p in (0,1):
        drcoll_iIc = rce_pic[p,:,None,:] - rcecoll_pIc[p,None,:,:]
        if isselfcoll:
            drcoll_iIc[range(entries),range(entries),:] = 1.e9
        bad_iIc = np.logical_not(ok_pic[p,:,None,:] & chkcoll_pIc[p,None,:,:])
        drcoll_iIc[bad_iIc] = 1.e9

        for i in range(entries):                
            Iclose_c = np.argmin(np.abs(drcoll_iIc[i]),axis=0)                    
            drclose_c = drcoll_iIc[i,Iclose_c,range(cols)]
            drcoll_c = (Rows_pi[p,i] + Rowscoll_pI[p,Iclose_c])/2.
            iscoll1_c = ((drclose_c >= 0.) & ( drclose_c < drcoll_c))        
            iscoll2_c = ((drclose_c <= 0.) & (-drclose_c < drcoll_c))        
            r12_dpic[0,p,i,iscoll1_c] = r12coll_dpIc[1,p,Iclose_c[iscoll1_c],iscoll1_c]
            r12_dpic[1,p,i,iscoll2_c] = r12coll_dpIc[0,p,Iclose_c[iscoll2_c],iscoll2_c]
            rint12_dc = np.clip(np.round(r12_dpic[:,p,i]).astype(int),0,rows-1)
                            
            for c in range(cols):
                if (r12_dpic[0,p,i,c] == 0): continue
                targetmap_prc[p,rint12_dc[0,c]:(rint12_dc[1,c]+1),c] = i+1 
            
    return targetmap_prc,r12_dpic    

#----------------------------------------------
def arcflex(img_rc,rce_c,Rows):
  # find qmax points.  center at mean, width is difference.
  # iterate to pull in profile at edge of first-guess window
    rows,cols = img_rc.shape
    drow=0
    rint_c = np.clip(np.round(rce_c).astype(int),0,rows-1)    
    for iter in (0,1,2,3):
        prof_R = np.zeros(Rows)                 
        for R in range(Rows):            
            r_c = np.clip(rint_c+drow+R-Rows/2,0,rows-1)
            prof_R[R] = img_rc[r_c,range(cols)].sum()
        prof_R -= prof_R.min()                                         
        argmax = np.argmax(prof_R)
        isprof_R = (prof_R > prof_R[argmax]/4.)
        argqmax_d = np.array([0,Rows-1])
        if (not (isprof_R[argmax::-1].all())): 
            argqmax_d[0] = argmax - np.argmin(isprof_R[argmax::-1])
        if (not (isprof_R[argmax:].all())): 
            argqmax_d[1] = argmax + np.argmin(isprof_R[argmax:])        
        drow += int(np.round(argqmax_d.mean())) - Rows/2
           
    rflex = drow + 1 + int(np.round(argqmax_d.mean())) - Rows/2
    rwidth = np.diff(argqmax_d)[0]
        
    return rflex, rwidth, prof_R[argmax]

#----------------------------------------------
def findlines(sci_c,var_c,ok_c,Lines=100,label="",logfile='salt.log'):
    cols = sci_c.shape[0]
    colwav_L = np.zeros(Lines)    
    colerr_L = np.zeros(Lines)
    colerr1_L = np.zeros(Lines)
    fitwidth_L = np.zeros(Lines)
    fmax_L = np.zeros(Lines)    
        
  # find strongest line and get its 1/2 width (use for all).  
  #   Need at least +/-2 good pix around max   

    if label:
        goodfile=open("findlines_"+label+".txt",'w')
        rejectfile=open("findlines_reject_"+label+".txt",'w')
            
    cmax = np.argmax(sci_c*ok_c)

    while (not ok_c[(cmax-2):(cmax+3)].all()):
        if label: print >>rejectfile, -1, "badstart ", cmax, ok_c[(cmax-2):(cmax+3)].astype(int) , sci_c[(cmax-2):(cmax+3)], var_c[(cmax-2):(cmax+3)]  
        ok_c[(cmax-2):(cmax+3)] = False
        cmax = np.argmax(sci_c*ok_c)
                
    isline_c = (sci_c > sci_c[cmax]/2.)    
    halfwidth = max(np.argmin(isline_c[cmax::-1]),np.argmin(isline_c[cmax:]),2)

  # fit to quadratic over halfwidth (minimum +/-2) to get wavelength
    sidesample = max(3,halfwidth)
    use_c = ((np.abs(np.arange(cols)-cmax) < sidesample) & ok_c)  
    a_C = np.arange(cols)[use_c]-cmax
    a_Cd = np.vstack((a_C**2,a_C,np.ones_like(a_C))).T   
    quadfit_d = la.lstsq(a_Cd,sci_c[use_c])[0]
    dcmax = -0.5*quadfit_d[1]/quadfit_d[0]
    colwav_L[0] = np.clip(cmax + dcmax,0,cols-1)
    eps_dd = la.inv((a_Cd[:,:,None]*a_Cd[:,None,:]).sum(axis=0))
    err_d = np.sqrt(np.diagonal(eps_dd)*var_c[use_c].sum()/a_C.shape[0])        
    cov_01 = eps_dd[0,1]
    colerr_L[0] =  0.5*np.abs(quadfit_d[1]/quadfit_d[0])*    \
        np.sqrt((err_d[1]/quadfit_d[1])**2 + (err_d[0]/quadfit_d[0])**2     \
        - 2.*cov_01/(quadfit_d[0]*quadfit_d[1]))
    fitwidth_L[0] = -np.sqrt(0.5*(quadfit_d[1]**2 - 4.*quadfit_d[0]*quadfit_d[2]))/quadfit_d[0]
    fmax_L[0] = sci_c[cmax]
          
  # continue, removing each line until 100 lines or max < 5 sigma
    ok1_c = np.copy(ok_c)    
    L = 0
    if label: print >>goodfile, (("%3i %4i "+3*"%8.3f "+3*"%10.3f ")%   \
        ((L, cmax, dcmax, colerr_L[0], fitwidth_L[0])+tuple(quadfit_d)))           
     
    while (L < (Lines-1)):
      # erasing previous line, find new one
        ok1_c[np.clip(cmax-halfwidth,0,cols-1):np.clip(cmax+halfwidth+1,0,cols-1)] = False                
        cmax = np.argmax(sci_c*ok1_c)
                    
      # quit if too weak          
        sigmax = sci_c[cmax]/np.sqrt(var_c[cmax])                      
        if (sigmax < 5.): 
            if label: print >>rejectfile, (" %2i " % (L+1)), "sigmax ", cmax, sigmax, ok1_c.sum()        
            break 
                
      # if new max is at edge of previous line or a gap, erase max only and continue          
        while (not ok1_c[(cmax-1):(cmax+2)].all()):
            if label: print >>rejectfile, (" %2i " % (L+1)), "edge   ", cmax, ok1_c[(cmax-1):(cmax+2)].astype(int)      
            ok1_c[cmax] = False
            cmax = np.argmax(sci_c*ok1_c)

      # if too few points, skip to next line                
        use_c = ((np.abs(np.arange(cols)-cmax) < sidesample) & ok1_c)
        if (use_c.sum() < 5):   
            if label: print >>rejectfile, (" %2i " % (L+1)), "points ", cmax, use_c.sum()
            continue

      # update cmax to use mean column of three highest points; fit down to halfmax with quadratic
        cmax = int(np.round(np.argsort((sci_c*ok1_c)[use_c])[-3:].mean())) + np.where(use_c)[0][0]
        use_c = ((np.abs(np.arange(cols)-cmax) < sidesample) & ok1_c)
        a_C = np.arange(cols)[use_c]-cmax
        a_Cd = np.vstack((a_C**2,a_C,np.ones_like(a_C))).T 
        quadfit_d = la.lstsq(a_Cd,sci_c[use_c])[0]
        dcmax = -0.5*quadfit_d[1]/quadfit_d[0]
        eps_dd = la.inv((a_Cd[:,:,None]*a_Cd[:,None,:]).sum(axis=0))        
        err_d = np.sqrt(np.diagonal(eps_dd)*var_c[use_c].sum()/a_C.shape[0])        
        colerr =  0.5*np.abs(quadfit_d[1]/quadfit_d[0])*    \
            np.sqrt((err_d[1]/quadfit_d[1])**2 + (err_d[0]/quadfit_d[0])**2     \
            - 2.*cov_01/(quadfit_d[0]*quadfit_d[1]))
                            
      # if fit is inverted,colerr large,or puts max at least 1.5 pix away, skip to next line
        if ((quadfit_d[0] >= 0.) | ((quadfit_d[1]**2 - 4.*quadfit_d[0]*quadfit_d[2]) <= 0.) |   \
            (colerr > 1.) | (np.abs(dcmax) >= 1.5)):  
            if label: print >>rejectfile, (" %2i " % (L+1)), "fit    ", cmax, dcmax, colerr, np.where(use_c)[0],sci_c[use_c],quadfit_d
            continue
                
        fitwidth = -np.sqrt(0.5*(quadfit_d[1]**2 - 4.*quadfit_d[0]*quadfit_d[2]))/quadfit_d[0]                
      # if fit is too wide, skip to next line
        if (fitwidth > 2.*fitwidth_L[0]):  
            if label: print >>rejectfile, (" %2i " % (L+1)), "width  ", cmax, fitwidth
            continue

        L += 1 
        colwav_L[L] = np.clip(cmax + dcmax,0,cols-1)
        colerr_L[L] =  colerr
        fitwidth_L[L] = fitwidth
        fmax_L[L] = sci_c[cmax]
        
        if label: print >>goodfile, (("%3i %4i "+3*"%8.3f "+3*"%10.3f ")%   \
            ((L, cmax, dcmax, colerr, fitwidth)+tuple(quadfit_d)))                             

    return colwav_L,colerr_L,fitwidth_L[0],fitwidth_L/fitwidth_L[0],fmax_L

#----------------------------------------------
def lineid(calhdr,lampwav_m,oklamp_m,okline_pl,mbrtid_pl,obscol_pl,fmax_pl,wav_pc,logfile='salt.log',debugname=''):
  # id lines for a target (O and E)
  # mbrtid_pl   identifications from initial bright-line search

    grating = calhdr['GRATING'].strip()
      
  # first, keep only lines seen both O and E, within 2.5 wavelength bin
    lines = obscol_pl.shape[1]
    cols = wav_pc.shape[1]
    usewav_pc = (wav_pc > 0.)    
    disp0 = np.polyfit((np.arange(cols)-cols/2)[usewav_pc[0]],wav_pc[0,usewav_pc[0]],3)[2]                 
    obswav_pl = np.zeros((2,lines))
    for p in (0,1):
        obswav_pl[p] = np.interp(obscol_pl[p],np.where(usewav_pc[p])[0],wav_pc[p,usewav_pc[p]])        
    oediff_oe = np.abs(obswav_pl[0][:,None] - (okline_pl[1]*obswav_pl[1])[None,:])
    e_o = np.argmin(oediff_oe,axis=1)
    diff_o = oediff_oe[range(lines),e_o]
    o_e = np.argmin(oediff_oe,axis=0)
    diff_e = oediff_oe[o_e,range(lines)]
    obswav_l = (obswav_pl[0] + obswav_pl[1,e_o])/2.           
    useo_l = (okline_pl[0] & (obswav_pl[0] > 0.) & (diff_o < 2.5*disp0))                
    oArray = np.where(useo_l)[0]
    oList = list(oArray[np.argsort(obswav_l[useo_l])])
    eList = list(e_o[oList])      
    l_pb = np.array([np.array(oList),np.array(eList)])

    obswav_b = obswav_l[oList]    
    obswav_pb = np.array([obswav_pl[0,oList],obswav_pl[1,eList]])
    obscol_pb = np.array([obscol_pl[0,oList],obscol_pl[1,eList]])
    fmax_pb = np.array([fmax_pl[0,oList],fmax_pl[1,eList]])    
                  
  # id lines, using OE mean obswav. _b, _m = obs,lamp
    bwavs = obswav_b.shape[0]           
    bmdiff_bm =  obswav_b[:,None] - lampwav_m[None,:]        
    m_b = np.argmin(np.abs(bmdiff_bm),axis=1)
    waverr_b = bmdiff_bm[range(bwavs),m_b]

  # cull lines id'd with rejected lamp lines, multiple id's          
    ok_b = oklamp_m[m_b]                
    m_M,count_M = np.unique(m_b[ok_b],return_counts=True)
    besterr = np.median(waverr_b[ok_b])        
    for M in np.where(count_M>1)[0]:
        bList = list(np.where(m_b==m_M[M])[0])                        
        argmin = np.argmin(np.abs((waverr_b[bList]-besterr)))
        bList.remove(bList[argmin])                                    
        ok_b[bList] = False
    ok1_b = np.copy(ok_b)

  # cull in between lines         
    m2_b = np.argsort(np.abs(bmdiff_bm),axis=1)[:,1]
    waverr2_b = bmdiff_bm[range(bwavs),m2_b]
    inbetween_b = (((np.abs(waverr2_b) - np.abs(waverr_b)) < disp0) &  \
        (np.sign(waverr_b*waverr2_b) == -1))
    ok_b[inbetween_b] = False
    ok2_b = np.copy(ok_b)

  # cull by fmax E/O       
    fmaxOE_b = np.log10(fmax_pb[0]/fmax_pb[1])
    legx_b = 2*obscol_pb[0]/cols-1.
    fitorder = [1,0][grating == 'PG0300']
    xlim_d = legx_b[[0,-1]]        
    fmaxfit_d,ok_b,err_b,fiterr_b,fiterr_X,cullLog =  \
        legfit_cull(legx_b,fmaxOE_b,ok_b,fitorder,xlim_d=xlim_d,IQcull=3.)
    ok3_b = np.copy(ok_b)

  # restore bright-line id's, if necessary (must be ok in both O and E)
    okbrt_b = ((mbrtid_pl[0][l_pb[0]] >= 0) & (mbrtid_pl[1][l_pb[1]] >= 0))
    m_b[okbrt_b] = mbrtid_pl[0][l_pb[0][okbrt_b]]
    ok_b = (ok_b | okbrt_b)   

    if debugname:
        debugfile = open("iddebug_"+debugname+".txt",'w') 
        mcullArray = m_b[np.where(ok2_b & np.logical_not(ok_b))[0]]
        print >>debugfile, (("fmaxOEcull: "+len(mcullArray)*"%3i \n" % tuple(mcullArray)))
        print >>debugfile, " m lamp ok1 ok2 ok3 ok_b    lamp     fmaxO     fmaxE     colO     colE     wavO     WavE"
        for b in range(len(oList)):
            print >>debugfile, ((6*"%3i "+3*"%9.2f "+2*"%8.3f "+2*"%9.2f ") %   \
                ((m_b[b],oklamp_m[m_b[b]],ok1_b[b],ok2_b[b],ok3_b[b],ok_b[b],lampwav_m[m_b[b]])+   \
                tuple(fmax_pb[:,b])+tuple(obscol_pb[:,b])+tuple(obswav_pb[:,b])))
    
    return m_b,obswav_pb,obscol_pb,l_pb,ok_b
   


