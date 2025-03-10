
# polarimetry utilities, including:

# datedfile(filename,date)
# datedline(filename,date)
# greff(grating,grang,artic,dateobs,wav)
# rssdtralign(datobs,trkrho)
# rssmodelwave(grating,grang,artic,trkrho,cbin,cols,datobs)
# hwcal(calfile,YX_ds,MByx_ds,wav_w)
# configmap(infilelist,confitemlist,debug='False')
# image_number(image_name)
# list_configurations(infilelist, log)
# configmapset(obs_tab, config_list=('GRATING','GR-ANGLE', 'CAMANG'))
# list_configurations_old(infilelist, log)
# blksmooth1d(ar_x,blk,ok_x)
# angle_average(ang_d)
# rotate2d(yx_ds, rot, center=np.zeros(2))
# fargmax(arr)
# printstdlog(string,logfile)

import os, sys, glob, shutil, inspect
import numpy as np
from astropy.io import fits as pyfits, ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import interp1d
from scipy import interpolate as ip

from saltobslog import obslog
from astropy.table import Table,unique

DATADIR = os.path.dirname(__file__) + '/data/'

# ------------------------------------

def datedfile(filename,date):
    """ select file based on observation date and latest version

    Parameters
    ----------
    filename: text file name pattern, including "yyyymmdd_vnn" place holder for date and version
    date: yyyymmdd of observation

    Returns: file name

    """

    filelist = sorted(glob.glob(filename.replace('yyyymmdd_vnn','????????_v??')))
    if len(filelist)==0: return ""
    dateoffs = filename.find('yyyymmdd')
    datelist = [file[dateoffs:dateoffs+8] for file in filelist]
    file = filelist[0]
    for (f,fdate) in enumerate(datelist):
        if date < fdate: continue
        for (v,vdate) in enumerate(datelist[f:]):
            if vdate > fdate: continue
            file = filelist[f+v]
 
    return file     
# ------------------------------------

def datedline(filename,date):
    """ select line from file based on observation date and latest version

    Parameters
    ----------
    filename: string
        skips lines without first field [yyyymmdd]_v[nn] datever label
    date: string or int yyyymmdd of observation

    Returns: string which is selected line from file (including datever label)

    """
    line_l = [ll for ll in open(filename) if ll[8:10] == "_v"]
    datever_l = [line_l[l].split()[0] for l in range(len(line_l))]

    line = ""
    for (l,datever) in enumerate(datever_l):
        if (int(date) < int(datever[:8])): continue
        for (v,vdatever) in enumerate(datever_l[l:]):
            if (int(vdatever[:8]) > int(datever[:8])): continue
            datever = datever_l[l+v]
            line = line_l[l+v]
 
    return line

#--------------------------------------------------------

def greff(grating,grang,artic,dateobs,wav):
#   grating efficiency, zero outside 1st order and eff < grateffedge
#   p/s added 9 July, 2017 khn
#   wav may be 1D array

    grname=np.loadtxt(DATADIR+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=np.loadtxt(DATADIR+"gratings.txt",usecols=(1,2),unpack=True)
    gr300wav,gr300eff,gr300ps=np.loadtxt(DATADIR+"grateff_0300.txt",usecols=(0,1,2),unpack=True)
    grng,grdn,grthick,grtrans,grbroaden=np.loadtxt(DATADIR+"grateff_v1.txt", \
        usecols=(1,2,3,4,5),unpack=True)
    spec_dp=np.array(datedline(DATADIR+"RSSspecalign.txt",dateobs).split()[1:]).astype(float)

    Grat0 = spec_dp[0]
    grateffedge = 0.04

    grnum = np.where(grname==grating)[0][0]
    lmm = grlmm[grnum]
    alpha_r = np.radians(grang+Grat0)

    if grnum == 0:          # SR grating
        eff = interp1d(gr300wav,gr300eff,kind='cubic',bounds_error=False)(wav)
        ps = interp1d(gr300wav,gr300ps,kind='cubic',bounds_error=False)(wav) 
    else:                   # Kogelnik gratings
        ng = grng[grnum]
        dn = grdn[grnum]
        thick = grthick[grnum]
        tran = grtrans[grnum]
        broaden = grbroaden[grnum]
        beta_r = np.arcsin(wav*lmm/1.e7 - np.sin(alpha_r))
        betag_r = np.arcsin(np.sin(beta_r)/ng)
        alphag_r = np.arcsin(np.sin(alpha_r)/ng)
        KK = 2.*np.pi*lmm/1.e7
        nus = (np.pi*dn/wav)*thick/np.cos(alphag_r)
        nup = nus*np.cos(alphag_r + betag_r)
        xi = (thick/(2*np.cos(alphag_r)))*(KK*np.sin(alphag_r) - KK**2*wav/(4.*np.pi*ng))/broaden
        sins = np.sin(np.sqrt(nus**2+xi**2))
        sinp = np.sin(np.sqrt(nup**2+xi**2))
        effs = sins**2/(1.+(xi/nus)**2)
        effp = sinp**2/(1.+(xi/nup)**2)
        eff = np.where((sins>0)&(sinp>0)&((effs+effp)/2. > grateffedge),tran*(effs+effp)/2.,0.)
        ps = np.where(eff > 0., effp/effs, 0.)
    return eff,ps
    
# ------------------------------------

def rssdtralign(datobs,trkrho):
    """return predicted RSS optic axis relative to detector center (in unbinned pixels), plate scale
    Correct for flexure, and use detector alignment history.

    Parameters 
    ----------
    datobs: int
        yyyymmdd of first data of applicability  
    trkrho: float
        tracker rho in degrees

    """

  # optic axis is center of imaging mask.  In columns, same as longslit position
    rc0_pd=np.loadtxt(DATADIR+"RSSimgalign.txt",usecols=(1,2))
    flex_p = np.array([np.sin(np.radians(trkrho)),np.cos(np.radians(trkrho))-1.])
    rcflex_d = (rc0_pd[0:2]*flex_p[:,None]).sum(axis=0)

    row0,col0,C0 = np.array(datedline(DATADIR+"RSSimgalign.txt",datobs).split()[1:]).astype(float)
    row0,col0 = np.array([row0,col0]) + rcflex_d        # sign changed 20190610

    return row0, col0, C0
# ------------------------------------

def rssmodelwave(grating,grang,artic,trkrho,cbin,cols,datobs):
    """compute wavelengths from model of RSS

    Parameters 
    ----------
    datobs: int
        yyyymmdd of first data of applicability    

     TODO:  replace using PySpectrograph
  
    """

    row0,col0 = rssdtralign(datobs,trkrho)[:2]
    spec_dp=np.array(datedline(DATADIR+"RSSspecalign.txt",datobs).split()[1:]).astype(float)
    Grat0,Home0,ArtErr,T2Con,T3Con = spec_dp[:5]
    FCampoly=spec_dp[5:]

    grname=np.loadtxt(DATADIR+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=np.loadtxt(DATADIR+"gratings.txt",usecols=(1,2),unpack=True)
    grnum = np.where(grname==grating)[0][0]
    lmm = grlmm[grnum]
    alpha_r = np.radians(grang+Grat0)
    beta0_r = np.radians(artic*(1+ArtErr)+Home0) - 0.015*col0/FCampoly[0] - alpha_r
    gam0_r = np.radians(grgam0[grnum])
    lam0 = 1e7*np.cos(gam0_r)*(np.sin(alpha_r) + np.sin(beta0_r))/lmm
    modelcenter = 3162.  #   image center (unbinned pixels) for wavelength calibration model
    ww = lam0/1000. - 4.
    fcam = np.polyval(FCampoly[::-1],ww)
    disp = (1e7*np.cos(gam0_r)*np.cos(beta0_r)/lmm)/(fcam/.015)
    dfcam = (modelcenter/1000.)*disp*np.polyval([FCampoly[5-x]*(5-x) for x in range(5)],ww)

    T2 = -0.25*(1e7*np.cos(gam0_r)*np.sin(beta0_r)/lmm)/(fcam/47.43)**2 + T2Con*disp*dfcam
    T3 = (-1./24.)*modelcenter*disp/(fcam/47.43)**2 + T3Con*disp
    T0 = lam0 + T2
    T1 = modelcenter*disp + 3*T3
    X = (np.array(range(cols))-cols/2)*cbin/modelcenter
    lam_X = T0+T1*X+T2*(2*X**2-1)+T3*(4*X**3-3*X)
    return lam_X

# ----------------------------------------------------------
def hwcal(hwcalfile,YX_dt,Tyx_d,rho,wav_w):
    """get P and PA (HWPol), Q and U (HWZero) vs wavelength from hw cal file    
    Parameters
    ----------
    hwcalfile: text
        name of hwcal file to use
    YX_dt: numpy 2d array
        _d: Y,X (mm) at RSS FP
        _t: targets
    Tyx_d: numpy 1d array
        _d:TRKY,TRKX (m) from fits header (same for all targets)
    rho: TRKRHO (deg) from fits header (same for all targets)
    wav_w: numpy 1d array
        _w: wavelengths (Ang)

    Returns: numpy 2d arrays:
    -------
    hwcal_ptw[0,1]: 2 numpy 2d arrays
        hwcal polarization stokes fits for each target and wavelength.  =0 where invalid
    
    """
  # input hwcal file data, _p=0,1 are stacked
    wav_pW = np.loadtxt(hwcalfile,usecols=0).reshape((2,-1))
    Wavs = wav_pW.shape[1]
    targets = YX_dt.shape[1]    
    wavs = wav_w.shape[0]
    
    if (Wavs==1):                         # skip this for now 
        hwcal_ptw = np.zeros(((2,targets,wavs)))
        ok_tw = np.ones((targets,wavs),dtype=bool)
        return hwcal_ptw, ok_tw        
    
    wav_W = wav_pW[0]
    hwcal_pcW = np.loadtxt(hwcalfile)[:,1:].reshape((2,Wavs,-1)).transpose((0,2,1))
    ok_W = (hwcal_pcW[0,0] != 0.)
    cofs = hwcal_pcW.shape[1]
    
    varList = ['Y','X','Ty','Tx','r']   
    Vars = open(hwcalfile).read().count("#") - 2        
    hdrlineList = open(hwcalfile).readlines()[1:(Vars+2)]
    
    PAoff = np.fromstring(hdrlineList[0][8:],dtype=float,sep=" ")[0]
    varidx_V = np.zeros(Vars,dtype=int)
    fntype_V = np.empty(Vars,dtype=str)
    fn_Vc = np.zeros((Vars,cofs),dtype=int)
 
    for V in range(Vars): 
        varname,fntype_V[V] = hdrlineList[V+1][1:8].split()
        varidx_V[V] = varList.index(varname)
        fn_Vc[V] = np.fromstring(hdrlineList[V+1][8:],dtype=int,sep=" ")

  # HWPol (Wavs>1) hwcal_p at cal wavelength grid for each target
  # HWZero (Wavs=1) hwcal_p for each target

    YXnorm = 50.
    ok_t = ((np.sqrt((YX_dt**2).sum(axis=0)) < 1.1*YXnorm) & (np.abs(YX_dt[0]) < 0.55*YXnorm))    
    var_vt = YX_dt/YXnorm    
    TYX_d = rotate2d(Tyx_d[:,None],-rho)[:,0]

    var_vt = np.vstack((var_vt,np.repeat(TYX_d,targets).reshape((2,-1)),np.repeat(rho,targets)))
    
    term_Vct = np.zeros((Vars,cofs,targets))    
    for V in range(Vars):
        var_t = var_vt[varidx_V[V]]
        if fntype_V[V]=='l':              # legendre polynomials
            term_ft = np.array([np.ones(targets),var_t,(3.*var_t**2-1.)/2.])
        elif fntype_V[V]=='s':            # sinusoidal
            term_ft = np.array([np.ones(targets),np.sin(np.radians(var_t)),np.cos(np.radians(var_t))])
        term_Vct[V] = term_ft[fn_Vc[V],:]
            
    if (Wavs == 1):
        Wnorm = 3000.
        W0 = 6500.
        W_w = (wav_w - W0)/Wnorm
        x_ds = np.empty((dims,0))
        for t in range(x_dt.shape[1]):
            x_ds = np.hstack((x_ds,np.vstack((np.repeat(x_dt[:,t][:,None],  \
                    wavs,axis=1),W_w.reshape((1,-1))))))  

  # evaluate fit terms, interpolate HWPol to desired wavelengths
    if (Wavs == 1):
        a_cs = (x_ds[:,None,:]**pow_dc[:,:,None]).prod(axis=0) 
        hwcal_ptw = (a_cs[None,:,:,None]*hwcal_pcW[:,:,None,:]).sum(axis=1).reshape((2,targets,wavs))
        ok_tw = np.repeat(ok_t[:,None],wavs,axis=1)
    else:
        a_ct = term_Vct.prod(axis=0)
        hwcal_ptW = (a_ct[None,:,:,None]*hwcal_pcW[:,:,None,:]).sum(axis=1)            
        ok_w = (wav_w > (wav_W[ok_W]).min()) & (wav_w < (wav_W[ok_W]).max())
        ok_tw = (ok_t[:,None] & ok_w[None,:])
        hwcal_ptw = np.zeros((2,targets,wavs))
        for p in [0,1]:
            hwcal_ptw[p,ok_tw] = interp1d(wav_W[ok_W],hwcal_ptW[p,ok_t][:,ok_W], \
                kind='cubic',bounds_error=False)(wav_w[ok_w]).flatten()                               
        hwcal_ptw[1,ok_tw] = hwcal_ptw[1,ok_tw] + PAoff
 
    return hwcal_ptw, ok_tw
 
# ----------------------------------------------------------
def configmap(infilelist,confitemlist,debug='False'):
    """general purpose mapper of observing configurations

    Parameters
    ----------
    infilelist: list
        List of filenames 
    confitemlist: list
        List of header keywords which define a configuration

    Returns
    -------
    obs_i: numpy 1d array
        observation number (unique object, config) for each file
    config_i: numpy 1d array
        config number (unique config) for each file
    obstab: astropy Table
        object name and config number for each observation     
    configtab: astropy Table
        config items for each config
     
    """
  # create the observation log
    obsdict = obslog(infilelist)
    images = len(infilelist)

  # make table of unique polarimetric configurations
    confdatlisti = []
    for i in range(images):
        if obsdict['BS-STATE'][i] == 'Removed': continue
        confdatlist = []
        for item in confitemlist:
            confdatlist.append(obsdict[item][i])
        confdatlisti.append(confdatlist)

    dtypelist = map(type,confdatlist)           
    configtab = Table(np.array(confdatlisti),names=confitemlist,dtype=dtypelist) 
    config_i = np.array([np.where(configtab[i]==unique(configtab))   \
                        for i in range(images)]).flatten()
    configtab = unique(configtab)
                        
  # make table of unique observations
    obsdatlisti = []
    for i in range(images):
        object = obsdict['OBJECT'][i].replace(' ','')
        obsdatlisti.append([object, config_i[i]])

    obstab = Table(np.array(obsdatlisti),names=['object','config'],dtype=[str,int])
    obs_i = np.array([np.where(obstab[i]==unique(obstab))   \
                        for i in range(images)]).flatten()
    obstab = unique(obstab)
                        
    return obs_i,config_i,obstab,configtab
# ------------------------------------

def image_number(image_name):
    """Return the number for an image"""
    return int(os.path.basename(image_name).split('.')[0][-4:])
# ------------------------------------

def list_configurations(infilelist, log):
    """Produce a list of files of similar configurations

    Parameters
    ----------
    infilelist: str
        list of input files

    log: ~logging
        Logging object. 

    Returns
    -------
    iarc_a: list
        list of indices for arc images

    iarc_i:
        list of indices for images

    imageno_i: 
        list of image numbers
    
    
    """
    # set up the observing dictionary
    arclamplist = ['Ar','CuAr','HgAr','Ne','NeAr','ThAr','Xe']
    obs_dict=obslog(infilelist)

    # hack to remove potentially bad data
    for i in reversed(range(len(infilelist))):
        if int(obs_dict['BS-STATE'][i][1])!=2: del infilelist[i]
    obs_dict=obslog(infilelist)

    # inserted to take care of older observations
    old_data=False
    for date in obs_dict['DATE-OBS']:
        if int(date[0:4]) < 2015: old_data=True

    if old_data:
        log.message("Configuration map for old data", with_header=False)
        iarc_a, iarc_i, confno_i, confdatlist = list_configurations_old(infilelist, log)
        arcs = len(iarc_a)
        config_dict = {}
        for i in set(confno_i):
            image_dict={}
            image_dict['arc']=[infilelist[iarc_a[i]]]
            ilist = [infilelist[x] for x in np.where(iarc_i==iarc_a[i])[0]]
            ilist.remove(image_dict['arc'][0])
            image_dict['object'] = ilist
            config_dict[confdatlist[i]] = image_dict
        return config_dict

    # delete bad columns
    obs_dict = obslog(infilelist)
    for k in obs_dict.keys():
        if len(obs_dict[k])==0: del obs_dict[k]
    obs_tab = Table(obs_dict)

    # create the configurations list
    config_dict={}
    confdatlist = configmapset(obs_tab, config_list=('GRATING', 'GR-ANGLE', 'CAMANG', 'BVISITID'))

    infilelist = np.array(infilelist)
    for grating, grtilt, camang, blockvisit in confdatlist:
        image_dict = {}
        #things with the same configuration 
        mask = ((obs_tab['GRATING']==grating) *  
                     (obs_tab['GR-ANGLE']==grtilt) * 
                     (obs_tab['CAMANG']==camang) *
                     (obs_tab['BVISITID']==blockvisit)
               )

        objtype = obs_tab['CCDTYPE']    # kn changed from OBJECT: CCDTYPE lists ARC more consistently
        lamp = obs_tab['LAMPID']
        isarc = ((objtype == 'ARC') | np.in1d(lamp,arclamplist))
                                        # kn added check for arc lamp when CCDTYPE incorrect        
        image_dict['arc'] = infilelist[mask * isarc]

        # if no arc for this config look for a similar one with different BVISITID
        if len(image_dict['arc']) == 0:
            othermask = ((obs_tab['GRATING']==grating) *  \
                     ((obs_tab['GR-ANGLE'] - grtilt) < .03) * ((obs_tab['GR-ANGLE'] - grtilt) > -.03) * \
                     ((obs_tab['CAMANG'] - camang) < .05) * ((obs_tab['CAMANG'] - camang) > -.05) *   \
                     (obs_tab['BVISITID'] != blockvisit))
            image_dict['arc'] = infilelist[othermask * (objtype == 'ARC')]
            if len(image_dict['arc']) > 0:
                log.message("Warning: using arc from different BLOCKID", with_header=False)                
            
        image_dict['flat'] = infilelist[mask * (objtype == 'FLAT')]
        image_dict['object'] = infilelist[mask * ~isarc *  (objtype != 'FLAT')]
        if len(image_dict['object']) == 0: continue
        config_dict[(grating, grtilt, camang, blockvisit)] = image_dict

    return config_dict
# ------------------------------------

def configmapset(obs_tab, config_list=('GRATING','GR-ANGLE', 'CAMANG')):
    """Determine how many different configurations are in the list

    Parameters
    ----------
    obstab: ~astropy.table.Table
        Observing table of image headers

    Returns
    -------
    configs: list
        Set of configurations
    """
    return list(set(zip(*(obs_tab[x] for x in config_list))))
# ------------------------------------

def list_configurations_old(infilelist, log):
    """For data observed prior 2015

    """
    obs_dict=obslog(infilelist)

    # Map out which arc goes with which image.  Use arc in closest wavcal block of the config.
    # wavcal block: neither spectrograph config nor track changes, and no gap in data files
    infiles = len(infilelist)
    newtrk = 5.                                     # new track when rotator changes by more (deg)
    trkrho_i = np.array(map(float,obs_dict['TRKRHO']))
    trkno_i = np.zeros((infiles),dtype=int)
    trkno_i[1:] = ((np.abs(trkrho_i[1:]-trkrho_i[:-1]))>newtrk).cumsum()

    infiles = len(infilelist)
    grating_i = [obs_dict['GRATING'][i].strip() for i in range(infiles)]
    grang_i = np.array(map(float,obs_dict['GR-ANGLE']))
    artic_i = np.array(map(float,obs_dict['CAMANG']))
    configdat_i = [tuple((grating_i[i],grang_i[i],artic_i[i])) for i in range(infiles)]
    confdatlist = list(set(configdat_i))          # list tuples of the unique configurations _c
    confno_i = np.array([confdatlist.index(configdat_i[i]) for i in range(infiles)],dtype=int)
    configs = len(confdatlist)

    imageno_i = np.array([image_number(infilelist[i]) for i in range(infiles)])
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
    return iarc_a, iarc_i, confno_i, confdatlist
#--------------------------------------

def blksmooth1d(ar_x,blk,ok_x):
# blkaverage, then spline interpolate result

    blks = ar_x.shape[0]/blk
    offset = (ar_x.shape[0] - blks*blk)/2
    ar_X = ar_x[offset:offset+blks*blk]                  # cut to integral blocks
    ok_X = ok_x[offset:offset+blks*blk]    
    ar_by = ar_X.reshape(blks,blk)
    ok_by = ok_X.reshape(blks,blk)

    okcount_b = (ok_by.sum(axis=1) > blk/2)
    ar_b = np.zeros(blks)
    for b in np.where(okcount_b)[0]: ar_b[b] = ar_by[b][ok_by[b]].mean()
    gridblk = np.arange(offset+blk/2,offset+blks*blk+blk/2,blk)
    grid = np.arange(ar_x.shape[0])
    arsm_x = ip.griddata(gridblk[okcount_b], ar_b[okcount_b], grid, method="cubic", fill_value=0.)
    oksm_x = (arsm_x != 0.)

    return arsm_x,oksm_x
# ----------------------------------------------------------

def angle_average(ang_d):
# average nparray of angles (floating deg), allowing for wrap at 360
  
    sc_d = SkyCoord(ang_d * u.deg,0. * u.deg)
    sep_d = sc_d[0].separation(sc_d)
    sep_d[sc_d[0].position_angle(sc_d).deg == 270.] *= -1.
    angmean = (sc_d[0].ra + sep_d.mean()).deg % 360.
    return angmean
# ----------------------------------------------------------

def rotate2d(yx_ds, rot, center=np.zeros(2)):
    """rotate an array of 2d coordinates

    Parameters:
    yx_ds: 2d numarray of 2d coordinates
        _d: 0,1 for y,x
        _s: index of coordinates
    rot: amount to rotate (degrees)
    center: y,x coordinates of rotation center (default 0,0)

    """

    c = np.cos(np.radians(rot))
    s = np.sin(np.radians(rot))
    rotate = np.transpose([[c, s],[-1.*s, c]])
    yx1_ds = yx_ds - center[:,None]
    yxout_ds = (np.dot(yx1_ds.T,rotate)).T
    yxout_ds +=  center[:,None]
    return yxout_ds
# ----------------------------------------------------------

def fargmax(arr):
    """returns simple floating argmax using quad fit to top 3 points

    Parameters:
    arr: 1d numarray

    """
    argmax = np.clip(np.argmax(arr),1,len(arr)-2)
    fm,f0,fp = tuple(arr[argmax-1:argmax+2])
    darg = -0.5*(fp - fm)/(fp + fm - 2.*f0)     
    return (argmax + darg)
# ----------------------------------------------------------

def printstdlog(string,logfile):
    print string
    print >>open(logfile,'a'), string
    return 

