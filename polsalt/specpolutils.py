import os
import numpy as np

from astropy.table import Table

from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog

DATADIR = os.path.dirname(__file__) + '/data/'

def rssmodelwave(grating,grang,artic,cbin,cols):
    """compute wavelengths from model of RSS

     TODO:  replace using PySpectrograph
  
    """
    spec=np.loadtxt(DATADIR+"spec.txt",usecols=(1,))
    Grat0,Home0,ArtErr,T2Con,T3Con=spec[0:5]
    FCampoly=spec[5:11]
    grname=np.loadtxt(DATADIR+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=np.loadtxt(DATADIR+"gratings.txt",usecols=(1,2),unpack=True)

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

def image_number(image_name):
    """Return the number for an image"""
    return int(os.path.basename(image_name).split('.')[0][-4:])

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
    confdatlist = configmap(obs_tab, config_list=('GRATING', 'GR-ANGLE', 'CAMANG', 'BVISITID'))

    infilelist = np.array(infilelist)
    for grating, grtilt, camang, blockvisit in confdatlist:
        image_dict = {}
        #things with the same configuration 
        mask = ((obs_tab['GRATING']==grating) *  
                     (obs_tab['GR-ANGLE']==grtilt) * 
                     (obs_tab['CAMANG']==camang) *
                     (obs_tab['BVISITID']==blockvisit)
               )
        objtype = obs_tab['CCDTYPE']          # kn changed from OBJECT: CCDTYPE lists ARC consistently
        image_dict['arc'] = infilelist[mask * (objtype == 'ARC')]

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
        image_dict['object'] = infilelist[mask * (objtype != 'ARC') *  (objtype != 'FLAT')]
        config_dict[(grating, grtilt, camang)] = image_dict
    return config_dict

def configmap(obs_tab, config_list=('GRATING','GR-ANGLE', 'CAMANG')):
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

