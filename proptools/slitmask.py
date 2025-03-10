# -*- coding: utf-8 -*

import numpy as np
import numpy.lib.recfunctions as rfn
import pywcs
from slitlets import Slitlets
from xml.dom import minidom, getDOMImplementation
import xml.parsers.expat

from PyQt4.QtCore import Qt, QVariant, QObject, SIGNAL, QAbstractTableModel, QModelIndex

class SlitMaskError(Exception):
    """Base class for exceptions from the slitmaks class"""
    pass


class SlitMask(QObject):
    def __init__(self, ui, center_ra=None, center_dec=None, position_angle=0, target_name='', mask_name='',
                equinox=2000, proposal_code='', proposer='', creator='', polarimetry=False, impol=False, filter='', 
                grating='', grang=None, artic=None, validated=False):
        super(QObject, self).__init__();
        self.__center_ra = None
        self.__center_dec = None
        self.__position_angle = None
        self.__equinox = None

        self.add_center_ra(center_ra)
        self.add_center_dec(center_dec)
        self.add_position_angle(position_angle)
        self.add_equinox(equinox)
        
        self.target_name = target_name
        self.mask_name = mask_name
        self.proposal_code = proposal_code
        self.proposer = proposer
        self.creator = creator
        self.polarimetry = polarimetry
        self.impol = impol        
        self.filter = filter
        self.grating = grating
        self.grang = grang
        self.artic = artic                
        self.validated = validated
        
        self.impollenList = [31.3,26.4,22.5,16.5,10.7]               # saltpol.xlsx, to 10000 Ang
        self.impolfilterList = ["PC00000","PC03200","PC03400","PC03850","PC04600"]        
        
        #create the slitlets
        self.ui = ui        
        self.slitlets = Slitlets()

        #needed for writing xml
        self.impl = getDOMImplementation()

#        self.connect(self, SIGNAL('xmlloaded'), self.printcreator)
        
    @property
    def center_ra(self):
        return self.__center_ra
    @property
    def center_dec(self):
        return self.__center_dec
    @property
    def position_angle(self):
        return self.__position_angle
    @property
    def equinox(self):
        return self.__equinox  

    def add_center_ra(self, new_center_ra):
        if new_center_ra is None:
            return
        if new_center_ra < 0 or new_center_ra >= 360:
            self.__center_ra=None
            return
                        
        self.__center_ra = new_center_ra

        return

    def add_center_dec(self, new_center_dec):
        if new_center_dec is None : 
            return
        if new_center_dec < -90 or new_center_dec > 90:
            self.__center_dec=None
            return

        self.__center_dec = new_center_dec

        return

    def add_position_angle(self, new_position_angle):
        """Update the value of the position angle"""

        if new_position_angle is None:
            return
        #if a str is given, try to update to a float

        if new_position_angle < 0 or new_position_angle > 360:
            self.__position_angle=None
            return

        self.__position_angle = new_position_angle

        return

    def add_equinox(self, new_equinox):
        if new_equinox is None:
            self.__equinox=None
            return
        if new_equinox < 0:
            self.__equinox=None
            return

        self.__equinox = new_equinox

        return
        
    def set_MaskPosition(self):
       """Sets the central mask position from the current slit positions.  The setting of the mask position works on the following
          priority: 1) If an object is given with a priority of 2, that object sets the masks position.  2) If objects are
          pre-selected to be in the mask, they set the center position of the mask.  3) Set the center position to be at the 
          center of the catalog
       """
       #get the arrays that you might need
       priority = self.slitlets.data['priority']
       inmask = self.slitlets.data['inmask_flag']
       ra = self.slitlets.data['targ_ra']
       dec = self.slitlets.data['targ_dec']

       if (priority >= 2).any():                                # khn fix logic error       
          cen_ra = ra[priority == 2].mean()
          cen_dec = dec[priority == 2].mean()
          
       elif inmask.any():
          cen_ra = ra[inmask == 1].mean()
          cen_dec = dec[inmask == 1].mean()
       else:
          cen_ra = ra.mean()
          cen_dec = dec.mean()  
 
       self.add_center_ra(cen_ra)
       self.add_center_dec(cen_dec)
       self.add_position_angle(0)

    #
    # ... analogous methods for other properties ...
    #

    def loadxmlinfo(self,par):

        self.proposal_code = par['proposalcode']
        self.proposer = par['pi']
        self.creator = par['creator']
        self.mask_name = par['masknum']
        self.validated = bool(par['validated'])
        
        print 'add_center called in loadxmlinfo'
               
        self.add_center_ra(float(par['centerra']))
        self.add_center_dec(float(par['centerdec']))
        self.add_position_angle(float(par['rotangle']))
        try:
            self.impol = bool(par['impol'])
            self.polarimetry = self.impol
        except:
            self.impol = False        
            try:
                self.polarimetry = bool((par['polarimetry']))
            except:
                self.polarimetry = False                                       
        try:
            self.filter = par['filter']
        except:
            pass
        try:
            self.target_name = par['target']
        except:
            pass

    def readxml(self,slits_dict,refstars_dict):
        '''
        read the dictionaries produced by slitmask.readmaskxml and populate the
        slitlets rec array.
        '''
        xmldnames=('name', 'slit_ra', 'slit_dec', 'mag', 'priority', 'width', 'len1','len2','tilt','itemtype')
        xmldformat=('S30', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4','f4','f4','S6')
        xmldtype=zip(xmldnames, xmldformat)

        # map the dict values to the recarray names
        map = {'name':'id','slit_ra':'slit_ra', 'slit_dec':'slit_dec','mag':'mag',\
        'priority':'priority','width':'width','len1':'len1','len2':'len2','tilt':'tilt','itemtype':'itemtype'}

        slitlist = []
        for i in slits_dict.keys():
            tmp = [slits_dict[i][map[j]] for j in xmldnames]
            slitlist.append(tmp)

        slits_arr = np.rec.array(slitlist,xmldtype)
        #determine the missing values
        mnames=[]
        mtypes=[]
        for i in range(len(self.slitlets.dnames)):
            if self.slitlets.dnames[i] not in xmldnames:
               mnames.append(self.slitlets.dnames[i])
               mtypes.append(self.slitlets.dformat[i])
        #set up the default values
        default_list=[np.zeros(len(slits_arr))]*len(mnames)
        slits_arr=rfn.append_fields(slits_arr, names=mnames, data=default_list,\
            dtypes=mtypes, fill_value=0, usemask=False)
        slits_arr['itemtype'] = 'slit'

        refstarlist = []
        for i in refstars_dict.keys():
            tmp = [refstars_dict[i][map[j]] for j in xmldnames]
            refstarlist.append(tmp)

        refstars_arr = np.rec.array(refstarlist,xmldtype)
        mnames=[]
        mtypes=[]
        for i in range(len(self.slitlets.dnames)):
            if self.slitlets.dnames[i] not in xmldnames:
               mnames.append(self.slitlets.dnames[i])
               mtypes.append(self.slitlets.dformat[i])

        #set up the default itemtype and priority values
        default_list=[np.zeros(len(refstars_arr))]*len(mnames)
        itemtype_arr=rfn.append_fields(refstars_arr, names=mnames, data=default_list, dtypes=mtypes,
                 fill_value=0, usemask=False)
        refstars_arr['priority'] = -1
        refstars_arr['itemtype'] = 'ref'
        
        object_arr = self.slitlets.add_arrays(slits_arr, refstars_arr)
        slits = len(object_arr['name'])
        self.slitlets.data['inmask_flag'] *= 0
        disp_pa = self.position_angle + 90. - 90.*self.impol                
        
        #substitute slit data in the catalog if it already exists, otherwise add it
        if self.slitlets.data is None:
            self.slitlets.data=object_arr
        else:        
            for slitidx in range(len(object_arr)):
                if object_arr['name'][slitidx] in self.slitlets.data['name']:
                    catidx = np.where(object_arr['name'][slitidx]==self.slitlets.data['name'])[0][0]
                    for dname in xmldnames:                
                        self.slitlets.data[dname][catidx]=object_arr[dname][slitidx]
                                        
                    self.slitlets.data['inmask_flag'][catidx] = 1
                  # compute slit offset from difference between catalog targ position and slit center position                    
                    ra = self.slitlets.data['targ_ra'][catidx]
                    dec = self.slitlets.data['targ_dec'][catidx]                                                             
                    drasecs = -3600.*(object_arr['slit_ra'][slitidx] - ra)*np.cos(np.radians(dec)) 
                    ddecsecs = 3600.*(object_arr['slit_dec'][slitidx] - dec)            
                    dlen = np.sign(drasecs+ddecsecs)*np.sqrt(drasecs**2 + ddecsecs**2)                                            
                    self.slitlets.data['len2'][catidx] = object_arr['len2'][slitidx] + dlen
                    self.slitlets.data['len1'][catidx] = object_arr['len1'][slitidx] - dlen                                        
                    self.outFoV_row(catidx)              
                                                                                 
                else:
                    print "Object ", object_arr['name'][slitidx], " is not in catalog: omit"
        self.nobjects=len(self.slitlets.data)
        
    def readmaskxml(self, dom):
        # read all the parameters into dictionaries
        parameters = dom.getElementsByTagName('parameter')

        Param = {}
        for param in parameters:
            Param[str(param.getAttribute('name')).lower()] \
            = str(param.getAttribute('value'))

        # read all the reference stars into dictionaries
        Refstars = {}
        refstars = dom.getElementsByTagName('refstar')
        for refstar in refstars:
            t = {}
            t['id'] = str(refstar.getAttribute('id'))
            t['length'] = float(refstar.getAttribute('length'))
            t['len1'] = float(refstar.getAttribute('length')) / 2.
            t['len2'] = float(refstar.getAttribute('length')) / 2.            
            t['mag'] = float(refstar.getAttribute('mag'))
            t['priority'] = -1
            t['tilt'] = float(refstar.getAttribute('tilt'))               
            t['width'] = float(refstar.getAttribute('width'))                                   
            t['slit_ra'] =  float(refstar.getAttribute('xce'))
            t['slit_dec'] =  float(refstar.getAttribute('yce'))
            t['itemtype'] = 'ref'
            Refstars[refstar.getAttribute('id')] = t

        # read all the slits into dictionaries
        Slits = {}
        slits = dom.getElementsByTagName('slit')
        for slit in slits:
            t = {}
            t['id'] = str(slit.getAttribute('id'))
            t['length'] = float(slit.getAttribute('length'))
            t['len1'] = float(slit.getAttribute('length')) / 2. # will be reset once we know targ
            t['len2'] = float(slit.getAttribute('length')) / 2.
            t['mag'] = float(slit.getAttribute('mag'))            
            t['priority'] = float(slit.getAttribute('priority'))
            t['tilt'] = float(slit.getAttribute('tilt'))             
            t['width'] = float(slit.getAttribute('width'))
            t['slit_ra'] = float(slit.getAttribute('xce'))
            t['slit_dec'] = float(slit.getAttribute('yce'))            
            if t['priority'] > 0:
                t['itemtype'] = 'slit'
            else:
                t['itemtype'] = 'avoid'
            Slits[slit.getAttribute('id')] = t
         
        self.loadxmlinfo(Param)        
        self.readxml(Slits, Refstars)

    def addxmlparameter(self,name,value):
        parameter = self.doc.createElement("parameter")
        parameter.setAttribute('name','%s'%name)
        parameter.setAttribute('value','%s'%value)
        return parameter

    def writexml(self):
        '''
        write out the slitmask and slits info to a xml file.
        '''

        print 'writing xml...'
         #create the xml documents and the main Element called slitmask
        self.doc = self.impl.createDocument(None, "slitmask", None)
        slitmask= self.doc.documentElement
        header = self.doc.createElement("header")
        slitmask.appendChild(header)

        header.appendChild(self.addxmlparameter("VERSION","1.1"))
        header.appendChild(self.addxmlparameter("PROPOSALCODE","%s"%self.proposal_code))
        header.appendChild(self.addxmlparameter("MASKNUM","%s"%self.mask_name))
        header.appendChild(self.addxmlparameter("TARGET","%s"%self.target_name))
        header.appendChild(self.addxmlparameter("PI","%s"%self.proposer))
        header.appendChild(self.addxmlparameter("CREATOR","%s"%self.creator))
        header.appendChild(self.addxmlparameter("ROTANGLE","%s"%self.position_angle))
        header.appendChild(self.addxmlparameter("CENTERRA","%f"%self.center_ra))
        header.appendChild(self.addxmlparameter("CENTERDEC","%f"%self.center_dec))
        header.appendChild(self.addxmlparameter("NSMODE","0"))
        header.appendChild(self.addxmlparameter("VALIDATED","%s"%str(self.validated)))
        if self.impol:
            header.appendChild(self.addxmlparameter("IMPOL","%s"%str(self.impol)))
        else:
            header.appendChild(self.addxmlparameter("POLARIMETRY","%s"%str(self.polarimetry)))
        if self.filter:
            header.appendChild(self.addxmlparameter("FILTER","%s"%self.filter))                    
#        header.appendChild(self.addxmlparameter("SPECLENGTH","12400"))
#        header.appendChild(self.addxmlparameter("SPECOFFSET","0"))
#        header.appendChild(self.addxmlparameter("SPECHEIGHT","0"))

        for i in range(0,len(self.slitlets.data)):
          if self.slitlets.data['inmask_flag'][i]:
            slitcard = self.slitlets.asxml(i)
            slitmask.appendChild(slitcard)

        xml = self.doc.toprettyxml(indent="  ")
        return xml

    def myWcs(self):        
        myWcs = pywcs.WCS(naxis=2)
        myWcs.wcs.crpix = [0., 0.]
        myWcs.wcs.cdelt = np.array([-1.,1.]) / 3600.                  # a "pixel" = 1 arcsec
        myWcs.wcs.crval = [self.center_ra, self.center_dec]
        myWcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        myWcs.wcs.crota = [-self.position_angle, -self.position_angle] # rotate SKY this amount
        myWcs.wcs.equinox = self.equinox
        return myWcs        

    def outFoV_row(self,i):
        '''
        checks if a single row in the slitlets.data array is outside the FoV
        input is a slitlets.data row
        '''

        # first check that the mask info has been defined, otherwise all
        # objects fall outside the FoV
        if self.center_ra == None or self.center_dec == None\
            or self.position_angle == None or self.equinox == None:
                self.slitlets.data[i]['fov_flag'] = 0
                return
        else:
            pass

        # units are arcsec relative to center of FOV

        dcr = 4.* 60. # radius of field (arcsec)

        # convert onsky coords to relative arcsec
        xp,yp = self.myWcs().wcs_sky2pix(self.slitlets.data[i]['targ_ra'],\
                                self.slitlets.data[i]['targ_dec'], 1)
        
        inFOV = (np.sqrt(xp**2+yp**2) < dcr)
        if self.polarimetry:
            inFOV &= (np.abs(yp) < dcr/2.)                

        # set the FoV flag for targets inside the FoV
        self.slitlets.data[i]['fov_flag'] = 1 * inFOV
        self.slitlets.data[i]['inmask_flag'] *= inFOV

        return

    def outFoV(self):
        '''
        this function goes through *all* slitlets.data and populates the FoV flag
        using the outFoV_row function for each entry in the array
        '''
        ra,dec,pa = self.center_ra, self.center_dec, self.position_angle        
        self.slitlets.updategaps(ra,dec,pa)
                         
        for i in range(0,len(self.slitlets.data)):
            if self.slitlets.data[i]['itemtype']=='gap': continue
            if (self.slitlets.data[i]['itemtype'] != 'avoid'):    # objects to avoid 
                self.slitlets.data[i]['inmask_flag'] = 1        
            self.outFoV_row(i)
            
        return


    def set_specbox(self):
        '''
        sets spectrum ra, dec center, and length, width projected to the FOV 
        for now, it is same for all similar-shaped slits
        '''

        catidxs = len(self.slitlets.data['targ_ra'])
        disp_pa = self.position_angle + 90. - 90.*self.impol
        havefilterdata = (self.filter != "")
        
        if havefilterdata:
            filterfile = self.ui.filterdir+self.filter+".txt"            
            wav_l,feff_l = np.loadtxt(filterfile,dtype=float,unpack=True,comments="!")
            lmax = np.argmax(feff_l)
            l1 = np.where(feff_l > feff_l[lmax]/10.)[0][0]
            l2 = (list(np.where(feff_l[lmax:] <     \
                feff_l[lmax]/10.)[0])+[len(feff_l)-lmax])[0]-1 + lmax
            wav_d = wav_l[[l1,l2]]                               # wavelength edges of filter

        havegratingdata = ((self.grating != "")&(self.grang != None)&(self.artic != None))
        if havegratingdata:
            collmag = (11000./150.)
            grname_g=np.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
            grnum = np.where(grname_g==grating)[0][0]
            lmm = np.loadtxt(datadir+"gratings.txt",usecols=(1))[grnum]
            alphar = np.radians(grang)                        
                    
        for i in range(catidxs):        
            self.slitlets.data['specdir'][i] = self.impol          
            # impol gap and bright star avoidance checking:                   
            if ((self.slitlets.data['itemtype'][i] == 'gap')| (self.slitlets.data['itemtype'][i] == 'avoid')):
                self.slitlets.data['slit_ra'][i] = self.slitlets.data['targ_ra'][i]
                self.slitlets.data['slit_dec'][i] = self.slitlets.data['targ_dec'][i]                
                self.slitlets.data['speclen1'][i] = self.slitlets.data['len1'][i]
                self.slitlets.data['speclen2'][i] = self.slitlets.data['len2'][i]
                self.slitlets.data['specwidth'][i] = self.slitlets.data['width'][i]                 
                continue

            ra = self.slitlets.data['targ_ra'][i]
            dec = self.slitlets.data['targ_dec'][i]                       
            xp, yp = self.myWcs().wcs_sky2pix(ra, dec, 1)                
            dlen = (self.slitlets.data['len2'][i]-self.slitlets.data['len1'][i])/2.       
            self.slitlets.data['slit_ra'][i] = ra - dlen*(np.cos(np.radians(disp_pa))/np.cos(np.radians(dec)))/3600.
            self.slitlets.data['slit_dec'][i] = dec + dlen*np.sin(np.radians(disp_pa))/3600.                                            
            slittilt = self.slitlets.data['tilt'][i] - 90.*self.slitlets.data['specdir'][i] 
            specwidth = (self.slitlets.data['len2'][i] + self.slitlets.data['len1'][i])*   \
                np.cos(np.radians(slittilt))             
            self.slitlets.data['specwidth'][i] = specwidth                          
            if self.impol:
                try :
                    filteridx = self.impolfilterList.index(self.filter)
                except:
                    filteridx = 0
                self.slitlets.data['speclen1'][i] = self.impollenList[filteridx]/2.
                self.slitlets.data['speclen2'][i] = self.impollenList[filteridx]/2.            
            else:
                if (havefilterdata & havegratingdata):
                    gammar = np.radians(yp/3600.)*collmag
                    dalphar = np.radians(xp/3600.)*collmag    
                    betar_d = np.arcsin((wav_d*lmm/1.e7)/np.cos(gammar) - np.sin(alphar + dalphar))
                    xedge_d = -np.degrees((betar_d + alphar - np.radians(artic))/collmag)*3600.
                else:
                    xedge_d = [xp + 360.,360. - xp]             # full length of detector                            
                self.slitlets.data['speclen1'][i] = xedge_d[0]
                self.slitlets.data['speclen2'][i] = xedge_d[1]

        return         
                
    def find_collisions(self):
        '''
        this function checks for slit collisions
        '''
        catidxs = len(self.slitlets.data['targ_ra'])
        self.slitlets.data['xoverlap'] = np.zeros(catidxs)
        self.slitlets.data['yoverlap'] = np.zeros(catidxs)        
        self.slitlets.data['collision_id'] = np.repeat([''],catidxs)

        if self.center_ra == None or self.center_dec == None\
            or self.position_angle == None or self.equinox == None:
            return

        # print "Checking for collisions"
        
        # convert onsky coords to pix
        xp_i, yp_i = self.myWcs().wcs_sky2pix(self.slitlets.data['targ_ra'], self.slitlets.data['targ_dec'], 1)
        islitList = list(np.where(((self.slitlets.data['itemtype'] == 'slit') &     \
                            (self.slitlets.data['fov_flag'] == 1) & \
                            (self.slitlets.data['inmask_flag'] == 1)))[0])
        if (len(islitList)==0):  
           print 'No objects in mask'
           return

        # first update FOV against gaps for refs and avoids, and for slits in ImPol mode
        gapavoid = 1.
        igapList = list(np.where(self.slitlets.data['itemtype'] == 'gap')[0])
        irefList = list(np.where(self.slitlets.data['itemtype'] == 'ref')[0])
        iavoidList = list(np.where(self.slitlets.data['itemtype'] == 'avoid')[0])        
        xpleft_g = xp_i[igapList] - self.slitlets.data['width'][igapList]/2. 
        xpright_g = xp_i[igapList] + self.slitlets.data['width'][igapList]/2.
       
        for i in [irefList+iavoidList,irefList+iavoidList+islitList][self.impol]:
            ingap = ((xp_i[i] > (xpleft_g - gapavoid)) & (xp_i[i] < (xpright_g + gapavoid))).any()
            if ingap:
                self.slitlets.data['fov_flag'][i] = 0
                self.slitlets.data['inmask_flag'][i] = 0                         
        islitList = list(np.where(((self.slitlets.data['itemtype'] == 'slit') & \
                            (self.slitlets.data['fov_flag'] == 1) &             \
                            (self.slitlets.data['inmask_flag'] == 1)))[0])
        irefList =  list(np.where(((self.slitlets.data['itemtype'] == 'ref') & \
                            (self.slitlets.data['fov_flag'] == 1) &             \
                            (self.slitlets.data['inmask_flag'] == 1)))[0])
        iavoidList =  list(np.where(((self.slitlets.data['itemtype'] == 'avoid') &    \
                            (self.slitlets.data['fov_flag'] == 1)))[0])
        if (len(islitList)==0): 
            print 'No objects in mask after gap check'
            return
            
        # now check for overlaps of slits against avoiders, slits, refs, and gaps (impol only) perpendicular to dispersion            
        self.set_specbox()
        slittilt_i = self.slitlets.data['tilt'] - 90.*self.slitlets.data['specdir']
        specwidth_i = self.slitlets.data['specwidth']        
        dspecwidth_i = (self.slitlets.data['len1'] - self.slitlets.data['len2'])* np.cos(np.radians(slittilt_i))
        specbox_i = np.array([-self.slitlets.data['speclen1'], self.slitlets.data['speclen2'], \
                              -(specwidth_i+dspecwidth_i)/2., (specwidth_i-dspecwidth_i)/2.])
        slitbox_i = np.array([-self.slitlets.data['width']/2., self.slitlets.data['width']/2.,  \
                              -self.slitlets.data['len1'], self.slitlets.data['len2']])        
        self.slitlets.data['xoverlap'] = np.zeros(catidxs)
        self.slitlets.data['yoverlap'] = np.zeros(catidxs)
        spacing = float(self.ui.tableWidgetOptcollParam.item(0,0).text())
        xspacing = [0.,spacing][self.impol]            # only use spacing perp to dispersion
        yspacing = [spacing,0.][self.impol]  
      
        for i in islitList:        
            tcoll_ids=[]
            # first check slits against avoid objects
            if self.impol:
                ymin_i,ymax_i,xmin_i,xmax_i = np.array([yp_i,yp_i,xp_i,xp_i]) + slitbox_i      
            else:
                xmin_i,xmax_i,ymin_i,ymax_i = np.array([xp_i,xp_i,yp_i,yp_i]) + slitbox_i              
            for j in iavoidList:
                if i != j:
                    xoverlap = max(0, min(xmax_i[i], xmax_i[j]) - max(xmin_i[i], xmin_i[j]) +   \
                        xspacing) / (xmax_i[i]-xmin_i[i])
                    yoverlap = max(0, min(ymax_i[i], ymax_i[j]) - max(ymin_i[i], ymin_i[j]) +   \
                        yspacing) / (ymax_i[i]-ymin_i[i])
                    overlap = xoverlap*yoverlap
                    if (overlap > 0.):
                        self.slitlets.data['xoverlap'][i] += 100.*xoverlap          # in percent
                        self.slitlets.data['yoverlap'][i] += 100.*yoverlap

                        collidx = self.slitlets.data['catidx'][j]
                        if (self.slitlets.data['itemtype'][j]=='gap'):
                            collidx = self.slitlets.data['name'][j]
                        tcoll_ids.append(collidx)
                        
            # next check slit spectra against other spectra (and gap, for imPol)
            if self.impol:
                icheckList = (islitList + irefList + igapList)            
                ymin_i,ymax_i,xmin_i,xmax_i = np.array([yp_i,yp_i,xp_i,xp_i]) + specbox_i      
            else:
                icheckList = (islitList + irefList )            
                xmin_i,xmax_i,ymin_i,ymax_i = np.array([xp_i,xp_i,yp_i,yp_i]) + specbox_i              
            for j in icheckList:
                if i != j:
                    xoverlap = max(0, min(xmax_i[i], xmax_i[j]) - max(xmin_i[i], xmin_i[j]) +   \
                        xspacing) / (xmax_i[i]-xmin_i[i])
                    yoverlap = max(0, min(ymax_i[i], ymax_i[j]) - max(ymin_i[i], ymin_i[j]) +   \
                        yspacing) / (ymax_i[i]-ymin_i[i])
                    overlap = xoverlap*yoverlap
                    if (overlap > 0.):
                        self.slitlets.data['xoverlap'][i] += 100.*xoverlap          # in percent
                        self.slitlets.data['yoverlap'][i] += 100.*yoverlap

                        collidx = self.slitlets.data['catidx'][j]
                        if (self.slitlets.data['itemtype'][j] == 'gap'):
                            collidx = self.slitlets.data['name'][j]
                        tcoll_ids.append(collidx)

            if (self.slitlets.data['xoverlap']*self.slitlets.data['yoverlap'])[i]:
               self.slitlets.data['collision_id'][i] = " ".join(['%s' % x for x in tcoll_ids])
        
    def update_fov_slitlets(self):
        changed = False
        if changed:
            # send signal
            pass

    def add_slitlet(self, slitlet, in_mask=False):
        self.__all_slitlets[slitlet.id] = slitlet
        # connect to slit
        # send signal
        if in_mask:
            self.__mask_slitlets.add(slitlet.id)
            # send signal

    def remove_slitlet(self, slitlet):
        del(self.__all_slitlets[slitlet.id])
        # disconnect from slitlet
        # send signal
        if slitlet.id in self.__mask_slitlets:
            self.__mask_slitlets.remove(slitlet.id)
            # send signal
        if slitlet.id in self.__fov_slitlets:
            self.__fov_slitlets.remove(slitlet.id)
            # send signal

    def add_to_mask(self, slitlet):
        if not slitlet.id in self.__all_slitlets.keys():
            raise SlitError()
        if slitlet.id in self.__mask_slitlets:
            return
        self.__mask_slitlets.add(slitlet.id)
        # send signal
        # update FOV list?
        # update collisions

    def remove_from_mask(self, slitlet):
        if not slitlet.id in self.__mask_slitlets:
            return
        self.__mask_slitlets.remove(slitlet.id)
        # send signal
        # update FOV list?
        # update collisions

    def slitlet_changed(self, slitlet):
        """callback method for handling slit changes
        """
        # update FOV list?
        # update collisions
        pass

    @staticmethod
    def is_in_fov(slitlet):
        pass

    @staticmethod
    def read_from_file():
        pass

#    def FOVTest(cra,cdec,equinox,rotang,slitra,slitdec,slit_length,tilt):
    def outFoV_all(self):

        # first check that the mask info has been defined, otherwise all
        # slits fall outside the FoV
        if self.center_ra == None or self.center_dec == None\
            or self.position_angle == None or self.equinox == None:
                self.slitlets.data['fov_flag'] = 0
                return
        else:
            pass

        pixscale = 0.2507 / 2. # unbinned pixels

        dcr = 4. / 60. # radius of field (deg)
        # global CCD parameters:
        ccd_dx = 2034.
        ccd_xgap = 70.
        ccd_dy = 4102.

        # define centre in pixel coords
        ccd_cx = (2.*(ccd_dx + ccd_xgap) + ccd_dx) / 2.
        ccd_cy = ccd_dy / 2.

        # setup the field WCS coords.
        wcs = pywcs.WCS(naxis=2)
        wcs.wcs.crpix = [ccd_cx,ccd_cy]
        wcs.wcs.cdelt = np.array([-pixscale, pixscale]) / 3600. # set in degrees
        wcs.wcs.crval = [self.center_ra, self.center_dec]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crota = [self.position_angle, self.position_angle] # rotate SKY this amount?
        wcs.wcs.equinox = self.equinox

        # convert onsky coords to pix
        xp, yp = wcs.wcs_sky2pix(self.slitlets.data['targ_ra'],
                                self.slitlets.data['targ_dec'], 1)

        # testing for all four slit corners
        # upper left corner
        ulx = xp - (self.slitlets.data['slit_width'] /2. ) / pixscale
        uly = yp + self.slitlets.data['len1'] / pixscale
        # upper right corner
        urx = xp + (self.slitlets.data['slit_width'] / 2.) / pixscale
        ury = yp + self.slitlets.data['len1'] / pixscale
        # lower left corner
        llx = xp - (self.slitlets.data['slit_width'] / 2.) / pixscale
        lly = yp - self.slitlets.data['len2'] / pixscale
        # lower right corner
        lrx = xp + (self.slitlets.data['slit_width'] /2. ) / pixscale
        lry = yp - self.slitlets.data['len2'] / pixscale

        # determine the distances to each corner
        uldist = (uly - ccd_cy)**2 + (ulx - ccd_cx)**2
        urdist = (ury - ccd_cy)**2 + (urx - ccd_cx)**2
        lldist = (lly - ccd_cy)**2 + (llx - ccd_cx)**2
        lrdist = (lry - ccd_cy)**2 + (lrx - ccd_cx)**2

        # test if each corner lies outside the FoV, the boolean array returns
        # true if a corner lies outside the FoV
        maxdist = (dcr * 3600.0 / pixscale)**2
        ultest = uldist <= maxdist
        urtest = urdist <= maxdist
        lltest = lldist <= maxdist
        lrtest = lrdist <= maxdist

        # the slitlet lies in the FoV only if all tests were passed
        # if only one test is failed its False
        pass_test = ultest * urtest * lltest * lrtest

        # convert the list to a np array
        pass_test = np.array(pass_test)

        # set the FoV flag for slits inside the FoV

        self.slitlets.data['fov_flag'] = 1 * pass_test

        return


if __name__=='__main__':
    import sys
   
