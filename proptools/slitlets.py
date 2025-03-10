
import os,sys
import numpy as np
import numpy.lib.recfunctions as rfn
from xml.dom.minidom import Document

polsaltdir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
datadir = polsaltdir+'/polsalt/data/'
sys.path.extend((polsaltdir+'/polsalt/',))

class SlitError(Exception):
    """Base class for exceptions from the slitlet class"""
    pass


class Slitlets:
    """Slitlets is a class describing mask features.   It has properties
       which describe the slitlet including information that would be
       included by the user and the shape of the slitlet.

       name: string character description of the target
       ra : right ascension of the target in decimal degrees
       dec: declination of the target in decimal degrees
       equinox:  equinox of the ra/dec 
       priority:  priority of the target
       mag: magnitude of the target
       band: pass band the magnitude was measured in
       itemtype:  string: 'slit', 'ref', 'avoid', or 'gap'       
       inmask_flag:  whether the item is currently selected for a mask
       fov_flag:  whether the item is currently in the field of view       

    """
    def __init__(self, shape=(10), default_width=1.5, default_length=10.0):
        self.dnames=('name', 'catidx', 'targ_ra', 'targ_dec', 'equinox', 'mag', 'band', 'priority', 'width', 'len1', 'len2', 'tilt',
                     'slit_ra','slit_dec','specwidth', 'speclen1', 'speclen2', 'specdir', 'xoverlap', 'yoverlap',
                     'itemtype', 'collision_id','inmask_flag','fov_flag'
                     )
        self.dformat=('S30', 'i4', 'f4', 'f4', 'i4', 'f4', 'S1', 'f4', 'f4', 'f4',  'f4', 'f4',
                      'f4', 'f4', 'f4', 'f4', 'f4', 'i4', 'f4', 'f4',
                      'S6', 'S30', 'i4', 'i4'
                      )
        self.dtype=zip(self.dnames, self.dformat)
        self.data=None
        self.default_width=default_width
        self.default_length=default_length

    def add_arrays(self, x,y):
        return rfn.stack_arrays((x,y), usemask=False)

    def findslitlet(self, value):
        """Find the slitlet object with a name given by value"""
        name=self.data['name']
        return np.where(name == value)[0][0]

    def findtarget(self, ra, dec):
        """Find the slitlet object with the closest ra and dec"""
        dist = ((self.data['targ_ra'] - ra)**2 + (self.data['targ_dec'] - dec)**2)**0.5
        return dist.argmin()

    def addtomask(self, sid):
        """Add object with a value of sid to the mask"""
        self.data['inmask_flag'][sid] = 1

    def updatevalue(self, sid, value, column):
        """Given a slitlet, update a column with the appropriate value"""
        dformat=self.dformat[self.dnames.index(column)]
        value=self.parseValue(value, dformat)
        self.data[column][sid] = value

    def parseValue(self, value, dformat):
        if dformat.count('S'):
           return str(value)
        elif dformat.count('i'):
           return int(value)
        elif dformat.count('f'):
           return float(value)
        else:
            message='%s is not a supported  format' % dformat
            raise SlitError(message)

    def make_default(self, dformat):
        '''
        depending on the format of the data type, either return a empty string
        or a 0
        '''
        if dformat.count('S'):
           return ''
        return 0
    
    def create_default_slitlet(self):
        '''
        create a default slit using the class dtype
        '''
        
        slitlet = np.rec.array(tuple([self.make_default(i) for i in self.dformat]),dtype=self.dtype)
        return slitlet

    def add_slitlet(self,isrefstar=0):
        '''
        add a default slitlet to the rec array and set the inmask flag = 1 so
        that the new slitlet shows in the slit table or in the reference star
        table
        '''
        
        slitlet = self.create_default_slitlet()
        if isrefstar:
            slitlet['itemtype'] = 'ref'
        slitlet['inmask_flag'] = 1
        self.data = self.add_arrays(self.data, slitlet)
        
    def updategaps(self,ra,dec,pa):       
        geomline = open(datadir+"RSSgeom.dat").readlines()[-1]
        alignline = open(datadir+"RSSimgalign.txt").readlines()[-1]        
        cc,dc1,dc2 = np.array(geomline.split())[[1,2,5]].astype(float)
        dcc = float(alignline.split()[2])
        xgap_dg = (np.array([[-1024.-cc+dc1,-1024.],[1024.,1024+cc+dc2]]) + dcc)/7.95   # in arcsec from optic axis at dtr
        xgap_dg = -xgap_dg[:,::-1]                     # mask view rotated 180 from detector view. Gap1 on the right. 
        
        for d in (0,1):
            gapname = "Gap%1i" % (d+1)
            slitList = np.where(self.data['name']==gapname)[0]
            if len(slitList)==0:
                slitlet = self.create_default_slitlet()
                slitlet['name'] = gapname
                slitlet['itemtype'] = 'gap'
                self.data = self.add_arrays(self.data, slitlet)                
            sid = np.where(self.data['name']==gapname)[0][0]                        
            self.data['width'][sid] = np.diff(xgap_dg[d])
            self.data['len1'][sid] = self.data['len2'][sid] = 240.
            self.data['targ_ra'][sid] = ra - (xgap_dg[d].mean()*np.cos(np.radians(pa))/np.cos(np.radians(dec)))/3600.
            self.data['targ_dec'][sid] = dec + xgap_dg[d].mean()*np.sin(np.radians(pa))/3600.
            self.data['priority'][sid] = -10
            self.data['slit_ra'] = self.data['targ_ra']
            self.data['slit_dec'] = self.data['targ_dec']            
            
    def readascii(self, infile, form='short'):
        #check the number of columns in the catalog
        l = len(open(infile).readline().split())
        #print l
        if l==10: form='long'      

        if form=='short':
            dnames=('name', 'targ_ra', 'targ_dec', 'equinox', 'mag', 'band', 'priority')
            dformat=('S30', 'f4', 'f4', 'i4', 'f4', 'S1', 'f4')
            object_arr=np.loadtxt(infile, dtype={'names': dnames, 'formats': dformat},
                         converters={1:ra_read, 2:dec_read})
            #determine the missing values
            mnames=[]
            mtypes=[]
            for i in range(len(self.dnames)):
                if self.dnames[i] not in dnames:
                   mnames.append(self.dnames[i])
                   mtypes.append(self.dformat[i])
            #set up the default values
            default_list=[np.zeros(len(object_arr))]*len(mnames)
            default_list[0]=default_list[0]+self.default_width
            default_list[1]=default_list[1]+0.5*self.default_length
            default_list[2]=default_list[2]+0.5*self.default_length
            object_arr=rfn.append_fields(object_arr, names=mnames, data=default_list, dtypes=mtypes,
                     fill_value=0, usemask=False)
        elif form=='long':
            dnames=('name', 'targ_ra', 'targ_dec', 'equinox', 'mag', 'band', 'priority', 'width', 'length', 'tilt')
            dformat=('S30', 'f4', 'f4', 'i4', 'f4', 'S1', 'f4', 'f4', 'f4', 'f4')
            object_arr=np.loadtxt(infile, dtype={'names': dnames, 'formats': dformat},
                         converters={1:ra_read, 2:dec_read})
            #determine the missing values
            mnames=[]
            mtypes=[]
            for i in range(len(self.dnames)):
                if self.dnames[i] not in dnames:
                   mnames.append(self.dnames[i])
                   mtypes.append(self.dformat[i])
            #set up the default values
            default_list=[np.zeros(len(object_arr))]*len(mnames)
            length=object_arr['length']
            object_arr=rfn.append_fields(object_arr, names=mnames, data=default_list, dtypes=mtypes,
                     fill_value=0, usemask=False)
            object_arr['len1'] = 0.5 * length
            object_arr['len2'] = 0.5 * length
            #print object_arr
        else:
            message='This format is not supported'
            raise SlitError(message)

        #set objects that are preselected
        object_arr['inmask_flag'] = 1.0*(object_arr['priority'] >= 1.0)
        #set reference stars
        object_arr['itemtype'][(object_arr['priority'] > 0.)] = 'slit'        
        object_arr['itemtype'][(object_arr['priority'] == 0.)] = 'avoid'                
        object_arr['itemtype'][(object_arr['priority'] == -1.0)] = 'ref'
             
        #stack the data if it already exists
        if self.data is None:  
           self.data=object_arr
        else:
           self.data=self.add_arrays(self.data, object_arr)
        # total number of objects:
        self.nobjects=len(self.data)
#        self.update_flags()
        
    def asregion(self, i, pa=0.):
        """Create a ds9 region string from the shape of the object 
        """        
        ra=self.data['targ_ra'][i] 
        dec=self.data['targ_dec'][i]        
        if (self.data['itemtype'][i] == 'ref'):
            shape='circle'
            radius=3#self.data['radius'][i]
            regstr='%s(%f,%f,%f") # color={yellow} tag = {ref} ' % (shape, ra, dec, radius)
        elif (self.data['itemtype'][i] == 'avoid'):
            shape='box'
            length = width = self.data['width'][i]
            regstr='%s(%f,%f,%f",%f") # color={white} tag = {ref} ' % (shape, ra, dec, width, length)                           
        else:
            shape = 'box'
            name = self.data['name'][i]
            width = self.data['width'][i]
            length = self.data['len1'][i]+self.data['len2'][i]
            tilt = self.data['tilt'][i] + pa
            paspec = pa + 90.*(1.-self.data['specdir'][i])           
            raspec = self.data['slit_ra'][i]
            decspec = self.data['slit_dec'][i]
            
            if (self.data['itemtype'][i] == 'gap'):
                regstr = '%s(%f,%f,%f",%f",%f) # color=red dashlist=8 3 dash=1 tag = {slit} tag = {%s}\n ' % (shape, raspec, decspec, width, length, tilt, name)
            else:
                color = 'green'
                regstr = '%s(%f,%f,%f",%f",%f) # color={%s} tag = {slit} tag = {%s}\n ' % (shape, raspec, decspec, width, length, tilt, color, name)
        return regstr

    def asregionspec(self, i, pa=0.):
        """
        Create a ds9 spec region string from the spectral shape of the object, ovelaying spectrum 
        """

        speclen1 = self.data['speclen1'][i]
        speclen2 = self.data['speclen2'][i]
        speclength = speclen1 + speclen2
        dspeclength = (speclen2 - speclen1)/2.
        slittilt = self.data['tilt'][i] - 90.*self.data['specdir'][i] 
        specwidth = self.data['specwidth'][i]  
        dspecwidth = (self.data['len2'][i] - self.data['len1'][i])* np.cos(np.radians(slittilt)) /2.                               
        specpa = pa + 90. - 90.*self.data['specdir'][i]       
        dra = -((dspecwidth*np.cos(np.radians(specpa)) + dspeclength*np.sin(np.radians(specpa))) /    \
            np.cos(np.radians(self.data['targ_dec'][i])))/3600.
        ddec = (dspecwidth*np.sin(np.radians(specpa)) - dspeclength*np.cos(np.radians(specpa)))/3600. 
        targra = self.data['targ_ra'][i]           
        targdec = self.data['targ_dec'][i]        
                        
        catidx = self.data['catidx'][i]        
        txtoff = [specwidth,speclength][self.data['specdir'][i]]/2. + 8.
        txtra = targra - txtoff*(np.sin(np.radians(-pa))/np.cos(np.radians(targdec)))/3600.
        txtdec = targdec + txtoff*np.cos(np.radians(-pa))/3600.
        regstr = 'text (%f,%f,{%s}) # textangle=%f \n' % (txtra,txtdec,str(catidx),pa)

        specra = targra + dra           
        specdec = targdec + ddec            
        shape = 'box'
        if (self.data['itemtype'][i] == 'ref'):
            properties = 'color={black}'
        else:
            properties = 'color={green}'                               
        regstr += '%s(%f,%f,%f",%f",%f) # %s tag = {spec} tag = {%s} \n' %  \
            (shape, specra, specdec, specwidth, speclength, specpa, properties, catidx)

        shape = 'vector'
        properties = 'color={black} vector=0 dashlist=2 2 dash=1'                   
        regstr += '%s(%f,%f,%f",%f) # %s tag = {spec} tag = {%s} \n' %  \
            (shape, targra, targdec, speclen1, specpa+90., properties, catidx)
        regstr += '%s(%f,%f,%f",%f) # %s tag = {spec} tag = {%s} ' %    \
            (shape, targra, targdec, speclen2, specpa+270., properties, catidx)

        return regstr

    def asxml(self, i):
        doc=Document()
        if (self.data['itemtype'][i] == 'ref'):
            card=doc.createElement("refstar")
            card.setAttribute("id", "%s"%(str(self.data['name'][i])))
            card.setAttribute("xce", "%f"%(self.data['targ_ra'][i]))
            card.setAttribute("yce", "%f"%(self.data['targ_dec'][i]))
            card.setAttribute("width", "%f"%(self.data['width'][i]))
            card.setAttribute("length", "%f"%(self.data['len1'][i]+self.data['len2'][i]))
            card.setAttribute("tilt", "%f"%(self.data['tilt'][i]))            
            card.setAttribute("priority", "%f"%(self.data['priority'][i]))
            card.setAttribute("mag", "%f"%(self.data['mag'][i]))
        else:
            card=doc.createElement("slit")
            card.setAttribute("id", "%s"%(str(self.data['name'][i])))
            card.setAttribute("xce", "%f"%(self.data['slit_ra'][i]))
            card.setAttribute("yce", "%f"%(self.data['slit_dec'][i]))
            card.setAttribute("width", "%f"%(self.data['width'][i]))
            card.setAttribute("length", "%f"%(self.data['len1'][i]+self.data['len2'][i]))
            card.setAttribute("tilt", "%f"%(self.data['tilt'][i]))
            card.setAttribute("priority", "%f"%(self.data['priority'][i]))
            card.setAttribute("mag", "%f"%(self.data['mag'][i]))
        return card
 

class slitshape:
    def __init__(self, ra, dec, width=1, length=5, angle=0):
        """Describe the shape for a typical slit.  It is a rectangle
           with a set height and width.  The RA and DEC are the center
           position of the slit and maybe different from the target

           ra--Central Right Ascension in decimal degrees of slit
           dec--Central Declination in decimal degress of slit
           width--width in arcseconds
           height--height in arcseconds
           angle--position angle in degrees with respect to the position angle
                  of the slitmask
        """

        self.ra=ra
        self.dec=dec
        self.width=width
        self.length=length
        self.angle=angle
        self.name='rectangle'

    def findslitlet(value):
        """For a value given for the slitlet name, return that slitlet"""
        try:
            x=np.where(self.data['name']=='8')[0][0]
            return self.data[x]
        except Exception,e:
            raise SlitError(e)


class starshape:
    def __init__(self, ra, dec, radius=1):
        """Describe the shape for the reference star.  It is a circle
           with a set size and radius

           ra--Right Ascension in decimal degrees
           dec--Declination in decimal degress
           radius--radius in arcseconds
        """

        self.ra=ra
        self.dec=dec
        self.radius=radius
        self.name='circle'


def sex2dec(x):
    x=x.split(':')
    if float(x[0])>=0:
        return float(x[0])+float(x[1])/60.0+float(x[2])/3600.0
    else:
        return -(abs(float(x[0]))+float(x[1])/60.0+float(x[2])/3600.0)

def ra_read(x):
    try:
       return float(x)
    except ValueError,e:
       try:
           return 15*sex2dec(x)
       except: 
           return None

def dec_read(x):
    try:
       return float(x)
    except ValueError,e:
       try:
           return sex2dec(x)
       except ValueError,e:
           return None
 


if __name__=='__main__':
   import sys
   s=Slitlets()
   s.readascii(sys.argv[1], form='short')
   print s.data[s.data['inmask_flag']==1]
   print s.data.dtype
   #object_arry, slit_dict=readasciicatalog(sys.argv[1], form='short')
   #print object_arry[0]
   #s=slit_dict['177']
   #print s.shape.name, s.selected  
   #print s.asregion()
