# -*- coding: utf-8 -*-
import os,sys

print "python version: ",(sys.version)

import numpy as np

from astropy.io import fits as pyfits   # khn fix
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import SIGNAL, SLOT, QObject

from slitlets import Slitlets, ra_read, dec_read
from slitmask import SlitMask

from rsmt_gui import Ui_MainWindow
from infotab import InfoTab
from catalogtab import CatalogTab
from slittab import SlitTab
from reftab import RefTab
from optimizetab import OptimizeTab
from finalizetab import FinalizeTab
from ImageDisplay import ImageDisplay

# added these two import to avoid a seg fault
from pyraf import iraf
from iraf import pysalt

#import warnings
#warnings.filterwarnings("error") 

class SlitMaskGui(QtGui.QMainWindow, InfoTab, CatalogTab, OptimizeTab, SlitTab, RefTab, FinalizeTab):
    def __init__(self, parent=None, infile=None, inimage=None, center_ra=None, center_dec=None,     \
            position_angle=None, polarimetry=False, debug=False):
        QtGui.QWidget.__init__(self, parent)
   
        #set up the main UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)        
        self.ui.filterdir = str(iraf.osfn("pysalt$data/rss/filters/") )       
     
        #set up the slitmask
        self.polarimetry=polarimetry        
        self.slitmask=SlitMask(self.ui,center_ra=center_ra, center_dec=center_dec,  \
            position_angle=position_angle, polarimetry=self.polarimetry)
        self.slitlets=self.slitmask.slitlets
        
        # setup default values for the optimizer
        self.optimize=OptimizeTab(self.ui,opt_spacing=1.,opt_crossmaxshift=25., opt_allowspeccoll=5.,setupDict=None)
        self.optimize.updatecollparamtable(self.optimize.spacing,0)        
        self.optimize.updatecollparamtable(self.optimize.crossmaxshift,1)
        self.optimize.updatecollparamtable(self.optimize.allowspeccoll,2)            

        #read in the input data if available
        self.debug = debug
        self.infile = infile 
        if infile:
            self.ui.radioButtonInfo_Catalogue.setChecked(True)
            self.setmode2cat()
            self.entercatalog(infile)            
            if self.slitmask.center_ra is None and self.slitmask.center_dec is None:
                self.slitmask.set_MaskPosition()                
            ra,dec,pa = self.slitmask.center_ra, self.slitmask.center_dec, self.slitmask.position_angle
            self.slitlets.updategaps(ra,dec,pa)
            self.slitmask.set_specbox()                                
            self.ui.lineEditMain_CenRA.setText(str(ra))
            self.ui.lineEditMain_CenDEC.setText(str(dec))
            self.ui.lineEditMain_PA.setText(str(pa))
            self.ui.lineEditMain_Equinox.setText(str(self.slitmask.equinox))            
            self.ui.lineEditMain_TargetName.setText(self.slitmask.target_name)
            self.ui.lineEditMain_MaskName.setText(self.slitmask.mask_name)
            self.ui.lineEditInfo_Creator.setText(self.slitmask.creator)
            self.ui.lineEditInfo_Proposer.setText(self.slitmask.proposer)
            self.ui.lineEditInfo_ProposalCode.setText(self.slitmask.proposal_code)
        else:
            self.ui.radioButtonInfo_Catalogue.setChecked(False)
            self.ui.radioButtonInfo_Manual.setChecked(True)
            self.setmode2manual()
            self.ui.toolButtonCat_Load.setEnabled(True)

        #setup the image interaction
        self.imagedisplay=ImageDisplay(target='pySlitMask:5909')        
        self.position_angle=position_angle         
        if inimage:
            self.loadimage(inimage)         
            self.displayslits()            
        #set up some variables that will be needed later
        self.xmlfile=None
        self.fcfile=None

        if infile: self.updatetabs()
   
        #Listen to different signals

        #menu items
        QtCore.QObject.connect(self.ui.actionLoad_Catalogue, QtCore.SIGNAL("triggered()"), self.loadcatalog)
        QtCore.QObject.connect(self.ui.actionLoad_Image, QtCore.SIGNAL("triggered()"), self.loadimage)
        QtCore.QObject.connect(self.ui.actionLoad_Xml, QtCore.SIGNAL("triggered()"), self.enterxml)        

        #main tabs
        QtCore.QObject.connect(self.ui.lineEditMain_CenRA, QtCore.SIGNAL("returnPressed()"), self.loadCenRA)
        QtCore.QObject.connect(self.ui.lineEditMain_CenDEC, QtCore.SIGNAL("returnPressed()"), self.loadCenDEC)
        QtCore.QObject.connect(self.ui.lineEditMain_PA, QtCore.SIGNAL("returnPressed()"), self.loadpositionangle)
        QtCore.QObject.connect(self.ui.lineEditMain_Equinox, QtCore.SIGNAL("returnPressed()"), self.loadequinox)
        QtCore.QObject.connect(self.ui.lineEditMain_TargetName, QtCore.SIGNAL("editingFinished()"), self.loadtargetname)
        QtCore.QObject.connect(self.ui.lineEditMain_MaskName, QtCore.SIGNAL("editingFinished()"), self.loadmaskname)
        
        #info tabs
        QtCore.QObject.connect(self.ui.lineEditInfo_ProposalCode, QtCore.SIGNAL("editingFinished()"), self.loadproposalcode)
        QtCore.QObject.connect(self.ui.lineEditInfo_Proposer, QtCore.SIGNAL("editingFinished()"), self.loadproposer)
        QtCore.QObject.connect(self.ui.lineEditInfo_Creator, QtCore.SIGNAL("editingFinished()"), self.loadcreator)
        QtCore.QObject.connect(self.ui.checkBoxInfo_Polarimetry, QtCore.SIGNAL("stateChanged(int)"), self.setpolarimetrymode)
        QtCore.QObject.connect(self.ui.checkBoxInfo_ImPol, QtCore.SIGNAL("stateChanged(int)"), self.setimpolmode)        
        QtCore.QObject.connect(self.ui.comboBoxInfo_Filter, QtCore.SIGNAL("activated(QString)"), self.loadfilter)
        QtCore.QObject.connect(self.ui.comboBoxInfo_Grating, QtCore.SIGNAL("activated(QString)"), self.loadgrating)

#        QtCore.QObject.connect(self.slitmask, SIGNAL('xmlloaded'), self.setcreator)

        QtCore.QObject.connect(self.ui.radioButtonInfo_Catalogue, QtCore.SIGNAL("clicked()"), self.setmode2cat)
        QtCore.QObject.connect(self.ui.radioButtonInfo_Manual, QtCore.SIGNAL("clicked()"), self.setmode2manual)
        QtCore.QObject.connect(self.ui.checkBoxInfo_CentroidOn, QtCore.SIGNAL("clicked()"), self.setmodecentroiding)
        
        #catalog tabs
        QtCore.QObject.connect(self.ui.toolButtonCat_Load, QtCore.SIGNAL("clicked(bool)"), self.loadcatalog)
        QtCore.QObject.connect(self.ui.pushButtonCat_AddSlits, QtCore.SIGNAL("clicked(bool)"), self.addslitfromcatalog)
        QtCore.QObject.connect(self.ui.pushButtonCat_Clear, QtCore.SIGNAL("clicked()"), self.clearContents)

        #slit tab
        QtCore.QObject.connect(self.ui.pushButtonSlit_SaveSlitFile, QtCore.SIGNAL("clicked()"), self.writeslittab)        
        QtCore.QObject.connect(self.ui.pushButtonSlit_ClearSlits, QtCore.SIGNAL("clicked()"), self.clearslittable)
        QtCore.QObject.connect(self.ui.pushButtonSlit_AddSlitImage, QtCore.SIGNAL("clicked()"), self.addslitletsfromimage)
        QtCore.QObject.connect(self.ui.pushButtonSlit_AddSlitfromCat, QtCore.SIGNAL("clicked()"), self.addslitletsfromcatalogue)
        QtCore.QObject.connect(self.ui.pushButtonSlit_AddSlit, QtCore.SIGNAL("clicked()"), self.addslitmanually)
        QtCore.QObject.connect(self.ui.pushButtonSlit_DeleteSlit, QtCore.SIGNAL("clicked()"), self.deleteslitmanually)
        QtCore.QObject.connect(self.ui.pushButtonSlit_DeleteSlitImage, QtCore.SIGNAL("clicked()"), self.deleteslitfromimage)
        QtCore.QObject.connect(self.ui.tableWidgetSlits, QtCore.SIGNAL("itemSelectionChanged()"), self.setposition)
        QtCore.QObject.connect(self.ui.tableWidgetSlits, QtCore.SIGNAL("cellChanged(int, int)"), self.slitchanged)

        #ref stars
        QtCore.QObject.connect(self.ui.pushButtonRef_ClearRefstars, QtCore.SIGNAL("clicked()"), self.clearrefstartable)
        QtCore.QObject.connect(self.ui.pushButtonRef_AddRefstarImage, QtCore.SIGNAL("clicked()"), self.addslitletsfromimage)
        QtCore.QObject.connect(self.ui.pushButtonRef_AddRefstarsfromCat, QtCore.SIGNAL("clicked()"), self.addrefstarsfromcatalogue)
        QtCore.QObject.connect(self.ui.pushButtonRef_AddRefstar, QtCore.SIGNAL("clicked()"), self.addrefstarmanually)
        QtCore.QObject.connect(self.ui.pushButtonRef_DeleteRefstar, QtCore.SIGNAL("clicked()"), self.deleterefstarmanually)
        QtCore.QObject.connect(self.ui.pushButtonRef_DeleteRefstar_2, QtCore.SIGNAL("clicked()"), self.deleteslitfromimage)
        QtCore.QObject.connect(self.ui.tableWidgetRefstars, QtCore.SIGNAL(" itemSelectionChanged()"), self.setrefposition)
        QtCore.QObject.connect(self.ui.tableWidgetRefstars, QtCore.SIGNAL("cellChanged(int, int)"), self.refchanged)

        #optimize tab      
        QtCore.QObject.connect(self.ui.pushButtonOpt_Optimize, QtCore.SIGNAL("clicked(bool)"), self.do_optimize)
        QtCore.QObject.connect(self.ui.pushButtonOpt_Adoptit, QtCore.SIGNAL("clicked(bool)"), self.adopt_best)        
        QtCore.QObject.connect(self.ui.radioButtonOpt_Start, QtCore.SIGNAL("clicked(bool)"), self.optdisplaystart)
        QtCore.QObject.connect(self.ui.radioButtonOpt_Best, QtCore.SIGNAL("clicked(bool)"), self.optdisplaybest)
        QtCore.QObject.connect(self.ui.tableWidgetOptcollParam, QtCore.SIGNAL("cellChanged(int, int)"), self.collparamchanged)
        QtCore.QObject.connect(self.ui.tableWidgetOptsetup, QtCore.SIGNAL("cellChanged(int, int)"), self.optsetupchanged)
        QtCore.QObject.connect(self.ui.tableWidgetOptNudge, QtCore.SIGNAL("cellClicked(int, int)"), self.optnudge)        
        
        # finalize tab
        QtCore.QObject.connect(self.ui.pushButtonFin_Validate, QtCore.SIGNAL("clicked(bool)"),self.validator)
        QtCore.QObject.connect(self.ui.pushButtonFin_WriteXML, QtCore.SIGNAL("clicked(bool)"), self.writexml)
        QtCore.QObject.connect(self.ui.toolButtonFin_WriteRSMT, QtCore.SIGNAL("clicked(bool)"), self.writersmt)
        QtCore.QObject.connect(self.ui.pushButtonFin_CreateFChart_Current, QtCore.SIGNAL("clicked(bool)"), self.writeFC_Current)
        QtCore.QObject.connect(self.ui.pushButtonFin_CreateFChart_DSS, QtCore.SIGNAL("clicked(bool)"), self.writeFC_DSS)
 
        self.ui.tabWidget.setCurrentIndex(0)

    def clearContents(self):
        self.slitlets.data = None
        self.ui.tableWidgetCat.clearContents()
        self.ui.tableWidgetCat.setRowCount(0)
        #TODO: Set the number of rows to the current data length
        #print 'nope not doing it'

    # loads mask coordinates
    def loadCenRA(self):
        newcen_ra = ra_read(self.ui.lineEditMain_CenRA.text())
        if newcen_ra==None: return
        self.slitmask.validated = False        

        if self.slitmask.center_ra == None:
            print ('CenRA signal: None %12.5f' % newcen_ra)        
            unchanged = False
            palette = self.setPalette('error')
                                        
        else:
            print ('CenRA signal: %12.5f %12.5f' % (self.slitmask.center_ra,newcen_ra))
            unchanged = (np.abs(newcen_ra - self.slitmask.center_ra) < .001)
            palette = self.setPalette('normal')

        self.ui.lineEditMain_CenRA.setPalette(palette)
        if unchanged: return
        self.slitmask.add_center_ra(ra_read(self.ui.lineEditMain_CenRA.text()))
        print "CenRA: updating FOV, all slits"              
        self.slitmask.outFoV()
        self.slitmask.find_collisions()
        self.updatetabs()

    def loadCenDEC(self):
        newcen_dec = dec_read(self.ui.lineEditMain_CenDEC.text())
        if newcen_dec==None: return            
        self.slitmask.validated = False

        if self.slitmask.center_dec == None:
            print ('CenDEC signal:  None %12.5f' % newcen_dec)
            unchanged = False                   
            palette = self.setPalette('error')
        else:
            print ('CenDEC signal: %12.5f %12.5f' % (self.slitmask.center_dec,newcen_dec))
            unchanged = (np.abs(newcen_dec - self.slitmask.center_dec) < .001)        
            palette = self.setPalette('normal')

        self.ui.lineEditMain_CenDEC.setPalette(palette)
        if unchanged: return            
        self.slitmask.add_center_dec(dec_read(self.ui.lineEditMain_CenDEC.text()))        
        print "CenDEC: updating FOV, all slits"                 
        self.slitmask.outFoV()
        self.slitmask.find_collisions()            
        self.updatetabs()

    def loadpositionangle(self):
        self.slitmask.validated = False
        self.slitmask.add_position_angle(dec_read(self.ui.lineEditMain_PA.text()))
        
        print 'add_position_angle called in loadpositionangle signal: ', self.slitmask.position_angle
                
        if self.slitmask.position_angle == None:
            palette = self.setPalette('error')
            self.ui.lineEditMain_PA.setPalette(palette)
        else:
            palette = self.setPalette('normal')
            self.ui.lineEditMain_PA.setPalette(palette)
            self.ui.lineEditMain_PA.setText(str(self.slitmask.position_angle))
            print "Positional Angle: updating FOV, all slits"               
            self.slitmask.outFoV()
            self.slitmask.find_collisions()              
            self.imagedisplay.rotate(-self.slitmask.position_angle)
            self.updatetabs()

    def loadequinox(self):
        self.slitmask.validated = False
        self.slitmask.add_equinox(dec_read(self.ui.lineEditMain_Equinox.text()))
        if self.slitmask.equinox == None:
            palette = self.setPalette('error')
            self.ui.lineEditMain_Equinox.setPalette(palette)
        else:
            palette = self.setPalette('normal')
            self.ui.lineEditMain_Equinox.setPalette(palette)
            self.ui.lineEditMain_Equinox.setText(str(self.slitmask.equinox))
            print "equinox: updating FOV, all slits"                 
            self.slitmask.outFoV()
            self.slitmask.find_collisions()
            self.updatetabs()                        
            
     # load info from the main window
    def loadtargetname(self):
        self.slitmask.target_name=str(self.ui.lineEditMain_TargetName.text()).strip()
        if self.slitmask.validated:  
           if len(self.slitmask.target_name)==0: self.slitmask.validated=False

    def loadmaskname(self):
        self.slitmask.mask_name=str(self.ui.lineEditMain_MaskName.text()).strip()
        if self.slitmask.validated:  
           if len(self.slitmask.mask_name)==0: self.slitmask.validated=False
   
    def loadValue(self):
        self.updatetabs()

    def updatetabs(self):
        """Update all of the information after changes to the slitlet class"""
        # print "pySlitMask updatetabs"
        self.updatecatalogtable()
        self.updateslittable()
        self.updaterefstartable()    
        self.displayslits()

    def displayslits(self): 
        """Add the gaps, slits, fov to the image in frame 1 """
        self.imagedisplay.setframe(1)               
        self.imagedisplay.deleteregions()         
        slit_s = np.where((self.slitlets.data['inmask_flag']==1))[0]
        gap_s = np.where((self.slitlets.data['itemtype']=='gap'))[0]
        avoid_s = np.where((self.slitlets.data['itemtype']=='avoid')&(self.slitlets.data['fov_flag']==1))[0]        
        fout = open('tmp1.reg', 'w')
        fout.write('# Region file format: DS9 version 4.1\n# Filename: sgpR.fits\n')
        fout.write('global color=red width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        fout.write('fk5\n')
        for i in gap_s:                       
            fout.write(self.slitlets.asregion(i,self.slitmask.position_angle)+'\n')
        for i in avoid_s:
            fout.write(self.slitlets.asregion(i,self.slitmask.position_angle)+'\n')                   
        for i in slit_s:
            fout.write(self.slitlets.asregion(i,self.slitmask.position_angle)+'\n')

      # show RSS FOV if maskcenter defined
        ra,dec,pa = self.slitmask.center_ra, self.slitmask.center_dec, self.slitmask.position_angle
        if ra:
            fout.write('global color=orange width=2\n'+'\n')
            fout.write('circle(%f,%f,4\')\n ' % (ra,dec)+'\n')
            if self.slitmask.polarimetry:
                dr, dpa = (4./60., 60.)          # where 4' circle meets 2' line
                dra1,dra2 = dr*np.sin(np.radians(pa+np.array([1.,-1.])*dpa))/np.cos(np.radians(dec))
                ddec1,ddec2 =  dr*np.cos(np.radians(pa+np.array([1.,-1.])*dpa))
                fout.write('line(%f,%f,%f,%f)\n ' % (ra+dra1,dec+ddec1,ra+dra2,dec+ddec2)+'\n')
                fout.write('line(%f,%f,%f,%f)\n ' % (ra-dra1,dec-ddec1,ra-dra2,dec-ddec2)+'\n')
        fout.close()
        self.imagedisplay.regionfromfile('tmp1.reg')
        
    def displayspectra(self): 
        """Add the gaps, detector boundary, spec footprint to frame 2 """
        self.readyspectrumdisplay()              
        self.imagedisplay.setframe(2)               
        self.imagedisplay.deleteregions()        
        slit_s = np.where((self.slitlets.data['inmask_flag']==1))[0]
        gap_s = np.where((self.slitlets.data['itemtype']=='gap'))[0]
        avoid_s = np.where((self.slitlets.data['itemtype']=='avoid')&(self.slitlets.data['fov_flag']==1))[0]        
        fout = open('tmp2.reg', 'w')
        fout.write('# Region file format: DS9 version 4.1\n# Filename: sgpR.fits\n')
        fout.write('global color=red width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        fout.write('fk5\n')
        for i in gap_s:                       
            fout.write(self.slitlets.asregion(i,self.slitmask.position_angle)+'\n')

      # show detector boundary
        ra,dec,pa = self.slitmask.center_ra, self.slitmask.center_dec, self.slitmask.position_angle
        if ra:
            fout.write('global color=orange width=2\n'+'\n')
            fout.write('box(%f,%f,6\',4\')\n ' % (ra,dec)+'\n')
                       
            if self.slitmask.polarimetry:
                fout.write('vector(%f,%f,-3\',0,vector=0)\n ' % (ra,dec)+'\n') 
                fout.write('vector(%f,%f,3\',0,vector=0)\n ' % (ra,dec)+'\n')               
                dr, dpa = (4./60., 60.)          # where 4' circle meets 2' line
                dra1,dra2 = dr*np.sin(np.radians(pa+np.array([1.,-1.])*dpa))/np.cos(np.radians(dec))
                ddec1,ddec2 =  dr*np.cos(np.radians(pa+np.array([1.,-1.])*dpa))
                fout.write('line(%f,%f,%f,%f)\n ' % (ra+dra1,dec+ddec1,ra+dra2,dec+ddec2)+'\n')
                fout.write('line(%f,%f,%f,%f)\n ' % (ra-dra1,dec-ddec1,ra-dra2,dec-ddec2)+'\n')

      # show spectra
        fout.write('global color=red width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        self.slitmask.set_specbox()          
        for i in slit_s:
            fout.write(self.slitlets.asregionspec(i,self.slitmask.position_angle)+'\n')
        fout.close()             
        self.imagedisplay.regionfromfile('tmp2.reg')        

    def loadimage(self, inimage=None):
        if not inimage:
             #launch a file IO dialog
             ldir = os.getcwd()
             inimage = QtGui.QFileDialog.getOpenFileName(caption="Open Catalog", directory=ldir)
        self.inimage=str(inimage)       
        self.imagedisplay.displayfile(inimage, 1, pa=self.position_angle)    
        
    def deleteslitfromimage(self):
        #download the slits in the image
        newslits=self.imagedisplay.getregions()
        
        #loop through the list and see which one is missing
        try:
            index=np.where(self.slitlets.data['inmask_flag']==1)[0]
        except:
            return
        
 
        #check to see if it is in the mask
        for i in index:
            sra=self.slitlets.data['targ_ra'][i]
            sdec=self.slitlets.data['targ_dec'][i]
            found=False
            for k in newslits.keys():
              ra = float(newslits[k][0][0])
              dec = float(newslits[k][0][1])
              if abs(sra-ra) < 0.0003 and abs(sdec-dec) < 0.0003:
                 found=True
            if not found:
               self.slitlets.data['inmask_flag'][i]=0

        #update the tabs
        self.updatetabs()      


    def addslitletsfromimage(self):
        """Download the slits from the image and add them to the slitlet or catalog

          If catalog is selected, it will search the catalog for a corresponding object and center the slit on that object

          **TODO**
          If manual centroided is selected, it will centroid around that value in the image and use that position
          If manual uncentroided is selected, it will just use the slit position

        """
        #download the slits in the image
        newslits=self.imagedisplay.getregions()
        #loop through the objects--if they are already in the catalog check to see if they
        #need updating.  If not, then add them to the catalog

        #print "Sorting through regions"

        for i in newslits:
            if newslits[i][1]:
              #if it is tagged we assume it is already in the slitmask
              #print newslits[i]
              pass
            else:
              ra = float(newslits[i][0][0])
              dec = float(newslits[i][0][1])
              #print i,ra,dec
           #width=str(newslits[i][0])
           #height=str(newslits[i][0])
           #tilt=str(newslits[i][0])
              #TODO: This searches that catalog and adds the target that matches the slit drawn
              sid=self.slitlets.findtarget(ra,dec)
              self.slitlets.addtomask(sid)

         
        self.updatetabs()

    def setPalette(self,mode):
        palette = QtGui.QPalette()

        if mode == 'error':
            brush = QtGui.QBrush(QtGui.QColor(255, 148, 148))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)

            return palette

        if mode == 'normal':
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)

            return palette

if __name__ == "__main__":
  infile = None
  inimage= None
  if len(sys.argv)>1:
     infile=sys.argv[1]
  if len(sys.argv)>2:
     inimage=sys.argv[2]
  kwargs = dict(x.split('=', 1) for x in sys.argv[1:] if x.count('.')==0)
  if len(kwargs): kwargs = {k:bool(v) for k,v in kwargs.iteritems()}       
  app = QtGui.QApplication([])
  myapp = SlitMaskGui(infile=infile, inimage=inimage, **kwargs)
  myapp.show()
  sys.exit(app.exec_())

# debug:
# cd ~/salt/polarimetry/GrPol/M30
# python2.7 ~/src/salt/polsaltcurrent/proptools/pySlitMask.py

-khn-oderbolz-73-> python2.7 ~/src/salt/polsaltcurrent/proptools/pySlitMask.py
python version:  2.7.3 (default, Dec 26 2014, 21:13:26) [GCC]
host: Undefined variable.
host: Undefined variable.
catalog: updating FOV, collisions, all slits
Optimizer search: 
Start
 i  j  k dPA(deg)   dX(")  dY(") P1s targs refs %P1coll  P1shft cr cs ss     ra          dec
 0  0  0     0.00     0.0     0.0  11  42   7 13490.26     0.00 

Grid Search 
 0  0  0     0.00     0.0     0.0   5  10   4     0.00     0.00  3 32  2  325.09949  -23.17536 
Traceback (most recent call last):
  File "/usr/users/khn/src/salt/polsaltcurrent/proptools/optimizetab.py", line 495, in do_optimize
    self.optdisplaybest(dpabest,dxbest,dybest)
  File "/usr/users/khn/src/salt/polsaltcurrent/proptools/optimizetab.py", line 98, in optdisplaybest
    self.displayspectra() 
  File "/usr/users/khn/src/salt/polsaltcurrent/proptools/pySlitMask.py", line 309, in displayspectra
    self.readyspectrumdisplay()              
AttributeError: 'SlitMaskGui' object has no attribute 'readyspectrumdisplay'

