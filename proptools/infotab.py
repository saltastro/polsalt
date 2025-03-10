# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import zipfile
from xml.dom import minidom
import xml.parsers.expat
from PyQt4 import QtCore, QtGui
'''

'''

class InfoTab():
    def __init__(self, ui, slitmask, xmlfile=None):
        super(QObject, self).__init__()
        self.ui = ui
        self.xmlfile=xmlfile
    
    # set the mode of operation:
    def setmode2cat(self):
        self.ui.lineEditMain_Mode.setText('Catalogue')
        self.ui.toolButtonCat_Load.setEnabled(True)
        self.mode = 'catalogue'
 
    def setmode2manual(self):
        self.ui.lineEditMain_Mode.setText('Manual')
        self.mode = 'manual'

    def setmodecentroiding(self):
        '''
        set the text on the centroiding display
        '''
        if self.ui.checkBoxInfo_CentroidOn.isChecked():
            self.ui.labelMain_CentroidingOnOff.setText('ON')
        else:
            self.ui.labelMain_CentroidingOnOff.setText('OFF')
            
    def getxml(self,xmlfile):
        """Return the xml dom file """
        try:
            zip = zipfile.ZipFile(xmlfile,'r')
            zip.extract('Slitmask.xml')
            try:
                dom = minidom.parse('Slitmask.xml')
            except xml.parsers.expat.ExpatError, e:
                raise ReadXMLError(e)

        except zipfile.BadZipfile:
            try:
                dom = minidom.parse(xmlfile)
            except xml.parsers.expat.ExpatError,e:
                raise ReadXMLError(e)
        return dom            

    def loadxml(self):
        """Locate a file and then enter it into the ui"""

        #launch a file IO dialog
        ldir = os.getcwd()
        xmlfile = QtGui.QFileDialog.getOpenFileName(caption="Open Catalog", directory=ldir)
        
        #load that file as slits
        self.enterxml(str(xmlfile))
        
        # set the new mask values on the display
        self.ui.lineEditMain_CenRA.setText(str(self.slitmask.center_ra))
        self.ui.lineEditMain_CenDEC.setText(str(self.slitmask.center_dec))
        self.ui.lineEditMain_PA.setText(str(self.slitmask.position_angle))
        self.ui.lineEditMain_Equinox.setText(str(self.slitmask.equinox))
        
        if self.slitmask.target_name:
            self.ui.lineEditMain_TargetName.setText(self.slitmask.target_name)
        if self.slitmask.mask_name:
            self.ui.lineEditMain_MaskName.setText(self.slitmask.mask_name)
        if self.slitmask.creator:
            self.ui.lineEditInfo_Creator.setText(self.slitmask.creator)
        if self.slitmask.proposer:
            self.ui.lineEditInfo_Proposer.setText(self.slitmask.proposer)
        if self.slitmask.proposal_code:
            self.ui.lineEditInfo_ProposalCode.setText(self.slitmask.proposal_code)
        if self.slitmask.filter:
            filteridx = self.slitmask.impolfilterList.index(self.slitmask.filter)
            self.ui.comboBoxInfo_Filter.setCurrentIndex(filteridx)
                
        if self.slitmask.polarimetry:        
            self.ui.checkBoxInfo_Polarimetry.setChecked(True)
        else:        
            self.ui.checkBoxInfo_Polarimetry.setChecked(False)                    
        if self.slitmask.impol:        
            self.ui.checkBoxInfo_ImPol.setChecked(True)
        else:        
            self.ui.checkBoxInfo_ImPol.setChecked(False)  
                                 
        self.displayslits()

    def enterxml(self, xmlfile=None):
        """Given an xml slit file, enter it into the tables in the ui"""
   
        #double check that a file exists and if not, then load it
        self.xmlfile = xmlfile
        if self.xmlfile is None: 
           self.loadxml()
           return

        # load the xml slitmask info, checking for FOV
        self.slitmask.readmaskxml(self.getxml(str(xmlfile)))

        print "xml: updating FOV, collisions, xml slits"
        self.slitlets.data['catidx'] = np.arange(len(self.slitlets.data['catidx']))
        ra,dec,pa = self.slitmask.center_ra, self.slitmask.center_dec, self.slitmask.position_angle        
        self.slitlets.updategaps(ra,dec,pa)          
        self.slitmask.find_collisions()
        
        #update the tables
        self.updatecatalogtable()
        self.updateslittable()
        self.updaterefstartable()   

    # load the mask info
    def loadcreator(self):
        self.slitmask.validated = False
        self.slitmask.creator=self.ui.lineEditInfo_Creator.text()
        if len(self.ui.lineEditInfo_Creator.text()) == 0:
            self.slitmask.creator = None

    def loadproposer(self):
        self.slitmask.validated = False
        self.slitmask.proposer=self.ui.lineEditInfo_Proposer.text()
        if len(self.ui.lineEditInfo_Proposer.text()) == 0:
            self.slitmask.proposer = None

    def loadproposalcode(self):
        self.slitmask.validated = False
        self.slitmask.proposal_code=self.ui.lineEditInfo_ProposalCode.text()
        if len(self.ui.lineEditInfo_ProposalCode.text()) == 0:
            self.slitmask.proposal_code = None

    def setpolarimetrymode(self):
        self.slitmask.validated = False
        self.slitmask.polarimetry=self.ui.checkBoxInfo_Polarimetry.isChecked()
        if (not self.slitmask.polarimetry):
            self.ui.checkBoxInfo_ImPol.setChecked(False)         
        
    def setimpolmode(self):
        self.slitmask.validated = False
        self.slitmask.impol=self.ui.checkBoxInfo_ImPol.isChecked()        
        if self.slitmask.impol:
            self.ui.checkBoxInfo_Polarimetry.setChecked(True)

    def loadfilter(self):
        newfilter = str(self.ui.comboBoxInfo_Filter.currentText())
        if newfilter==None: return        
        self.slitmask.validated = False
        
        if self.slitmask.filter == None:
            print ('filter signal: None %s' % newfilter)
            unchanged = False
        else:
            print ('filter signal: %s %s' % (self.slitmask.filter,newfilter))
            unchanged = (self.slitmask.filter == newfilter)

        if unchanged: return
        self.slitmask.filter=newfilter
        print "Filter: updating FOV, all slits"                 
        self.slitmask.outFoV()
        self.slitmask.find_collisions()            
        self.updatetabs()

    def loadgrating(self):
        self.slitmask.validated = False
        self.slitmask.grating=self.ui.comboBoxInfo_Grating.currentText()
        if len(self.ui.comboBoxInfo_Grating.currentText()) == 0:
            self.slitmask.grating = None
