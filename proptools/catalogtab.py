# -*- coding: utf-8 -*-
import os, sys
import numpy as np

from PyQt4 import QtCore, QtGui


from slitmask import SlitMask
from slitlets import Slitlets

class ReadXMLError(Exception):
    """Class for handling XML reading errors"""
    pass

catcolumnList=['name', 'catidx', 'targ_ra', 'targ_dec', 'width', 'len1', 'len2', 'tilt', 'mag', 'priority', 'fov_flag', 'inmask_flag']
catformatList=['%s',   '%4i',    '%10.5f',  '%10.5f',   '%6.1f', '%6.1f','%6.1f','%7.2f','%7.2f','%6.1f',   '%3i',       '%3i']

class CatalogTab:
    def __init__(self, ui, infile=None):
        self.ui = ui
        self.slitlets = Slitlets()
        self.slitmask = Slitmask()        
        self.infile = infile

    def loadcatalog(self):
        """Locate a file and then enter it into the ui"""

        #launch a file IO dialog
        ldir = os.getcwd()
        infile = QtGui.QFileDialog.getOpenFileName(caption="Open Catalog", directory=ldir)

        #load that file into the catalog
        self.entercatalog(str(infile))
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
         
        self.displayslits()

    def entercatalog(self, infile=None, form='short'):
        """Given a catalog, enter it into table in the ui"""
   
        #double check that a file exists and if not, then load it
        self.infile = infile
        if self.infile is None: 
           self.loadcatalog()
           return

        # load it as an ascii file. Enter the file information into the slit_arr
        self.slitlets.readascii(self.infile, form=form)
        self.slitmask.set_MaskPosition()

        # check whether it is an ImPol file
        if ((self.slitlets.data['tilt']==90.).all() & (len(self.slitmask.grating)==0)):
            self.ui.checkBoxInfo_Polarimetry.setChecked(True)
            self.ui.checkBoxInfo_ImPol.setChecked(True)            
            self.slitmask.polarimetry = True            # for init, before signals set
            self.slitmask.impol = True

        #check for objects outside the FoV
        print "catalog: updating FOV, collisions, all slits"
        self.slitlets.data['catidx'] = np.arange(len(self.slitlets.data['catidx']))                
        self.slitmask.outFoV()
        self.slitmask.find_collisions()
        
        #update the tables
        self.updatecatalogtable()
        self.updateslittable()
        self.updaterefstartable()

    def updatecatalogtable(self):
        self.ui.tableWidgetCat.setRowCount(0)
        #enter the information into the table
        for i in range(self.slitlets.nobjects):
            self.ui.tableWidgetCat.insertRow(i)           
            for j in range(len(catcolumnList)):
                item=QtGui.QTableWidgetItem(catformatList[j] % (self.slitlets.data[catcolumnList[j]][i]))                        
                self.ui.tableWidgetCat.setItem(i,j,item)

    def updatetabs(self):
       print "catalogtab updatetabs"
       self.updatecatalogtable()

    def addslitfromcatalog(self):
       """Determine slits elected in the catalog and add them to the catalog"""

       #get the selected items
       sel_list = self.ui.tableWidgetCat.selectedItems()
       #print self.ui.tableWidgetCat.selectedRanges()
       
       #for each item in sel_list, 
       #get the item, and determine the parameters from it 
       #and activite the object
       for selitem in sel_list:
           selitem.row() 
           i = selitem.row()
           stext = self.ui.tableWidgetCat.item(i,0).text()
           sid = self.slitlets.findslitlet(str(stext))
           self.slitlets.addtomask(sid)
       self.updatetabs()
           
        
    def parseItem(self, x):                     # OBSOLETE
       """Parse an object so it can be entered into the table"""
       if isinstance(x, str):
           return QtGui.QTableWidgetItem(x)
       elif isinstance(x, np.float32):
           return QtGui.QTableWidgetItem('%f' % x)
       elif isinstance(x, float):
           return QtGui.QTableWidgetItem('%f' % x)
       elif isinstance(x, int):
           return QtGui.QTableWidgetItem('%i' % x)
       return QtGui.QTableWidgetItem('')

  



