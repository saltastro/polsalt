# -*- coding: utf-8 -*-
import os, sys
import numpy as np

from PyQt4 import QtCore, QtGui

from slitlets import Slitlets

slitcolumnList=['name', 'catidx', 'targ_ra', 'targ_dec', 'width', 'len1', 'len2', 'tilt', 'mag', 'priority',  \
                'fov_flag', 'inmask_flag', 'xoverlap', 'yoverlap', 'collision_id']
slitformatList=['%s',   '%4i',   '%10.5f',  '%10.5f',   '%6.1f', '%6.1f','%6.1f','%7.2f','%7.2f','%6.1f',    \
                '%3i',      '%3i',         '%6.1f',     '%6.1f',  '%s']

class SlitTab:
    def __init__(self, ui, infile=None):
        self.ui = ui
        self.slitlets=Slitlets()
        self.infile=infile
        self.cell = None
        self.rows = None

    def unique(self,seq):
        '''
        return a unique number of rows from the table selection
        '''
        seen = set()
        seen_add = seen.add
        return [ x for x in seq if x not in seen and not seen_add(x)]

    def setposition(self):
        '''
        determine the which rows are selected for the deletion on slits
        '''
        self.rows = []
        indexes = self.ui.tableWidgetSlits.selectedIndexes()
        for index in indexes:
            self.rows.append(index.row())
        self.rows = self.unique(self.rows)
        #print self.rows


    def clearslittable(self):
        '''
        set all the in_mask flags to 0 and update the slit table
        '''
        rows = self.ui.tableWidgetSlits.rowCount()
        #print rows
        self.slitlets.data['inmask_flag'] = 0
        self.updatetabs()
#        self.ui.tableWidgetSlits.clear()
#        for i in range(0,rows):
#            print i
#            self.ui.tableWidgetSlits.removeRow(i)

    def addslitmanually(self):
        '''
        add an empy row to the slit table that the user must fill in manually
        '''
#        rows = self.ui.tableWidgetSlits.rowCount()
#        print rows
#        if rows > 0:
#            self.ui.tableWidgetSlits.setRowCount(rows+1)
##            self.ui.tableWidgetSlits.insertRow(rows+1)
#            for j in range(len(slitcolumnList)):
#                item = self.parseItem('')
#                self.ui.tableWidgetSlits.setItem(rows+1,j,item)
#        else:
#            self.ui.tableWidgetSlits.setRowCount(0)
#            self.ui.tableWidgetSlits.insertRow(0)
#            for j in range(len(slitcolumnList)):
#                item = self.parseItem('')
#                self.ui.tableWidgetSlits.setItem(rows+1,j,item)

        ######### TESTING: ###########
        self.slitlets.add_slitlet()
        self.slitmask.outFoV()
        self.updatetabs()

    def deleteslitmanually(self):
        '''
        set the selected slits inmask_flag to 0, if none were slected, do nothing
        '''

        if len(self.rows) == 0:
            return
        else:
            #print self.rows
            for i in self.rows:
                item = self.ui.tableWidgetSlits.item(i,0)
                name = str(item.text())
                ai = np.where(self.slitlets.data['name'] == name)
                self.slitlets.data[ai[0][0]]['inmask_flag'] = 0
        self.slitlets.update_flags()
        self.updatetabs()
        return

    def addslitletsfromcatalogue(self):
        '''
        if a slit has a priority >0 add it to the slits table.
        * if slits have been added manually before adding from the catalogue
        the row count is set accordingly
        '''
        slits = np.where((self.slitlets.data['priority'] > 0) * (self.slitlets.data['fov_flag'] == 1))        

        for i in slits[0]:
            self.slitlets.data[i]['inmask_flag'] = 1
            self.slitlets.data[i]['itemtype'] = 'slit'            
        self.slitmask.outFoV_all()
        self.slitmask.find_collisions()
        self.updatetabs()

    def writeslittab(self):
        '''
        writes the current slit info to an user specified .csv file
        '''
        ldir=os.getcwd()
        slitoutfilename = QtGui.QFileDialog.getSaveFileName(caption="Save Current slit file as .csv", directory=ldir)

        # make sure that the output filename contains the .csv extension
        outfile = str(slitoutfilename).strip('.csv') + '.csv'
        self.slitfile=outfile        
        if os.path.isfile(outfile): os.remove(outfile)
        header = ','.join(slitcolumnList)
        slittabtxt = [header+'\n']
                       
        inmask_i = np.where((self.slitlets.data['inmask_flag'] == 1)*(self.slitlets.data['itemtype'] == 'slit'))[0]
        for i in inmask_i:
            line = ""
            for j in range(len(slitcolumnList)):
                line += ((slitformatList[j]+",") % (self.slitlets.data[slitcolumnList[j]][i]))
            slittabtxt.append(line+'\n')
        slitFile = open(outfile,'w')
        slitFile.writelines(slittabtxt)
        slitFile.close()

    def updateslittable(self):
        """Using the slitlet object, update the slit table"""

        # check which entries are in the mask and not reference stars
        inmask = np.where((self.slitlets.data['inmask_flag'] == 1)*(self.slitlets.data['itemtype'] == 'slit'))
        
        #enter the information into the table
        self.ui.tableWidgetSlits.setRowCount(0)
        nobj = 0
        for i in inmask[0]:
            self.ui.tableWidgetSlits.insertRow(nobj)
            for j in range(len(slitcolumnList)):
                item=QtGui.QTableWidgetItem(slitformatList[j] % (self.slitlets.data[slitcolumnList[j]][i]))
                self.ui.tableWidgetSlits.blockSignals(True)
                self.ui.tableWidgetSlits.setItem(nobj,j,item)
                self.ui.tableWidgetSlits.blockSignals(False)
            nobj += 1

    def slitchanged(self, x ,y):
        """When ever a cell is changed, updated the information about the slit"""
        #identify what slit changed and what attribute of that slit changed
        item = self.ui.tableWidgetSlits.item(x,0)
        name = str(item.text())
        ai = self.slitlets.findslitlet(name)

        #update the attribute
        item=self.ui.tableWidgetSlits.item(x,y)
        self.slitlets.updatevalue(ai, str(item.text()), slitcolumnList[y])

        #update all the tabs
        self.slitmask.outFoV_row(ai)
        self.slitmask.find_collisions()
        self.updatetabs()

    def updatetabs(self):
       print "slittab updatetabs"
       self.updateslittable()
       self.updatecatalogtable()

