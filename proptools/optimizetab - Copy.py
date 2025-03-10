# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from astropy.coordinates import Latitude,Longitude,Angle

from PyQt4 import QtCore, QtGui

from slitmask import SlitMask
from slitlets import Slitlets, ra_read,dec_read

optsetupkeyList = ['PAcount','PAstep','RAcount','RAstep','Deccount','Decstep']
optsetupdefaultList = [1,'10d',1,'5s',1,'10s',] 
optsetupfmtList = ['%5i','%6s','%5i','%6s','%5i','%6s']
optresultkeyList = ['PA','RA','Dec','P1Targs','TotTargs','Refs','P1Colltot','P1Shifttot']
optresultfmtList = ['{:>9.2f}','{:>8.1f}','{:>8.1f}','{:>7d}','{:>7d}','{:>7d}','{:>9.2f}','{:>9.2f}']
optnudgelblList = ['-PA','+PA','-x','+x','-y','+y']   

class OptimizeTab:
    def __init__(self, ui, opt_spacing=1., opt_crossmaxshift=25., opt_allowspeccoll=5., setupDict=None):
        self.ui = ui
        self.slitlets=Slitlets()
        self.spacing = opt_spacing               
        self.crossmaxshift = opt_crossmaxshift
        self.allowspeccoll = opt_allowspeccoll        
        if (setupDict==None):     
            self.optsetupDict = dict(zip(optsetupkeyList, optsetupdefaultList))
            for key in optsetupkeyList:
                self.updatesetuptable(key, self.optsetupDict[key])
            self.optresultDict = dict(zip(optresultkeyList, [0,0,0,0,0,0,0]))                
        else:
            self.optsetupDict = setupDict
        for i,j in np.ndindex(3,2):
            self.ui.tableWidgetOptNudge.setItem(i,j,QtGui.QTableWidgetItem(optnudgelblList[2*i+j]))     

    def updatecollparamtable(self,newvalue,row):
        """update the collision parameter table"""
        item = QtGui.QTableWidgetItem(('%7.1f' % newvalue))
        #enter the information into the table
        self.ui.tableWidgetOptcollParam.blockSignals(True)
        self.ui.tableWidgetOptcollParam.setItem(0,row,item)
        self.ui.tableWidgetOptcollParam.blockSignals(False)
        
    def updateoptdisplay(self,isnew=True):
        """update display radio buttons"""
        self.ui.radioButtonOpt_Start.setChecked(not isnew)
        self.ui.radioButtonOpt_Best.setChecked(isnew)            
                    
    def updatesetuptable(self,key,newvalue):
        """update the confsetup table"""
        # get location of this key in table
        k = optsetupkeyList.index(key)
        i,j = np.unravel_index(k,(3,2))
        
        #enter the information into the table
        item=QtGui.QTableWidgetItem(optsetupfmtList[k] % newvalue)
        self.ui.tableWidgetOptsetup.blockSignals(True)
        self.ui.tableWidgetOptsetup.setItem(i,j,item)
        self.ui.tableWidgetOptsetup.blockSignals(False)

    def updateresulttable(self,key,newvalue,col):
        """update the confresult table"""
        # get location of this key in table
        row = optresultkeyList.index(key)
        
        #enter the information into the table and array
        item=QtGui.QTableWidgetItem(optresultfmtList[row].format(newvalue))
        self.ui.tableWidgetOptresult.setItem(row,col,item)
        self.optresultDict[key] = newvalue        

    def optdisplaystart(self):
        """When start display button pressed, set start and not best"""
        #print 'opt_start'        
        if ((not hasattr(self.slitlets, 'datastart'))): return
        self.optimize.isstart = True
        self.slitmask.slitlets.data = np.copy(self.slitlets.datastart)
        self.slitmask.add_position_angle(self.optimize.posstart_d[0])                        
        self.slitmask.add_center_ra(self.optimize.posstart_d[1])
        self.slitmask.add_center_dec(self.optimize.posstart_d[2])
        self.slitmask.find_collisions()
        self.updateresult(0.,0.,0.,isnew=False)                        
        self.updatetabs()        
            
    def optdisplaybest(self,dpa,dx,dy):
        """When best display button pressed, set best and not start"""
        #print 'opt_best'                
        if ((not hasattr(self.slitlets, 'databest'))): return
                
        self.optimize.isstart = False            
        self.slitlets.data = np.copy(self.slitlets.databest)
        self.slitmask.add_position_angle(self.optimize.posbest_d[0])                        
        self.slitmask.add_center_ra(self.optimize.posbest_d[1])        
        self.slitmask.add_center_dec(self.optimize.posbest_d[2])        
        self.slitmask.find_collisions()
        
        self.updateresult(dpa,3600.*dx,3600.*dy,isnew=True)        
        self.updatetabs()
        
    def adopt_best(self):
        """When adopt it button pressed, reset main RA,Dec,PA, display"""
        print "adopt best?"
        pachanged = (np.abs(self.optimize.posbest_d[0] -   \
            dec_read(self.ui.lineEditMain_PA.text())) > .001)              
        rachanged = (np.abs(self.optimize.posbest_d[1] -    \
            ra_read(self.ui.lineEditMain_CenRA.text())) > .001)            
        decchanged = (np.abs(self.optimize.posbest_d[2] -   \
            dec_read(self.ui.lineEditMain_CenDEC.text())) > .001)            

        print "old position: %8.2f %10.5f %10.5f" % (dec_read(self.ui.lineEditMain_PA.text()),   \
            ra_read(self.ui.lineEditMain_CenRA.text()),dec_read(self.ui.lineEditMain_CenDEC.text()))
        print "new position: %8.2f %10.5f %10.5f" % tuple(self.optimize.posbest_d)

        if  not (rachanged | decchanged | pachanged): 
            print "no change"
            return
        print "adopt Best RA,Dec,PA"
        
        delattr(self.slitlets,'datastart')
        if (rachanged | decchanged):         # slit set to be center is no longer          
            iscenterslit_i = (self.slitmask.slitlets.data['priority'] > 1)
            self.slitmask.slitlets.data['priority'][iscenterslit_i] = 1
        self.ui.lineEditMain_PA.setText(str(self.optimize.posbest_d[0]))                                        
        self.ui.lineEditMain_CenRA.setText(str(self.optimize.posbest_d[1]))
        self.ui.lineEditMain_CenDEC.setText(str(self.optimize.posbest_d[2]))

        self.imagedisplay.rotate(-self.slitmask.position_angle)
        self.imagedisplay.deleteregions()           
        self.displayslits()                 
        self.updatetabs()
                
    def collparamchanged(self, i, j):
        """When ever a cell is changed, update the optimization collision parameters"""    
        newvalue = float(self.ui.tableWidgetOptcollParam.item(i,j).text())
        if (i==0):            
            self.optimize.spacing = newvalue
            print "Optimizer crossdisp spacing: ", self.optimize.spacing            
        elif (i==1):
            self.optimize.crossmaxshift = newvalue
            print "Optimizer crossdisp %maxshift: ", self.optimize.crossmaxshift            
        else:        
            self.optimize.allowspeccoll = newvalue
            print "Optimizer max allowed spec %overlap: ", self.optimize.allowspeccoll                  
        self.optimize.updatecollparamtable(newvalue, i)

    def optsetupchanged(self, i ,j):
        """When ever a cell is changed, update the optimization setup"""
        text = str(self.ui.tableWidgetOptsetup.item(i,j).text())
        key = np.array(optsetupkeyList).reshape((3,2))[i,j]
        if (type(self.optimize.optsetupDict[key]) is int):
            newvalue = int(text)
        else:
            newvalue = text        
        if (j==0):
            newvalue=2*int(newvalue/2)+1                 # ensure steps is odd
        self.optimize.optsetupDict[key] = newvalue
        self.optimize.updatesetuptable(key, newvalue)
        print "Optimizer setup change: ", i,j,key,newvalue

    def optnudge(self, i ,j):
        """When ever a cell is changed, change backgound"""
        self.ui.tableWidgetOptNudge.setStyleSheet("QTableView::item:selected{color:white; background:red}")
        self.do_nudge(i, j)
        self.ui.tableWidgetOptNudge.setStyleSheet("QTableView::item:selected{color:black; background:white}")        
        #print "Optimizer nudge: ", i,j

    def updateresult(self,dpa,dx,dy,isnew=True):
        """Update result table from current slitmask"""
        isPone_i = (self.slitlets.data['priority'] >= 1)
        refs = ((self.slitlets.data['inmask_flag'] * (self.slitlets.data['itemtype'] == 'ref'))).sum()
        targs = ((self.slitlets.data['inmask_flag'] * isPone_i)).sum()
        tottargs = self.slitlets.data['inmask_flag'].sum() - refs
        colltot = (self.slitlets.data['yoverlap']*self.slitlets.data['xoverlap'])[isPone_i].sum()/100.
        if (colltot < .001): colltot=0.                 # avoid rounding problems in optimization
        shifttot = (np.abs((self.slitlets.data['len1'] - self.slitlets.data['len2'])/   \
            (self.slitlets.data['len1'] + self.slitlets.data['len2']))*self.slitlets.data['inmask_flag'])[isPone_i].sum()
        for row,key in enumerate(optresultkeyList):
            value = [dpa,dx,dy,targs,tottargs,refs,colltot,shifttot][row]
            self.optimize.updateresulttable(key,value,isnew)
            
        self.updateoptdisplay(isnew)            
            
    def cullrefs(self):
        """Cull ref stars causing collisions, down to minimum of 4 refs, worst first"""
        # print "Culling colliding ref stars"
        refmin = 4
        self.slitmask.find_collisions()
        lenoverlap_i = self.slitlets.data[['yoverlap','xoverlap'][self.slitmask.impol]]        
        refList = list(np.where((self.slitlets.data['inmask_flag'] *        \
            (self.slitlets.data['itemtype'] == 'ref'))==1)[0])
        collideeList = list(np.where((self.slitlets.data['inmask_flag'] * lenoverlap_i) > 0)[0])
        colliderList_i = np.array([x.split() for x in self.slitlets.data['collision_id']])           
        collrList = list(np.where(np.in1d([str(x) for x in refList],sum(list(colliderList_i),[])))[0])
        refcolliderList = [refList[x] for x in collrList]                                  
        if (len(refcolliderList)==0): return min(0,len(refList)-refmin)
        
        lenoverlap_c = np.zeros(len(refcolliderList))        
        for c,refidx in enumerate(refcolliderList):
            refcollideeList = [i for (i,x) in enumerate(colliderList_i) if x.count(str(refidx))]               
            lenoverlap_c[c] = (lenoverlap_i[refcollideeList]).max()
            
        icullList = [refcolliderList[x] for x in np.argsort(lenoverlap_c)][::-1]                
        culls = min(len(refcolliderList),len(refList)-refmin)        
        if culls: self.slitlets.data['inmask_flag'][icullList[:culls]] = 0
        
        return culls

    def slideslits(self):
        """
        cull targets with collision perp dispersion > 2*crossmaxshift, lower priority or fainter target
        cull targets with crossdisp collisions in opposite directions 
        for other collisions shift slits up to crossmaxshift in direction opposite to collider
        """
        # print "culling slits with unacceptable collisions"
        self.slitmask.find_collisions()
        xp_i, yp_i = self.slitmask.myWcs().wcs_sky2pix(self.slitlets.data['targ_ra'], self.slitlets.data['targ_dec'], 1)
        lencenter_i = [yp_i,xp_i][self.slitmask.impol]
        priority_i = self.slitlets.data['priority']
        mag_i = self.slitlets.data['mag']
        length_i = self.slitlets.data['len1'] + self.slitlets.data['len2']
        refiList = list(np.where(self.slitlets.data['itemtype'] == 'ref')[0])
        avoidiList = list(np.where(self.slitlets.data['itemtype'] == 'avoid')[0])        
        gapidxDict = dict(zip(['Gap1','Gap2'],[list(self.slitlets.data['name']).index(x) for x in ['Gap1','Gap2']]))
        spacing = float(self.ui.tableWidgetOptcollParam.item(0,0).text())

        # do culls, one by one, worst first
        docull = True
        slitculls = 0
        
        if self.debug: 
            print "slits inmask before cull: ", \
                ((self.slitlets.data['inmask_flag'] * (self.slitlets.data['itemtype'] != 'ref'))==1).sum()
        
        while docull:
            lenoverlap_i = self.slitlets.data[['yoverlap','xoverlap'][self.slitmask.impol]]
            specoverlap_i = self.slitlets.data[['xoverlap','yoverlap'][self.slitmask.impol]]
            collideeList = list(np.where((self.slitlets.data['inmask_flag'] == 1) & \
                (lenoverlap_i > 0) & (specoverlap_i > self.optimize.allowspeccoll))[0])
            lenoverlap_c = lenoverlap_i[collideeList]
            collideeList = [collideeList[x] for x in np.argsort(lenoverlap_c)[::-1]]
            lenoverlap_c = lenoverlap_i[collideeList]
            colliderList_c = np.array([x.split() for x in self.slitlets.data['collision_id']])[collideeList] 
            colls_c = np.array([len(x) for x in colliderList_c])
            colltgts = len(collideeList)

            # compute collider information                
            colldir_c = np.zeros(colltgts)
            colliderpri_c = np.zeros(colltgts)
            collidermag_c = np.zeros(colltgts)
            hasgapcoll_c = np.zeros(colltgts,dtype=bool)
            hasrefcoll_c = np.zeros(colltgts,dtype=bool)
            hasavoidcoll_c = np.zeros(colltgts,dtype=bool)            
            for c in range(colltgts):
                isgap_j = np.array([x.startswith('Gap') for x in colliderList_c[c]])
                hasgapcoll_c[c] = isgap_j.any()
                if hasgapcoll_c[c]:
                    jgap = np.where(isgap_j)[0][0] 
                    colliderList_c[c][jgap] = gapidxDict[colliderList_c[c][jgap]]

                colliderList_c[c] = map(int,colliderList_c[c])                    
                hasrefcoll_c[c] = np.in1d(refiList,colliderList_c[c]).any()
                hasavoidcoll_c[c] = np.in1d(avoidiList,colliderList_c[c]).any()                
                colldir_j =  np.sign(lencenter_i[colliderList_c[c]] - lencenter_i[collideeList[c]])
                colldir_c[c] = np.median(colldir_j)             # sign if all same, otherwise 0
                colliderpri_c[c] = priority_i[colliderList_c[c]].max()
                collidermag_c[c] = mag_i[colliderList_c[c]].min()                      

            # cull slits with uncorrectable collisions
            okcull_c = (hasgapcoll_c | hasrefcoll_c | hasavoidcoll_c)
            okcull_c |= (priority_i[collideeList] < colliderpri_c)
            okcull_c |= ((priority_i[collideeList] == colliderpri_c)   \
                        & (mag_i[collideeList] > collidermag_c))
            shiftsallow_c = 2. - (hasgapcoll_c | hasrefcoll_c | hasavoidcoll_c)
            
            cullcList = list(np.where(okcull_c &   \
                        (lenoverlap_c > shiftsallow_c*self.optimize.crossmaxshift))[0])
            cullcList += list(np.where(colldir_c==0)[0])
            cullidxList = [collideeList[c] for c in cullcList]            
            docull = (len(cullidxList) > 0)
            if docull:                                          # cull and update collisions, if necessary
                self.slitlets.data['inmask_flag'][cullidxList[0]] = 0
                self.slitmask.find_collisions()
                slitculls += 1

        if self.debug: 
            print "slits inmask after first cull: ", \
                ((self.slitlets.data['inmask_flag'] * (self.slitlets.data['itemtype'] != 'ref'))==1).sum()

        dogroupcull = True                              # allow for further cull in collisions caused by shifting
        while dogroupcull:
            collidees = len(collideeList)
            colliderList = [int(colliderList_c[c][0]) for c in range(collidees)] # collisions should be down to pairs                        
            i_dc = np.sort(np.vstack((collideeList,colliderList)),axis=0)
            i0_dC = i_dc[:,np.unique(i_dc[0],return_index=True)[-1]]             # _C = collision pairs before shift
                    
          # shift all correctable slits: lenoverlap and crossmaxshift are in percent of length                
            absshift_c = ((lenoverlap_c + 100.*spacing/length_i[collideeList])/shiftsallow_c)    \
                *length_i[collideeList]/100.
            lenmin_c = length_i[collideeList]*(0.5 - self.optimize.crossmaxshift/100.) 
            lenmax_c = length_i[collideeList]*(0.5 + self.optimize.crossmaxshift/100.)
                          
            for c in range(collidees):                                         
                shift = -(colldir_c*absshift_c)[c]     
                len1 = self.slitlets.data['len1'][collideeList[c]] - shift
                len1 += np.clip(len1, lenmin_c[c], lenmax_c[c]) - len1                               
                self.slitlets.data['len1'][collideeList[c]] = len1               
                self.slitlets.data['len2'][collideeList[c]] = length_i[collideeList[c]] - len1              

            self.slitmask.find_collisions()

            badoverlap_i = 100.*spacing/length_i        # this lenoverlap corresponds to spacing=0
            lenoverlap_i = self.slitlets.data[['yoverlap','xoverlap'][self.slitmask.impol]]
            dogroupcull = (lenoverlap_i > badoverlap_i)[(self.slitlets.data['inmask_flag'] == 1)].any()        
            if dogroupcull:                            
                specoverlap_i = self.slitlets.data[['xoverlap','yoverlap'][self.slitmask.impol]]
                collideeList = list(np.where((self.slitlets.data['inmask_flag'] == 1) & \
                    (lenoverlap_i > 0) & (specoverlap_i > self.optimize.allowspeccoll))[0])
                lenoverlap_c = lenoverlap_i[collideeList]
                collideeList = [collideeList[x] for x in np.argsort(lenoverlap_c)[::-1]]
                colliderList_c = np.array([x.split() for x in self.slitlets.data['collision_id']])[collideeList] 
                collidees = len(collideeList)
                colliderList = [int(colliderList_c[c][0]) for c in range(collidees)]
                i_dc = np.sort(np.vstack((collideeList,colliderList)),axis=0)
                i1_dC = i_dc[:,np.unique(i_dc[0],return_index=True)[-1]]
                
              # find collision groups which may be shifting slits into each other 
              #   by comparing collision pairs before (i0_dC) and after (i1_dC) shift
                C0_dC = -np.ones(i1_dC.shape,dtype=int)
                isrepeat1_dC = np.in1d(i1_dC.flatten(),i0_dC.flatten()).reshape((2,-1))
                for d,C in np.array(np.where(isrepeat1_dC)).T:
                    C0_dC[d,C] = np.where(i0_dC==i1_dC[d,C])[1][0]
                groupC1List = np.where((isrepeat1_dC.sum(axis=0)==2) & (np.diff(C0_dC,axis=0)[0] != 0))[0]

                if self.debug:
                    print "\npreshift colls: ", i0_dC
                    print "postshift colls: ", i1_dC
                    print "coll repeat?: ", isrepeat1_dC
                    print "coll in preshift: ", C0_dC
                    print "group list: ", groupC1List

                dogroupcull = (len(groupC1List) > 0)

                if (not dogroupcull): continue
                groupC1List = [groupC1List[0],]
                for C in range(i1_dC.shape[1]):
                    if (C == groupC1List[0]): continue                                      
                    if np.in1d(C0_dC[:,C],C0_dC[:,groupC1List[0]]).any(): groupC1List.append(C)
                groupiArray = np.sort(i1_dC[:,groupC1List].flatten())                       
                lenmean = [yp_i,xp_i][self.slitmask.impol][groupiArray].mean()
                cullpri = priority_i[groupiArray].min()
                culliArray = groupiArray[np.where(priority_i[groupiArray] == cullpri)[0]]           
                icull = culliArray[np.argmin(np.abs([yp_i,xp_i][self.slitmask.impol] - lenmean)[culliArray])]

                if self.debug:
                    print "\nculling slit from collision group:"
                    print "  group: ", groupiArray
                    print "  lenposns ",[yp_i,xp_i][self.slitmask.impol][groupiArray] 
                    print "  cull slitidx: ",icull
                
              # cull, reassess collisions, prepare for additional shifts
                self.slitlets.data['inmask_flag'][icull] = 0                                                                                
                self.slitmask.find_collisions()
                lenoverlap_i = self.slitlets.data[['yoverlap','xoverlap'][self.slitmask.impol]]
                specoverlap_i = self.slitlets.data[['xoverlap','yoverlap'][self.slitmask.impol]]
                
                collideeList = list(np.where((self.slitlets.data['inmask_flag'] == 1) & \
                    (lenoverlap_i > 0) & (specoverlap_i > self.optimize.allowspeccoll))[0])
                collidees = len(collideeList)                    
                colliderList_c = np.array([x.split() for x in self.slitlets.data['collision_id']])[collideeList]                
                colliderList = [int(colliderList_c[c][0]) for c in range(collidees)]                                
                i_dc = np.sort(np.vstack((collideeList,colliderList)),axis=0)
                i2_dC = i_dc[:,np.unique(i_dc[0],return_index=True)[-1]]             
                lenoverlap_c = lenoverlap_i[collideeList] 
                colldir_c = np.array([np.sign(lencenter_i[colliderList[c]] -    \
                    lencenter_i[collideeList[c]]) for c in range(collidees)])
                shiftsallow_c = 2.*np.ones(collidees)
                                
                slitculls += 1

        if self.debug: 
            print "slits inmask after second cull: ", \
                ((self.slitlets.data['inmask_flag'] * (self.slitlets.data['itemtype'] != 'ref'))==1).sum()
                                         
        return slitculls,collidees
        
    def do_optimize(self):
        print "Optimizer search: "
        #get, save starting data, if necessary
        if (not hasattr(self.slitlets, 'datastart')):     
            self.slitlets.datastart = np.copy(self.slitlets.data)
            self.optimize.posstart_d =  \
                np.array([self.slitmask.position_angle,self.slitmask.center_ra,self.slitmask.center_dec])        
            self.updateresult(0.,0.,0.,isnew=False)
            self.startresulttxt = (3*'%2i '+'%8.2f '+2*'%7.1f '+3*'%3i '+2*'%8.2f ') % ((0,0,0)+ \
                tuple([self.optimize.optresultDict[key] for key in optresultkeyList]))   
        
        print "Start"
        print " i  j  k dPA(deg)   dX(\")  dY(\") P1s targs refs %P1coll  P1shft cr cs ss     ra          dec"
        print self.startresulttxt   
        self.slitlets.data = np.copy(self.slitlets.datastart)
                            
        # set up search grid
        
        countList = [self.optimize.optsetupDict[x] for x in optsetupkeyList if x.endswith('count')]
        stepList = [self.optimize.optsetupDict[x] for x in optsetupkeyList if x.endswith('step')]
        count_d = np.array(countList).astype(int)                 
        step_d = np.array([Angle(x).deg for x in stepList])        
        pa0,ra0,dec0 = self.optimize.posstart_d                          
        resulttxtList = []        
        besttargs = 0
        bestcoll = 1.e9
        bestshft = 1.e9    
            
        print "\nGrid Search",
        
        # evaluate best separately for each pa
        for i in range(count_d[0]):
            pabesttargs = 0
            pabestcoll = 1.e9
            pabestshft = 1.e9        
        
            for j,k in np.ndindex(*count_d[1:]):
                didx_d = np.array([i,j,k],dtype=int) - (count_d - 1)/2
                dpa,dx,dy = didx_d*step_d
                pa = (pa0 + dpa + 360.) % 360.   
                ra =  ra0  - (dx*np.cos(np.radians(pa0))-dy*np.sin(np.radians(pa0)))/np.cos(np.radians(dec0))
                dec = dec0 + (dx*np.sin(np.radians(pa0))+dy*np.cos(np.radians(pa0)))                                    
                self.slitmask.add_position_angle(pa)
                self.slitmask.add_center_ra(ra)
                self.slitmask.add_center_dec(dec)
                self.slitlets.data = np.copy(self.slitlets.datastart)                        
                self.slitmask.outFoV()          
                refculls = self.cullrefs()            
                slitculls,slitshifts = self.slideslits()
                self.updateresult(dpa,3600.*dx,3600.*dy,isnew=True)        
                self.updatetabs()               
                okrefs =(self.optimize.optresultDict['Refs'] >= 3)
                
                # check for best for this pa
                moretargs = (self.optimize.optresultDict['P1Targs'] > pabesttargs)
                sametargs = (self.optimize.optresultDict['P1Targs'] == pabesttargs)
                bettercolls = (self.optimize.optresultDict['P1Colltot'] < pabestcoll)
                nocolls = (self.optimize.optresultDict['P1Colltot'] == 0)            
                bettershifts = (self.optimize.optresultDict['P1Shifttot'] < pabestshft)
                if (okrefs & (moretargs | sametargs & (bettercolls | (nocolls & bettershifts)))):
                    pabesttargs = self.optimize.optresultDict['P1Targs']
                    pabestrefs = self.optimize.optresultDict['Refs']        
                    pabestcoll = self.optimize.optresultDict['P1Colltot']  
                    pabestshft = self.optimize.optresultDict['P1Shifttot']                                
                    (ipabest,jpabest,kpabest) = (i,j,k)

                # check for best for all pas                    
                moretargs = (self.optimize.optresultDict['P1Targs'] > besttargs)
                sametargs = (self.optimize.optresultDict['P1Targs'] == besttargs)
                bettercolls = (self.optimize.optresultDict['P1Colltot'] < bestcoll)
                nocolls = (self.optimize.optresultDict['P1Colltot'] == 0)            
                bettershifts = (self.optimize.optresultDict['P1Shifttot'] < bestshft)
                if (okrefs & (moretargs | sametargs & (bettercolls | (nocolls & bettershifts)))):
                    self.slitlets.databest = np.copy(self.slitlets.data)
                    self.optimize.posbest_d = (pa,ra,dec)
                    besttargs = self.optimize.optresultDict['P1Targs']
                    bestrefs = self.optimize.optresultDict['Refs']        
                    bestcoll = self.optimize.optresultDict['P1Colltot']  
                    bestshft = self.optimize.optresultDict['P1Shifttot']                                
                    (ibest,jbest,kbest) = (i,j,k)
                    (dpabest,dxbest,dybest) = (dpa,dx,dy)                    
                
                resulttxt = ((3*'%2i '+'%8.2f '+2*'%7.1f '+3*'%3i '+2*'%8.2f '+3*"%2i "+2*"%10.5f ") % (tuple(didx_d)+ \
                    tuple([self.optimize.optresultDict[key] for key in optresultkeyList]) + \
                    (refculls,slitculls,slitshifts,ra,dec)))
                #print resulttxt
                resulttxtList.append(resulttxt)

            if (pabesttargs == 0):
                print "No usable slitmask found for pa ", pa
                continue      
            print '\n',resulttxtList[np.ravel_multi_index((ipabest,jpabest,kpabest),count_d)],

        if (besttargs == 0):
            print "\nNo usable slitmask found for any pa "
            return
        if (count_d.max > 1):
            with open('optresult.txt', 'w') as f:
                f.write('\n'.join(resulttxtList))
        self.optdisplaybest(dpabest,dxbest,dybest)
        self.ui.pushButtonOpt_Adoptit.setEnabled(True)                  
        if (count_d[0]==1):
            print "Best\n" 
            return        
        print '\n\n',resulttxtList[np.ravel_multi_index((ibest,jbest,kbest),count_d)], "Best\n"        
                

    def do_nudge(self,i,j):
        #get, save starting (unshifted) data, if no opt done yet
        if (not hasattr(self.slitlets, 'datastart')):     
            self.slitlets.datastart = np.copy(self.slitlets.data)
            self.optimize.posstart_d =  \
                np.array([self.slitmask.position_angle,self.slitmask.center_ra,self.slitmask.center_dec])        
            self.updateresult(0.,0.,0.,isnew=False)
            self.startresulttxt = (3*'%2i '+'%8.2f '+2*'%7.1f '+3*'%3i '+2*'%8.2f ') % ((0,0,0)+ \
                tuple([self.optimize.optresultDict[key] for key in optresultkeyList]))              

        # nudge from current slitmask position(0), but report relative to start (unshifted) slitlets
        pastart,rastart,decstart = self.optimize.posstart_d
        dx0, dy0 = -np.array(self.slitmask.myWcs().wcs_sky2pix(rastart, decstart, 1))[:,0]/3600.
        dpa0 = ((self.slitmask.position_angle - pastart + 180.) % 360.) - 180.  # current relative to start               
        didx_d = [-1,1][j]*np.ones(3)*(i==np.arange(3,dtype=int))
        dpa,dx,dy = didx_d*np.array([0.5,Angle('1s').deg,Angle('1s').deg])      # nudge from current
        
        pa = (pastart + dpa0 + dpa + 360.) % 360.        
        ra =  rastart  - ((dx0+dx)*np.cos(np.radians(pastart))-(dy0+dy)*np.sin(np.radians(pastart)))/   \
            np.cos(np.radians(decstart))
        dec = decstart + ((dx0+dx)*np.sin(np.radians(pastart))+(dy0+dy)*np.cos(np.radians(pastart)))
                
        self.slitmask.add_position_angle(pa)
        self.slitmask.add_center_ra(ra)
        self.slitmask.add_center_dec(dec)

        self.slitlets.data = np.copy(self.slitlets.datastart)        
        self.slitmask.outFoV()          
        refculls = self.cullrefs()            
        slitculls,slitshifts = self.slideslits()
        self.updateresult(dpa0+dpa,3600.*(dx0+dx),3600.*(dy0+dy),isnew=True)        
        self.updatetabs()  
        resulttxt = ((3*'%2i '+'%8.2f '+2*'%7.1f '+3*'%3i '+2*'%8.2f '+3*"%2i "+2*"%10.5f ") % (tuple(didx_d)+ \
            tuple([self.optimize.optresultDict[key] for key in optresultkeyList]) + \
            (refculls,slitculls,slitshifts,ra,dec)))
        print resulttxt, "  Nudge"
        self.ui.pushButtonOpt_Adoptit.setEnabled(True)         
            
    def updatetabs(self):
        print "optimizetab updatetabs"    
        self.slitmask.outFoV_all()
        self.slitmask.find_collisions()
        self.slitlets.update_flags()
#        pass

