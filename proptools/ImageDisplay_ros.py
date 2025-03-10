################################# LICENSE ##################################
# Copyright (c) 2009, South African Astronomical Observatory (SAAO)        #
# All rights reserved.                                                     #
#                                                                          #
############################################################################


#!/usr/bin/env python

"""
ImageDisplay--Class for displaying and interacting with ds9

Author                 Version      Date
-----------------------------------------------
S M Crawford (SAAO)    0.1          19 Jun 2011

"""
import os
import pyds9 as ds9

class ImageDisplay:
    def __init__(self, target='ImageDisplay:*'):
        self.ds9 = ds9.ds9('var/folders/cp/0g276mys5_v748v51144ghtw0000gp/T//DS9_ds9.29452')

    def display(self, filename, pa=None):
        cmd='file %s'  % filename
        self.ds9.set(cmd)
        self.ds9.set('zscale')
        self.ds9.set('match frames wcs')
#        print pa
        if pa:
           self.ds9.set('rotate to %f' % pa)
        else:
           self.ds9.set('rotate to %f' % 0)

    def regions(self, rgnstr):
        cmd = 'regions %s'

    def rssregion(self, ra, dec):
        """Plot the FOV for RSS"""
#        cmd='color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=0 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\ncircle(%s, %s, 4\')' % (ra, dec)
#        cmd='fk5\ncircle(%s, %s, 4\')  # color=yellow background dashlist=8 3 width=1 font="helvetica 10 normal roman" select=0 highlite=1 dash=0 fixed=0 edit=0 move=0 delete=0 include=1 source=1\n' % (ra, dec)

#        ds9.set(cmd)
        self.ds9.set('regions', 'fk5; circle(%f,%f,4\') # color=yellow background dashlist=8 3 width=3 font="helvetica 10 normal roman" select=0 highlite=1 dash=0 fixed=0 edit=0 move=0 delete=0 include=1 source=1'%(ra, dec))

    def rotate(self, angle):
        """Rotate the image"""
        self.ds9.set('rotate to %f' % angle)


    def regionfromfile(self, regfile, d=None, rformat='ds9'):
        cmd='regions %s -format %s' % (regfile, rformat)
        self.ds9.set(cmd)

    def deleteregions(self):
        """Delete all regions in the frame"""
        cmd='regions delete all'
        self.ds9.set(cmd)

    def getregions(self):
        """Return a list of regions"""
        rgnstr=self.ds9.get('regions -system fk5')
        i = 0
        newslits = {}
        #print rgnstr
        for l in rgnstr.split('\n'): 
            tags = ''
            # work out how to use tags and just deal with "slit" tags
            if l.startswith('box'):
                #first look for tags
                l = l[4:].split('#')
                if len(l) > 1:
                    tags = l[-1]
                l = l[0][:-2].split(',')
                newslits[i] = [l, tags]
                i += 1
            elif l.startswith('circle'):
                l = l[7:].split('#')
                #print l
                if len(l) > 1:
                    tags=l
                l = l[0][:-2].split(',')
                newslits[i] = [l, tags]
                i += 1
        return newslits
         
