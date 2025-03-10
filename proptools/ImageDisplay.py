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
opt1 = "-geometry 640x840 -view info no -view panner no -view magnifier no -view colorbar no -view wcsa no -view buttons no "
opt2 = "-tile yes -tile row -frame 2 -cmap grey -cmap invert yes -scale mode 99.5 "
opt3 = "-match frame wcs -match scale -match colorbar"
# opt = opt1+opt2+opt3

class ImageDisplay:
    def __init__(self,target='ImageDisplay:*'):
        self.ds9 = ds9.DS9(start=opt1,verify=False,wait=240)
        
    def readyspectrumdisplay(self):
        frames = len(self.ds9.get('ds9 frame all #'))
        if (frames==1):
            self.ds9.set(opt2+opt3)          

    def displayfile(self, filename, frameno, pa=None):
        pacmd = 'rotate to %f' % [pa,0][pa==None]
        self.ds9.set('frame %i' % frameno)    
        self.ds9.set('file %s'  % filename)
        self.ds9.set(pacmd)        
        return self.ds9.get_pyfits()        
        
    def displayhdu(self, hdul, frameno, pa=None):
        pacmd = 'rotate to %f' % [pa,0][pa==None]
        self.ds9.set('frame %i' % frameno)    
        self.ds9.set_pyfits(hdul)
        self.ds9.set(pacmd)        
        self.ds9.set('match frames wcs')        

    def setframe(self,frameno):
        self.ds9.set('frame %i' % frameno)

    def regions(self, rgnstr):
        cmd = 'regions %s'

    def rssregion(self, ra, dec):
        """Plot the FOV for RSS"""
        print "ra, dec:", ra, dec
        if ra:
            print 'regions shape circle( %fd %fd 4\')' % (ra,dec)
            self.ds9.set('regions shape circle( %fd %fd 4\')' % (ra,dec))

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
         
