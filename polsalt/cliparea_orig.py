#! /usr/bin/env python

import os, sys, time, glob, shutil
import numpy as np

# implementation of Sutherland-Hodgman polygon clipping
# http://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping
 
def clip(subjectPolygon, clipPolygon):
   def inside(p):
      print "    ",(cp2[0]-cp1[0]),(p[1]-cp1[1]) , (cp2[1]-cp1[1]),(p[0]-cp1[0])
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      print "      ",dc,n1
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      print " ",cp2
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         print "  ",e
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
               print "AA ",computeIntersection()
            outputList.append(e)
            print "AB ",e
         elif inside(s):
            outputList.append(computeIntersection())
            print "C  ",computeIntersection()
         s = e
      cp1 = cp2
      print outputList
   return(outputList)

# implementation of "shoelace formula"
# https://en.wikipedia.org/wiki/Shoelace_formula
# http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

# -----------------------------------------------------------------
if __name__=='__main__':
    inbin=sys.argv[1]
    outbin=sys.argv[2]
    subjectPolygon=np.loadtxt(inbin,dtype=float).tolist()
    clipPolygon=np.loadtxt(outbin,dtype=float).tolist()
    clipbin = clip(subjectPolygon, clipPolygon)
    for v in range(len(clipbin)): print ('%8.3f %8.3f' % tuple(clipbin[v]))
    print 'area = %8.3f' % PolygonArea(clipbin)
