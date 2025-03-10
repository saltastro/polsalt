#! /usr/bin/env python

# implementation of Sutherland-Hodgman polygon clipping
# http://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping
# special case of unit square clip, quadralateral subject polygon 
# numpy-vectorized
# _d dimension (eg 0/1 = x/y in vertex)
# _c vertex in unit square clip polygon (0-3)
# _o vertex in clipped polygon (0-7)
# _z cases (vectorization)
# _v _oz flattened to allow for varying numbers of vertices
# _x 0/1 = vertex/intersection in SH clipping
# Polygon _vd vertices _v=0-7,dimensions _d=0,1.  vertices listed *counterclockwise*

import os, sys, time, glob, shutil
import numpy as np
 
def cliparea(subjectPolygon):
    """ Compute clipped areas between subject quadrilaterals and a unit square bin
 
    Parameters 
    ----------
    subjectPolygon: >=2d ndarray _sd
        quadrateral polygons to be clipped

    Returns
    -------
    area: >=0d ndarray of clipped areas
    outPolygon: >=2d ndarray _sd
        Wave map of wavelengths correspond to pixels

    """
    cases = subjectPolygon.size/8                                      
    vertices_z = np.repeat(4,cases)
    v_z = np.arange(cases)*4        # v-address of first vertex for each case
                                                       
    outPolygon_vd = (subjectPolygon).reshape(cases*4,2)
    outPolygon_bytes = memoryview(subjectPolygon)

    z = 849234
    print 'cases = ',cases
    print 'vertices = ',vertices_z.sum()
    print "z = ",z
    print outPolygon_vd[v_z[z]:v_z[z]+vertices_z[z]]

    time0 = time.time() 
    for c in range(4):
        vroll_vd = np.roll(outPolygon_vd,1,axis=0)
        vroll_vd[v_z] = outPolygon_vd[v_z+vertices_z-1]
        print "A ",time.time()-time0

        if c==0:
            usevert_v = (0 < outPolygon_vd[:,0])
            useint_v =  ((0 < vroll_vd[:,0]) ^ usevert_v)
        elif c==1:
            usevert_v = (outPolygon_vd[:,1] > 0)
            useint_v =  ((vroll_vd[:,1] > 0) ^ usevert_v)
        elif c==2:
            usevert_v = (0 > (outPolygon_vd[:,0] -1))
            useint_v =  ((0 > (vroll_vd[:,0] -1)) ^ usevert_v)
        elif c==3:
            usevert_v = ((outPolygon_vd[:,1] -1) < 0)
            useint_v =  (((vroll_vd[:,1] -1) < 0) ^ usevert_v)
        print "B ",time.time()-time0

        dv_Vd = (outPolygon_vd - vroll_vd)[useint_v] 
        print "C ",time.time()-time0
                                               
        qv_V = (vroll_vd[:,0]*outPolygon_vd[:,1] - vroll_vd[:,1]*outPolygon_vd[:,0])[useint_v]
        print "D ",time.time()-time0

        if c==0:
            intersect_Vd = (qv_V[:,None]*np.array([0,-1]))/ (dv_Vd[:,0])[:,None]
        elif c==1:
            intersect_Vd = (qv_V[:,None]*np.array([1,0]))/ (dv_Vd[:,1])[:,None]
        elif c==2:
            intersect_Vd = (-dv_Vd + qv_V[:,None]*np.array([0,1,]))/ (-dv_Vd[:,0])[:,None]
        elif c==3:
            intersect_Vd = (-dv_Vd + qv_V[:,None]*np.array([-1,0]))/ (-dv_Vd[:,1])[:,None]
        print "E ",time.time()-time0
        print "F ",time.time()-time0       
    # map to new _zo
        use_vx = np.array([useint_v,usevert_v],dtype=bool).T
        print "G ",time.time()-time0

        idxout_vx = use_vx.astype(int).ravel().cumsum(out=use_vx).reshape((-1,2)) - 1
        intout = idxout_vx[:,0].compress(useint_v,axis=0)
        vertout = idxout_vx[:,1].compress(usevert_v,axis=0)
        print "H ",time.time()-time0

        vertices_z[:] = (usevert_v.cumsum() \
                        + useint_v.cumsum())[v_z+vertices_z-1]
        vertices_z[cases-1:0:-1] -= vertices_z[cases-2::-1]
        print "I ",time.time()-time0

        newPolygon_vd = np.zeros((vertices_z.sum(),2))
        newPolygon_bytes = memoryview(newPolygon_vd.ravel())
        print "J ",time.time()-time0

        intersect_bytes = memoryview(intersect_Vd.ravel())
        newPolygon_bytes[8*intout] = intersect_bytes
        print "K ",time.time()-time0

        newPolygon_vd[vertout] = outPolygon_vd[usevert_v]
        print "L ",time.time()-time0

        v_z[1:] = vertices_z[:-1].cumsum()
        print "M ",time.time()-time0

        outPolygon_vd = newPolygon_vd
        print "N ",time.time()-time0

        print outPolygon_vd[v_z[z]:v_z[z]+vertices_z[z]]

    vroll_vd = np.roll(outPolygon_vd,-1,axis=0)
    vroll_vd[v_z+vertices_z-1] = outPolygon_vd[v_z]
    area0_v = (outPolygon_vd[:,0]*vroll_vd[:,1]).cumsum() 
    area1_v = (outPolygon_vd[:,1]*vroll_vd[:,0]).cumsum() 
    area_z = ((area0_v[v_z+vertices_z-1] - area0_v[v_z]) \
            - (area1_v[v_z+vertices_z-1] - area1_v[v_z]))/2.

    print "Z",time.time()-time0
    print area_z[z]

    return area_z

# -----------------------------------------------------------------
if __name__=='__main__':
    outbin=sys.argv[1]
    clipadx=sys.argv[2]
    subjectPolygon=np.loadtxt(outbin,dtype=float).reshape(-1,8)
    clipaddress=np.loadtxt(clipadx,dtype=float).reshape(-1,2)
    area = cliparea(subjectPolygon, clipaddress)

