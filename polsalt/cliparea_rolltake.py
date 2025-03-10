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

    buffer1 = np.empty(cases*16,dtype='float32')
    buffer2 = np.empty(cases*16,dtype='float32')
    rollbuffer = np.empty(cases*16,dtype='float32')

    outPolygon_vd = buffer1[:cases*8].reshape((-1,2))
    outPolygon_vd[:] = subjectPolygon.reshape(cases*4,2).astype('float32')

    vertices_z = np.repeat(4,cases)
    vertices = cases*4
    isz0_v = (np.arange(vertices)%4 == 0)           # flags first vertex for each case
    iszn_v = np.roll(isz0_v,-1)                     # flags last vertex for each case

    z = 849234
    print 'cases = ',cases
    print 'vertices = ',vertices
    print "z = ",z
    print outPolygon_vd[4*z:4*z+4]

    time0 = time.time() 
    for c in range(4):
        print "A ",time.time()-time0

        if c%2:
            oldbuffer = buffer2
            newbuffer = buffer1
        else:
            oldbuffer = buffer1
            newbuffer = buffer2
        print "AA ",time.time()-time0

        vroll_vd = rollbuffer[:vertices*2].reshape((-1,2))
        oldbuffer[2*vertices:2*(vertices+cases)] = outPolygon_vd.compress(iszn_v)
        rmap_v = (np.arange(vertices)-1)*(~iszn_v) + 
            iszn_v.cumsum() np.arange(vertices)*iszn_v
        print "AB ",time.time()-time0
        vroll_vd[:] = np.take(oldbuffer[:vertices*2].reshape((-1,2)),rmap_v,axis=0)

        print "AC ",time.time()-time0

        print "B ",time.time()-time0

        use_vx = np.zeros((vertices,2),dtype=bool)
        useint_v = use_vx[:,0]
        usevert_v = use_vx[:,1]
        if c==0:
            usevert_v[:] = (0 < outPolygon_vd[:,0])
            useint_v[:] =  ((0 < vroll_vd[:,0]) ^ usevert_v)
        elif c==1:
            usevert_v[:] = (outPolygon_vd[:,1] > 0)
            useint_v[:] =  ((vroll_vd[:,1] > 0) ^ usevert_v)
        elif c==2:
            usevert_v[:] = (0 > (outPolygon_vd[:,0] -1))
            useint_v[:] =  ((0 > (vroll_vd[:,0] -1)) ^ usevert_v)
        elif c==3:
            usevert_v[:] = ((outPolygon_vd[:,1] -1) < 0)
            useint_v[:] =  (((vroll_vd[:,1] -1) < 0) ^ usevert_v)
        print "C ",time.time()-time0

        dv_Vd = (outPolygon_vd - vroll_vd).compress(useint_v,axis=0) 
        print "D ",time.time()-time0
                                               
        qv_V = (vroll_vd[:,0]*outPolygon_vd[:,1] - vroll_vd[:,1]*outPolygon_vd[:,0])    \
                .compress(useint_v,axis=0)
        print "E ",time.time()-time0

        intersects = useint_v.sum()
        intersect_Vd = oldbuffer[vertices*2:(vertices+intersects)*2].reshape((-1,2))
        if c==0:
            intersect_Vd[:] = (qv_V[:,None]*np.array([0,-1]))/ (dv_Vd[:,0])[:,None]
        elif c==1:
            intersect_Vd[:] = (qv_V[:,None]*np.array([1,0]))/ (dv_Vd[:,1])[:,None]
        elif c==2:
            intersect_Vd[:] = (-dv_Vd + qv_V[:,None]*np.array([0,1,]))/ (-dv_Vd[:,0])[:,None]
        elif c==3:
            intersect_Vd[:] = (-dv_Vd + qv_V[:,None]*np.array([-1,0]))/ (-dv_Vd[:,1])[:,None]
        print "F ",time.time()-time0

        Vertices = use_vx.sum()
        print "G ",time.time()-time0

    # map to new _zo = _V
        arg_vx = use_vx.astype(int)
        arg_vx[:,0] = arg_vx[:,0].cumsum() + vertices-1
        arg_vx[:,1] = np.arange(vertices,dtype=int)
        vmap_V = arg_vx.ravel().compress(use_vx.ravel())
        print "H ",time.time()-time0

        count_v =  use_vx[:,0].astype(int) + use_vx[:,1].astype(int)
        Vertices_z = count_v.cumsum(out=count_v).take(v_z+vertices_z-1)
        Vertices_z[cases-1:0:-1] -= Vertices_z[cases-2::-1]
        print "I ",time.time()-time0

        newPolygon_vd = newbuffer[:Vertices*2].reshape((-1,2))
        newPolygon_vd[:] = np.take(oldbuffer[:(vertices+intersects)*2].reshape((-1,2)),vmap_V,axis=0)
        print "J ",time.time()-time0

        print "K ",time.time()-time0
        outPolygon_vd = newPolygon_vd
        vertices_z = Vertices_z
        vertices = Vertices
        v_z[1:] = vertices_z[:-1].cumsum()
        print "L ",time.time()-time0

        print outPolygon_vd[v_z[z]:v_z[z]+vertices_z[z]]

    vroll_vd = rollbuffer[:vertices*2].reshape((-1,2))  
    vroll_vd[:-1] = outPolygon_vd[1:]
    vroll_vd[v_z+vertices_z-1] = outPolygon_vd[v_z]
    area_v = ((outPolygon_vd[:,0]*vroll_vd[:,1]) \
            -(outPolygon_vd[:,1]*vroll_vd[:,0])).cumsum(dtype='float64') 
    area_z = (area_v[v_z+vertices_z-1] - area_v[v_z])/2.

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

