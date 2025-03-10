#! /usr/bin/env python

# implementation of Sutherland-Hodgman polygon clipping
# http://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping
# special case of unit square clip, quadralateral subject polygon 
# numpy-vectorized
# _d dimension (eg 0/1 = x/y in vertex)
# _c vertex in unit square clip polygon (0-3)
# _o vertex in clipped polygon (0-7)
# _z cases (vectorization)
# _Z cases before optional masking
# _v _oz flattened to allow for varying numbers of vertices
# _x 0/1 = vertex/intersection in SH clipping
# Polygon _vd vertices _v=0-7,dimensions _d=0,1.  vertices listed *counterclockwise*

import os, sys, time, glob, shutil
import numpy as np
 
def cliparea(subjectPolygon,mask=False):
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
    if type(mask) is np.ndarray:
        okbin_Z = np.copy(mask.ravel())
        Cases = okbin_Z.shape[0]
        cases = okbin_Z.sum()
        mask = True
    else:
        Cases = subjectPolygon.size/8
        cases = Cases                                      

    buffer1 = np.empty(cases*16,dtype='float32')
    buffer2 = np.empty(cases*16,dtype='float32')
    rollbuffer = np.empty(cases*16,dtype='float32')

    outPolygon_vd = buffer1[:cases*8+2].reshape((-1,2))
    if mask:
        outPolygon_vd[:-1] = \
          np.compress(okbin_Z,subjectPolygon.reshape(-1,4,2).astype('float32'),axis=0).reshape(cases*4,2)
        Z_z = np.where(okbin_Z)[0]        
    else:
        outPolygon_vd[:-1] = subjectPolygon.reshape(cases*4,2).astype('float32')
        Z_z = np.arange(Cases)

    vertices_z = np.repeat(4,cases)
    vertices = vertices_z.sum()
    v0_z = np.arange(cases)*4        # v-address of first vertex for each case
    vn_z = v0_z + 3

    z = 0
#    print 'cases = ',cases
#    print 'vertices = ',vertices
#    print "z = ",z
#    print outPolygon_vd[v0_z[z]:vn_z[z]+1]

    time0 = time.time() 
    for c in range(4):
        if c%2:
            oldbuffer = buffer2
            newbuffer = buffer1
        else:
            oldbuffer = buffer1
            newbuffer = buffer2
        vroll_vd = rollbuffer[:(vertices+1)*2].reshape((-1,2))
        vroll_vd[1:-1] = outPolygon_vd[:-2]
        endvert_zd = np.take(outPolygon_vd,vn_z,axis=0)
        vput0_z = np.copy(v0_z)
        np.put(vput0_z,np.where(vertices_z==0),vertices)   # safekeeping at the end for empty polygons
        np.put(vroll_vd[:,0],v0_z, endvert_zd[:,0])
        np.put(vroll_vd[:,1],v0_z, endvert_zd[:,1])

        use_vx = np.zeros((vertices,2),dtype=bool)
        useint_v = use_vx[:,0]
        usevert_v = use_vx[:,1]
        if c==0:
            usevert_v[:] = (0 < outPolygon_vd[:-1,0])
            useint_v[:] =  ((0 < vroll_vd[:-1,0]) ^ usevert_v)
        elif c==1:
            usevert_v[:] = (outPolygon_vd[:-1,1] > 0)
            useint_v[:] =  ((vroll_vd[:-1,1] > 0) ^ usevert_v)
        elif c==2:
            usevert_v[:] = (0 > (outPolygon_vd[:-1,0] -1))
            useint_v[:] =  ((0 > (vroll_vd[:-1,0] -1)) ^ usevert_v)
        elif c==3:
            usevert_v[:] = ((outPolygon_vd[:-1,1] -1) < 0)
            useint_v[:] =  (((vroll_vd[:-1,1] -1) < 0) ^ usevert_v)

        outPolygon_Vd = outPolygon_vd.compress(useint_v,axis=0)
        vroll_Vd = vroll_vd.compress(useint_v,axis=0)       
        dv_Vd = outPolygon_Vd - vroll_Vd                                               
        qv_V = vroll_Vd[:,0]*outPolygon_Vd[:,1] - vroll_Vd[:,1]*outPolygon_Vd[:,0]

        intersects = useint_v.sum()
        intersect_Vd = oldbuffer[vertices*2:(vertices+intersects)*2].reshape((-1,2))
        intersect_Vd[:] = int(c/2)

        if c==0:
            intersect_Vd[:,1] = -qv_V/dv_Vd[:,0]
        elif c==1:
            intersect_Vd[:,0] =  qv_V/dv_Vd[:,1]
        elif c==2:
            intersect_Vd[:,1] = (-dv_Vd[:,1] + qv_V)/ (-dv_Vd[:,0])
        elif c==3:
            intersect_Vd[:,0] = (-dv_Vd[:,0] - qv_V)/ (-dv_Vd[:,1])

        Vertices = use_vx.sum()

    # map to new _zo = _V
        arg_vx = use_vx.astype(int)
        np.cumsum(arg_vx[:,0],out=arg_vx[:,0]) 
        arg_vx[:,0] += vertices-1                           
        arg_vx[:,1] = np.arange(vertices,dtype=int)
        vmap_V = np.extract(use_vx,arg_vx)
        count_v = np.zeros(vertices+1,dtype='int64')   # extra allows for wrap from empty polygon at beginning        
        count_v[:vertices] = (use_vx[:,0].astype(int) + use_vx[:,1].astype(int)).cumsum(out=count_v[:vertices])
        Vertices_z = count_v.take(vn_z)
        Vertices_z[cases-1:0:-1] -= Vertices_z[cases-2::-1]

        newPolygon_vd = newbuffer[:(Vertices+1)*2].reshape((-1,2))
        newPolygon_vd[:-1] = np.take(oldbuffer[:(vertices+intersects)*2].reshape((-1,2)),vmap_V,axis=0)
        outPolygon_vd = newPolygon_vd
        vertices_z = Vertices_z
        vertices = Vertices
        v0_z[1:] = vertices_z[:-1].cumsum()
        vn_z = v0_z + vertices_z-1

#        print c, " bins left: ",(vertices_z>0).sum()
#        print outPolygon_vd[v0_z[z]:vn_z[z]+1]

    vroll_vd = rollbuffer[:(vertices+1)*2].reshape((-1,2))  
    vroll_vd[:-2] = outPolygon_vd[1:-1]
    vert0_zd = np.take(outPolygon_vd,v0_z,axis=0)
    vputn_z = np.copy(vn_z)
    np.put(vputn_z,np.where(vertices_z==0),vertices)    # safekeeping at the end for empty polygons
    np.put(vroll_vd[:,0],vputn_z, vert0_zd[:,0])
    np.put(vroll_vd[:,1],vputn_z, vert0_zd[:,1])

    area_v = ((outPolygon_vd[:,0]*vroll_vd[:,1]).cumsum(dtype='float64')  \
                    -(outPolygon_vd[:,1]*vroll_vd[:,0]).cumsum(dtype='float64'))/2.

#   combine with mask........................
    ok_z = (vertices_z>0)
    goodcases = ok_z.sum()
    area_z = np.take(area_v,vn_z[ok_z])
    area_z[goodcases-1:0:-1] -= area_z[goodcases-2::-1]
    area_Z = np.zeros(Cases)
    np.put(area_Z,Z_z[ok_z],area_z)

#    print "area for bin ",z,": ",area_z[z]
    print "cliparea time: ",time.time()-time0,'\n'
    return area_Z

# -----------------------------------------------------------------
if __name__=='__main__':
    outbin=sys.argv[1]
    clipadx=sys.argv[2]
    subjectPolygon=np.loadtxt(outbin,dtype=float).reshape(-1,8)
    clipaddress=np.loadtxt(clipadx,dtype=float).reshape(-1,2)
    area = cliparea(subjectPolygon, clipaddress)

