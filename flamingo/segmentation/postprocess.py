import numpy as np
import collections


def remove_disjoint(segments):
    '''Remove disjoint regions in segmentation

    Remove disjoint regions in segmentation by running a region
    growing algorihtm for each segment. Any segment that appears to
    consist out of multiple disconnected parts is splitted. The
    biggest part remains as placeholder of the existing superpixel.
    The smaller parts are joined with the neighbouring superpixels.
    If multiple neighbouring superpixels exist, the one that shares
    the largest border is chosen.

    Parameters
    ----------
    segments : np.ndarray
        NxM matrix with segment numbering

    Returns
    -------
    np.ndarray
        NxM matrix with alternative segment numbering with connected
        segments
    '''
    
    seg_new = np.asarray(segments).copy()
    r = []
    e = []
    maxind = []
    for i in range(seg_new.max()+1):
        region,edges = region_growing(seg_new == i)
        
        reglen = [len(x) for x in region]
        maxind.append(reglen.index(max(reglen)))

        r.append(region)
        e.append(edges)
        
    for i in range(len(r)):
        inds = range(len(r[i]))
        inds.pop(maxind[i])
        
        for j in inds:
            neighsegs = [seg_new[ecoor] for ecoor in e[i][j] if ecoor in r[seg_new[ecoor]][maxind[seg_new[ecoor]]]]
            
            if neighsegs == []:
                neighsegs = [seg_new[ecoor] for ecoor in e[i][j]]
            
            d = collections.defaultdict(int)
            for ns in neighsegs:
                d[ns] += 1
                
            seg_new[zip(*r[i][j])] = max(d,key=d.get)

    return seg_new


def region_growing(mask, connectivity=8):
    '''Simple region growing algorithm

    Parameters
    ----------
    mask : np.ndarray
        Binary matrix indicating what pixels are within a region and
        what are not
    connectivity : int, 4 or 8
        Number of neighbouring pixels taken into account

    Returns
    -------
    list
        List of 2-tuples with coordinates within a region
    list
        List of 2-tuples with coordinates at the edge of the region
    '''
    
    inds = {tuple(i) for i in np.array(np.where(mask)).transpose()}
    inds_todo = inds.copy()
    
    bbox = mask.shape
    
    if connectivity == 8:
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    elif connectivity == 4:
        neigh = [(-1,0),(0,-1),(0,1),(1,0)]
    else:
        raise InputError(connectivity,"Only 4- or 8-connectivity is allowed as valid input.")
    
    regions = []
    edges = []
    
    while not inds_todo == set():
        Q = set()   # queue
        r = set()   # regions
        e = set()   # edges
        
        for itd in inds_todo:
            break

        Q.add(itd)

        while not Q == set():
            t = Q.pop()
            x = t[0]
            y = t[1]

            for n in neigh:
                xn = x+n[0]
                yn = y+n[1]
                if (xn >= 0) and (xn < bbox[0]) and (yn >= 0) and (yn < bbox[1]):
                    if ((xn,yn) not in Q) and ((xn,yn) in inds_todo):
                        Q.add((xn,yn))
                    elif ((xn,yn) not in inds):
                        e.add((xn,yn))

            r.add(t)
            inds_todo.discard(t)
            
        regions.append(r)
        edges.append(e)

    return regions, edges
    

def regularize(segments, nx, ny):
    '''Create a regular grid from a collection of image segments

    The number of segments supplied is supposed to be larger than the
    number of segments in the target grid (nx*ny). A regular grid of
    size nx*ny over the image grid NxM is constructed. Subsequently,
    the segments are ordered based on size. the nx*ny largest segments
    are preserved and assigned to a single grid cell in the regular
    grid based on least squares fit. The smaller segments are added to
    the preserved segment that is closest based on their centroids.

    Parameters
    ----------
    segments : np.ndarray
        NxM matrix with segment numbering
    nx, ny : int
        Dimensions of the target superpixel grid

    Returns
    -------
    np.ndarray
        NxM matrix with alternative segment numbering with segments in
        a regular grid
    '''
    
    # create regular target grid
    m_segments = nx * ny
    s1, s2 = segments.shape[0]/nx, segments.shape[1]/ny
    sy, sx = np.meshgrid(np.arange(s2/2, segments.shape[1], s2),
                         np.arange(s1/2, segments.shape[0], s1))
    sxf, syf = sx.flatten(), sy.flatten()

    # create image coordinate matrices
    v, u = np.meshgrid(np.arange(segments.shape[1]),
                       np.arange(segments.shape[0]))

    uw = np.zeros(np.max(segments)+1)
    vw = np.zeros(np.max(segments)+1)
    sz = np.zeros(np.max(segments)+1)
    n  = np.zeros(np.max(segments)+1)
    
    # determine sizes and centroids
    for i in np.unique(segments):
        idx = segments == i
        uw[i] = np.mean(u[idx])
        vw[i] = np.mean(v[idx])
        sz[i] = np.sum(idx)
        n[i]  = i

    # order based on size and select largest superpixels
    n, sz, uw, vw = zip(*sorted(
            zip(n, sz, uw, vw), key=lambda x: x[1], reverse=True))
    ns, szs, uws, vws = n[:m_segments], sz[:m_segments], \
        uw[:m_segments], vw[:m_segments]
    szs = list(szs)

    # assign smaller superpixels to closest larger superpixel
    rsegments = segments.copy()
    for i in range(m_segments, len(n)):
        d = (uws - uw[i])**2 + (vws - vw[i])**2
        ii = np.argmin(d)
        idx = segments == n[i]
        rsegments[idx] = ns[ii]
        szs[ii] += sz[i]

    # initially sort based on vertical centroid location
    ns, szs, uws, vws = zip(*sorted(zip(ns, szs, uws, vws), 
                                    key=lambda x: x[0]))

    ns  = list(ns)
    szs = list(szs)
    uws = list(uws)
    vws = list(vws)
    
    # update superpixel sorting such that the least square difference
    # with the regular target grid is minimal
    n_changes = 1
    while n_changes > 0: # this loop is a dirty fix
    
        i = 0
        n_changes = 0
        while i+1 < len(ns):

            changed = False
            for j in range(i+1, len(ns)):
                d1 = (uws[i]-sxf[i])**2 + (vws[i]-syf[i])**2 + \
                     (uws[j]-sxf[j])**2 + (vws[j]-syf[j])**2
                d2 = (uws[j]-sxf[i])**2 + (vws[j]-syf[i])**2 + \
                     (uws[i]-sxf[j])**2 + (vws[i]-syf[j])**2

                if d2 < d1:
                    ns[i],  ns[j]  = ns[j],  ns[i]
                    szs[i], szs[j] = szs[j], szs[i]
                    uws[i], uws[j] = uws[j], uws[i]
                    vws[i], vws[j] = vws[j], vws[i]
                    changed = True
                    break

            if not changed:
                i += 1
            else:
                n_changes += 1
        
    # assign new superpixel numbers
    rsegments_ordered = rsegments.copy()
    for i, n in enumerate(ns):
        rsegments_ordered[rsegments==n] = i

    return rsegments_ordered
