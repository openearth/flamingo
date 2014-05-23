import numpy as np
import collections
from .. import log

def remove_disjoint(segments):
    '''Remove disjoint regions in segmentation'''
    
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
    log.memory_usage('after region growing')
        
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
    log.memory_usage('after region update')

    return seg_new

def region_growing(mask,connectivity=8):
    '''Simple region growing algorithm'''
    
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

    return regions,edges
    
