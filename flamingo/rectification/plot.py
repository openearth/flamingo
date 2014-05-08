import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections
import matplotlib.patches

def plot_rectified(X, Y, imgs,
                   rotation=None, translation=None, max_distance=1e4,
                   axs=None, figsize=(30,20), color=True):

    # create figure
    if axs is None:
        fig, axs = plt.subplots(figsize=figsize)
    else:
        fig = axs.figure

    # loop over images
    for x, y, img in zip(X, Y, imgs):

        # find horizon based on maximum distance
        o = _find_horizon_offset(x, y, max_distance=max_distance)

        # rotate to world coordinate system
        x, y = rotate_translate(x, y, rotation=rotation, translation=translation)

        # plot
        im = axs.pcolormesh(x[o:,:], y[o:,:], np.mean(img[o:,...], -1), cmap='Greys')

        # add colors
        if color:
            rgba = _construct_rgba_vector(img[o:,...])
            im.set_array(None) # remove the array
            im.set_edgecolor('none')
            im.set_facecolor(rgba)

    axs.set_aspect('equal')

    return fig, axs

def plot_coverage(X, Y,
                  rotation=None, translation=None, max_distance=1e4, 
                  axs=None, figsize=(30,20), cmap=cm.jet, alpha=0.4):

    # create figure
    if axs is None:
        fig, axs = plt.subplots(figsize=figsize)
    else:
        fig = axs.figure

    xl = [0,0]
    yl = [0,0]

    # loop over images
    patches = []
    for x, y in zip(X, Y):

        # find horizon based on max_distance
        o = _find_horizon_offset(x, y, max_distance=max_distance)

        # rotate to world coordinate system
        x, y = rotate_translate(x, y, rotation=rotation, translation=translation)

        xl[0] = np.min((np.min(x), xl[0]))
        xl[1] = np.max((np.max(x), xl[1]))
        yl[0] = np.min((np.min(y), yl[0]))
        yl[1] = np.max((np.max(y), yl[1]))
        
        # create patch
        xy = np.vstack((x[[o,o,-1,-1],[0,-1,-1,0]], y[[o,o,-1,-1],[0,-1,-1,0]])).T
        patches.append(matplotlib.patches.Polygon(xy, closed=True))
    
    # create collection from patches
    p = matplotlib.collections.PatchCollection(patches, cmap=cmap, alpha=alpha)
    p.set_array(np.array(range(len(X))))

    # plot
    axs.add_collection(p)
    axs.set_xlim(xl)
    axs.set_ylim(yl)
    axs.set_aspect('equal')

    return fig, axs

def rotate_translate(x, y, rotation=None, translation=None):
    
    if rotation is not None:
        shp = x.shape
        rotation = rotation / 180 * np.pi

        R = np.array([[ np.cos(rotation),np.sin(rotation)],
                      [-np.sin(rotation),np.cos(rotation)]])

        xy = np.dot(np.hstack((x.reshape((-1,1)),
                               y.reshape((-1,1)))), R)
    
        x = xy[:,0].reshape(shp)
        y = xy[:,1].reshape(shp)

    if translation is not None:
        x += translation[0]
        y += translation[1]
    
    return x, y

def _find_horizon_offset(x, y, max_distance=1e4):
    offset = 0
    if max_distance is not None:
        try:
            th = (np.abs(x)>max_distance)|(np.abs(y)>max_distance)
            offset = np.max(np.where(np.any(th, axis=1))) + 1
        except:
            pass

    return offset

def _construct_rgba_vector(img):
    if np.any(img > 1):
        img /= 255.0

    rgb = img[:,:-1,:].reshape((-1,3)) # we have 1 less faces than grid cells
    rgba = np.concatenate((rgb, np.ones((rgb.shape[0],1))), axis=1)
    
    return rgba
