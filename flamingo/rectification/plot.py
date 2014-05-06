import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_rectified(X, Y, imgs, axs=None, figsize=(30,20), threshold=1e4):

    # create figure
    if axs is None:
        fig, axs = subplots(figsize=figsize)
    else:
        fig = axs.figure

    # loop over images
    for x, y, img in zip(X, Y, imgs):

        # find horizon based on threshold
        o = _find_horizon_offset(x, y, threshold=threshold)

        # construct rgba vector
        rgba = _construct_rgba_vector(img)

        im = axs.pcolormesh(x[o:,:], y[o:,:], np.mean(img[o:,:,:], -1))
        im.set_array(None) # remove the array
        im.set_edgecolor('none')
        im.set_facecolor(rgba)

    axs.set_aspect('equal')

    return fig, axs

def plot_footprint(X, Y, imgs, axs=None, figsize=(30,20), cmap=cm.jet, alpha=0.4, threshold=1e4):

    # create figure
    if axs is None:
        fig, axs = subplots(figsize=figsize)
    else:
        fig = axs.figure

    # loop over images
    patches = []
    for x, y, img in zip(X, Y, imgs):

        # find horizon based on threshold
        o = _find_horizon_offset(x, y, threshold=threshold)

        xy = np.vstack((x[[o,o,-1,-1],[0,-1,-1,0]], y[[o,o,-1,-1],[0,-1,-1,0]])).T
        patches.append(matplotlib.patches.Polygon(xy, closed=True))
    
    p = matplotlib.collections.PatchCollection(patches, cmap=cmap, alpha=alpha)
    p.set_array(np.array(range(len(imgs))))

    axs.add_collection(p)
    axs.set_aspect('equal')

    return fig, axs

def _find_horizon_offset(x, y, threshold=1e4):
    th = (np.abs(x)>threshold)|(np.abs(y)>threshold)
    offset = np.max(np.where(np.any(th, axis=1))) + 1

    return offset

def _construct_rgba_vector(img):
    rgb = img[o:,:-1,:].reshape((-1,3)) / 255.0 # we have 1 less faces than grid cells
    rgba = np.concatenate((rgb, np.ones((rgb.shape[0],1))), axis=1)
    
    return rgba
