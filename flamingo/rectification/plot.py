import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections
import matplotlib.patches


def plot_rectified(X, Y, imgs, slice=1,
                   rotation=None, translation=None, max_distance=1e4,
                   ax=None, figsize=(30,20), cmap='Greys',
                   color=True, n_alpha=0):
    '''Plot the projection of multiple RGB images in a single axis.

    Plot a list of images using corresponding lists of real-world
    x and y coordinate matrices. The resulting composition can be
    rotated and translated seperately.

    Points projected at infinite distance can be ignored by
    specifying a maximum distance.

    Parameters
    ----------
    X : list of np.ndarrays
        List of NxM matrix containing real-world x-coordinates
    Y : list of np.ndarrays
        List of NxM matrix containing real-world y-coordinates
    imgs : list of np.ndarrays
        List of NxMx1 or NxMx3 image matrices
    slice : int
        Slice to limit resolution
    rotation : float, optional
        Rotation angle in degrees
    translation : list or tuple, optional
        2-tuple or list with x and y translation distances
    max_distance : float, optional
        Maximum distance from origin to be included in the plot.
        Larger numbers are considered to be beyond the horizon.
    ax : matplotlib.axes.AxesSubplot, optional
        Axis object used for plotting
    figsize : tuple, optional
        2-tuple or list containing figure dimensions
    color : bool, optional
        Whether color image should be plotted or grayscale
    n_alpha : int
        Number of border pixels to use to increase alpha

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing axis object
    matplotlib.axes.AxesSubplot
        Axis object containing plot
    '''

    # create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    s = slice

    # loop over images
    for x, y, img in zip(X, Y, imgs):

        # find horizon based on maximum distance
        o = find_horizon_offset(x, y, max_distance=max_distance)

        # rotate to world coordinate system
        x, y = rotate_translate(x, y, rotation=rotation, translation=translation)

        # plot
        im = ax.pcolormesh(x[o::s,::s], y[o::s,::s],
                           np.mean(img[o::s,::s,...], -1), cmap=cmap, rasterized=True)

        # add colors
        if color:
            rgba = _construct_rgba_vector(img[o::s,::s,...], n_alpha=n_alpha)
            im.set_array(None) # remove the array
            im.set_edgecolor('none')
            im.set_facecolor(rgba)

    ax.set_aspect('equal')

    return fig, ax


def plot_coverage(X, Y,
                  rotation=None, translation=None, max_distance=1e4, 
                  ax=None, figsize=(30,20), cmap=cm.jet, alpha=0.4):
    '''Plot the coverage of the projection of multiple images in a single axis.

    Plot the outline of lists of real-world x and y coordinate
    matrices. The resulting composition can be rotated and
    translated seperately.

    Points projected at infinite distance can be ignored by
    specifying a maximum distance.

    Parameters
    ----------
    X : list of np.ndarrays
        List of NxM matrix containing real-world x-coordinates
    Y : list of np.ndarrays
        List of NxM matrix containing real-world y-coordinates
    rotation : float, optional
        Rotation angle in degrees
    translation : list or tuple, optional
        2-tuple or list with x and y translation distances
    max_distance : float, optional
        Maximum distance from origin to be included in the plot.
        Larger numbers are considered to be beyond the horizon.
    ax : matplotlib.axes.AxesSubplot, optional
        Axis object used for plotting
    figsize : tuple, optional
        2-tuple or list containing figure dimensions
    cmap : matplotlib.colors.Colormap, optional
        Colormap to determine colors for individual patches
    alpha : float, optional
        Alpha value for patches

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing axis object
    matplotlib.axes.AxesSubplot
        Axis object containing plot
    '''

    # create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xl = [0,0]
    yl = [0,0]

    # loop over images
    patches = []
    for x, y in zip(X, Y):

        # find horizon based on max_distance
        o = find_horizon_offset(x, y, max_distance=max_distance)

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
    ax.add_collection(p)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_aspect('equal')

    return fig, ax


def rotate_translate(x, y, rotation=None, translation=None):
    '''Rotate and/or translate coordinate system

    Parameters
    ----------
    x : np.ndarray
        NxM matrix containing x-coordinates
    y : np.ndarray
        NxM matrix containing y-coordinates
    rotation : float, optional
        Rotation angle in degrees
    translation : list or tuple, optional
        2-tuple or list with x and y translation distances

    Returns
    -------
    np.ndarrays
        NxM matrix containing rotated/translated x-coordinates
    np.ndarrays
        NxM matrix containing rotated/translated y-coordinates
    '''
    
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


def find_horizon_offset(x, y, max_distance=1e4):
    '''Find minimum number of pixels to crop to guarantee all pixels are within specified distance

    Parameters
    ----------
    x : np.ndarray
        NxM matrix containing real-world x-coordinates
    y : np.ndarray
        NxM matrix containing real-world y-coordinates
    max_distance : float, optional
        Maximum distance from origin to be included in the plot.
        Larger numbers are considered to be beyond the horizon.

    Returns
    -------
    float
        Minimum crop distance in pixels (from the top of the image)
    '''

    offset = 0
    if max_distance is not None:
        try:
            th = (np.abs(x)>max_distance)|(np.abs(y)>max_distance)
            offset = np.max(np.where(np.any(th, axis=1))) + 1
        except:
            pass

    return offset


def _construct_rgba_vector(img, n_alpha=0):
    '''Construct RGBA vector to be used to color faces of pcolormesh

    Parameters
    ----------
    img : np.ndarray
        NxMx3 RGB image matrix
    n_alpha : int
        Number of border pixels to use to increase alpha

    Returns
    -------
    np.ndarray
        (N*M)x4 RGBA image vector
    '''

    alpha = np.ones(img.shape[:2])    
    
    if n_alpha > 0:
        for i, a in enumerate(np.linspace(0, 1, n_alpha)):
            alpha[:,[i,-2-i]] = a
        
    rgb = img[:,:-1,:].reshape((-1,3)) # we have 1 less faces than grid cells
    rgba = np.concatenate((rgb, alpha[:,:-1].reshape((-1, 1))), axis=1)

    if np.any(img > 1):
        rgba[:,:3] /= 255.0
    
    return rgba
