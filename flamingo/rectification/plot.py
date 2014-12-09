#!/usr/bin/env python

'''Module to plot projected images into a real-world coordinate system

This module provides functions to plot projected images onto a
real-world coordinate system. The projection itself is handled by the
accompanying rectification module.

Author: Bas Hoonhout
E-mail: bas.hoonhout@deltares.nl
License: GPL
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections
import matplotlib.patches

__author__ = "Bas Hoonhout"
__copyright__ = "Copyright 2014, The NEMO Project"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Bas Hoonhout"
__email__ = "bas.hoonhout@deltares.nl"
__status__ = "Production"


def plot_rectified(X, Y, imgs,
                   rotation=None, translation=None, max_distance=1e4,
                   ax=None, figsize=(30,20), color=True):
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
    rotation : float
        Rotation angle in degrees
    translation : list or tuple
        2-tuple or list with x and y translation distances
    max_distance : float
        Maximum distance from origin to be included in the plot.
        Larger numbers are considered to be beyond the horizon.
    ax : matplotlib.axes.AxesSubplot
        Axis object used for plotting
    figsize : tuple
        2-tuple or list containing figure dimensions
    color : bool
        Whether color image should be plotted or grayscale

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

    # loop over images
    for x, y, img in zip(X, Y, imgs):

        # find horizon based on maximum distance
        o = find_horizon_offset(x, y, max_distance=max_distance)

        # rotate to world coordinate system
        x, y = rotate_translate(x, y, rotation=rotation, translation=translation)

        # plot
        im = ax.pcolormesh(x[o:,:], y[o:,:], np.mean(img[o:,...], -1), cmap='Greys')

        # add colors
        if color:
            rgba = _construct_rgba_vector(img[o:,...])
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
    rotation : float
        Rotation angle in degrees
    translation : list or tuple
        2-tuple or list with x and y translation distances
    max_distance : float
        Maximum distance from origin to be included in the plot.
        Larger numbers are considered to be beyond the horizon.
    ax : matplotlib.axes.AxesSubplot
        Axis object used for plotting
    figsize : tuple
        2-tuple or list containing figure dimensions
    cmap : matplotlib.colors.Colormap
        Colormap to determine colors for individual patches
    alpha : float
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
    rotation : float
        Rotation angle in degrees
    translation : list or tuple
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
    max_distance : float
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


def _construct_rgba_vector(img):
    '''Construct RGBA vector to be used to color faces of pcolormesh

    Parameters
    ----------
    img : np.ndarray
        NxMx3 RGB image matrix

    Returns
    -------
    np.ndarray
        (N*M)x4 RGBA image vector
    '''

    if np.any(img > 1):
        img /= 255.0

    rgb = img[:,:-1,:].reshape((-1,3)) # we have 1 less faces than grid cells
    rgba = np.concatenate((rgb, np.ones((rgb.shape[0],1))), axis=1)
    
    return rgba
