#!/usr/bin/env python

__author__ = "Bas Hoonhout"
__copyright__ = "Copyright 2014, The NEMO Project"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Bas Hoonhout"
__email__ = "bas.hoonhout@deltares.nl"
__status__ = "production"

import numpy as np
import cStringIO
import matplotlib.pyplot as plt
import skimage.segmentation

import superpixels


def plot(img, segments, mark_boundaries=True, shuffle=False,
         average=False, slice=1, raw=False):
    '''Plot segmentation result

    Parameters
    ----------
    img : np.ndarray
        NxM or NxMx3 array with greyscale or colored image information
        respectively
    segments : np.ndarray
        NxM matrix with segment numbering
    mark_boundaries : bool, optional
        Draw boundaries in image
    shuffle : bool, optional
        Shuffle segment numbering for more scattered coloring (ignored
        when *average* is used)
    average : bool, optional
        Average colors per segment
    slice : int, optional
        Use slice to reduce the image size
    raw : bool, optional
        Return raw binary output

    Returns
    -------
    str or 2-tuple
        Binary image data or 2-tuple with matplotlib.figure.Figure and
        matplotlib.axes.AxesSubplot objects
    '''

    # mark boundaries
    if mark_boundaries:
        boundaries = skimage.segmentation.find_boundaries(segments)

    # shuffle pixels  
    if shuffle and not average:
        segments = superpixels.shuffle_pixels(segments)

    # average colors per superpixel    
    if average:
        segments = superpixels.average_colors(img, segments)

    # mark boundaries of superpixels        
    if mark_boundaries:
        segments = img
        if len(segments.shape) > 2:
            for i in range(segments.shape[2]):
                img_channel             = segments[:,:,i]
                img_channel[boundaries] = 0
                segments[:,:,i]         = img_channel
        else:
            segments[boundaries] = 0
    
    # render superpixel image
    segments = plot_image(segments, slice=slice, raw=raw)
    
    return segments


def plot_image(img, cmap='Set2', dpi=96, slice=0,
               transparent=True, raw=False):
    '''Get binary image data

    Parameters
    ----------
    img : np.ndarray
        NxM or NxMx3 array with greyscale or colored image information
        respectively
    cmap : matplotlib.colors.Colormap, optional
        Colormap to determine colors for individual patches
    dpi : int, optional
        Image resolution
    slice : int, optional
        Use slice to reduce the image size
    transparent : bool, optional
        Plot background transparent
    raw : bool, optional
        Return raw binary output

    Returns
    -------
    str or 2-tuple
        Binary image data or 2-tuple with matplotlib.figure.Figure and
        matplotlib.axes.AxesSubplot objects
    '''

    if slice > 0:
        img = img[::slice,::slice] 

    i = float(img.shape[1] + 44)/dpi # correct for "invisible" tick label space
    j = float(img.shape[0] + 17)/dpi

    fig, ax = plt.subplots(figsize=(i,j))

    ax.imshow(img, aspect='normal', cmap=cmap)
    ax.set_axis_off()

    plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    plt.tight_layout(pad=0., h_pad=0., w_pad=0.)

    if raw:
        return get_image_data(fig, dpi=dpi)
    else:
        return fig, ax


def get_image_data(fig, dpi=96, axis_only=True, transparent=True):
    '''Get binary image data

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object containing axis object
    dpi : int, optional
        Image resolution
    axis_only : bool, optional
        Only include contents of axis
    transparent : bool, optional
        Plot background transparent

    Returns
    -------
    str
        Binary image data
    '''

    imgdata = cStringIO.StringIO()
    #imgdata = io.BytesIO()

    if axis_only:
        extent = fig.axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(imgdata, format='png', bbox_inches=extent, pad_inches=0, dpi=dpi, transparent=transparent)

    else:
        fig.savefig(imgdata, format='png', pad_inches=0, dpi=dpi, transparent=transparent)

    imgdata.seek(0)

    return imgdata.read()
