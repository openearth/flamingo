#!/usr/bin/env python

import numpy as np
import matplotlib.path
import logging
from flamingo import filesys

# initialize log
logger = logging.getLogger(__name__)


def paint_roi(img,roi): 
    '''Give uniform color to pixels outside the ROI

    To assist the segmentation algorithm, pixels outside the ROI
    are given a uniform, odd color. This enhances the probability
    of superpixel boundaries coinciding with the ROI boundary and 
    therefore helps avoiding splitting of superpixels by the ROI.

    Parameters
    ----------
    img : np.ndarray
        NxMx3 image matrix
    roi : np.ndarray
        Kx2 matrix containing ROI vertices in UV coordinates

    Returns
    -------
    np.ndarray
        NxMx3 image matrix with colored pixels outside ROI
    '''

    logger.info('Adjusting pixel intensity outside ROI...')
    
    iroi = get_roi_mask(roi,img.shape)
    
    cols = [251,110,82]

    for i in range(3):
        img[:,:,i][iroi] = cols[i]

    return img


def segments_roi(seg,roi): 
    '''Adjust segmentation for ROI

    All pixels outside the ROI are assigned to a single superpixel.
    Segment numbers are recalculated to maintain sequential numbering.

    Parameters
    ----------
    seg : np.ndarray
        NxM matrix containing segment numbers
    roi : np.ndarray
        Kx2 matrix containing ROI vertices in UV coordinates

    Returns
    -------
    np.ndarray
        NxM matrix containing adjusted segment numbers
    '''
    
    iroi = get_roi_mask(roi,seg.shape)
    
    nseg = seg.max() + 1
    seg[iroi] = nseg

    segnums = np.unique(seg)
    nseg = len(segnums)
    for i in range(nseg):
        seg[seg == segnums[i]] = i

    return seg


def get_roi_mask(roi,imshape):
    '''Get mask for image based on ROI

    Parameters
    ----------
    roi : np.ndarray
        Kx2 matrix containing ROI vertices in UV coordinates
    imshape : tuple
        tuple containing image dimensions

    Returns
    -------
    np.ndarray
        NxM mask matrix
    '''

    proi = matplotlib.path.Path(roi)
    ny,nx = imshape[:2]
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    imcoor = np.vstack((x,y)).T

    iroi = proi.contains_points(imcoor)
    iroi = np.invert(iroi.reshape((ny,nx)))
    
    return iroi
