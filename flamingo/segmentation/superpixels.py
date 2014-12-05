
import matplotlib.cm as cm
import skimage.segmentation
from skimage.util import img_as_float
import numpy as np
import inspect
import pandas
import cv2

import postprocess

from argus2 import plot

def get_segmentation(img, method='slic', method_params={},
                     extract_contours=False, remove_disjoint=True):

    # first segmentation step
    segments = get_superpixel(img,
                              method=method, method_params=method_params,
                              remove_disjoint=remove_disjoint)
    
    # extract contours
    if extract_contours:
        contours = get_contours(segments)
    else:
        contours = None

    return segments, contours


def get_superpixel(img, method='slic', method_params={}, remove_disjoint=True):
    'Run segmentation algorithm'
    
    #img = img_as_float(img)
    
    if method.lower() == 'slic':
        method_params = {x:method_params[x]
                         for x in ['n_segments',
                                   'ratio',
                                   'compactness',
                                   'convert2lab',
                                   'sigma'] if method_params.has_key(x)}
        if remove_disjoint and __supports_connectivity():
            img_superpix = skimage.segmentation.slic(img,
                                                     enforce_connectivity=True,
                                                     **method_params)
        else:
            img_superpix = skimage.segmentation.slic(img, **method_params)

            if remove_disjoint:
                img_superpix = postprocess.remove_disjoint(img_superpix)

    elif method.lower() == 'quickshift':
        method_params       = {x:method_params[x]
                               for x in ['ratio', 'convert2lab']
                               if method_params.has_key(x)}
        img_superpix = skimage.segmentation.quickshift(img, **method_params)
    elif method.lower() == 'felzenszwalb':
        img_superpix = skimage.segmentation.felzenszwalb(img)
    elif method.lower() == 'random_walker':
        img_superpix = skimage.segmentation.random_walker(img)
    else:
        raise ValueError('Unknown superpixel method [%s]' % method)

    return img_superpix


def shuffle_pixels(img):
    'Shuffle classification identifiers'

    mn = img.min()
    mx = img.max()+1

    x = np.arange(mn,mx)
    np.random.shuffle(x)

    img_shuffle = np.zeros(img.shape) * np.nan

    for i, value in enumerate(x):
        img_shuffle[img==i] = value

    return img_shuffle


def average_colors(img, img_superpix):
    'Apply average color of original pixels within superpixel'

    nd = img.shape[2]

    cat1 = [('cat', img_superpix.flatten())]                # superpixel category
    cat2 = [(i, img[:,:,i].flatten()) for i in range(nd)]   # colorspace dimensions

    cat  = dict(cat1 + cat2)

    df         = pandas.DataFrame(cat)
    df_grouped = df.groupby('cat', sort=True)
    df_meaned  = df_grouped.aggregate(np.mean)
    df_meaned  = np.array(df_meaned)

    img_avg    = np.zeros((np.prod(img.shape[:-1]), nd))

    for j in range(len(df_meaned)):
        img_avg[cat['cat'] == j,:] = df_meaned[j,:]

    img_avg    = img_avg.reshape(img.shape)

    return img_avg

  
def plot(img, img_superpix, mark_boundaries=True, shuffle=False, average=False, slice=1):

    if mark_boundaries:
        boundaries = skimage.segmentation.find_boundaries(img_superpix)

    # shuffle pixels  
    if shuffle and not average:
        img_superpix = shuffle_pixels(img_superpix)

    # average colors per superpixel    
    if average:
        img_superpix = average_colors(img, img_superpix)

    # mark boundaries of superpixels        
    if mark_boundaries:
        img_superpix = img
        if len(img_superpix.shape) > 2:
            for i in range(img_superpix.shape[2]):
                img_channel             = img_superpix[:,:,i]
                img_channel[boundaries] = 0
                img_superpix[:,:,i]       = img_channel
        else:
            img_superpix[boundaries] = 0
    
    # render superpixel image
    img_superpix = plot.plot_image(img_superpix, slice=slice)
    
    return img_superpix

    
def mark(img_superpix, cat=1):

    nx, ny = img_superpix.shape

    img = np.ones((nx,ny,1))
    img[img_superpix==cat] = 0
    
    #boundaries = skimage.segmentation.find_boundaries(img)
    
    img_alpha = 1 * np.abs(1-img)
    #img_alpha[boundaries] = 1
    
    img_ones = np.ones((nx,ny,1))
    
    img = np.concatenate((img_ones,img,img,img_alpha),axis=2);
    
    return plot.plot_image(img, transparent=True)


def get_superpixel_grid(segments, img_shape):
    '''Return shape of superpixels grid'''
    
    K = segments.max()
    height, width = img_shape
    superpixelsize = width * height / float(K);
    step = np.sqrt(superpixelsize)
    nx = int(round(width / step))
    ny = int(round(height / step))

#    assert(np.max(segments) == nx*ny - 1)
    
    return (ny,nx)


def get_contours(segments):
    '''Return contours of superpixels'''

    contours = []
    for i in range(np.max(segments)+1):
        c, h = cv2.findContours((segments==i).astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
        contours.append([i.tolist() for i in c])

    return contours


def check_segmentation(segments, nx, ny):
    # check if total number of segments is ok
    if not np.max(segments) + 1 == nx * ny:
        return False

    # check if all segments are present
    if not np.all([i in np.asarray(segments) for i in range(np.max(segments)+1)]):
        return False
        
    return True


def __supports_connectivity():
    return 'enforce_connectivity' in inspect.getargspec(skimage.segmentation.slic).args
