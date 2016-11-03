import matplotlib.cm as cm
import skimage.segmentation
from skimage.util import img_as_float
import numpy as np
import inspect
import pandas
import cv2

import postprocess


def get_segmentation(img, method='slic', method_params={},
                     extract_contours=False, remove_disjoint=True):
    '''Return segmentation of image

    Parameters
    ----------
    img : np.ndarray
        NxM or NxMx3 array with greyscale or colored image information
        respectively
    method : str, optional
        Segmentation method to use, supported by scikit-image toolbox
    method_params : dict, optional
        Extra parameters supplied to segmentation method
    extract_contours : bool, optional
        Also extract contours of segments
    remove_disjoint : bool, optional
        Ensure that the output contains connected segments only
        **and** that the superpixels form a more or less regular grid.
        In case the segmentation method does not provide both
        constraints, the constraint is ensured in a postprocessing
        step.

    Returns
    -------
    np.ndarray
        NxM matrix with segment numbering

    Examples
    --------
    >>> img = argus2.rest.get_image(station='kijkduin')[0]
    >>> segments = get_segmentation(img)
    '''

    # first segmentation step
    segments = _get_segmentation(img,
                                 method=method,
                                 method_params=method_params,
                                 remove_disjoint=remove_disjoint)
    
    # extract contours
    if extract_contours:
        contours = get_contours(segments)
    else:
        contours = None

    return segments, contours


def _get_segmentation(img, method='slic', method_params={}, remove_disjoint=True):
    '''Return segmentation of image

    Parameters
    ----------
    img : np.ndarray
        NxM or NxMx3 array with greyscale or colored image information
        respectively
    method: str, optional
        Segmentation method to use, supported by scikit-image toolbox
    method_params: dict, optional
        Extra parameters supplied to segmentation method
    remove_disjoint: bool, optional
        Ensure that the output contains connected segments only
        **and** that the superpixels form a more or less regular grid.
        In case the segmentation method does not provide both
        constraints, the constraint is ensured in a postprocessing
        step.

    Note
    ----
    This is a helper function to the
    :func:flamingo.segmentation.segmentation.get_segmentation():
    function

    Returns
    -------
    np.ndarray
        NxM matrix with segment numbering
    '''
    
    #img = img_as_float(img)
    
    if method.lower() == 'slic':
        method_params = {x:method_params[x]
                         for x in ['n_segments',
                                   'ratio',
                                   'compactness',
                                   'convert2lab',
                                   'sigma'] if method_params.has_key(x)}

        if remove_disjoint and __supports_connectivity():
            if method_params.has_key('n_segments'):
                n_segments = method_params['n_segments']
            else:
                n_segments = 100.
            nx, ny = get_superpixel_grid(n_segments, img.shape[:2])
            img_superpix = skimage.segmentation.slic(img,
                                                     enforce_connectivity=True,
                                                     min_size_factor=.2,
                                                     **method_params)
            img_superpix = postprocess.regularize(img_superpix, nx, ny)
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
    '''Shuffle class identifiers

    Parameters
    ----------
    img : np.ndarray
        NxM matrix with segment numbering

    Returns
    -------
    np.ndarray
        NxM matrix with shuffled segment numbering

    Examples
    --------
    >>> seg = get_segmentation(img)
    >>> fig, axs = plt.subplots(1, 2)
    >>> axs[0].imshow(seg)
    >>> axs[1].imshow(shuffle_pixels(seg))
    '''

    mn = img.min()
    mx = img.max()+1

    x = np.arange(mn,mx)
    np.random.shuffle(x)

    img_shuffle = np.zeros(img.shape) * np.nan

    for i, value in enumerate(x):
        img_shuffle[img==i] = value

    return img_shuffle


def average_colors(img, segments):
    '''Average colors per superpixels

    Returns an image where each pixel has the average color of the
    superpixel that it belongs to.
    
    Parameters
    ----------
    img : np.ndarray
        NxM or NxMx3 array with greyscale or colored image information
        respectively
    segments : np.ndarray
        NxM matrix with segment numbering

    Returns
    -------
    np.ndarray
        NxM or NxMx3 matrix with averaged image
    '''

    nd = img.shape[2]

    cat1 = [('cat', segments.flatten())]                # superpixel category
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


def get_superpixel_grid(segments, img_shape):
    '''Return shape of superpixels grid

    Parameters
    ----------
    segments : np.ndarray
        NxM matrix with segment numbering
    img_shape : 2-tuple or list
        Dimensions of image

    Returns
    -------
    2-tuple
        tuple containing M and N dimension of regular superpixel grid
    '''
    
    if type(segments) is int:
        K = segments
    else:
        K = segments.max()

    height, width = img_shape
    superpixelsize = width * height / float(K);
    step = np.sqrt(superpixelsize)
    nx = int(np.round(width / step))
    ny = int(np.round(height / step))

#    assert(np.max(segments) == nx*ny - 1)
    
    return (ny,nx)

  
def get_contours(segments):
    '''Return contours of superpixels

    Parameters
    ----------
    segments : np.ndarray
        NxM matrix with segment numbering

    Returns
    -------
    list
        list of lists for each segment in *segments*. Each segment
        list contains one or more contours. Each contour is defined by
        a list of 2-tuples with an x and y coordinate.

    Examples
    --------
    >>> contours = get_contours(segments)
    >>> plot(contours[0][0][0][0], contours[0][0][0][1]) # plot first contour of first segment
    '''

    contours = []
    for i in range(np.max(segments)+1):
        c, h = cv2.findContours((segments==i).astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
        contours.append([i.tolist() for i in c])

    return contours


def check_segmentation(segments, nx, ny):
    '''Checks if segmentation data is complete

    Checks if the segmentation data indeed contains nx*ny segments and
    if the set of segment numbers is continuous.

    Parameters
    ----------
    segments : np.ndarray
        NxM matrix with segment numbering
    nx, ny : int
        Size of supposed segmentation grid

    Returns
    -------
    bool
        Returns true if segmentation is valid and false otherwise
    '''

    # check if total number of segments is ok
    if not np.max(segments) + 1 == nx * ny:
        return False

    # check if all segments are present
    if not np.all([i in np.asarray(segments) for i in range(np.max(segments)+1)]):
        return False
        
    return True


def __supports_connectivity():
    '''Checks if slic algorithm supports connectivity constraint'''
    
    return 'enforce_connectivity' in inspect.getargspec(skimage.segmentation.slic).args
