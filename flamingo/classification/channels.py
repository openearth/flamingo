#!/usr/bin/env python

__author__ = "Bas Hoonhout"
__copyright__ = "Copyright 2014, The NEMO Project"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Bas Hoonhout"
__email__ = "bas.hoonhout@deltares.nl"
__status__ = "production"

from itertools import product
import numpy as np
import skimage.color


_FREQS = np.arange(0.05, 0.30, 0.1)
_SIGMAS = range(1, 21, 7)
_SIGMAS2 = [3, 5, 7]
_THETAS = np.arange(0, np.pi, 0.25*np.pi)


def add_channels(img, colorspace='rgb',
                 methods=['gabor', 'gaussian', 'sobel'],
                 methods_params=None):
    '''Add artificial channels to an image

    Parameters
    ----------
    img : np.ndarray
        NxMx3 array with colored image data
    colorspace : str, optional
        String indicating colorspace of *img* (rgb/hsv/etc.)
    methods : list, optional
        List of strings indicating channels to be added
    methods_params : dict, optional
        Dictionairy with named options for channel functions
        
    Notes
    -----
    Currently implemented channels are:
    * gabor, with options *frequencies* and *thetas*
    * gaussian, with option *sigmas*
    * sobel, without any options

    Returns
    -------
    np.ndarray
        NxMx(3+P) array with image data with extra channels where P
        is the number of channels added
    '''

    if colorspace.lower() == 'rgb':
        gray = img.mean(-1)
    elif colorspace.lower() == 'hsv':
        gray = img[..., -1]
    else:
        raise ValueError(
            'Unsupported colorspace [%s]' % colorspace)

    # default settings
    methods_params = _update_params(methods_params)

    # create extra channels
    channels = []

    # Create a set of gabor kernels and apply them
    if 'gabor' in methods:
        for i, (frequency, sigma) in enumerate(product(methods_params['frequencies'],
                                                       methods_params['sigmas2'])):
            response = []
            for theta in methods_params['thetas']:
                real, imag = skimage.filter.gabor_filter(
                    gray,
                    frequency=frequency,
                    sigma_x=sigma,
                    sigma_y=sigma,
                    theta=theta)
                response.append(np.sqrt(real ** 2 + imag ** 2))
        
            response = np.array(response).max(axis=0)

            channels.append(response)

    # Create relative greyishness
    if 'gaussian' in methods:
        for i, sigma1 in enumerate(methods_params['sigmas']):
            # blur
            response1 = skimage.filter.gaussian_filter(gray, sigma1)
            # blur on a higher level
            response2 = skimage.filter.gaussian_filter(gray, sigma1+1)
            # create relative darkness
            response = response2 - response1
        
            channels.append(response)

    # Create sobel filter channel
    if 'sobel' in methods:
        channels.append(skimage.filter.sobel(gray))
        
    # Combine all channels
    allchannels = np.concatenate([
        # greyscale
        np.uint8(gray)[..., np.newaxis],
        # color
        img
    ] + [
        # other channels
        channel[..., np.newaxis]
        for channel
        in channels
    ], -1)

    return allchannels


def get_channel_bounds(methods=['gabor', 'gaussian', 'sobel'],
                       methods_params=None):
    '''Get theoretical bounds of channel values

    Parameters
    ----------
    methods : list, optional
        List of strings indicating channels to be added
    methods_params : dict, optional
        Dictionairy with named options for channel functions
        
    Returns
    -------
    list
        List of dicts with keys *min* and *max* indicating the
        theoretical boundaries of the channel values
    '''

    # default settings
    methods_params = _update_params(methods_params)

    # initialize all bounds to 0 and 255
    stats = [{'max': 0., 'min': 255.}
             for i in range(get_number_channels(methods, methods_params))]

    # update gabor channel bounds, if requested
    if 'gabor' in methods:
        for i, frequency in enumerate(methods_params['frequencies']):
            k = skimage.filter.gabor_kernel(frequency=frequency, theta=0)
            resp = np.zeros(k.shape) + 255.
            resp[np.real(k) < 0] = 0.
            resp_real = np.sum(resp*np.real(k))
            resp_imag = np.sum(resp*np.imag(k))
            resp = np.sqrt(resp_real**2 + resp_imag**2)
            stats[i]['max'] = resp.max()
    
    return stats


def get_number_channels(methods=['gabor', 'gaussian', 'sobel'],
                        methods_params=None):
    '''Get number of artificial channels

    Parameters
    ----------
    methods : list, optional
        List of strings indicating channels to be added
    methods_params : dict, optional
        Dictionairy with named options for channel functions
        
    Returns
    -------
    int
        Number of channels added when using the specified settings
    '''

    methods_params = _update_params(methods_params)

    n = 0
    if 'gabor' in methods:
        n += len(methods_params['frequencies']) * \
            len(methods_params['sigmas2'])

    if 'gaussian' in methods:
        n += len(methods_params['sigmas'])

    if 'sobel' in methods:
        n += 1

    return n


def normalize_channel(channel, channelstats):
    '''Scale channel to uint8 based on maximum possible filter response

    Parameters
    ----------
    channel : np.ndarray
        Array with channel values
    channelstats : dict
        Dictionary with fields *min* and *max* indicating the channel
        value bounds of the dataset and used for normalization

    Returns
    -------
    np.ndarray
        Array with normalized channel values
    '''

    return np.uint8((channel-channelstats['min']) / 
                    (channelstats['max']-channelstats['min'])*255)


def _update_params(params):
    '''Set channel options to module defaults'''

    methods_params = {'frequencies':_FREQS,
                      'sigmas':_SIGMAS,
                      'sigmas2':_SIGMAS2,
                      'thetas':_THETAS}

    methods_params.update(params)

    return methods_params
