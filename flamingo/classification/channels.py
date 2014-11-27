from itertools import product

import numpy as np
import skimage.color

_FREQS = np.arange(0.05, 0.30, 0.1)
_SIGMAS = [3, 5, 7]
_SIGMA1 = range(1, 21, 7)
_THETAS = np.arange(0, np.pi, 0.25*np.pi)

def add_channels(img, colorspace, channelstats=None):
    """add channels to an image.
    Channels are:
    - 0: greyscale
    - 1,2,3: colorspace
    - extra channels
    """
    if colorspace.lower() == 'rgb':
        gray = img.mean(-1)
    elif colorspace.lower() == 'hsv':
        gray = img[..., -1]
    else:
        raise ValueError(
            'Unsupported colorspace [%s]' % colorspace)

    # create extra channels
    channels = []

    # Create a set of gabor kernels and apply them
    for i,(frequency, sigma) in enumerate(product(_FREQS,_SIGMAS)):
        response = []
        for theta in _THETAS:
            real, imag = skimage.filter.gabor_filter(
                gray,
                frequency=frequency,
                sigma_x=sigma,
                sigma_y=sigma,
                theta=theta
                )
            response.append(np.sqrt(real ** 2 + imag ** 2))
        
        response = np.array(response).max(axis=0)

        if channelstats:
            response = _normalize_channel(response, channelstats[i])
        channels.append(response)

    # Create relative greyishness
    for i,sigma1 in enumerate(_SIGMA1):
        # blur
        response1 = skimage.filter.gaussian_filter(gray, sigma1)
        # blur on a higher level
        response2 = skimage.filter.gaussian_filter(gray, sigma1 + 1)
        # create relative darkness
        response = response2 - response1
        
        if channelstats:
            response = _normalize_channel(response,channelstats[i + len(_FREQS)*len(_SIGMAS)])
        channels.append(response)
        
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

def get_channel_bounds():
    channelstats = [{'min': 0., 'max': []} for i in range(len(_FREQS)*len(_SIGMAS))]
    for i,(frequency,sigma) in enumerate(product(_FREQS,_SIGMAS)):
        k = skimage.filter.gabor_kernel(frequency=frequency,theta=0, sigma_x=sigma, sigma_y=sigma)
        resp = np.zeros(k.shape) + 255.
        resp[np.real(k) < 0] = 0.
        resp_real = np.sum(resp*np.real(k))
        resp_imag = np.sum(resp*np.imag(k))
        resp = np.sqrt(resp_real**2 + resp_imag**2)
        channelstats[i]['max'] = resp.max()
    
    channelstats.append([{'min':-255.,'max':255.},{'min':-255.,'max':255.},{'min':-255.,'max':255.}])

    return channelstats

def get_number_channels():
    return len(_FREQS)*len(_SIGMAS) + len(_SIGMA1)

def _normalize_channel(channel, channelstats):
    # Scale channel to uint8 based on maximum possible filter response
    return np.uint8((channel-channelstats['min'])/(channelstats['max']-channelstats['min'])*255)
