from itertools import product

import numpy as np
import skimage.color

_FREQS = np.arange(0.05, 0.30, 0.1)
_SIGMAS = range(1, 21, 7)
_THETAS = np.arange(0, np.pi, 0.25*np.pi)

def add_channels(img, colorspace='rgb',
                 methods=['gabor', 'gaussian', 'sobel'],
                 methods_params=None):
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

    # default settings
    if not methods_params:
        methods_params = {'frequencies':_FREQS,
                          'sigmas':_SIGMAS,
                          'thetas':_THETAS}

    # create extra channels
    channels = []

    # Create a set of gabor kernels and apply them
    if 'gabor' in methods:
        for i, frequency in enumerate(methods_params['frequencies']):
            response = []
            for theta in methods_params['thetas']:
                real, imag = skimage.filter.gabor_filter(
                    gray,
                    frequency=frequency,
                    #sigma_x=sigma,
                    #sigma_y=sigma,
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


def get_channel_bounds(gabor=False):
    stats = [{'max': 0., 'min': 255.}
             for i in range(get_number_channels())]

    if gabor:
        for i, frequency in enumerate(_FREQS):
            k = skimage.filter.gabor_kernel(frequency=frequency, theta=0)
            resp = np.zeros(k.shape) + 255.
            resp[np.real(k) < 0] = 0.
            resp_real = np.sum(resp*np.real(k))
            resp_imag = np.sum(resp*np.imag(k))
            resp = np.sqrt(resp_real**2 + resp_imag**2)
            stats[i]['max'] = resp.max()
    
    return stats


def get_number_channels():
    return len(_FREQS) + len(_SIGMAS) + 1


def normalize_channel(channel, channelstats):
    # Scale channel to uint8 based on maximum possible filter response
    return np.uint8((channel-channelstats['min']) / 
                    (channelstats['max']-channelstats['min'])*255)
