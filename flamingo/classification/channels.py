from itertools import product

import numpy as np
import skimage.color


def add_channels(img, colorspace):
    """add channels to an image.
    Channels are:
    - 0:2/3 colorspace
    - greyscale
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
    frequencies = np.arange(0.05, 0.30, 0.1)
    sigmas = [3, 5, 7]
    thetas = np.arange(0, np.pi, 0.25*np.pi)
    for frequency, sigma, theta in product(frequencies, sigmas, thetas):
        real, imag = skimage.filter.gabor_filter(
            gray,
            frequency=frequency,
            sigma_x=sigma,
            sigma_y=sigma,
            theta=theta
        )
        response = np.sqrt(real ** 2 + imag ** 2)

        response = _normalize_channel(response,frequency,sigma,theta)
        channels.append(response)

    # Create relative greyishness
    for sigma1 in range(1, 21, 7):
        # blur
        response1 = skimage.filter.gaussian_filter(gray, sigma1)
        # blur on a higher level
        response2 = skimage.filter.gaussian_filter(gray, sigma1 + 1)
        # create relative darkness
        response = response2 - response1

        response = _normalize_channel(response)
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

def _normalize_channel(channel, frequency=None, sigma=None, theta=None):
    # Determine maximum possible filter response
    if any([frequency,sigma,theta]):
        k = skimage.filter.gabor_kernel(frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
        resp = np.zeros(k.shape) + 255.
        resp[np.real(k) < 0] = 0.
        resp_real = sum(resp*np.real(k))
        resp_imag = sum(resp*np.imag(k))
        resp = np.sqrt(resp_real**2 + resp_imag**2)
        maxval = resp.max()
        minval = 0.
    else:
        maxval = 255.
        minval = -255.
    
    # Scale channel between 0 and 255 based on maximum possible filter response
    return np.uint8((channel-minval)/(maxval-minval)*255)
