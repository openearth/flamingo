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
    assert colorspace == 'rgb', 'channels only supported for rgb for now'
    gray = skimage.color.rgb2gray(img)
    # add the greyscale after the colors

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
        channels.append(response)

    # Create relative greyishness
    for sigma1 in range(1, 21, 7):
        # blur
        response1 = skimage.filter.gaussian_filter(gray, sigma1)
        # blur on a higher level
        response2 = skimage.filter.gaussian_filter(gray, sigma1 + 1)
        # create relative darkness
        response = response2 - response1
        channels.append(response)

    allchannels = np.concatenate([
        # color
        img,
        # greyscale
        gray[..., np.newaxis]
    ] + [
        # other channels
        channel[..., np.newaxis]
        for channel
        in channels
    ], -1)
    return allchannels


