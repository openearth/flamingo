import re
import pandas
import inspect
import numpy as np
import skimage.measure
import skimage.feature

_PROP_PIXEL = {'area',
               'coords',
               'convex_area',
               'centroid',
               'equivalent_diameter',
               'perimeter',
               'moments_central',
               'solidity',
               'euler_number',
               'extent',
               'moments_normalized',
               'eccentricity',
               'convex_image',
               'label',
               'filled_image',
               'orientation',
               'major_axis_length',
               'moments',
               'image',
               'filled_area',
               'bbox',
               'minor_axis_length',
               'moments_hu'}

_PROP_INTENSITY = {'label',
                   'max_intensity',
                   'mean_intensity',
                   'min_intensity',
                   'weighted_moments_central',
                   'weighted_centroid',
                   'weighted_moments_normalized',
                   'weighted_moments_hu',
                   'weighted_moments'}

_TEMP_FEATURES = {'image',
                  'image_masked',
                  'convex_image',
                  'filled_image',
                  'coords'}

def list_blocks():
    '''List all block extraction functions in module'''
    return {k:v for k,v in globals().iteritems() if k.startswith('extract_blocks_')}

def extract_blocks(data, segments, colorspace='rgb', blocks=None):
    '''Extract all blocks in right order'''

    features = []
    features0 = []
    features_in_block = {}

    if not blocks:
        blocks = list_blocks()

    done = []
    while True:
        n = len(done)
        for block, fcn in blocks.iteritems():
            if not block in done:
                try:
                    features = fcn(data,
                                   segments,
                                   features=features,
                                   colorspace=colorspace)
                    features_in_block[block] = [x for x in features.columns if x not in features0]
                    features0 = features.columns

                    done.append(block)
                except ValueError as e:
                    continue

        if len(done) == len(blocks):
            break

        if len(done) == n:
            # nothing done, raise error
            raise StopIteration('Cannot determine order of blocks')

    return features, features_in_block

def __extract_blocks(grayscale=True, channel=False, derived_from=()):
    '''Feature extraction decorator.'''
    def wrapper(f):
        def extract(data, segments, features=None, colorspace='rgb', **kwargs):
            features0 = features

            # initialize feature list
            if type(features) is not pandas.DataFrame:
                if (features is None) or (features == []):
                    features = []
                    if len(derived_from) > 0:
                        raise ValueError('Derived features, but no feature set provided')
            else:
                features = [features]

            # check if necessary features are present for derived features
            for feature in derived_from:
                if feature not in features0: #.keys():
                    raise ValueError('Cannot compute derived feature. Feature "%s" is missing.' % feature)

            # compute feature for grayscale channel
            if grayscale:
                if colorspace.lower() == 'rgb':
                    gs = data.mean(-1)
                elif colorspace.lower() == 'hsv':
                    gs = data[...,-1]
                else:
                    raise ValueError('Unsupported colorspace [%s]' % colorspace)

                if type(features0) is pandas.DataFrame:
                    f0 = features0.filter(derived_from)
                else:
                    f0 = None

                features.append(f(gs,
                                  segments,
                                  features=f0))

            # compute feature for each channel individually
            if channel:

                df_sum = None
                for i in range(data.shape[-1]):
                    if type(features0) is pandas.DataFrame:
                        f0 = features0.filter(['%s.%d' % (x,i) if '%s.%d' % (x,i) in features0.keys() else x for x in derived_from]) \
                                      .rename(columns=lambda x: re.sub('\.\d+$','',x))
                    else:
                        f0 = None

                    df = f(data[...,i],
                           segments,
                           features=f0)

                    # incremental sum of all channel values

                    if df_sum is None:
                        df_sum = df.copy()
                    else:
                        df_sum += df

                    df.columns = ['%s.%d' % (x,i) for x in df.columns] # add channel index to column name

                    features.append(df)

                df_sum.columns = ['%s.sum' % x for x in df_sum.columns]

                features.append(df_sum)

            # merge features for different channels and with previous feature list
            return merge_blocks(features)

        return extract

    return wrapper

def _empty_block_frame(block, n=1, columns=[], data=None, fill_value=np.empty((0))):
    '''Create empty feature dataframe'''

    if data is None:
        data = [{k:fill_value for k in columns} for i in range(n)]

    #ix = pandas.MultiIndex.from_arrays([[block] * len(columns), columns], names=['block','feature'])
    df = pandas.DataFrame(data, columns=columns, index=np.arange(n)+1)
    #df = df.set_index('label')

    return df

@__extract_blocks(grayscale=True, channel=False)
def extract_blocks_pixel(data, segments, **kwargs):

    regionprops = skimage.measure.regionprops(np.asarray(segments)+1, intensity_image=data)
    features = [{key:feature[key] for key in _PROP_PIXEL} for feature in regionprops] # if feature._slice is not None]

    df = _empty_block_frame('pixel', np.max(segments)+1, _PROP_PIXEL, data=features)

    return df

@__extract_blocks(grayscale=False, channel=True)
def extract_blocks_intensity(data, segments, **kwargs):

    regionprops = skimage.measure.regionprops(np.asarray(segments)+1, intensity_image=data)
    features = [{key:feature[key] for key in _PROP_INTENSITY} for feature in regionprops] # if feature._slice is not None]

    df = _empty_block_frame('intensity', np.max(segments)+1, _PROP_INTENSITY, data=features)

    return df

@__extract_blocks(grayscale=True, channel=False, derived_from=['area','centroid','perimeter','convex_area'])
def extract_blocks_shape(data, segments, features=[], **kwargs):

    df = _empty_block_frame('shape', np.max(segments)+1, ['size','position.n','position.m','shape','holeyness'])

    for i, feature in features.iterrows():

        # size is represented by the portion of the image covered by the region
        df.ix[i,'size'] = np.prod(segments.shape)/feature['area']

        # position is represented using the coordinates of the region center of mass normalized by the image dimensions
        df.ix[i,'position.n'] = feature['centroid'][0]/segments.shape[0]
        df.ix[i,'position.m'] = feature['centroid'][1]/segments.shape[1]

        # shape is represented by the ratio of the area to the perimeter squared
        # seems to be the same as solidity
        df.ix[i,'shape'] = feature['area']/(feature['perimeter']**2)

        # holeyness is defined as the convex area divided by the area
        df.ix[i,'holeyness'] = feature['convex_area']/feature['area']

    return df

@__extract_blocks(grayscale=True, channel=True, derived_from=['bbox','image'])
def extract_blocks_mask(data, segments, features=[], **kwargs):

    df = _empty_block_frame('mask', np.max(segments)+1, ['image_masked'])

    for i, feature in features.iterrows():

        # color
        minn, minm, maxn, maxm = feature['bbox']

        # select the pixels that are not in the image
        mask = np.logical_not(features.ix[i,'image'])

        # select the image pixels and apply the mask
        df.ix[i,'image_masked'] = np.ma.masked_array(data[minn:maxn, minm:maxm], mask=mask)

    return df

@__extract_blocks(grayscale=True, channel=True, derived_from=['image_masked'])
def extract_blocks_grey(data,
                           segments,
                           features=[],
                           distances=[5,7,11],
                           angles=np.linspace(0,1*np.pi, num=6, endpoint=False),
                           props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'], **kwargs):

    df = _empty_block_frame('grey', np.max(segments)+1, ['grey_%s' % x for x in props])

    for i, feature in features.iterrows():
        greyprops = skimage.feature.greycomatrix(features.ix[i,'image_masked'],
                                                 distances=distances,
                                                 angles=angles)
        for prop in props:
            df.ix[i,'grey_%s' % prop] = skimage.feature.greycoprops(greyprops, prop)

    return df

@__extract_blocks(grayscale=True, channel=True, derived_from=['image_masked','image_masked.sum'])
def extract_blocks_intensitystatistics(data, segments, features=[], histogram_bins=5, **kwargs):

    df = _empty_block_frame('intensitystatistics', np.max(segments)+1, ['variance_intensity', 'mean_relative_intensity', 'variance_relative_intensity', 'histogram'])

    for i, feature in features.iterrows():

        df.ix[i,'variance_intensity'] = features.ix[i,'image_masked'].var()
        df.ix[i,'mean_relative_intensity'] = (features.ix[i,'image_masked'].astype('float')/features.ix[i,'image_masked.sum']).mean()
        df.ix[i,'variance_relative_intensity'] = (features.ix[i,'image_masked'].astype('float')/features.ix[i,'image_masked.sum']).var()

        # intensity histogram
        counts, bins = np.histogram(features.ix[i,'image_masked'], bins=np.linspace(0, 255, endpoint=True, num=histogram_bins+1))
        df.ix[i,'histogram'] = counts

    return df

def merge_blocks(features):
    if len(features) > 1:
        df = features[0]
        for i in range(1,len(features)):
            df = pandas.merge(df,
                              features[i],
                              how='outer',
                              left_index=True,
                              right_index=True)
    elif len(features) > 0:
        df = features[0]
    else:
        df = features

    return df
