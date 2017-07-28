from __future__ import absolute_import

import re
import pandas
import numpy as np
import uuid


_LENGTH_FEATURES = {'equivalent_diameter',
                    'major_axis_length',
                    'minor_axis_length',
                    'perimeter'}

_AREA_FEATURES = {'area',
                  'convex_area',
                  'filled_area'}

_COORDINATE_FEATURES = {'centroid',
                        'weighted_centroid'}

_POP_FEATURES = {'moments',
                 'moments_central',
                 'weighted_moments',
                 'weighted_moments_central',
                 'size',
                 'position.m',
                 'position.n'}

_MAX_SIZE = 50


def linearize(features):
    '''convert all items in each matrix feature into individual features'''

    features_lin = [{} for i in range(len(features))]
    for name, values in features.iteritems():
        for i, value in values.iteritems():
            arr = np.asarray(value)
            if np.prod(arr.shape) > _MAX_SIZE:
                # skip features with more than _MAX_SIZE items, probably an
                # image
                pass
            elif np.prod(arr.shape) > 1:
                for j, item in enumerate(arr.ravel()):
                    features_lin[i - 1]['%s.%d' % (name, j)] = item
            else:
                features_lin[i - 1][name] = value

    df_features = pandas.DataFrame(features_lin)

    for k, v in df_features.iteritems():
        if all(np.isnan(v.values.astype(np.float64))):
            df_features = df_features.drop(k, 1)

    return df_features


def extend_feature_blocks(features, features_in_block):
    blocks = {}
    for block, cols in features_in_block.items():
        blocks[block] = []
        for col in cols:
            blocks[block].extend([c for c in features.columns if re.sub(
                '[\.\d]+$', '', c) == re.sub('[\.\d]+$', '', col)])
        blocks[block] = np.unique(blocks[block])

    return blocks


def remove_large_features(features):
    for k, v in features.ix[1].iteritems():
        if type(v) is np.ndarray or type(v) is np.ma.core.MaskedArray:
            if np.prod(v.shape) > _MAX_SIZE:
                del features[k]

    return features


def compute_feature_stats(features):

    df =  pandas.DataFrame({'uuid':uuid.uuid4().hex,
                            'avg':features.mean(),
                            'var':features.var(),
                            'min':features.min(),
                            'max':features.max(),
                            'sum':features.sum(),
                            'n':features.shape[0]})
    df.index.name = 'feature'
    df = df.set_index('uuid', append=True)
    return df


def aggregate_feature_stats(stats):
     
    df_concat = pandas.concat(stats)
    if df_concat.index.name != 'feature' and 'feature' not in df_concat.index.names:
        df_concat.index.name = 'feature'
        df_concat = df_concat.set_index('uuid', append=True)

    # number of items
    ilen = df_concat['n'].sum(level='feature')

    # combined weighed average
    avg = (df_concat['avg'] * df_concat['n']).sum(level='feature') / ilen
    avg_r = __repeat_series(avg, stats)

    # combined weighed variance
    var = (df_concat['n'] * (df_concat['var'] + (df_concat['avg'] - avg_r['avg'])**2)).sum(level='feature') / ilen

    df_aggregated = pandas.DataFrame({'avg':avg, 'var':var})
    df_aggregated['min'] = df_concat['min'].min(level='feature')
    df_aggregated['max'] = df_concat['max'].max(level='feature')
    df_aggregated['sum'] = df_concat['sum'].sum(level='feature')

    return df_aggregated

def normalize_features(features, df_aggregated):

    # convert to standard normal distribution: (x - mu)/sigma
    sigma = df_aggregated['var'].replace(0,1).apply(np.sqrt)
    df_normalized = (features - df_aggregated['avg']) / sigma

    # remove all-nan columns (again) in case column is missing in stats
    for k,v in df_normalized.items():
        if all(np.isnan(v.values.astype(np.float64))):
            df_normalized = df_normalized.drop(k,1)

    return df_normalized

def __repeat_series(avg, stats):
    avg_r = []
    for i in range(len(stats)):
        df = pandas.DataFrame({'avg':avg})
        df['uuid'] = stats[i].reset_index(1)['uuid']
        avg_r.append(df)
    avg_r = pandas.concat(avg_r)
    avg_r = avg_r.set_index('uuid', append=True)

    return avg_r


def scale_features(data,
                   features,
                   length_features=_LENGTH_FEATURES,
                   area_features=_AREA_FEATURES,
                   coordinate_features=_COORDINATE_FEATURES,
                   pop_features=_POP_FEATURES):

    for ext in ['.%d' % i for i in range(4)] + ['']:

        feat = 'histogram' + ext
        if feat in features.columns and 'area' in features.columns:
            features[feat] = features[feat] / np.float64(features['area'])
            
        for lfeat in length_features:
            feat = lfeat + ext
            if feat in features.columns:
                 features[feat] = features[feat].apply(lambda x: x / np.sqrt(np.float64(data.size)))
        
        for afeat in area_features:
            feat = afeat + ext
            if feat in features.columns:
                features[feat] = features[feat].apply(lambda x: x / np.float64(data.size))
                
        for cfeat in coordinate_features:
            feat = cfeat + ext
            if feat in features.columns:
                features[feat] = features[feat].apply(lambda x: x / np.array(np.float64(data.shape))[[0,1]])
        
        feat = 'bbox' + ext
        if feat in features.columns:
            features[feat] = features[feat].apply(lambda x: x / np.array(np.float64(data.shape))[[0,1,0,1]])

#        for pfeat in pop_features:
#            feat = pfeat + ext
#            if feat in features.columns:
#                del features[feat]

    return features
