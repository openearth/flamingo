import re
import pandas
import numpy as np

_MAX_SIZE = 50

def linearize(features):
    '''convert all items in each matrix feature into individual features'''

    features_lin = [{} for i in range(len(features))]
    for name, values in features.iteritems():
        for i,value in values.iteritems():
            arr = np.asarray(value)
            if np.prod(arr.shape) > _MAX_SIZE:
                pass # skip features with more than _MAX_SIZE items, probably an image
            elif np.prod(arr.shape) > 1:
                for j, item in enumerate(arr.ravel()):
                    features_lin[i-1]['%s.%d' % (name, j)] = item
            else:
                features_lin[i-1][name] = value

    df_features = pandas.DataFrame(features_lin)

    for k,v in df_features.iteritems():
        if all(np.isnan(v)):
            df_features = df_features.drop(k,1)

    return df_features

def extend_feature_blocks(features, features_in_block):
    blocks = {}
    for block, cols in features_in_block.iteritems():
        blocks[block] = []
        for col in cols:
            blocks[block].extend([c for c in features.columns if re.sub('[\.\d]+$','',c) == re.sub('[\.\d]+$','',col)])
        blocks[block] = np.unique(blocks[block])

    return blocks

def remove_large_features(features):
    for k,v in features.ix[1].iteritems():
        if type(v) is np.ndarray:
            if np.prod(v.shape) > _MAX_SIZE:
                del features[k]

    return features
