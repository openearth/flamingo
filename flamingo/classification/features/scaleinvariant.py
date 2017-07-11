import numpy as np

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
