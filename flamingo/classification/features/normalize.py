import numpy as np
import pandas
import uuid
import cPickle as pickle

def compute_feature_stats(features):

    return pandas.DataFrame({'uuid':uuid.uuid4(),
                             'avg':features.mean(),
                             'var':features.var(),
                             'min':features.min(),
                             'max':features.max(),
                             'sum':features.sum(),
                             'n':features.shape[0]})

def aggregate_feature_stats(stats):
     
    df_concat = pandas.concat(stats)
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
    for k,v in df_normalized.iteritems():
        if all(np.isnan(v.values.astype(np.float64))):
            df_normalized = df_normalized.drop(k,1)

    return df_normalized

def __repeat_series(avg, stats):
    avg_r = []
    for i in range(len(stats)):
        df = pandas.DataFrame({'avg':avg})
        df['uuid'] = stats[i]['uuid']
        avg_r.append(df)
    avg_r = pandas.concat(avg_r)
    avg_r = avg_r.set_index('uuid', append=True)

    return avg_r
