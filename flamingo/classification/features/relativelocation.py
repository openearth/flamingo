import numpy as np
import pandas
import scipy.ndimage as nd
import matplotlib.pyplot as plt

import flamingo.classification.models

def compute_prior(annotations, centroids, image_size, superpixel_grid, n=100):
    '''
    Compute relative location prior according to Gould et al. (2008)

    Parameters
    ----------
    ds : string
        String indicating the dataset to be used.

    Returns
    -------
    maps : pandas.Panel4D
        4D panel containing the relative location prior maps: maps[<other class>][<given class>]
        gives a n*n dataframe representing the dimensionless image map

    Other parameters
    ----------------
    n : integer
        Half the size of the dimensionless image map

    '''

    nm, nn = image_size
    nx, ny = superpixel_grid

    annotations = np.reshape(annotations,(nx,ny))
    classes = np.unique(annotations)

    # allocate normalized relation maps
    mapi = {c1:{c2:np.zeros((2*n,2*n)) for c2 in classes} for c1 in classes}

    # get centroids
    centroids = zip(*list(centroids))
    m_mat = np.array(centroids[0]).reshape((nx,ny))
    n_mat = np.array(centroids[1]).reshape((nx,ny))

    # normalize centroid coordinates to map grid
    m_mat = np.round(m_mat.astype(np.float32)/(nm-1)*(n-1))
    n_mat = np.round(n_mat.astype(np.float32)/(nn-1)*(n-1))

    # loop over all superpixels
    for i in range(nx):
        for j in range(ny):
            # list indices in dimensionless image map that match the superpixel centroids
            ind_m = (m_mat - m_mat[i,j] + n - 1).ravel().astype(np.uint)
            ind_n = (n_mat - n_mat[i,j] + n - 1).ravel().astype(np.uint)

            # determine class of current superpixel
            c0 = annotations[i,j]

            # loop over classes in current image
            for c in classes:

                # add score matrix with offset to relation map
                mapi[c][c0][ind_m,ind_n] += (annotations == c).astype(np.uint).ravel()

    return mapi

def aggregate_maps(maplist):

    classes = np.unique([k for m in maplist for k in m.keys()])
    n = maplist[0].values()[0].values()[0].shape[0] / 2

    # allocate normalized relation maps
    maps = pandas.Panel4D(labels=classes, items=classes, major_axis=range(2*n), minor_axis=range(2*n))
    maps = maps.fillna(1) # Use a simple, small, uniform prior to deal with empty (zero) counts in the histogram

    for mapi in maplist:
        for c, mapi_from in mapi.iteritems():
            for c0, mapi_to in mapi_from.iteritems():
                maps[c][c0] += mapi_to

    # normalization
    tot = maps.sum(axis = 'labels')
    for c1 in classes:
        maps.ix[:,c1,:,:] = maps.ix[:,c1,:,:].divide(tot[c1]).as_matrix()

    return maps

def smooth_maps(maps, sigma=2):
    '''
    Convolve relative location prior maps with a gaussian filter for smoothing purposes

    Parameters
    ----------
    ds : string
        String indicating the dataset to be used.

    maps : pandas.Panel4D
        4D panel containing the relative location prior maps: maps[<other class>][<given class>]
        gives a n*n dataframe representing the dimensionless image map

    Returns
    -------
    maps : pandas.Panel4D
        4D panel containing the smoothed relative location prior maps.

    Other parameters
    ----------------
    sigma : integer
        Size of the gaussian kernel that is to be convolved with the relative location prior maps

    '''

    classes = maps.labels

    # convolution per map
    for c0 in classes:
        for c1 in classes:
            maps[c1][c0] = nd.gaussian_filter(maps[c1][c0].as_matrix(), sigma = sigma)

    # normalize per given class
    tot = maps.sum(axis = 'labels')
    for c0 in classes:
        maps.ix[:,c0,:,:] = maps.ix[:,c0,:,:].divide(tot[c0]).as_matrix()

    return maps

def vote_sets(Y, maps):

    votes = []

    for i, row in enumerate(Y):
        votes.append([])
        for j, col in enumerate(row):
            votes[i].append([])
            for Yi in col:
                votes[i][j].append(vote_image(Yi, maps)[0])

    return votes

def vote_image(Y, maps, centroids=None, img_size=None, winner_takes_all_mode=False):
    '''
    Class voting based on 1st order prediction and relative location prior maps

    Parameters
    ----------
    ds : string
        String indicating the dataset to be used.

    Ipred : list of lists of tuple of lists of arrays with size [n_models][n_partitions](training,testing)[n_images]
        Arrays contain the 1st order prediction of the labelled images.

    Returns
    -------
    votes : pandas.Panel
        Panel containing the votes for all classes and superpixels: maps[<class>]
        gives a nx*ny dataframe representing the probability of every superpixel to be <class>

    Ivote : np.array
        Labelled image based on classes in votes with maximum probability for every superpixel

    '''

    Ipred = np.asarray(Y)
    nx, ny = Ipred.shape
    
    classes = maps.keys()
    maps = dict_to_panel(maps)

    votes = pandas.Panel(items = classes,
                         major_axis = range(nx),
                         minor_axis = range(ny))
    votes = votes.fillna(0)

    n = maps.shape[-1] / 2

    # compute normalized centroids (FIXME: remove None options)
    if centroids is None:
        xn, xs = np.linspace(0, n, nx, endpoint=False, retstep=True)
        yn, ys = np.linspace(0, n, ny, endpoint=False, retstep=True)
        cxn, cyn = np.meshgrid(yn + ys/2, xn + xs/2)
    else:
        cx, cy = [np.asarray(c).reshape((nx,ny)).astype(np.float32) for c in zip(*list(centroids))]
        
        if img_size is None:
            cxn = cx / np.max(cx) * (n-1)
            cyn = cy / np.max(cy) * (n-1)
        else:
            cxn = cx / (img_size[0]-1) * (n-1)
            cyn = cy / (img_size[1]-1) * (n-1)

    # for every superpixel, loop over the relative location maps
    for i in range(nx):
        for j in range(ny):
            rx = np.round(cxn - cxn[i,j] + n - 1).flatten().astype(np.uint)
            ry = np.round(cyn - cyn[i,j] + n - 1).flatten().astype(np.uint)

            c0 = Ipred[i,j]

            # get assignments with highest percentage
            if winner_takes_all_mode:
                highest_perc = np.zeros((nx, ny))
                assignments = np.array(['' for ii in range(nx*ny)], dtype='a%d' % np.max([len(c) for c in classes])).reshape((nx,ny))
                for c in classes:
                    p = maps[c][c0].as_matrix()[ry,rx].reshape((nx,ny))
                    idx = p > highest_perc
                    assignments[idx] = c
                    highest_perc[idx] = p[idx]
                for c in classes:
                    votes[c] += (assignments == c)
            else:
                # vote classes
                for c in classes:
                    votes[c] += maps[c][c0].as_matrix()[ry,rx].reshape((nx,ny))

    # normalize votes by superpixel grid size
    for c in votes.keys():
        votes[c] = votes[c] / np.prod((nx,ny))

    Ivote = pandas.DataFrame([votes.major_xs(i).idxmax(axis=1) for i in range(votes.shape[1])])

    return votes, Ivote.as_matrix()

def compute_features(models, sets, maps, stats, features):

    # remove relative location features from sets
    for j, sets_parts in enumerate(sets): # loop over partitions
        for k, sets_img in enumerate(sets_parts): # loop over images
            for c in maps.iterkeys():
                s = 'prob_%s' % c
                idx = list(features).index(s)

                sets[j][0][k][...,idx] = 0.0

    # do first order prediction
    Y_predicted = flamingo.classification.models.predict_models(models, [s[0] for s in sets])
    
    # do voting based on first prediction round and relative location prior
    votes = vote_sets(Y_predicted, maps)

    # translate voted probabilities to relative location features
    sets = votes_to_features(sets, votes, stats, features)

    return sets

def votes_to_features(sets, votes, stats, features):

    # copy sets for each model
    sets = [sets for i in votes]
    
    for i, vote_models in enumerate(votes): # loop over models
        for j, vote_parts in enumerate(vote_models): # loop over partitions
            for k, vote_img in enumerate(vote_parts): # loop over images
                sets[i][j][0][k] = votes_to_feature(sets[i][j][0][k], vote_img, stats, features)

    return sets

def votes_to_feature(X, votes, stats, features):
    for c in votes.items:
        s = 'prob_%s' % c
        avg = stats['avg'][s]
        std = np.sqrt(stats['var'][s])
        val = votes[c].as_matrix()
        idx = list(features).index(s)

        X[...,idx] = (val - avg) / std
        
    return X

def add_features(votes, features, features_in_block):
    features_in_block['relloc'] = []
    for c in votes.items:
        s = 'prob_%s' % c
        features[s] = votes[c].as_matrix().ravel()
        features_in_block['relloc'].append(s)

    return features, features_in_block

def remove_features(X, classes):
#    for i in range(len(X)):
    for c in classes:
#        X[i]['prob_%s' % c] = 0.0
        X['prob_%s' % c] = 0.0
            
    return X

def panel_to_dict(maps):
    return {c1:{c0: maps[c1][c0].as_matrix() for c0 in maps.labels} for c1 in maps.items}

def dict_to_panel(maps):

    return pandas.Panel4D(maps)

def plot_maps(maps, figsize=(20,20), cmap='Reds'):

    classes = maps.keys()
    n = len(classes)

    fig, axs = plt.subplots(n,n,figsize=figsize)
    for k1 in classes:
        for k2 in classes:
            i = maps.keys().index(k1)
            j = maps.keys().index(k2)
            
            axs[i,j].matshow(maps[k1][k2], cmap=cmap)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
        
    for i, classi in enumerate(classes):
        axs[i,0].set_ylabel(classi)
        axs[0,i].set_title(classi)

    return fig, axs

def plot_votes(y, figsize=(7,5)):

    classes = np.unique(y)
    img = np.zeros(np.asarray(y).shape)
    for i, c in enumerate(classes):
        img[y == c] = i

    fig, axs = plt.subplots(figsize=figsize)
    axs.matshow(img)

    return fig, axs
