#!/usr/bin/env python

__author__ = "Bas Hoonhout"
__copyright__ = "Copyright 2014, The NEMO Project"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Bas Hoonhout"
__email__ = "bas.hoonhout@deltares.nl"
__status__ = "production"

import numpy as np
import sys
import logging


def linearize_data(X=None, Y=None):
    '''Linearizes structured data

    Transforms structured data suitible for the use with CRF and SSVM
    into non-structrued data with a single dimension suitible for the
    use with LR or SVM.

    Parameters
    ----------
    X : list, optional
        List with np.ndarray for each image with feature data in three
        dimensions (u, v and feature number)
    Y : list, optional
        List with np.ndarrays for each images with label data in two
        dimensions (u and v)

    Returns
    -------
    np.ndarray or 2-tuple
        Either linearized X, linearized Y or both are returned
        depending on the input
    '''

    if X is not None:
        X = np.vstack([x.reshape((-1,x.shape[-1])) for x in X])
    if Y is not None:
        Y = np.concatenate([y.ravel() for y in Y])

    if X is not None and Y is not None:
        return X, Y
    elif X is not None:
        return X
    elif Y is not None:
        return Y


def delinearize_data(Y, X):
    '''De-linearizes structured label data

    Transforms linearized labell data suitible for the use with LR
    and SVM into structured data for the use with CRF and SSVM.

    Parameters
    ----------
    Y : list
        List with np.ndarrays for each images with label data in one
        dimensions (u*v)
    X : list
        List with np.ndarray for each image with feature data in two
        dimensions (u*v and feature number)

    Returns
    -------
    list
        Delinearized Y data
    '''

    return [Yi.reshape(Xi.shape[:2]) for Xi, Yi in zip(X, Y)]


def aggregate_classes(Y, aggregation=None):
    '''Aggregate class labels into a subsection of class labels

    Replaces all class labels in Y with substitutes from the
    dictionary *aggregation*.

    Parameters
    ----------
    Y : tuple, list or np.ndarray
        Array containing class labels
    class_aggregation : dict, optional
        Dictionary containing class replacements where each key
        is a the replacement value of all classes in the
        corresponding list

    Returns
    -------
    np.ndarray
        Aggregated class labels
    '''

    if aggregation is not None:
        if type(Y) is tuple:
            Y = list(Y)
        if type(Y) is list or type(Y) is set:
            try:
                for i in range(len(Y)):
                    Y[i] = aggregate_classes(Y[i], aggregation)
            except:
                logging.error('Unexpected aggregation error (list)')
        elif type(Y) is np.ndarray:
            try:
                if np.all([type(y) is np.ndarray for y in Y]):
                    for i in range(len(Y)):
                        Y[i] = aggregate_classes(Y[i], aggregation)
                elif np.all([type(y) is np.string_ or type(y) is np.unicode_ for y in Y]):
                    for k,v in aggregation.iteritems():
                        for vi in v:
                            Y[np.where(Y == vi)] = k
            except:
                logging.error('Unexpected aggregation error (numpy)')
        else:
            logging.warn('Unexpected type found for Y during class aggregation')
    return Y


def get_classes(Y):
    '''Get list of unique classes in Y

    Returns a list of unique classes in Y with all None values removed
    and regardless of the shape and type of Y.

    Parameters
    ----------
    Y : list or np.ndarray
        List with np.ndarrays or np.ndarray with class labels

    Returns
    -------
    np.ndarray
        Array with unique class labels in Y not being None
    '''

    classes = []

    try:
        classes.extend(np.unique(Y))
    except:
        for y in Y:
            classes.extend(get_classes(y))

    classes = np.unique(classes)
    classes = np.delete(classes, [i for i,c in enumerate(classes) if c is None])

    return classes


def check_sets(train_sets, test_sets, models=None):
    '''Checks if train sets, test sets and models have matching dimensions

    Parameters
    ----------
    train_sets : list
        List of tuples containing training data corresponding to the model list.
    test_sets : list
        List of tuples containing test data corresponding to the model list.
    models : list
        List of lists with each item a trained instance of a model.

    Raises
    ------
    ValueError
    '''

    if len(train_sets) != len(test_sets):
        raise ValueError('Number of train and test sets not equal')
    if np.any([len(im1) != len(im2) for im1,im2 in zip(train_sets, test_sets)]):
        raise ValueError('Not all train and test sets have equal size')
    if models is not None:
        if np.any([len(x) < len(train_sets) for x in models]):
            raise ValueError('Not as many models as training sets defined')


def labels2int(Y, classes=None):
    '''Transforms string class labels in numbers

    Parameters
    ----------
    Y : list
        List with np.ndarrays with class labels
    classes : list, optional
        List with unique class labels possibly in Y

    Returns
    -------
    np.ndarray
        Array with class numbers rather than labels
    '''

    if classes is None:
        classes = np.unique(linearize_data(Y=Y))

    Yint = [(np.zeros(y.shape)-1).astype(int, copy=True) for y in Y]
    for i, y in enumerate(Y):
        for j, c in enumerate(classes):
            Yint[i][y == c] = j
        
    return np.asarray(Yint)


def int2labels(Y, classes=None):
    '''Transforms class numbers in string class labels

    Parameters
    ----------
    Y : list
        List with np.ndarrays with class numbers
    classes : list, optional
        List with unique class labels possibly in Y

    Returns
    -------
    np.ndarray
        Array with class labels rather than numbers
    '''

    if classes is None:
        classes = np.unique(linearize_data(Y=Y))

    Ystr = [y.astype(unicode, copy=True) for y in Y]
    for i, y in enumerate(Y):
        for j, c in enumerate(classes):
            Ystr[i][y == j] = c
            
    return np.asarray(Ystr)

def labels2image(Y, seg, classes=None):
    '''Transforms class labels and segmentation into class image

    Parameters
    ----------
    Y : list
        List with np.ndarrays with class labels
    seg : np.ndarray
        MxN array with superpixel numbers
    classes : list, optional
        List with unique class labels possibly in Y

    Returns
    -------
    np.ndarray
        Unnormalized single-channel image of class assignments
    '''

    Y = Y.flatten()
    prediction = np.empty(seg.shape)
    
    if classes is None:
        classes = list(np.unique(Y))
    
    for i, c in enumerate(Y):
        prediction[seg == i] = classes.index(c)

    return prediction

    
