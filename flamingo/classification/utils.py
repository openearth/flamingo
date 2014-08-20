import numpy as np
import sys

def linearize_data(X=None, Y=None):

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
    return [Yi.reshape(Xi.shape[:2]) for Xi, Yi in zip(X, Y)]

def aggregate_classes(Y, aggregation=None):
    if aggregation is not None:
        if type(Y) is tuple:
            Y = list(Y)
        if type(Y) is list or type(Y) is set:
            try:
                for i in range(len(Y)):
                    Y[i] = aggregate_classes(Y[i], aggregation)
            except:
                print "Unexpected aggregation error:", sys.exc_info()[0].message
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
                print "Unexpected aggregation error:", sys.exc_info()[0].message
        else:
            print "Unexpected type found for Y during class aggregation"
    return Y

def get_classes(Y):
    classes = []

    try:
        classes.extend(np.unique(Y))
    except:
        for y in Y:
            classes.extend(get_classes(y))

    classes = np.unique(classes)
    classes = np.delete(classes, [i for i,c in enumerate(classes) if c is None])

    return classes

def save_figure(fig, filename, ext=''):
    '''Save figure to file

    Parameters
    ----------
    fig : object
        Figure object
    filename : string
        Path to output file
    ext : string, optional
        String to be added to the filename before the file extension

    '''

    if filename is not None:
        filename = re.sub('\..+$', ext + '\g<0>', filename)
        fig.savefig(filename)

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

    '''

    if len(train_sets) != len(test_sets):
        raise ValueError('Number of train and test sets not equal')
    if np.any([len(im1) != len(im2) for im1,im2 in zip(train_sets, test_sets)]):
        raise ValueError('Not all train and test sets have equal size')
    if models is not None:
        if np.any([len(x) < len(train_sets) for x in models]):
            raise ValueError('Not as many models as training sets defined')
