import numpy as np
import pandas
import matplotlib.pyplot as plt
import sklearn.linear_model

from models import *
from utils import *


def aggregate_scores(scores):
    '''Aggregate model scores over training and test sets

    Parameters
    ----------
    scores : pandas.DataFrame
        DataFrame with test scores for different models and training
        sets. Should have at least one level index named "model".

    Returns
    -------
    pandas.DataFrame
        DataFrame averaged over all indices except "model".

    '''

    return scores.mean(level='model')


def compute_learning_curve(models, train_sets, test_sets, step=10, **kwargs):
    '''Computes learning curves for combinations of models and training/test sets

    Parameters
    ----------
    models : list
        List of model objects. Model objects should have a fit() and
        score() method.
    train_sets : list
        List of tuples containing training data corresponding to the
        model list.
    test_sets : list
        List of tuples containing test data corresponding to the model
        list.
    step : integer, optional
        Step size of learning curve (default: 10)
    kwargs : dict-like
        All other named arguments are redirected to the function
        models.train_models()

    Returns
    -------
    all_scores : pandas.DataFrame
        MultiIndex DataFrame containing training and test scores.
        Indices "model" and "set" indicate the model and training set
        number used. Index "n" indicates the number of samples used
        during training. Columns "train" and "test" contain the train
        and test scores respectively.
    all_models : list
        List with trained models.
        Each item corresponds to a single point on the learning curve
        and can consist of several models organized in a NxM matrix
        where N is the original number of models trained and M is the
        number of training sets used.
    '''

    check_sets(train_sets, test_sets)

    m = len(test_sets)

    all_scores = []
    all_models = []
    set_lengths = [x[0].shape[0] for x in train_sets]
    for n in range(step, np.max(set_lengths), step):
        idx = [i for i,l in enumerate(set_lengths) if n < l]

        train_set = [s for i,s in enumerate(train_sets) if i in idx]
        test_set = [s for i,s in enumerate(test_sets) if i in idx]

        models_trained = train_models(models, train_set, **kwargs)
        scores = score_models(models_trained, train_set, test_set)

        scores['n'] = n
        scores = scores.set_index('n', append=True)

        all_scores.append(scores)
        all_models.append(models_trained)

    return pandas.concat(all_scores, axis=0), all_models


def compute_confusion_matrix(models, test_sets):
    '''Computes confusion matrix for combinations of models and training/test sets

    Parameters
    ----------
    models : list
        List of model objects. Model objects should have a fit() and
        score() method.
    test_sets : list
        List of tuples containing test data corresponding to the model
        list.

    Returns
    -------
    mtxs : list
        List of lists with np.ndarrays that contain confusion matrices
        for each combination of test set and model
    classes : list
        List with all unique classes in test sets and the axis labels
        of the confusion matrices
    '''

    # get model predictions
    Y = predict_models(models, [s[0] for s in test_sets])[0]

    # get all classes
    classes = list(np.unique(np.concatenate([np.unique(yi) for (x,y) in test_sets for yi in y])))

    # compute confusion matrices
    mtxs = []
    for i, Y1 in enumerate(Y):
        mtxs.append(np.zeros((len(classes),len(classes))))
        for j, (Y1j, Y2j) in enumerate(zip(Y1, test_sets[i][1])):
            for k, (Y1k, Y2k) in enumerate(zip(Y1j, Y2j)):
                for c1, c2 in zip(Y1k.flatten(), Y2k.flatten()):
                    i1 = classes.index(c1)
                    i2 = classes.index(c2)
                
                    mtxs[i][i2,i1] += 1
                
    return mtxs, classes
