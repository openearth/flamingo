import numpy as np
import pandas
import matplotlib.pyplot as plt
import sklearn.linear_model

from .. import filesys

from models import *
from utils import *

def aggregate_scores(scores):
    '''Aggregate model scores over training and test sets

    Parameters
    ----------
    scores : pandas.DataFrame
        DataFrame with test scores for different models and training sets.
        Should have at least one level index names "model".

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
        List of model objects. Model objects should have a fit() and score() method.
    train_sets : list
        List of tuples containing training data corresponding to the model list.
    test_sets : list
        List of tuples containing test data corresponding to the model list.
    step : integer, optional
        Step size of learning curve (default: 10)
    **kwargs : dict-like
        All other named arguments are redirected to the function models.train_models()

    Returns
    -------
    all_scores : pandas.DataFrame
        MultiIndex DataFrame containing training and test scores.
        Indices "model" and "set" indicate the model and training set number used.
        Index "n" indicates the number of samples used during training.
        Columns "train" and "test" contain the train and test scores respectively.
    all_models : list
        List with trained models.
        Each item corresponds to a single point on the learning curve and
            can consist of several models organized in a NxM matrix where
            N is the original number of models trained and M is the number
            of training sets used.

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

def compute_confusion_matrix(models, train_sets, test_sets):

    check_sets(train_sets, test_sets)

    # get model predictions
    Y = predict_models(models, [s[0] for s in test_sets])

    # get all classes
    classes = list(np.unique(np.concatenate([np.unique(yi) for (x,y) in test_sets for yi in y])))

    # compute confusion matrices
    mtxs = []
    for i, Y1 in enumerate(Y):
        mtxs.append([])
        for j, (Y1j, (X2j, Y2j)) in enumerate(zip(Y1, test_sets)):
            mtxs[i].append(np.zeros((len(classes),len(classes))))
            for k, (Y1k, Y2k) in enumerate(zip(Y1j, Y2j)):
                for c1, c2 in zip(Y1k.flatten(), Y2k.flatten()):
                    i1 = classes.index(c1)
                    i2 = classes.index(c2)
                
                    mtxs[i][j][i1,i2] += 1
                
    return mtxs, classes

def compute_regularization_curve(models, train_sets, test_sets):
    pass

def plot_learning_curve(scores, ylim=(.75,1), filename=None):
    '''Plots learning curves

    Parameters
    ----------
    scores : pandas.DataFrame
        DataFrame containing all scores used to plot one or more learning curves.
        Should at least have the index "n" indicating the number of training samples
        used.
    ylim : 2-tuple, optional
        Vertical axis limit for learning curve plots.
    filename : string, optional
        If given, plots are saved to indicated file path.

    Returns
    -------
    list
        List with figure handles for all plots
    list
        List with axes handles for all plots

    '''

    scores_iterate = scores.reset_index('n')

    figs = []
    axss = []

    idx = np.vstack([x.ravel() for x in np.meshgrid(*scores_iterate.index.levels)])
    for i in range(idx.shape[1]):
        cs = scores.xs(idx[:,i], level=scores_iterate.index.names)
    
        fig, axs = plt.subplots()
        axs.plot(cs.index, zip(cs['train'],cs['test']))
        axs.legend(('test score','training score'))
        axs.set_xlabel('Number of training images')
        axs.set_ylim(ylim)

        save_figure(fig, filename, ext=''.join(['_%s' % i for i in idx[:,i]]))

        figs.append(fig)
        axss.append(axs)

    if len(figs) == 1:
        return figs[0], axss[0]
    else:
        return figs, axss

def plot_confusion_matrix(mtxs, classes, cmap='Reds'):
    
    figs = []
    axss = []

    n = len(classes)

    for row in mtxs:
        for mtx in row:
            fig, axs = plt.subplots(1,2)

            mtx_norm = mtx / np.repeat(mtx.sum(axis = 1), n).reshape((n,n))
            mtx_norm[np.isnan(mtx_norm)] = 0

            axs[0].matshow(mtx_norm, cmap=cmap)
            axs[0].set_xticks(range(n))
            axs[0].set_yticks(range(n))
            axs[0].set_xticklabels(classes, rotation=90)
            axs[0].set_yticklabels(classes, rotation=0)
            axs[0].set_ylabel('ground truth')
            axs[0].set_xlabel('predicted')

            axs[1].matshow(mtx, cmap=cmap)
            axs[1].set_xticks(range(n))
            axs[1].set_yticks([])
            axs[1].set_xticklabels(classes, rotation=90)
            axs[1].set_xlabel('predicted')

            figs.append(figs)
            axss.append(axs)

    return figs, axss

def plot_regularization_curve(x, y):
    pass

def plot_feature_weights(model, meta, normalize=True, sort=True, figsize=(20,5), cmap='bwr'):

    weights = model.coef_

    # sort weights
    idx = range(weights.shape[1])
    if sort:
        idx = sorted(idx, key=lambda i: np.max(np.abs(weights[:,i])))
        idx.reverse()

    # normalize weights
    if normalize:
        weights = weights / np.tile(np.max(np.abs(weights), axis=0), (weights.shape[0], 1))

    features, classes = meta['features'], meta['classes']
    features.extend(['prob_%s' % c for c in classes])

    fig, axs = plt.subplots(figsize=figsize)
    axs.matshow(weights[:,idx], cmap=cmap)
    axs.set_xticks(range(len(features)))
    axs.set_xticklabels(np.asarray(features)[idx], rotation=90)
    axs.set_yticks(range(len(classes)))
    axs.set_yticklabels(classes)
    axs.set_xlim((-.5,len(features)-.5))
    axs.set_ylim((-.5,len(classes)-.5))

    return fig, axs

