import re
from flamingo.classification import utils
from matplotlib import pyplot as plt
import numpy as np


def plot_prediction(Y, seg, clist=None, cm='jet', axs=None):
    
    Y = Y.flatten()
    prediction = utils.labels2image(Y, seg, clist)

    if axs is None:
        fig, axs = plt.subplots()
    else:
        fig = axs.get_figure()
        
    axs.imshow(prediction, vmin=0, vmax=len(clist), cmap=cm)

    return fig, axs


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

    for mtx in mtxs:
        fig, axs = plt.subplots(1,2,figsize=(12,5))

        mtx_norm = mtx / np.repeat(mtx.sum(axis = 1), n).reshape((n,n))
        mtx_norm[np.isnan(mtx_norm)] = 0

        p0 = axs[0].matshow(mtx_norm, cmap=cmap, vmin=0, vmax=1)
        axs[0].set_xticks(range(n))
        axs[0].set_yticks(range(n))
        axs[0].set_xticklabels(classes, rotation=90, fontsize=14)
        axs[0].set_yticklabels(classes, rotation=0, fontsize=14)
        axs[0].set_ylabel('ground truth', fontsize=16)
        axs[0].set_xlabel('predicted', fontsize=16)
        axs[0].set_title('Row-normalized',y=1.3,fontsize=18)
        plt.colorbar(p0,ax=axs[0])

        p1 = axs[1].matshow(mtx, cmap=cmap)
        axs[1].set_xticks(range(n))
        axs[1].set_yticks([])
        axs[1].set_xticklabels(classes, rotation=90, fontsize=14)
        axs[1].set_xlabel('predicted', fontsize=16)
        axs[1].set_title('Absolute',y=1.3,fontsize=18)
        plt.colorbar(p1,ax=axs[1])

        plt.tight_layout()

        figs.append(figs)
        axss.append(axs)

    return figs, axss


def plot_feature_weights(model, meta, normalize=True, sort=True,
                         figsize=(20,5), cmap='bwr'):

    n = 162
    weights = model.w[:n].reshape((2,-1)) #coef_

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


def save_figure(fig, filename, ext='', figsize=None, dpi=30, **kwargs):
    '''Save figure to file

    Parameters
    ----------
    fig : object
        Figure object
    filename : string
        Path to output file
    ext : string, optional
        String to be added to the filename before the file extension

    Save a Matplotlib figure as an image without borders or frames.
       Args:
            fileName (str): String that ends in .png etc.

            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain 
            aspect ratio.
    '''

    if filename is not None:
        filename = re.sub('\.[^\.]+$', ext + '\g<0>', filename)

    fig_size = fig.get_size_inches()
    w, h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)

    if kwargs.has_key('orig_size'): # Aspect ratio scaling if required
        w,h = kwargs['orig_size']
        w2,h2 = fig_size[0],fig_size[1]
        fig.set_size_inches([(w2/w)*w,(w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())

    axs = fig.gca()
    axs.set_frame_on(False)
    axs.set_xticks([])
    axs.set_yticks([])

    plt.axis('off')

    if figsize is not None:
        fig.set_size_inches(figsize[0]/dpi, figsize[1]/dpi)

    fig.savefig(filename,
                dpi=dpi,
                transparent=True,
                bbox_inches='tight',
                pad_inches=0)
