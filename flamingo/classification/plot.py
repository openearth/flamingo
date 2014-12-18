import re
from flamingo import filesys #, classify # FIXME: do not do this!
from flamingo.classification import utils
from matplotlib import pyplot as plt
import matplotlib.colors
import numpy as np

def plot_prediction(ds, im, Y, clist, cm='jet', axs=None):

    Y = Y.flatten()
    
    seg = filesys.read_export_file(ds, im, 'segments')

    prediction = utils.labels2image(Y, seg, clist)

    if axs is None:
        fig, axs = plt.subplots()
    else:
        fig = axs.get_figure()
        
    axs.imshow(prediction, vmin=0, vmax=len(clist), cmap=cm)

    return fig, axs


def plot_predictions(ds, model, meta, test_sets, part=0, class_aggregation=None):
 
    # TODO: kick Max for doing this. And then kick BasHo! He demolished everything! Reverse engineering! Murder! Fire!
    #model, meta, train_sets, test_sets, prior_sets = classify.reinitialize_model(
    #    ds, model, class_aggregation=class_aggregation)

    if model is list:
        model = model[part]
    elif part > 0:
        raise IOError

    classlist = filesys.read_default_categories(ds)
    classlist = utils.aggregate_classes(np.array(classlist), class_aggregation)
    classlist = list(np.unique(classlist))

    eqinds = resolve_indices(ds, test_sets[0][1], meta, class_aggregation)

    n = np.shape(test_sets)[-1]

    fig,axs = plt.subplots(n,3,figsize=(20,10 * round(n / 2)))

    cdict = {
        'red'  :  ((0., .5, .5), (.2, .5, 1.), (.4, 1., .66), (.6, .66, .13), (.8, .13, .02), (1., .02, .02)),
        'green':  ((0., .5, .5), (.2, .5, .95), (.4, .95, 1.), (.6, 1., .69), (.8, .69, .24), (1., .24, .24)),
        'blue' :  ((0., .5, .5), (.2, .5, 0.), (.4, 0., 1.), (.6, 1., .30), (.8, .30, .75), (1., .75, .75))
        }
    cmap_argus = matplotlib.colors.LinearSegmentedColormap('argus_classes', cdict, 5)

    for i, fname in enumerate(meta['images_test'][:-1]):
        print 'Processing image number %i' %i
        if any(eqinds[:,1] == i):
            gind = np.where(eqinds[:,1] == i)[0][0]
            grnd = test_sets[part][1][gind]
            pred = model.predict([test_sets[part][0][gind]])[0]
            score = float(np.sum(pred == grnd)) / np.prod(grnd.shape) * 100
        
            img = filesys.read_image_file(ds,fname)
            axs[i,0].imshow(img)

            plot_prediction(ds,
                        fname,
                        grnd,
                        cm=cmap_argus,
                        clist=classlist,
                        axs=axs[i,1])

            plot_prediction(ds,
                        fname,
                        pred,
                        cm=cmap_argus,
                        clist=classlist,
                        axs=axs[i,2])
        
            axs[i,0].set_title(fname)
            axs[i,1].set_title('groundtruth')
            axs[i,2].set_title('prediction (%0.1f%%)' % score)

    return fig,axs


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

def resolve_indices(ds,Ytest,meta,agg):
    eqinds = np.empty((len(Ytest),2))
    for i in range(len(Ytest)):
        for j in range(len(meta['images_test'])):
            Y = filesys.read_export_file(ds,meta['images_test'][j],'classes')
            if Y:
                metim = filesys.read_export_file(ds,meta['images_test'][j],'meta')
                Ya = np.array(Y)
                if np.prod(metim['superpixel_grid']) == len(Ya):
                    Yr = Ya.reshape((metim['superpixel_grid']))
                    Ya = utils.aggregate_classes(Yr,agg)
                    if np.prod(Yr.shape) == np.prod(Ytest[i].shape):
                        if np.all(Yr == Ytest[i]):
                            eqinds[i,:] = np.array([i,j])
                            break

    return eqinds
