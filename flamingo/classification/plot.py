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
