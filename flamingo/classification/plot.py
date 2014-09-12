import re
from flamingo import filesys
from flamingo.classification import utils
from matplotlib import pyplot as plt
import numpy as np

def plot_prediction(ds, im, Y, axs=None):

    Y = Y.flatten()

    classes = list(np.unique(Y))

    img = filesys.read_image_file(ds, im)
    seg = filesys.read_export_file(ds, im, 'segments')
        
    prediction = np.empty(seg.shape)

    for i, c in enumerate(Y):
        prediction[seg == i] = classes.index(c)

    if axs is None:
        fig, axs = plt.subplots()
    else:
        fig = axs.get_figure()

    axs.imshow(prediction, vmin=0, vmax=len(classes))

    return fig, axs


def plot_predictions(ds,model,part=0,class_aggregation=None):
 
    model, meta, train_sets, test_sets, prior_sets = classify.reinitialize_model(
        ds, model, class_aggregation=class_aggregation)

    if model is list:
        model = model[part]
    elif part > 0:
        raise IOError

    classlist = filesys.read_default_categories(ds)
    classlist = utils.aggregate_classes(np.array(classlist),class_aggregation)
    classlist = np.unique(classlist)

    n = np.shape(test_sets)[1]

    fig,axs = plt.subplots(n,3,figsize=(20,10 * round(n / 2)))

    for i, fname in enumerate(meta['images_test']):

        grnd = test_sets[part][0][i]
        pred = model.predict(test_sets[part][0][i])
        score = float(np.sum(pred == grnd)) / len(grnd) * 100

        axs[i,0].imshow(img)

        plot_prediction(ds,
                        fname,
                        grnd,
                        axs=axs[i,1])

        plot_prediction(ds,
                        fname,
                        pred,
                        axs=axs[i,2])
        
        axs[i,0].set_title(fname)
        axs[i,1].set_title('groundtruth')
        axs[i,2].set_title('prediction (%0.1f%%)' % score)


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
