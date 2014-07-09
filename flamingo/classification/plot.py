from flamingo import classify,filesys
from flamingo.classification import utils
from matplotlib import pyplot as plt
import numpy as np

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
        
        img = filesys.read_image_file(ds, fname)
        seg = filesys.read_export_file(ds, fname, 'segments')
        
        pred = model.predict(test_sets[part][0][i])

        prednum = np.empty(seg.shape)
        groudnum = np.empty(seg.shape)

        for j, c in enumerate(pred):
            prednum[segments==j] = classlist.index(c)

        for j, c in enumerate(test_sets[part][1][i]):
            groundnum[segments==j] = classlist.index(c)

        axs[i,0].imshow(img)
        axs[i,0].set_title(fname)

        axs[i,1].imshow(groundnum, vmin = 0, vmax = len(classlist))
        axs[i,1].set_title('groundtruth')

        axs[i,2].imshow(prednum, vmin = 0, vmax = len(classlist))
        axs[i,2].set_title('prediction (%0.1f%%)' % (float(np.sum(pred == cls)) / len(cls) * 100))
