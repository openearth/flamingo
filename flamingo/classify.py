'''Train, score and use classification models on image datasets.

Usage:
    classify-images preprocess <dataset> [--segmentate] [--channels] [--features] [--extract] [--update] [--normalize] [--relloc] [--relloc_maps] [--images=FILE] [--config=FILE] [--overwrite] [--verbose]
    classify-images partition <dataset> [--n=N] [--frac=FRAC] [--images=FILE] [--config=FILE] [--verbose]
    classify-images train <dataset> [--type=NAME] [--partitions=N] [--images=FILE] [--config=FILE] [--verbose]
    classify-images score <dataset> [--model=NAME] [--images=FILE] [--config=FILE] [--verbose]
    classify-images predict <dataset> [--model=NAME] [--images=FILE] [--config=FILE] [--overwrite] [--verbose]
    classify-images regularization <dataset> [--type=NAME] [--images=FILE] [--config=FILE] [--verbose]

Positional arguments:
    dataset            dataset containing the images
    image              image file to be classified

Options:
    -h, --help         show this help message and exit
    --segmentate       create segmentation of images
    --channels         include channel extraction
    --features         include feature extraction
    --extract          extract channels/features
    --update           update channels/features
    --normalize        normalize channels/features
    --relloc           include relative location features
    --relloc_maps      compute new relative location maps
    --n=N              number of partitions [default: 5]
    --frac=FRAC        fraction of images used for testing [default: 0.25]
    --type=NAME        model type to train [default: LR]
    --partitions=N     only train these partitions
    --model=NAME       name of model to be scored, uses last trained if omitted
    --images=FILE      images to include in process
    --config=FILE      configuration file to use instead of command line options
    --overwrite        overwrite existing files
    --verbose          print logging messages
'''

import re
import os
import sys
import time
import json
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cross_validation import train_test_split

from flamingo import classification as cls
from flamingo import segmentation as seg
from flamingo import filesys, config
from flamingo.utils import printinfo
from flamingo.classification import channels
from flamingo.classification.features import relativelocation
from flamingo.classification import plot


# initialize log
logger = logging.getLogger(__name__)


def run_preprocessing(ds,
                      images='all',
                      segmentation=False,
                      channels=False,
                      features=False,
                      extract=False,
                      update=False,
                      normalize=False,
                      feature_blocks='all',
                      colorspace='rgb',
                      relative_location_maps=False,
                      relative_location_prior=False,
                      overwrite=False,
                      model_dataset=None,
                      class_aggregation=None,
                      cfg=None):

    logger.info('Preprocessing started for dataset "%s"...' % ds)

    # make sure log file exists
    filesys.write_log_file(
        ds, {'start_preprocess': time.strftime('%d-%b-%Y %H:%M')})

    # create image list
    images = create_image_list(ds, images)

    # segmentation
    if segmentation:
        run_segmentation(ds, images, overwrite=overwrite, cfg=cfg)
        
    # channel extraction
    if channels:
        if extract:
            run_channel_extraction(
                ds, images, colorspace=colorspace,
                overwrite=overwrite, cfg=cfg)

        if normalize:
            run_channel_normalization(
                ds, images, model_dataset=model_dataset,
                overwrite=overwrite, cfg=cfg)

    # feature extraction, first part
    if features and extract:
        run_feature_extraction(
            ds, images, feature_blocks=feature_blocks,
            colorspace=colorspace, model_dataset=model_dataset,
            overwrite=overwrite, cfg=cfg)

    # relative location prior maps, requires centroids
    # from feature extraction
    if relative_location_prior and relative_location_maps:
        run_relative_location_mapping(
            ds, n=100, sigma=2,
            class_aggregation=class_aggregation, cfg=cfg)

    # feature extraction, second part
    if features:
        if update:
            run_feature_update(
                ds, images, feature_blocks=feature_blocks,
                class_aggregation=class_aggregation,
                relative_location_prior=relative_location_prior,
                overwrite=overwrite)

        if normalize:
            run_feature_normalization(
                ds, images, feature_blocks=feature_blocks,
                overwrite=overwrite, model_dataset=model_dataset)

    logger.info('Preprocessing finished.')


@config.parse_config(['partition'])
def run_partitioning(ds,
                     images='all',
                     n_partitions=5,
                     frac_validation=0.0,
                     frac_test=0.25,
                     force_split=False,
                     cfg=None):

    images = create_image_list(ds, images)

    logger.info('Defining %d train/test partitions' % n_partitions)

    # make train test partitions if requested
    dslog = filesys.read_log_file(ds)
    if force_split or not dslog.has_key('training images'):
        images_part = multiple_partitions(
            images, n_partitions, frac_test)
        filesys.write_log_file(ds, images_part)


@config.parse_config(['features', 'training'])
def run_training(ds,
                 images='all',
                 feature_blocks='all',
                 model_type='LR',
                 class_aggregation=None,
                 partitions='all',
                 cfg=None):
    '''
    Batch function to train and test a PGM with multiple
    train/test partitions
    '''

    logger.info('Classification started for dataset "%s"...' % ds)

    if not type(model_type) is list:
        model_type = [model_type]

    # create feature block list
    feature_blocks = create_feature_list(feature_blocks)

    # create partitions list
    partitions = create_partitions_list(partitions)

    # initialize models, training and test sets
    logger.info(
        'Preparation of models, train and test sets started...')
    models, meta, train_sets, test_sets, prior_sets = \
        initialize_models(
        ds, images, feature_blocks, model_type=model_type,
        class_aggregation=class_aggregation, partitions=partitions)
    logger.info(
        'Preparation of models, train and test sets finished.')

    # fit models
    logger.info('Fitting of models started...')
    models = cls.models.train_models(
        models, train_sets, prior_sets,
        callback=lambda m,(i,j): filesys.write_model_file(
            ds, m, meta[i][j], ext='.%d.backup' % partitions[j]))
    filesys.write_model_files(ds, models, meta)
    logger.info('Fitting of models finished.')

    return models, train_sets, test_sets

 
@config.parse_config()
def run_scoring(ds, models=None, class_aggregation=None, cfg=None):

    if models is None:
        models = filesys.get_model_list(ds)[:1]
    elif type(models) is str:
        if ',' in models:
            models = models.split(',')
        elif not filesys.check_model_file(ds, models):
            models = [x for x in glob.glob(models)
                      if '.meta.' not in x and '.backup.' not in x]
        else:
            models = [models]

    logger.info('Testing of model started...')

    scores = []
    for model in models:
        logger.info('Scoring model "%s"...' % model)

        model, meta, train_sets, test_sets, prior_sets = \
            reinitialize_model(
            ds, model, class_aggregation=class_aggregation)

        scores.append(cls.models.score_models([[model]], train_sets, test_sets))

    logger.info('Testing of model finished.')

    if len(scores) > 0:
        return pd.concat(scores, axis=0)
    else:
        return None


@config.parse_config(['features'])
def run_prediction(ds, images='all', model=None,
                   model_dataset=None, colorspace='rgb',
                   feature_blocks='all', overwrite=False, cfg=None):

    if model_dataset is None:
        model_dataset = ds

    if model is None:
        model = filesys.get_model_list(model_dataset)[-1]
    if type(model) is str:
        logging.info('Using model %s' % model)
        model = filesys.read_model_file(model_dataset, model)[0]
    if not hasattr(model,'predict'):
        raise IOError('Invalid model input type') 

    # create image list
    images = create_image_list(ds, images)

    # create block list
    blocks = create_feature_list(feature_blocks)

    # read feature data
    X = get_data(ds,
                 images,
                 feature_blocks=blocks)[0]

    for i, im in iterate_images(ds, images, overwrite, 'predict'):

        shp = filesys.read_export_file(
            ds, im, 'meta')['superpixel_grid']
        X[i] = np.asarray(X[i]).reshape((shp[0], shp[1], -1))

        # run prediction
        classes = model.predict([X[i]])[0]

        # save raw data
        filesys.write_export_file(ds, im, 'predict', classes)

        # save plot
        fig, axs = plot.plot_prediction(ds, im, classes)
        plot.save_figure(
            fig, im, ext='.classes', figsize=(1.30*1392, 1.30*1024))


def plot_predictions(ds, model, meta, test_sets, part=0, class_aggregation=None):
 
    if model is list:
        model = model[part]
    elif part > 0:
        raise IOError

    classlist = filesys.read_default_categories(ds)
    classlist = cls.utils.aggregate_classes(np.array(classlist), class_aggregation)
    classlist = list(np.unique(classlist))

    eqinds = resolve_indices(ds, test_sets[0][1], meta, class_aggregation)

    n = np.shape(test_sets)[-1]

    fig,axs = plt.subplots(n,3,figsize=(20,10 * round(n / 2)))

    cdict = {
        'red'  :  ((0., .5, .5), (.2, .5, 1.), (.4, 1., .66),
                   (.6, .66, .13), (.8, .13, .02), (1., .02, .02)),
        'green':  ((0., .5, .5), (.2, .5, .95), (.4, .95, 1.),
                   (.6, 1., .69), (.8, .69, .24), (1., .24, .24)),
        'blue' :  ((0., .5, .5), (.2, .5, 0.), (.4, 0., 1.),
                   (.6, 1., .30), (.8, .30, .75), (1., .75, .75))
        }
    cmap_argus = matplotlib.colors.LinearSegmentedColormap('argus_classes', cdict, 5)

    for i, fname in enumerate(meta['images_test'][:-1]):
        if any(eqinds[:,1] == i):
            gind = np.where(eqinds[:,1] == i)[0][0]
            grnd = test_sets[part][1][gind]
            pred = model.predict([test_sets[part][0][gind]])[0]
            score = float(np.sum(pred == grnd)) / np.prod(grnd.shape) * 100
        
            img = filesys.read_image_file(ds,fname)
            axs[i,0].imshow(img)

            plot.plot_prediction(ds,
                                 fname,
                                 grnd,
                                 cm=cmap_argus,
                                 clist=classlist,
                                 axs=axs[i,1])

            plot.plot_prediction(ds,
                                 fname,
                                 pred,
                                 cm=cmap_argus,
                                 clist=classlist,
                                 axs=axs[i,2])
        
            axs[i,0].set_title(fname)
            axs[i,1].set_title('groundtruth')
            axs[i,2].set_title('prediction (%0.1f%%)' % score)

    return fig,axs
        

def resolve_indices(ds, Y, meta, agg):
    eqinds = np.empty((len(Y),2))
    for i in range(len(Y)):
        for j in range(len(meta['images_test'])):
            Y = filesys.read_export_file(ds, meta['images_test'][j],'classes')
            if Y:
                metim = filesys.read_export_file(ds, meta['images_test'][j],'meta')
                Ya = np.array(Y)
                if np.prod(metim['superpixel_grid']) == len(Ya):
                    Yr = Ya.reshape((metim['superpixel_grid']))
                    Ya = utils.aggregate_classes(Yr,agg)
                    if np.prod(Yr.shape) == np.prod(Ytest[i].shape):
                        if np.all(Yr == Ytest[i]):
                            eqinds[i,:] = np.array([i,j])
                            break

    return eqinds


@config.parse_config(['segmentation'])
def run_segmentation(ds, images=[], method='slic', method_params={},
                     extract_contours=False, remove_disjoint=True,
                     overwrite=False, cfg=None):

    logger.info('Segmentation started...')

    for i, im in iterate_images(ds, images, overwrite=overwrite, ext='segments'):

        img = filesys.read_image_file(ds, im)

        segments, contours = seg.superpixels.get_segmentation(
            img, method=method, method_params=method_params,
            extract_contours=extract_contours,
            remove_disjoint=remove_disjoint)

        nx, ny = seg.superpixels.get_superpixel_grid(
            segments, img.shape[:2])
        err = not seg.superpixels.check_segmentation(segments, nx, ny)

        meta = {'image_resolution_cropped': img.shape,
                'superpixel_grid': (nx, ny),
                'superpixel_grid_error': err,
                'last_segmented': time.strftime('%d-%b-%Y %H:%M')}

        filesys.write_export_file(ds, im, 'meta', meta, append=True)
        filesys.write_export_file(ds, im, 'segments', segments)
        filesys.write_export_file(ds, im, 'contours', contours)

    logger.info('Segmentation finished.')


@config.parse_config(['channels'])
def run_channel_extraction(ds, images=[], colorspace='rgb',
                           methods=['gabor', 'gaussian', 'sobel'],
                           methods_params=None, overwrite=False,
                           cfg=None):
    logger.info('Channel extraction started...')

    stats = [{'max': 0., 'min': 255.}
             for i in range(channels.get_number_channels(methods_params=methods_params, methods=methods))]

    for i, im in iterate_images(ds, images, overwrite, ['channels']):

        img = filesys.read_image_file(ds, im)

        # img is now [i,j,rgb]:                   grayscale channels   extra_channels
        # img becomes [i,j, channels] channels -> gray      (r g b)    (gabor sigmadiff)
        img = channels.add_channels(
            img, colorspace, methods=methods,
            methods_params=methods_params)
        filesys.write_export_file(ds, im, 'channels', img)

        stats = [{'max': np.max([stats[j]['max'], img[:,:,j+4].max()]),
                  'min': np.min([stats[j]['min'], img[:,:,j+4].min()])}
                 for j in range(img.shape[-1] - 4)]
    
    filesys.write_log_file(ds, {'channelstats': stats})

    logger.info('Channel extraction finished.')


@config.parse_config(['channels'])
def run_channel_normalization(ds, images=[], model_dataset=None,
                              overwrite=False, cfg=None):

    logger.info('Channel normalization started...')

    stats = filesys.read_log_file(
        model_dataset if model_dataset is not None else ds,
        'channelstats')
    if not stats:
        logger.info(
            'Using theoretical channel boundaries for normalization.')
        stats = channels.get_channel_bounds()

    for i, im in iterate_images(ds, images, overwrite, 'channels.normalized'):
        if filesys.check_export_file(ds, im, 'channels'):
            img = filesys.read_export_file(ds, im, 'channels')
            for j in range(4, img.shape[-1]):
                img[...,j] = channels.normalize_channel(img[...,j],
                                                        stats[i-4])
            filesys.write_export_file(ds, im, 'channels.normalized', img)

    logger.info('Channel normalization finished.')


@config.parse_config(['features'])
def run_feature_extraction(ds, images=[], feature_blocks=[],
                           colorspace='rgb', model_dataset=None,
                           overwrite=False, image_slice=1,
                           blocks_params={}, cfg=None):

    logger.info('Feature extraction started...')

    # create feature block list
    feature_blocks = create_feature_list(feature_blocks)

    for i, im in iterate_images(ds, images, overwrite,
                                ['features.%s' % re.sub('^extract_blocks_', '', k)
                                 for k in feature_blocks.keys()]):

        segments = filesys.read_export_file(ds, im, 'segments')

        if segments is None:
            logging.warning(
                'No segmentation found for image: %s' % im)
            continue

        meta = filesys.read_export_file(ds, im, 'meta')
        if meta['superpixel_grid_error']:
            logging.warning(
                'Invalid segmentation found for image: %s' % im)
            continue

        # load image
        img = filesys.read_export_file(ds, im, 'channels.normalized')
        if img is None:
            img = filesys.read_export_file(ds, im, 'channels')
        if img is None:
            img = filesys.read_image_file(ds, im)

        # extract features
        features, features_in_block = \
            cls.features.blocks.extract_blocks(img[::image_slice,::image_slice,:],
                                               segments[::image_slice,::image_slice],
                                               colorspace=colorspace,
                                               blocks=feature_blocks,
                                               blocks_params=blocks_params)

        # remove too large features
        features = cls.features.remove_large_features(features)

        # write features to disk
        filesys.write_feature_files(
            ds, im, features, features_in_block)

        meta = {
            'last feature extraction': time.strftime('%d-%b-%Y %H:%M')}
        filesys.write_export_file(ds, im, 'meta', meta, append=True)

    logger.info('Feature extraction finished.')


@config.parse_config(['features', 'relative_location_prior'])
def run_feature_update(ds, images=[], feature_blocks=[],
                       class_aggregation=None,
                       relative_location_prior=False,
                       overwrite=False, cfg=None):
    logger.info('Updating extracted features started...')

    # create feature block list
    feature_blocks = create_feature_list(feature_blocks)

    if relative_location_prior:
        maps = filesys.read_export_file(
            ds, None, 'relative_location_maps')

    for i, im in iterate_images(ds, images, overwrite,
                                ['features.linear.%s' % re.sub('^extract_blocks_', '', k)
                                 for k in feature_blocks.keys()]):

        # load image and features
        img = filesys.read_image_file(ds, im)
        features, features_in_block = filesys.read_feature_files(
            ds, im, feature_blocks.keys())

        # include relative location feature if requested
        if relative_location_prior:
            try:
                logger.info('Add relative location votes')

                Iann = filesys.read_export_file(ds, im, 'classes')
                meta = filesys.read_export_file(ds, im, 'meta')
                nx, ny = meta['superpixel_grid']
                nm, nn = meta['image_resolution_cropped'][:-1]
                Iann = np.reshape(Iann, meta['superpixel_grid'])

                centroids = filesys.read_feature_files(
                    ds, im, ['pixel'])[0].ix[:, 'centroid']
                Iann = cls.utils.aggregate_classes(
                    np.asarray(Iann), class_aggregation)

                votes = relativelocation.vote_image(
                    Iann, maps, centroids, (nm, nn))[0]

                features, features_in_block = \
                    relativelocation.add_features(
                    votes, features, features_in_block)
                filesys.write_feature_files(
                    ds, im, features, features_in_block)
            except:
                logging.warning(
                    'Adding relative location votes failed, using zeros')
                features = relativelocation.remove_features(
                    features, maps.keys())
                features_in_block['relloc'] = [
                    'prob_%s' % c for c in maps.keys()]

            meta = {'last relative location voting':
                    time.strftime('%d-%b-%Y %H:%M')}
            filesys.write_export_file(
                ds, im, 'meta', meta, append=True)

        # make features scale invariant
        logger.info('Make features scale invariant')
        features = cls.features.scaleinvariant.scale_features(
            img, features)
        filesys.write_feature_files(
            ds, im, features, features_in_block, ext='invariant')

        # linearize features
        logger.info('Linearize features')
        features = cls.features.linearize(features)
        features_in_block = cls.features.extend_feature_blocks(
            features, features_in_block)
        filesys.write_feature_files(
            ds, im, features, features_in_block, ext='linear')

        # get feature stats for image
        logger.info('Compute feature statistics')
        imstats = cls.features.normalize.compute_feature_stats(features)

        meta = {'stats': imstats,
                'last stats computation': time.strftime('%d-%b-%Y %H:%M')}
        filesys.write_export_file(ds, im, 'meta', meta, append=True)

    logger.info('Updating extracted features finished.')


@config.parse_config(['features'])
def run_feature_normalization(ds, images=[], feature_blocks=[],
                              model_dataset=None, feature_stats=None,
                              overwrite=False, cfg=None):
    logger.info('Normalizing features started...')

    logger.info('Aggregate feature statistics')
    
    if feature_stats is None:
        if model_dataset is not None and model_dataset != ds:
            images_model = filesys.get_image_list(model_dataset)
        else:
            images_model = images
        allstats = [filesys.read_export_file(ds, im, 'meta')['stats']
                    for im in images_model]
        feature_stats = \
            cls.features.normalize.aggregate_feature_stats(allstats)
        l = {'stats': feature_stats,
             'last stats computation': time.strftime('%d-%b-%Y %H:%M')}
        filesys.write_log_file(ds, l)

    # create feature block list
    feature_blocks = create_feature_list(feature_blocks)

    for i, im in iterate_images(ds, images, overwrite,
                                ['features.normalized.%s' % re.sub('^extract_blocks_', '', k)
                                 for k in feature_blocks.keys()]):

        feature_stats = filesys.read_log_file(ds, keys='stats')

        features, features_in_block = filesys.read_feature_files(
            ds, im, feature_blocks.keys() + ['relloc'], ext='linear')
        features = \
            cls.features.normalize.normalize_features(
            features, feature_stats)
        filesys.write_feature_files(
            ds, im, features, features_in_block, ext='normalized')

        meta = {'last normalized': time.strftime('%d-%b-%Y %H:%M')}
        filesys.write_export_file(ds, im, 'meta', meta, append=True)

    logger.info('Normalizing features finished.')


@config.parse_config(['relative_location_prior'])
def run_relative_location_mapping(ds, n=100, sigma=2,
                                  class_aggregation=None, cfg=None):
    logger.info('Computing relative location maps started...')

    # get image list
    images = filesys.get_image_list(ds)

    # loop over training samples
    maplist = []
    for i, im in iterate_images(ds, images):

        if not filesys.is_classified(ds, im):
            logging.warning(
                'Image %s not annotated, skipped' % im)
            continue

        annotations = filesys.read_export_file(ds, im, 'classes')
        meta = filesys.read_export_file(ds, im, 'meta')
        nx, ny = meta['superpixel_grid']
        nm, nn = meta['image_resolution_cropped'][:-1]

        if not len(annotations) == nx * ny:
            logging.warning(
                'Size mismatch for image %s, skipped' % im)
            continue

        centroids = filesys.read_feature_files(
            ds, im, ['pixel'])[0].ix[:, 'centroid']
        annotations = cls.utils.aggregate_classes(
            np.asarray(annotations), class_aggregation)

        maplist.append(relativelocation.compute_prior(annotations,
                                                      centroids,
                                                      (nm, nn),
                                                      (nx, ny),
                                                      n=n))

    maps = relativelocation.aggregate_maps(maplist)
    maps = relativelocation.smooth_maps(maps, sigma=sigma)
    maps = relativelocation.panel_to_dict(maps)

    filesys.write_export_file(ds, None, 'relative_location_maps', maps)

    l = {'last relative location prior computation':
         time.strftime('%d-%b-%Y %H:%M')}
    filesys.write_log_file(ds, l)

    logger.info('Computing relative location maps finished.')


@config.parse_config(['features', 'regularization'])
def run_regularization(ds, images='all', feature_blocks='all',
                       model_type='LR', class_aggregation=None,
                       partition=0, C=[.01,.03,.3,.7],
                       cfg=None):
    logger.info('Optimization of regularization coefficient started...')

    if not type(model_type) is list:
        model_type = [model_type]
    
    if not type(partition) is list:
        partition = [partition]

    if not type(C) is list:
        C = [C]

    # create feature block list
    feature_blocks = create_feature_list(feature_blocks)

    scores = pd.DataFrame(data=np.empty((len(C),2)), 
                          columns=['Train', 'Test'], 
                          index=C)

    # loop over range of regularization coefficients,
    # train model and determine score
    for i, cval in enumerate(C):

        # initialize models, training and test sets
        logger.info(
            'Regularization value %d of %d: C = %f' % (
                i, len(C), cval))
        
        logger.info('Preparing model...')
        models, meta, train_sets, \
            test_sets, prior_sets = initialize_models(
            ds,
            images,
            feature_blocks,
            model_type=model_type,
            class_aggregation=class_aggregation,
            partitions=partition,
            C=cval)

        # fit models
        logger.info('Fitting model...')
        models = cls.models.train_models(
            models, train_sets, prior_sets,
            callback=lambda m,(i,j): filesys.write_model_file(
                ds, m, meta[i][j], ext='.%d.backup' % (partition[0])))
        filesys.write_model_files(ds, models, meta)
        
    logger.info(
        'Finished fitting models with various regularization coefficients.')


def initialize_models(ds, images='all', feature_blocks='all',
                      model_type='LR', class_aggregation=None,
                      partitions='all', C=1.0):

    # create image list
    images = create_image_list(ds, images)

    # create feature block list
    feature_blocks = create_feature_list(feature_blocks)

    # retreive train test partitions
    dslog = filesys.read_log_file(
        ds, keys=['training images', 'testing images'])
    if not dslog:
        msg = 'Train and test partitions not found'
        logger.error(msg)
        raise ValueError(msg)

    images_train = dslog['training images']
    images_test = dslog['testing images']

    # get data
    X, Y, X_rlp = get_data(ds,
                           images,
                           feature_blocks=feature_blocks)

    # aggregate classes
    if class_aggregation is not None:
        logger.info('Aggregate classes...')
        Y = cls.utils.aggregate_classes(Y, class_aggregation)

    # create category list
    classes = cls.utils.get_classes(Y)

    # read relative location data
    if filesys.check_export_file(ds, None, 'relative_location_maps'):
        rlp_maps = filesys.read_export_file(
            ds, None, 'relative_location_maps')
        rlp_stats = filesys.read_log_file(ds, keys='stats')
    else:
        rlp_maps = None
        rlp_stats = None

    # number of features
    n_features = len(X[0].columns)

    # create partitions list
    partitions = create_partitions_list(partitions, len(images_train))

    # construct models
    if not type(model_type) is list:
        model_type = [model_type]

    models = [cls.models.get_model(model_type=m,
                                   n_states=len(classes),
                                   n_features=n_features,
                                   rlp_maps=rlp_maps,
                                   rlp_stats=rlp_stats,
                                   C=C) for m in model_type]

    # construct data arrays from dataframes and partitions
    train_sets, test_sets, prior_sets = \
        features_to_input(ds,
                          images,
                          images_train,
                          images_test,
                          X,
                          Y,
                          X_rlp,
                          partitions=partitions)

    # collect meta information
    meta = [[{'dataset': ds,
              'images': list(images),
              'images_train': list(images_train[i]),
              'images_test':list(images_test[i]),
              'feature_blocks':[re.sub('^extract_blocks_', '', x)
                                for x in feature_blocks.keys()],
              'features':list(X[0].columns),
              'classes':list(classes),
              'model_type':m} for i in partitions] for m in model_type]

    return models, meta, train_sets, test_sets, prior_sets


def reinitialize_model(ds, model, class_aggregation=None):

    models, meta = filesys.read_model_file(ds, model)

    X, Y, X_rlp = get_data(ds,
                           meta['images'],
                           feature_blocks=meta['feature_blocks'])

    # aggregate classes
    if class_aggregation is not None:
        logger.info('Aggregate classes...')
        Y = cls.utils.aggregate_classes(Y, class_aggregation)

    # construct data arrays from dataframes and partitions
    train_sets, test_sets, prior_sets = \
        features_to_input(ds,
                          meta['images'],
                          [meta['images_train']],
                          [meta['images_test']],
                          X,
                          Y,
                          X_rlp)

    return models, meta, train_sets, test_sets, prior_sets


def get_data(ds, images=[], feature_blocks=[]):
    X = []
    Y = []
    X_rlp = []

    if type(feature_blocks) is dict:
        feature_blocks = feature_blocks.keys()

    logger.info('Preparation of features and labels started...')
    for i, im in iterate_images(ds, images):

        meta = filesys.read_export_file(ds, im, 'meta')

        if meta is not None:
            if meta.has_key('superpixel grid error'):
                if meta['superpixel grid error']:
                    continue

        # read feature blocks
        X.append(filesys.read_feature_files(
            ds, im, feature_blocks, ext='normalized')[0])
        X_rlp.append(
            filesys.read_feature_files(
                ds, im, ['relloc'], ext='normalized')[0])

        # load classes and append as array
        classes = filesys.read_export_file(ds, im, 'classes')
        Y.append(np.asarray(classes))

    logger.info('Preparation of features and labels finished.')

    return X, Y, X_rlp


def create_image_list(ds, images):

    images_all = filesys.get_image_list(ds)

    if images == 'all':
        images = images_all

    imlist = []
    for im in images:
        if im in images_all:
            imlist.append(im)
        else:
            if os.path.exists(im):
                with open(im, 'r') as fp:
                    for line in fp:
                        if line.strip() in images_all:
                            imlist.append(line.strip())
            else:
                logger.warn(
                    'Image not found in dataset "%s": %s' % (ds, im))

    return imlist


def create_feature_list(feature_blocks):

    allfb = cls.features.blocks.list_blocks()

    if type(feature_blocks) is str:
        if feature_blocks == 'all':
            feature_blocks = allfb
        else:
            feature_blocks = feature_blocks.split(',')

    if type(feature_blocks) is dict:
        return feature_blocks
    elif type(feature_blocks) is list:
        feature_blocks = {
            k: v
            for k, v in allfb.iteritems()
            if k in feature_blocks or k.replace(
                'extract_blocks_', '') in feature_blocks}
    else:
        msg = 'Features should be a list of feature block names or the keyword "all"'
        logger.warn(msg)
        raise ValueError(msg)

    return feature_blocks


def create_partitions_list(partitions, n=5):

    if type(partitions) is str:
        if partitions == 'all':
            partitions = range(n)
        else:
            partitions = [int(x) for x in partitions.split(',')]

    return partitions


def create_feature_collection(feature_blocks):
    if not type(feature_blocks) is list:
        feature_blocks = [feature_blocks]

    try:
        for i, block in enumerate(feature_blocks):
            feature_blocks[i] = create_feature_list(block)
    except:
        feature_blocks = [create_feature_list(feature_blocks)]

    return feature_blocks


def train_test_part(images, frac_test=0.25):
    # Wrapper for sklearn function cross_validation.train_test_split

    images_train, images_test = train_test_split(
        images, test_size=frac_test)

    return images_train, images_test


def multiple_partitions(images, n_part=5, frac_test=0.25):
    # Make multiple train-test partitions for statistical model
    # performance check

    images_train = []
    images_test = []

    lists = [images_train, images_test]

    for i in range(n_part):
        for lst, part in zip(lists, train_test_part(images, frac_test)):
            lst.append(part)

    return {'training images': images_train, 'testing images': images_test}


def features_to_input(ds, images, images_train, images_test,
                       X, Y, X_rlp=[], partitions='all'):

    train_sets = []
    test_sets = []
    prior_sets = []
    
    # create partitions list
    partitions = create_partitions_list(partitions, len(images_train))

    for i in partitions:

        X_train, X_test, \
            Y_train, Y_test, \
            X_train_prior, X_test_prior = split_data(ds,
                                                     images,
                                                     images_train[i],
                                                     images_test[i],
                                                     X,
                                                     Y,
                                                     X_rlp)

        train_sets.append((np.asarray(X_train), np.asarray(Y_train)))
        test_sets.append((np.asarray(X_test), np.asarray(Y_test)))
        prior_sets.append(
            (np.asarray(X_train_prior), np.asarray(X_test_prior)))

    return train_sets, test_sets, prior_sets


def split_data(ds, images, images_train, images_test, X, Y, X_rlp=[]):

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_train_prior = []
    X_test_prior = []

    for i, im in enumerate(images):
        meta = filesys.read_export_file(ds, im, 'meta')
        shp = meta['superpixel_grid']

        if not np.prod(shp) == np.prod(np.asarray(Y[i]).shape):
            continue

        Xi = np.asarray(X[i]).reshape((shp[0], shp[1], -1))
        Yi = np.asarray(Y[i]).reshape(shp)

        if len(X_rlp) > 0:
            Xi_rlp = np.asarray(X_rlp[i]).reshape((shp[0], shp[1], -1))
        else:
            Xi_rlp = None

        if im in images_train:
            X_train.append(Xi)
            Y_train.append(Yi)
            X_train_prior.append(Xi_rlp)
        if im in images_test:
            X_test.append(Xi)
            Y_test.append(Yi)
            X_test_prior.append(Xi_rlp)

    return X_train, X_test, Y_train, Y_test, X_train_prior, X_test_prior


def iterate_images(ds, images, overwrite=False, ext=None):
    if type(ext) is not list:
        ext = [ext]

    for i, im in enumerate(images):
        if overwrite or not np.all([
                filesys.check_export_file(ds, im, e) for e in ext]):
            logger.info('Processing image %d of %d: %s' % (i+1, len(images), im))
            yield i, im
        else:
            logger.info('Skipped image %d of %d: %s' % (i+1, len(images), im))
            continue


def main():
    import docopt

    arguments = docopt.docopt(__doc__)

    # set verbose logging format
    if arguments['--verbose']:
        logging.basicConfig(
            format='%(asctime)-15s %(name)-8s %(levelname)-8s %(message)s')
        logging.root.setLevel(logging.NOTSET)

    cfg = config.read_config(arguments['--config'])

    if arguments['--images']:
        images = arguments['--images'].split(',')
    else:
        images = 'all'

    if arguments['--partitions']:
        partitions = [int(x) for x in arguments['partitions'].split(',')]
    else:
        partitions = 'all'

    if arguments['preprocess']:
        run_preprocessing(
            arguments['<dataset>'],
            images=images,
            segmentation=arguments['--segmentate'],
            channels=arguments['--channels'],
            features=arguments['--features'],
            extract=arguments['--extract'],
            update=arguments['--update'],
            normalize=arguments['--normalize'],
            relative_location_prior=arguments['--relloc'],
            relative_location_maps=arguments['--relloc_maps'],
            overwrite=arguments['--overwrite'],
            cfg=cfg
        )
    
    if arguments['partition']:
        run_partitioning(
            arguments['<dataset>'],
            n_partitions=int(arguments['--n']),
            frac_validation=0.0,
            frac_test=float(arguments['--frac']),
            force_split=True,
            cfg=cfg
        )
    
    if arguments['train']:
        run_training(
            arguments['<dataset>'],
            images=images,
            model_type=arguments['--type'],
            partitions=partitions,
            cfg=cfg
        )

    if arguments['score']:
        scores = run_scoring(
            arguments['<dataset>'],
            models=arguments['--model'],
            cfg=cfg
        )

        if scores is not None:
            print scores.to_string()
            print pd.DataFrame(scores.mean(), columns=['mean']).T.to_string()
        else:
            print 'No scoring results'

    if arguments['predict']:
        run_prediction(
            arguments['<dataset>'],
            images=images,
            model=arguments['--model'],
            overwrite=arguments['--overwrite'],
            cfg=cfg
        )

    if arguments['regularization']:
        run_regularization(
            arguments['<dataset>'],
            model_type=arguments['--type'],
            cfg=cfg
        )


if __name__ == '__main__':
    main()
