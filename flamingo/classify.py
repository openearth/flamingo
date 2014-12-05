import time
import re
import os
import logging
import json
import pandas

import numpy as np
import matplotlib.pyplot as plt
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
                      channel_extract=False,
                      channel_normalize=False,
                      feat_extract=False,
                      feat_update=False,
                      feat_normalize=False,
                      feature_blocks='all',
                      colorspace='rgb',
                      relloc_maps=False,
                      relloc=False,
                      overwrite=False,
                      class_aggregation=None,
                      cfg=None):
    '''Batch function to preprocess a dataset'''

    logger.info('Preprocessing started for dataset "%s"...' % ds)

    # make sure log file exists
    filesys.write_log_file(
        ds, {'start_preprocess': time.strftime('%d-%b-%Y %H:%M')})

    # create image list
    images = _create_imlist(ds, images)

    # create feature block list
    feature_blocks = _create_featlist(feature_blocks)

    # segmentation
    if segmentation:
        args = config.get_function_args(run_segmentation, cfg, 'segmentation')
        run_segmentation(ds, images, overwrite=overwrite, **args)
        
    # channel extraction
    if channel_extract:
        run_channel_extraction(ds, images, colorspace=colorspace)

    # channel normalization
    if channel_normalize:
        run_channel_normalization(ds, images, model_dataset=model_dataset)

    # feature extraction first part
    if feat_extract:
        run_feature_extraction(
            ds, images, feature_blocks=feature_blocks, colorspace=colorspace,
            model_dataset=model_dataset, overwrite=overwrite)

    # relative location prior maps, requires centroids from feature extraction
    if relloc and relloc_maps:
        run_relative_location_mapping(
            ds, n=100, sigma=2, class_aggregation=class_aggregation)

    # feature extraction, second part, requires relative location prior maps
    if feat_update:
        run_feature_update(ds, images, feature_blocks=feature_blocks,
                           class_aggregation=class_aggregation, relloc=relloc)

    # normalize features
    if feat_normalize:
        run_feature_normalization(ds, images, feature_blocks=feature_blocks, model_dataset=model_dataset)

    logger.info('Preprocessing finished.')


def run_partitioning(ds,
                     images='all',
                     n_partitions=5,
                     frac_validation=0.0,
                     frac_test=0.25,
                     force_split=False):

    images = _create_imlist(ds, images)

    logger.info('Defining %d train/test partitions' % n_part)

    # make train test partitions if requested
    dslog = filesys.read_log_file(ds)
    if forcesplit or not dslog.has_key('training images'):
        images_part = _multiple_partitions(images, n_part, test_frac)
        filesys.write_log_file(ds, images_part)


def run_classification(ds,
                       images='all',
                       feature_blocks='all',
                       model_type='LR',
                       class_aggregation=None,
                       partitions='all'):
    ''' Batch function to train and test a PGM with multiple train_test partitions '''

    logger.info('Classification started for dataset "%s"...' % ds)

    if not type(modtype) is list:
        modtype = [modtype]

    # initialize models, training and test sets
    logger.info('Preparation of models, train and test sets started...')
    models, meta, train_sets, test_sets, prior_sets = initialize_models(
        ds, images, feature_blocks, modtype=modtype,
        class_aggregation=class_aggregation, partition_start=partition_start)
    logger.info('Preparation of models, train and test sets finished.')

    # fit models
    logger.info('Fitting of models started...')
    models = cls.models.train_models(models, train_sets, prior_sets,
                                     callback=lambda m,(i,j): filesys.write_model_file(ds, m, meta[i][j], ext='.%d.backup' % (partition_start + j)))
    filesys.write_model_files(ds, models, meta)
    logger.info('Fitting of models finished.')

    return models, train_sets, test_sets

 
def run_scoring(ds, model=None, class_aggregation=None):

    if model is None:
        model = filesys.get_model_list(ds)[-1]

    model, meta, train_sets, test_sets, prior_sets = reinitialize_model(
        ds, model, class_aggregation=class_aggregation)

    logger.info('Testing of model started...')
    scores = cls.models.score_models([[model]], train_sets, test_sets)
    logger.info('Testing of model finished.')

    return scores


def run_prediction(ds, images='all', model=None,
                   model_dataset=None, colorspace='rgb', blocks='all', overwrite=False):

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
    images = _create_imlist(ds, images)

    # create block list
    blocks = _create_featlist(blocks)

    # read feature data
    X = get_data(ds,
                 images,
                 feature_blocks=blocks)[0]

    for i, im in enumerate(images):

        if not overwrite and os.path.isfile(filesys.get_export_file(ds, im, 'predict')):
            logger.info('Skipping image %d of %d: %s. Already predicted' % (i + 1, len(images), im))
            continue

        logger.info('Predicting image %d of %d: %s' % (i + 1, len(images), im))

        shp = filesys.read_export_file(ds, im, 'meta')['superpixel_grid']
        X[i] = np.asarray(X[i]).reshape((shp[0], shp[1], -1))

        # run prediction
        classes = model.predict([X[i]])[0]

        # save raw data
        filesys.write_export_file(ds, im, 'predict', classes)

        # save plot
        fig, axs = plot.plot_prediction(ds, im, classes)
        plot.save_figure(fig, im, ext='.classes', figsize=(1.30*1392, 1.30*1024))


def run_segmentation(ds, images=[], method='slic', method_params={},
                     extract_contours=False, remove_disjoint=True, overwrite=False):

    logger.info('Segmentation started...')

    for i, im in enumerate(images):

        if not overwrite and filesys.check_export_file(ds, im, 'segments'):
            logger.info(
                'Skipping image %d of %d: %s. Already segmented' % (i+1,
                                                                    len(images), im))
            continue

        logger.info('Processing image %d of %d: %s' % (i+1, len(images), im))

        img = filesys.read_image_file(ds, im)

        segments, contours = seg.superpixels.get_segmentation(
            img, method=method, method_params=method_params,
            extract_contours=extract_contours, remove_disjoint=remove_disjoint)

        nx, ny = seg.superpixels.get_superpixel_grid(segments, img.shape[:2])
        err = not seg.superpixels.check_segmentation(segments, nx, ny)

        meta = {'image_resolution_cropped': img.shape,
                'superpixel_grid': (nx, ny),
                'superpixel_grid_error': err,
                'last_segmented': time.strftime('%d-%b-%Y %H:%M')}

        filesys.write_export_file(ds, im, 'meta', meta, append=True)
        filesys.write_export_file(ds, im, 'segments', segments)
        filesys.write_export_file(ds, im, 'contours', contours)

    logger.info('Segmentation finished.')


def run_channel_extraction(ds, images=[], colorspace='rgb'):
    logger.info('Channel extraction started...')

    stats = [{'max': 0., 'min': 255.}
             for i in range(channels.get_number_channels())]

    for i, im in enumerate(images):
        logger.info('Processing image %d of %d: %s' % (i+1, len(images), im))
        img = filesys.read_image_file(ds,im)

        # img is now [i,j,rgb]:                   grayscale channels   extra_channels
        # img becomes [i,j, channels] channels -> gray      (r g b)    (gabor sigmadiff)
        img = channels.add_channels(img, colorspace)
        filesys.write_export_file(ds, im, 'channels', img)

        stats = [{'max': np.max([stats[j]['max'], img[:,:,j+4].max()]),
                  'min': np.min([stats[j]['min'], img[:,:,j+4].min()])}
                 for j in range(img.shape[-1] - 4)]
    
    filesys.write_log_file(ds, {'channelstats': stats})

    logger.info('Channel extraction finished.')


def run_channel_normalization(ds, images=[], model_dataset=None):

    stats = filesys.read_log_file(model_dataset
                                  if model_dataset is not None else ds, 'channelstats')
    if not stats:
        logger.info('Using theoretical channel boundaries for normalization.')
        stats = channels.get_channel_bounds()

    for i, im in enumerate(images):
        logger.info('Processing image %d of %d: %s' % (i + 1, len(images), im))
        img = filesys.write_export_file(ds, im, 'channels')
        for j in range(4, img.shape[-1]):
            img[...,j] = channels.normalize_channel(img[...,j], stats[i-4])
        filesys.write_export_file(ds, im, 'channels.normalized', img)


def run_feature_extraction(ds, images=[], feature_blocks=[],
                           colorspace='rgb', model_dataset=None, overwrite=False):
    logger.info('Feature extraction started...')
    for i, im in enumerate(images):
        if not overwrite and all([os.path.isfile(filesys.get_export_file(ds,im,'features.' + re.findall('(?<=extract_blocks_).*',k)[0])) for k,v in feature_blocks.iteritems()]):
            logger.info('Skipping image %d of %d: %s. Features already extracted' % (i + 1, len(images), im))
            continue
        logger.info('Processing image %d of %d: %s' % (i + 1, len(images), im))

        segments = filesys.read_export_file(ds, im, 'segments')

        if segments is None:
            logging.warning('No segmentation found for image: %s' % im)
            continue

        meta = filesys.read_export_file(ds, im, 'meta')
        if meta['superpixel_grid_error']:
            logging.warning('Invalid segmentation found for image: %s' % im)
            continue

        # load image
        img = filesys.read_image_file(ds, im)

        # extract features
        features, features_in_block = \
            cls.features.blocks.extract_blocks(img,
                                               segments,
                                               colorspace=colorspace,
                                               blocks=feature_blocks)

        # remove too large features
        features = cls.features.remove_large_features(features)

        # write features to disk
        filesys.write_feature_files(ds, im, features, features_in_block)

        meta = {'last feature extraction': time.strftime('%d-%b-%Y %H:%M')}
        filesys.write_export_file(ds, im, 'meta', meta, append=True)

    logger.info('Feature extraction finished.')


def run_feature_update(ds, images=[], feature_blocks=[],
                       class_aggregation=None, relloc=False):
    logger.info('Updating extracted features started...')

    if relloc:
        maps = filesys.read_export_file(ds, None, 'relative_location_maps')

    for i, im in enumerate(images):
        logger.info('Processing image %d of %d: %s' % (i + 1, len(images), im))

        # load image and features
        img = filesys.read_image_file(ds, im)
        features, features_in_block = filesys.read_feature_files(
            ds, im, feature_blocks.keys())

        # include relative location feature if requested
        if relloc:
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

                features, features_in_block = relativelocation.add_features(
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
            filesys.write_export_file(ds, im, 'meta', meta, append=True)

        # make features scale invariant
        logger.info('Make features scale invariant')
        features = cls.features.scaleinvariant.scale_features(img, features)
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


def run_feature_normalization(ds, images=[], feature_blocks=[],
                              model_dataset=None, stats=None):
    logger.info('Normalizing features started...')

    logger.info('Aggregate feature statistics')
    
    if stats is None:
        if model_dataset is not None and model_dataset != ds:
            images_model = filesys.get_image_list(model_dataset)
        else:
            images_model = images
        allstats = [filesys.read_export_file(ds, im, 'meta')['stats']
                    for im in images_model]
        stats = cls.features.normalize.aggregate_feature_stats(allstats)
        l = {'stats': stats,
             'last stats computation': time.strftime('%d-%b-%Y %H:%M')}
        filesys.write_log_file(ds, l)

    for i, im in enumerate(images):
        logger.info('Processing image %d of %d: %s' % (i + 1, len(images), im))
        stats = filesys.read_log_file(ds, keys='stats')

        features, features_in_block = filesys.read_feature_files(
            ds, im, feature_blocks.keys() + ['relloc'], ext='linear')
        features = cls.features.normalize.normalize_features(features, stats)
        filesys.write_feature_files(
            ds, im, features, features_in_block, ext='normalized')

        meta = {'last normalized': time.strftime('%d-%b-%Y %H:%M')}
        filesys.write_export_file(ds, im, 'meta', meta, append=True)

    logger.info('Normalizing features finished.')


def run_relative_location_mapping(ds, n=100, sigma=2, class_aggregation=None):
    logger.info('Computing relative location maps started...')

    # get image list
    images = filesys.get_image_list(ds)

    # loop over training samples
    maplist = []
    for k, im in enumerate(images):

        if not filesys.is_classified(ds, im):
            continue

        logger.info('Processing image %d of %d: %s' % (k, len(images), im))

        annotations = filesys.read_export_file(ds, im, 'classes')
        meta = filesys.read_export_file(ds, im, 'meta')
        nx, ny = meta['superpixel_grid']
        nm, nn = meta['image_resolution_cropped'][:-1]

        if not len(annotations) == nx * ny:
            logging.warning('Size mismatch for image %s, skipped' % im)
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


def run_regularization(ds, images='all', feature_blocks='all', modtype='LR',
                       class_aggregation=None, partition=0, C=[.1,1,10,100,1000,10000]):
    logger.info('Optimization of regularization coefficient started...')

    if not type(modtype) is list:
        modtype = [modtype]
        
    if not type(partition) is list:
        partition = [partition]

    if not type(C) is list:
        C = [C]

    scores = pandas.DataFrame(data=np.empty((len(C),2)), columns=['Train', 'Test'], index=C)
    # loop over range of regularization coefficients, train model and determine score
    for i,cval in enumerate(C):
        # initialize models, training and test sets
        logger.info('Regularization value %d of %d: C = %f' % (i, len(C), cval))
        
        logger.info('Preparing model...')
        models, meta, train_sets, test_sets, prior_sets = initialize_models(
            ds,
            images,
            feature_blocks,
            modtype=modtype,
            class_aggregation=class_aggregation,
            partition_start=partition,
            C=cval)

        # fit models
        logger.info('Fitting model...')
        models = cls.models.train_models(models, train_sets, prior_sets,
                                         callback=lambda m,(i,j): filesys.write_model_file(ds, m, meta[i][j], ext='.%d.backup' % (partition[0])))
        filesys.write_model_files(ds, models, meta)
        
    logger.info('Finished fitting models with various regularization coefficients.')


def initialize_models(ds, images='all', feature_blocks='all', modtype='LR',
                      class_aggregation=None, partition_start=0, C=1.0):

    # create image list
    images = _create_imlist(ds, images)

    # create feature block list
    feature_blocks = _create_featlist(feature_blocks)

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
    if os.path.exists(filesys.get_export_file(ds, None, 'relative_location_maps')):
        rlp_maps = filesys.read_export_file(ds, None, 'relative_location_maps')
        rlp_stats = filesys.read_log_file(ds, keys='stats')
    else:
        rlp_maps = None
        rlp_stats = None

    # number of features
    nfeat = len(X[0].columns)

    # construct models
    if not type(modtype) is list:
        modtype = [modtype]

    models = [cls.models.get_model(model_type=m,
                                   n_states=len(classes),
                                   n_features=nfeat,
                                   rlp_maps=rlp_maps,
                                   rlp_stats=rlp_stats,
                                   C=C) for m in modtype]

    # construct data arrays from dataframes and partitions
    train_sets, test_sets, prior_sets = _features_to_input(ds,
                                                           images,
                                                           images_train,
                                                           images_test,
                                                           X,
                                                           Y,
                                                           X_rlp,
                                                           partition_start=partition_start)

    if type(partition_start) is list:
        parts = partition_start
    else:
        parts = range(partition_start, len(images_train))
    # collect meta information
    meta = [[{'dataset': ds,
              'images': list(images),
              'images_train': list(images_train[i]),
              'images_test':list(images_test[i]),
              'feature_blocks':[re.sub('^extract_blocks_', '', x) for x in feature_blocks.keys()],
              'features':list(X[0].columns),
              'classes':list(classes),
              'model_type':m} for i in parts] for m in modtype]

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
    train_sets, test_sets, prior_sets = _features_to_input(ds,
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
    for i, im in enumerate(images):
        logger.info('Processing image %d of %d: %s' % (i + 1, len(images), im))

        meta = filesys.read_export_file(ds, im, 'meta')

        if meta is not None:
            if meta.has_key('superpixel grid error'):
                if meta['superpixel grid error']:
                    continue

        # read feature blocks
        X.append(filesys.read_feature_files(
            ds, im, feature_blocks, ext='normalized')[0])
        X_rlp.append(
            filesys.read_feature_files(ds, im, ['relloc'], ext='normalized')[0])

        # load classes and append as array
        classes = filesys.read_export_file(ds, im, 'classes')
        Y.append(np.asarray(classes))

    logger.info('Preparation of features and labels finished.')

    return X, Y, X_rlp


def _create_imlist(ds, images):

    images_all = filesys.get_image_list(ds)

    if images == 'all':
        images = images_all

    imlist = []
    for im in images:
        if im in images_all:
            imlist.append(im)
        else:
            logger.warn('Image not found in dataset "%s": %s' % (ds, im))
    return imlist


def _create_featlist(feature_blocks):
    allfb = cls.features.blocks.list_blocks()
    if type(feature_blocks) is list:
        feature_blocks = {k: v for k, v in allfb.iteritems() if k in feature_blocks or k.replace(
            'extract_blocks_', '') in feature_blocks}
    elif feature_blocks == 'all':
        feature_blocks = allfb
    else:
        msg = 'Features should be a list of feature block names or the keyword "all"'
        logger.warn(msg)
        raise ValueError(msg)

    return feature_blocks


def _create_featcollection(feature_blocks):
    if not type(feature_blocks) is list:
        feature_blocks = [feature_blocks]

    try:
        for i, block in enumerate(feature_blocks):
            feature_blocks[i] = _create_featlist(block)
    except:
        feature_blocks = [_create_featlist(feature_blocks)]

    return feature_blocks


def _train_test_part(images, frac_test=0.25):
    # Wrapper for sklearn function cross_validation.train_test_split

    images_train, images_test = train_test_split(images, test_size=frac_test)

    return images_train, images_test


def _multiple_partitions(images, n_part=5, frac_test=0.25):
    # Make multiple train-test partitions for statistical model performance
    # check

    images_train = []
    images_test = []

    lists = [images_train, images_test]

    for i in range(n_part):
        for lst, part in zip(lists, _train_test_part(images, frac_test)):
            lst.append(part)

    return {'training images': images_train, 'testing images': images_test}


def _features_to_input(ds, images, images_train, images_test,
                       X, Y, X_rlp=[], partition_start=0):

    train_sets = []
    test_sets = []
    prior_sets = []
    
    if type(partition_start) is list:
        parts = partition_start
    else:
        parts = range(partition_start, len(images_train))

    for i in parts:

        X_train, X_test, \
            Y_train, Y_test, \
            X_train_prior, X_test_prior = _split_data(ds,
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


def _split_data(ds, images, images_train, images_test, X, Y, X_rlp=[]):

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


def main():
    import docopt

    usage = """
Train, score and use classification models on image datasets.

Usage:
    classify-images preprocess <dataset> [--segmentate] [--channelstats] [--extract] [--update] [--normalize] [--relloc] [--relloc_maps] [--images=FILE] [--config=FILE] [--overwrite] [--verbose]
    classify-images partition <dataset> [--n=N] [--frac=FRAC] [--images=FILE] [--config=FILE] [--verbose]
    classify-images train <dataset> [--type=NAME] [--partition=N] [--images=FILE] [--config=FILE] [--verbose]
    classify-images score <dataset> [--model=NAME] [--images=FILE] [--config=FILE] [--verbose]
    classify-images predict <dataset> [--model=NAME] [--images=FILE] [--config=FILE] [--overwrite] [--verbose]
    classify-images regularization <dataset> [--type=NAME] [--partition=N] [--images=FILE] [--config=FILE] [--verbose]

Positional arguments:
    dataset            dataset containing the images
    image              image file to be classified

Options:
    -h, --help         show this help message and exit
    --segmentate       create segmentation of images
    --channelstats     compute channel statistics for normalization
    --extract          extract features
    --update           update features
    --normalize        normalize features
    --relloc           include relative location features
    --relloc_maps      compute new relative location maps
    --n=N              number of partitions [default: 5]
    --frac=FRAC        fraction of images used for testing [default: 0.25]
    --type=NAME        model type to train [default: LR]
    --partition=N      start training at this partition [default: 0]
    --model=NAME       name of model to be scored, uses last trained if omitted
    --images=FILE      images to include in process
    --config=FILE      configuration file to use instead of command line options
    --overwrite        overwrite existing files
    --verbose          print logging messages
"""

    arguments = docopt.docopt(usage)

    # set verbose logging format
    if arguments['--verbose']:
        logging.basicConfig(
            format='%(asctime)-15s %(name)-8s %(levelname)-8s %(message)s')
        logging.root.setLevel(logging.NOTSET)

    cfg = config.read_config(arguments['--config'])

    if arguments['--images']:
        # TODO: differentiate between image files, image file list and JSON shizzle
        images = arguments['--images'].split(',')
    else:
        images = 'all'

    if arguments['preprocess']:
        run_preprocessing(
            arguments['<dataset>'],
            images=images,
            segmentation=arguments['--segmentate'],
            channel_extract=arguments['--channelstats'],
            channel_normalize=arguments['--normalize'],
            feat_extract=arguments['--extract'],
            feat_update=arguments['--update'],
            feat_normalize=arguments['--normalize'],
            relloc=arguments['--relloc'],
            relloc_maps=arguments['--relloc_maps'],
            overwrite=arguments['--overwrite'],
            cfg=cfg
        )
    
    if arguments['partition']:
        run_partitioning(
            arguments['<dataset>'],
            n_part=int(arguments['--n']),
            frac_validation=0.0,
            frac_test=float(arguments['--frac']),
            force_split=True
        )
    
    if arguments['train']:
        run_classification(
            arguments['<dataset>'],
            modtype=arguments['--type'],
            partition_start=int(arguments['--partition'])
        )

    if arguments['score']:
        print run_scoring(
            arguments['<dataset>'],
            model=arguments['--model'],
        ).to_string()

    if arguments['predict']:
        run_prediction(
            arguments['<dataset>'],
            images=images,
            model=arguments['--model'],
            overwrite=arguments['--overwrite']
        )

    if arguments['regularization']:
        run_regularization(
            arguments['<dataset>'],
            modtype=arguments['--type'],
            partition=int(arguments['--partition'])
        )
