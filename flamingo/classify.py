import time
import re
import os
import logging
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from flamingo import classification as cls
from flamingo import segmentation as seg
from flamingo import filesys
from flamingo.utils import printinfo
from flamingo.classification import channels
from flamingo.classification.features import relativelocation


# initialize log
logger = logging.getLogger(__name__)


def run_preprocessing(ds,
                      images='all',
                      segmentation=False,
                      feat_extract=False,
                      feat_update=False,
                      feat_normalize=False,
                      feat_blocks='all',
                      colorspace='rgb',
                      relloc_maps=False,
                      relloc=False,
                      class_aggregation=None):
    '''Batch function to preprocess a dataset'''

    logger.info('Preprocessing started for dataset "%s"...' % ds)

    # make sure log file exists
    filesys.write_log_file(
        ds, {'start_preprocess': time.strftime('%d-%b-%Y %H:%M')})

    # create image list
    images = _create_imlist(ds, images)

    # create feature block list
    feat_blocks = _create_featlist(feat_blocks)

    # segmentation
    if segmentation:
        run_segmentation(ds, images)

    # feature extraction first part
    if feat_extract:
        run_feature_extraction(
            ds, images, feat_blocks=feat_blocks, colorspace=colorspace)

    # relative location prior maps, requires centroids from feature extraction
    if relloc and relloc_maps:
        run_relative_location_mapping(
            ds, n=100, sigma=2, class_aggregation=class_aggregation)

    # feature extraction, second part, requires relative location prior maps
    if feat_update:
        run_feature_update(ds, images, feat_blocks=feat_blocks,
                           class_aggregation=class_aggregation, relloc=relloc)

    # normalize features
    if feat_normalize:
        run_feature_normalization(ds, images, feat_blocks=feat_blocks)

    logger.info('Preprocessing finished.')


def run_partitioning(ds,
                     images='all',
                     n_part=5,
                     test_frac=.25,
                     forcesplit=False):

    images = _create_imlist(ds, images)

    logger.info('Defining %d train/test partitions' % n_part)

    # make train test partitions if requested
    dslog = filesys.read_log_file(ds)
    if forcesplit or not dslog.has_key('training images'):
        images_part = _multiple_partitions(images, n_part, test_frac)
        filesys.write_log_file(ds, images_part)


def run_classification(ds,
                       images='all',
                       feat_blocks='all',
                       modtype='LR',
                       class_aggregation=None):
    ''' Batch function to train and test a PGM with multiple train_test partitions '''

    logger.info('Classification started for dataset "%s"...' % ds)

    if not type(modtype) is list:
        modtype = [modtype]

    # initialize models, training and test sets
    logger.info('Preparation of models, train and test sets started...')
    models, meta, train_sets, test_sets, prior_sets = initialize_models(ds,
                                                                        images,
                                                                        feat_blocks,
                                                                        modtype=modtype,
                                                                        class_aggregation=class_aggregation)
    logger.info('Preparation of models, train and test sets finished.')

    # fit models
    logger.info('Fitting of models started...')
    models = cls.models.train_models(models, train_sets, prior_sets)
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


def run_prediction(ds, im, model=None, colorspace='rgb', blocks='all'):

    if model is None:
        model = filesys.get_model_list(ds)[-1]

    if os.path.exists(im):
        img = plt.imread(im)
    else:
        raise IOError('Image not found: %s' % im)

    logger.info('Prediction started...')

    # create feature block list
    blocks = _create_featlist(blocks)

    # segmentation
    logger.info('Segmentate image')
    segments_orig = seg.superpixels.get_superpixel(
        img, n_segments=600, compactness=20, sigma=0)

    # compute superpixel grid
    nx, ny = seg.superpixels.get_superpixel_grid(segments_orig, img.shape[:2])

    # fix disjoint segments
    logger.info('Remove disjoint segments')
    segments = seg.postprocess.remove_disjoint(segments_orig)

    # add channels
#    logger.info('Add image channels')
#    img = channels.add_channels(img, colorspace)

    # extract features
    logger.info('Extrect features')
    features, features_in_block = \
        cls.features.blocks.extract_blocks(img,
                                           segments,
                                           colorspace=colorspace,
                                           blocks=blocks)

    # remove too large features
    logger.info('Remove large features')
    features = cls.features.remove_large_features(features)

    # make features scale invariant
    logger.info('Make features scale invariant')
    features = cls.features.scaleinvariant.scale_features(img, features)

    # linearize features
    logger.info('Linearize features')
    features = cls.features.linearize(features)
    features_in_block = cls.features.extend_feature_blocks(
        features, features_in_block)

    # get feature stats for image
    logger.info('Compute feature statistics')
    stats = cls.features.normalize.compute_feature_stats(features)

    # normalize features
    logger.info('Normalize features')
    features = cls.features.normalize.normalize_features(features, stats)

    # make prediction
    classes = model.predict(features)
    
    logger.info('Prediction finished.')

    return classes


@printinfo
def run_segmentation(ds, images=[]):
    logger.info('Segmentation started...')
    for i, im in enumerate(images):
        logger.info('Processing image %d of %d: %s' % (i + 1, len(images), im))
        segments, contours, segments_orig, meta = _segmentate(ds, im)
        filesys.write_export_file(ds, im, 'meta', meta, append=True)
        filesys.write_export_file(ds, im, 'segments.original', segments_orig)
        filesys.write_export_file(ds, im, 'segments', segments)
        filesys.write_export_file(ds, im, 'contours', contours)
    logger.info('Segmentation finished.')


def run_feature_extraction(ds, images=[], feat_blocks=[], colorspace='rgb'):
    logger.info('Feature extraction started...')
    for i, im in enumerate(images):
        logger.info('Processing image %d of %d: %s' % (i + 1, len(images), im))

        segments = filesys.read_export_file(ds, im, 'segments')

        if segments is None:
            loglging.warning('No segmentation found for image: %s' % im)
            continue

        meta = filesys.read_export_file(ds, im, 'meta')
        if meta['superpixel_grid_error']:
            loglging.warning('Invalid sedmentation found for image: %s' % im)
            continue

        # load image
        img = filesys.read_image_file(ds, im)

        # Add extra channels
        # img is now [i,j,rgb]:                   grayscale channels   extra_channels
        # img becomes [i,j, channels] channels -> gray      (r g b)    (gabor sigmadiff)
        img = channels.add_channels(img, colorspace)
        filesys.write_export_file(ds, im, 'channels', img)

        # extract features
        features, features_in_block = \
            cls.features.blocks.extract_blocks(img,
                                               segments,
                                               colorspace=colorspace,
                                               blocks=feat_blocks)

        # remove too large features
        features = cls.features.remove_large_features(features)

        # write features to disk
        filesys.write_feature_files(ds, im, features, features_in_block)

        meta = {'last feature extraction': time.strftime('%d-%b-%Y %H:%M')}
        filesys.write_export_file(ds, im, 'meta', meta, append=True)

    logger.info('Feature extraction finished.')


def run_feature_update(ds, images=[], feat_blocks=[], class_aggregation=None, relloc=False):
    logger.info('Updating extracted features started...')

    if relloc:
        maps = filesys.read_export_file(ds, None, 'relative_location_maps')

    for i, im in enumerate(images):
        logger.info('Processing image %d of %d: %s' % (i + 1, len(images), im))

        # load image and features
        img = filesys.read_image_file(ds, im)
        features, features_in_block = filesys.read_feature_files(
            ds, im, feat_blocks.keys())

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
                loglging.warning(
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


def run_feature_normalization(ds, images=[], feat_blocks=[]):
    logger.info('Normalizing features started...')

    logger.info('Aggregate feature statistics')
    allstats = [filesys.read_export_file(
        ds, im, 'meta')['stats'] for im in images]
    stats = cls.features.normalize.aggregate_feature_stats(allstats)

    l = {'stats': stats,
         'last stats computation': time.strftime('%d-%b-%Y %H:%M')}
    filesys.write_log_file(ds, l)

    for i, im in enumerate(images):
        logger.info('Processing image %d of %d: %s' % (i + 1, len(images), im))
        stats = filesys.read_log_file(ds, keys='stats')

        features, features_in_block = filesys.read_feature_files(
            ds, im, feat_blocks.keys() + ['relloc'], ext='linear')
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
            loglging.warning('Size mismatch for image %s, skipped' % im)
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


def initialize_models(ds, images='all', feat_blocks='all', modtype='LR', class_aggregation=None):

    # create image list
    images = _create_imlist(ds, images)

    # create feature block list
    feat_blocks = _create_featlist(feat_blocks)

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
                           feat_blocks=feat_blocks)

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
                                   rlp_stats=rlp_stats) for m in modtype]

    # construct data arrays from dataframes and partitions
    train_sets, test_sets, prior_sets = _features_to_input(ds,
                                                           images,
                                                           images_train,
                                                           images_test,
                                                           X,
                                                           Y,
                                                           X_rlp)

    # collect meta information
    meta = [[{'dataset': ds,
              'images': list(images),
              'images_train': list(images_train[i]),
              'images_test':list(images_test[i]),
              'feature_blocks':[re.sub('^extract_blocks_', '', x) for x in feat_blocks.keys()],
              'features':list(X[0].columns),
              'classes':list(classes),
              'model_type':m} for i in range(len(images_train))] for m in modtype]

    return models, meta, train_sets, test_sets, prior_sets


def reinitialize_model(ds, model, class_aggregation=None):

    models, meta = filesys.read_model_file(ds, model)

    X, Y, X_rlp = get_data(ds,
                           meta['images'],
                           feat_blocks=meta['feature_blocks'])

    # aggregate classes
    if class_aggregation is not None:
        logger.info('Aggregate classes...')
        Y = cls.utils.aggregate_classes(Y, class_aggregation)

    # construct data arrays from dataframes and partitions
    train_sets, test_sets, prior_sets = _features_to_input(ds,
                                                           meta['images'],
                                                           [meta[
                                                               'images_train']],
                                                           [meta[
                                                               'images_test']],
                                                           X,
                                                           Y,
                                                           X_rlp)

    return models, meta, train_sets, test_sets, prior_sets


def get_data(ds, images=[], feat_blocks=[]):
    X = []
    Y = []
    X_rlp = []

    if type(feat_blocks) is dict:
        feat_blocks = feat_blocks.keys()

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
            ds, im, feat_blocks, ext='normalized')[0])
        X_rlp.append(
            filesys.read_feature_files(ds, im, ['relloc'], ext='normalized')[0])

        # load classes and append as array
        classes = filesys.read_export_file(ds, im, 'classes')
        Y.append(np.asarray(classes))

    logger.info('Preparation of features and labels finished.')

    return X, Y, X_rlp


@printinfo
def _segmentate(ds, im):

    # load image
    img = filesys.read_image_file(ds, im)

    # first segmentation step
    # TODO: use enforce_connectivity here when skimage 0.10 is released, segments_orig can then be removed
    # check for feature or check for version. Usually feature checking is better.
    # if distutils.version.LooseVersion(skimage.__version__) < distutils.version.LooseVersion('0.10'):
    segments_orig = seg.superpixels.get_superpixel(
        img, n_segments=600, compactness=20, sigma=0)

    # compute superpixel grid
    nx, ny = seg.superpixels.get_superpixel_grid(segments_orig, img.shape[:2])

    # fix disjoint segments
    segments = seg.postprocess.remove_disjoint(segments_orig)

    contours = seg.superpixels.get_contours(segments)

    meta = {'image_resolution_cropped': img.shape,
            'superpixel_grid': (nx, ny),
            'superpixel_grid_error': not seg.superpixels.check_segmentation(segments, nx, ny),
            'last_segmented': time.strftime('%d-%b-%Y %H:%M')}

    return segments, contours, segments_orig, meta


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


def _create_featlist(feat_blocks):
    allfb = cls.features.blocks.list_blocks()
    if type(feat_blocks) is list:
        feat_blocks = {k: v for k, v in allfb.iteritems() if k in feat_blocks or k.replace(
            'extract_blocks_', '') in feat_blocks}
    elif feat_blocks == 'all':
        feat_blocks = allfb
    else:
        msg = 'Features should be a list of feature block names or the keyword "all"'
        logger.warn(msg)
        raise ValueError(msg)

    return feat_blocks


def _create_featcollection(feat_blocks):
    if not type(feat_blocks) is list:
        feat_blocks = [feat_blocks]

    try:
        for i, block in enumerate(feat_blocks):
            feat_blocks[i] = _create_featlist(block)
    except:
        feat_blocks = [_create_featlist(feat_blocks)]

    return feat_blocks


def _train_test_part(images, test_frac=0.25):
    # Wrapper for sklearn function cross_validation.train_test_split

    images_train, images_test = train_test_split(images, test_size=test_frac)

    return images_train, images_test


def _multiple_partitions(images, n_part=5, test_frac=0.25):
    # Make multiple train-test partitions for statistical model performance
    # check

    images_train = []
    images_test = []

    lists = [images_train, images_test]

    for i in range(n_part):
        for lst, part in zip(lists, _train_test_part(images, test_frac)):
            lst.append(part)

    return {'training images': images_train, 'testing images': images_test}


def _features_to_input(ds, images, images_train, images_test, X, Y, X_rlp=[]):

    train_sets = []
    test_sets = []
    prior_sets = []
    for i in range(len(images_train)):

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
    classify-images preprocess <dataset> [--segmentate] [--extract] [--update] [--normalize] [--aggregate=FILE] [--verbose]
    classify-images partition <dataset> [--n=N] [--frac=FRAC] [--verbose]
    classify-images train <dataset> [--type=NAME] [--aggregate=FILE] [--verbose]
    classify-images score <dataset> [--model=NAME] [--aggregate=FILE] [--verbose]
    classify-images predict <dataset> <image> [--model=NAME] [--verbose]

Positional arguments:
    dataset           dataset containing the images
    image             image file to be classified

Options:
    -h, --help        show this help message and exit
    --segmentate      create segmentation of images
    --extract         extract features
    --update          update features
    --normalize       normalize features
    --n=N             number of partitions [default: 5]
    --frac=FRAC       fraction of images used for testing [default: 0.25]
    --type=NAME       model type to train [default: LR]
    --model=NAME      name of model to be scored, uses last trained if omitted
    --aggregate=FILE  use class aggregation from json file
    --verbose         print logging messages
"""

    arguments = docopt.docopt(usage)

    if arguments['--verbose']:
        logging.basicConfig()
        logging.root.setLevel(logging.NOTSET)

    class_aggregation = None
    if arguments['--aggregate'] is not None:
        if os.path.exists(arguments['--aggregate']):
            with open(arguments['--aggregate'], 'r') as fp:
                class_aggregation = json.load(fp)

    if arguments['preprocess']:
        run_preprocessing(
            arguments['<dataset>'],
            segmentation=arguments['--segmentate'],
            feat_extract=arguments['--extract'],
            feat_update=arguments['--update'],
            feat_normalize=arguments['--normalize'],
            class_aggregation=class_aggregation
        )
    
    if arguments['partition']:
        run_partitioning(
            arguments['<dataset>'],
            n_part=int(arguments['--n']),
            test_frac=float(arguments['--frac']),
            forcesplit=True
        )
    
    if arguments['train']:
        run_classification(
            arguments['<dataset>'],
            modtype=arguments['--type'],
            class_aggregation=class_aggregation
        )

    if arguments['score']:
        print run_scoring(
            arguments['<dataset>'],
            model=arguments['--model'],
            class_aggregation=class_aggregation
        ).to_string()

    if arguments['predict']:
        print run_prediction(
            arguments['<dataset>'],
            arguments['<image>'],
            model=arguments['--model']
        )
