import os
import re
import time
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt

from pkg_resources import Requirement, resource_filename

import classification.features.blocks as feature_blocks

PATH_DATASET = '/flamingo/datasets/'


def set_dataset_path(fpath):
    'Set path to datasets'

    # FIXME: if it's captitalized it's a constant.
    # Use a config file for this or command line args
    PATH_DATASET = fpath


def get_dataset_path():
    'Get path to datasets'
    
    if PATH_DATASET is not None:
        if os.path.exists(PATH_DATASET):
            return PATH_DATASET
    demopath = resource_filename(Requirement.parse('flamingo'), 'data')
    return demopath


def get_dataset_list():
    'Get list of available datasets'
    
    fpath = get_dataset_path()

    datasets = []
    for fname in os.listdir(fpath):
        if not fname.startswith('.') and os.path.isdir(os.path.join(get_dataset_path(), fname)):
            datasets.append(fname)
    return datasets


def get_image_path(ds):
    'Get absolute path to images within dataset'
    
    return os.path.join(get_dataset_path(), ds)


def get_image_location(ds, im):
    'Get absolute path to specific image in dataset'
    
    return os.path.join(get_image_path(ds), im)


def get_image_list(ds):
    'Get list with all images in dataset'
    
    images = []
    if ds is not None:
        fpath = get_image_path(ds)
        if os.path.exists(fpath):
            for im in os.listdir(fpath):
                if im.endswith('.jpg') or im.endswith('.JPG') or \
                   im.endswith('.png') or im.endswith('.PNG') or \
                   im.endswith('.jpeg') or im.endswith('.JPEG'):
                    # skip cropped versions of image
                    if not im.startswith('cropped_'):
                        images.append(im)
    return images


def get_export_file(ds, im=None, ext=None):
    'Get path to export file'
    
    if im is None and ext is None:
        return os.path.join(get_image_path(ds), '%s.pkl' % ds)
    elif im is None:
        return os.path.join(get_image_path(ds), '%s.pkl' % ext)
    elif ext is None:
        fpath = get_image_location(ds, im)
        return re.sub('\.[\w\d]+$', '.pkl', fpath)
    else:
        fpath = get_image_location(ds, im)
        return re.sub('\.[\w\d]+$', '.%s.pkl' % ext, fpath)


def check_export_file(ds, im, ext):
    'Check if export file exists'
    
    pklfile = get_export_file(ds, im, ext)
    return os.path.exists(pklfile)


def read_export_file(ds, im, ext):
    'Read contents of export file'
    
    contents = None
    pklfile = get_export_file(ds, im, ext)
    if os.path.exists(pklfile):

        success = False
        try:
            with open(pklfile, 'rb') as fp:
                contents = pickle.load(fp)
            success = True
        except:
            pass

        try:
            contents = pd.read_pickle(pklfile)
            success = True
        except:
            pass

        if not success:
            raise IOError('Error reading file %s' % pklfile)

    return contents


def write_export_file(ds, im, ext, contents, append=False):
    'Write contents to export file'
    
    pklfile = get_export_file(ds, im, ext)

    if append:
        if type(contents) is dict:
            current_contents = read_export_file(ds, im, ext)
            if current_contents is not None:
                current_contents.update(contents)
                contents = current_contents
        else:
            raise Exception('Only appending of dictionaries is supported')

    with open(pklfile, 'wb') as fp:
        pickle.dump(contents, fp, pickle.HIGHEST_PROTOCOL)


def read_log_file(ds, keys=None):
    'Read contents of log file'
    
    log = read_export_file(ds, None, 'log')
    if (keys is None) or (log is None):
        return log
    elif type(keys) is list:
        return {k: log[k] for k in keys if k in log.keys()}
    elif keys in log.keys():
        return log[keys]
    else:
        return None


def write_log_file(ds, contents):
    'Write contents to log file'
    
    return write_export_file(ds, None, 'log', contents, append=True)


def read_image_file(ds, im, crop=False):
    'Read image file'
    
    fpath = get_image_location(ds, im)

    img = None
    if os.path.exists(fpath):

        if crop:
            p, f = os.path.split(fpath)
            fpath_cropped = os.path.join(p, 'cropped_' + f)
            if not os.path.exists(fpath_cropped):
                img = plt.imread(fpath)
                img = img[8:-8,:,:]
                plt.imsave(fpath_cropped, img)
            else:
                img = plt.imread(fpath_cropped)
        else:
            img = plt.imread(fpath)

    return img


def write_feature_files(ds, im, features, features_in_block, ext=None):
    'Write features to a collection of export files depending on their feature block'
    
    for block, cols in features_in_block.iteritems():
        fname = __get_feature_filename(block, ext)
        cols = [c for c in cols if c in features.columns]
        write_export_file(ds, im, fname, features[cols])


def read_feature_files(ds, im, blocks=feature_blocks.list_blocks().keys(), ext=None):
    'Read features from a collection of export files including only selected feature blocks'
    
    features = []
    features_in_block = {}
    for block in blocks:
        fname = __get_feature_filename(block, ext)
        current_features = read_export_file(ds, im, fname)
        if not current_features is None:
            features.append(current_features)
            features_in_block[block] = current_features.columns
    features = feature_blocks.merge_blocks(features)

    return features, features_in_block


def get_model_list(ds):
    'Get list of model files in dataset'
    
    models = []
    if ds is not None:
        fpath = get_image_path(ds)
        if os.path.exists(fpath):
            for m in os.listdir(fpath):
                if m.startswith('model_') and m.endswith('.pkl') and not m.endswith('.meta.pkl'):
                    models.append(m)

    models.sort(key=lambda x: os.stat(os.path.join(fpath, x)).st_mtime)

    return models


def write_model_files(ds, models, meta, ext=''):
    'Write a series of models to export files including meta data'
    
    if len(ext) > 0 and not ext.startswith('.'):
        ext = '.%s' % ext
    for i, model in enumerate(models):
        for j in range(len(model)):
            write_model_file(ds, model[j], meta[i][j], ext='.%d%s' % (j, ext))


def write_model_file(ds, model, meta, ext=''):
    'Write a single model to export file including meta data'
    
    model_name = __get_model_filename(meta) + ext

    # write model file
    fname = os.path.join(get_image_path(ds), '%s.pkl' % model_name)
    with open(fname, 'wb') as fp:
        pickle.dump(model, fp)

    # write meta file
    fname = os.path.join(get_image_path(ds), '%s.meta.pkl' % model_name)
    with open(fname, 'wb') as fp:
        pickle.dump(meta, fp)


def read_model_file(ds, model_name):
    'Read a model from export file including meta data'
    
    model_name = re.sub('\.pkl$', '', model_name)

    # write model file
    fname = os.path.join(get_image_path(ds), '%s.pkl' % model_name)
    with open(fname, 'rb') as fp:
        model = pickle.load(fp)

    # write meta file
    fname = os.path.join(get_image_path(ds), '%s.meta.pkl' % model_name)
    with open(fname, 'rb') as fp:
        meta = pickle.load(fp)

    return model, meta


def check_model_file(ds, model_name):
    model_name = re.sub('\.pkl$', '', model_name)
    fname = os.path.join(get_image_path(ds), '%s.pkl' % model_name)
    return os.path.exists(fname)


def read_default_categories(ds):
    'Read list of uniquely defined classes in dataset'
    
    classes = []
    images = get_image_list(ds)
    for im in images:
        c = read_export_file(ds, im, 'classes')
        if c is not None:
            classes.extend(c)

    return list(np.unique(classes))


def is_classified(ds, im):
    'Determine if image is annotated'
    
    classes = read_export_file(ds, im, 'classes')
    if classes is not None:
        return None not in classes
    return False


def is_segmented(ds, im):
    'Determine if image is segmented'
    
    pklfile = get_export_file(ds, im, 'segments')
    return os.path.exists(pklfile)


def has_features(ds, im):
    'Determine if features for image are extracted'
    
    pklfile = get_export_file(ds, im, 'features')
    return os.path.exists(pklfile)


def __get_feature_filename(block, ext=None):
    'Get name of export file for feature block'
    
    block = re.sub('^extract_blocks_', '', block)

    if ext:
        fname = 'features.%s.%s' % (ext, block)
    else:
        fname = 'features.%s' % block

    return fname


def __get_model_filename(meta):
    'Get name of model file'
    
    model_name = 'model_%s_%s_I%d_B%d_%s' % (meta['model_type'].upper(),
                                             meta['dataset'].upper(),
                                             len(meta['images']),
                                             len(meta['feature_blocks']),
                                             time.strftime('%Y%m%d%H%M%S'))

    return model_name
