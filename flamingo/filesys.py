import os
import re
import time
import cPickle as pickle

from pkg_resources import Requirement, resource_filename

import matplotlib.pyplot as plt
import numpy as np

import classification.features.blocks as feature_blocks

PATH_DATASET = '/flamingo/datasets/'


def set_dataset_path(fpath):

    # FIXME: if it's captitalized it's a constant. 
    # Use a config file for this or command line args 
    PATH_DATASET = fpath
def get_dataset_path():
    if PATH_DATASET is not None:
        if os.path.exists(PATH_DATASET):
            return PATH_DATASET
    demopath = resource_filename(Requirement.parse("flamingo"),"data")
    return demopath

def get_dataset_list():
    fpath = get_dataset_path()

    datasets = []
    for fname in os.listdir(fpath):
        if not fname.startswith('.') and os.path.isdir(os.path.join(get_dataset_path(), fname)):
            datasets.append(fname);
    return datasets

def get_image_path(ds):
    return os.path.join(get_dataset_path(),ds)

def get_image_location(ds, im):
    return os.path.join(get_image_path(ds),im)


def get_image_list(ds):
    images = []
    if ds is not None:
        fpath = get_image_path(ds)
        if os.path.exists(fpath):
            for im in os.listdir(fpath):
                if im.endswith('.jpg') or im.endswith('.png') or im.endswith('.jpeg'):
                    if not im.startswith('cropped_'): # skip cropped versions of image
                        images.append(im)
    return images

def get_export_file(ds, im=None, ext=None):
    if im is None and ext is None:
        return os.path.join(get_image_path(ds), '%s.pkl' % ds)
    elif im is None:
        return os.path.join(get_image_path(ds), '%s.pkl' % ext)
    elif ext is None:
        fpath = get_image_location(ds,im)
        return re.sub('\.[\w\d]+$','.pkl',fpath)
    else:
        fpath = get_image_location(ds,im)
        return re.sub('\.[\w\d]+$','.%s.pkl' % ext,fpath)

def read_export_file(ds, im, ext):
    contents = None
    pklfile = get_export_file(ds,im,ext)
    if os.path.exists(pklfile):
        try:
            with open(pklfile, 'rb') as fp:
                contents = pickle.load(fp)
        except:
            raise IOError('Error reading file %s' % pklfile)
    return contents

def write_export_file(ds, im, ext, contents, append=False):
    pklfile = get_export_file(ds,im,ext)

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
    
def read_log_file(ds,keys = None):
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
    return write_export_file(ds, None, 'log', contents, append=True)

def read_image_file(ds, im, crop=True):
    fpath = get_image_location(ds,im)

    img = None
    if os.path.exists(fpath):

        if crop:
            p,f = os.path.split(fpath)
            fpath_cropped = os.path.join(p,'cropped_' + f)
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
    for block, cols in features_in_block.iteritems():
        fname = __get_feature_filename(block, ext)
        cols = [c for c in cols if c in features.columns]
        write_export_file(ds, im, fname, features[cols])

def read_feature_files(ds, im, blocks=feature_blocks.list_blocks().keys(), ext=None):
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
    if len(ext) > 0 and not ext.startswith('.'):
        ext = '.%s' % ext
    for i, model in enumerate(models):

        for j in range(len(model)):
            write_model_file(ds, model[j], meta[i][j], ext='.%d%s' % (j, ext))

def write_model_file(ds, model, meta, ext=''):
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
    model_name = re.sub('\.pkl$','',model_name)

    # write model file
    fname = os.path.join(get_image_path(ds), '%s.pkl' % model_name)
    with open(fname, 'rb') as fp:
        model = pickle.load(fp)
    
    # write meta file
    fname = os.path.join(get_image_path(ds), '%s.meta.pkl' % model_name)
    with open(fname, 'rb') as fp:
        meta = pickle.load(fp)

    return model, meta

def read_default_categories(ds):
    classes = []
    images  = get_image_list(ds)
    for im in images:
        c = read_export_file(ds, im, 'classes')
        if c is not None:
            classes.extend(c)
    
    return list(np.unique(classes))
    
def is_classified(ds,im):
    classes = read_export_file(ds,im,'classes')
    if classes is not None:
        return None not in classes
    return False

def is_segmented(ds,im):
    pklfile = get_export_file(ds,im,'segments')
    return os.path.exists(pklfile)

def has_features(ds,im):
    pklfile = get_export_file(ds,im,'features')
    return os.path.exists(pklfile)

def __get_feature_filename(block, ext=None):
    block = re.sub('^extract_blocks_','',block)
    
    if ext:
        fname = 'features.%s.%s' % (ext, block)
    else:
        fname = 'features.%s' % block
        
    return fname

def __get_model_filename(meta):
    model_name = 'model_%s_%s_I%d_B%d_%s' % (meta['model_type'].upper(),
                                             meta['dataset'].upper(),
                                             len(meta['images']),
                                             len(meta['feature_blocks']),
                                             time.strftime('%Y%m%d%H%M%S'))

    return model_name
