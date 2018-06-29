# This module implements the following classes:
#   * FlamingoBase
#     * FlamingoImage
#     * FlamingoDataset
#     * FlamingoDatasetPartition
#     * FlamingoModel
#   * FlamingoBlockchain
#   * FlamingoBlockchainException
#   * FlamingoConfig
#   * FlamingoHash
#   * FlamingoIO

from __future__ import absolute_import

import os
import io
import re
import cv2
import six
import time
import zlib
import json
import glob
import copy
import zipfile
import fnmatch
import hashlib
import logging
import numpy as np
import pandas as pd

# handle python 2/3 differences
from six.moves import cPickle as pickle

# FIXME: this should move to classification module
from sklearn.cross_validation import train_test_split
import sklearn.linear_model

# import subpackages
#import flamingo.calibration as fca
#import flamingo.rectification as fre
import flamingo.classification as fcl


# initialize logger
logger = logging.getLogger(__name__)


# set zlib compression level
zlib.Z_DEFAULT_COMPRESSION = zlib.Z_BEST_COMPRESSION


class FlamingoBlockchainException(Exception):
    '''Exception class for blockchain violations'''
    pass


def workflow(stage=0):
    '''Workflow decorator to guarantee data integrity
    
    The workflow decorator can be used to enforce blockchain integrity
    of an object. Blockchain integrity ensures that specific methods
    are called in the right order. Therefore the methods are grouped
    in stages. Once a method in a higher stage is called, methods in
    lower stages are disabled. Similarly, previously called methods in
    a higher stage are forced to be called again once a method in a
    lower stage is called.

    Any class that uses the workflow decorator should have an
    attribute ``_blockchain`` that contains a
    :class:`FlamingoBlockchain` object. The workflow decorator
    validates and appends the blockchain while methods are called and
    throws a :class:`FlamingoBlockchainException` whenever the
    blockchain integrity is violated.

    See :class:`FlamingoBlockchain` for more details on the blockchain
    implementation.

    Parameters
    ----------
    stage : int
      Blockchain stage that the methods belongs to.

    Returns
    -------
    function
      Decrorator function

    Raises
    ------
    FlamingoBlockchainException
      If the blockchain integrity is violated.

    See Also
    --------
    FlamingoBlockchain
    FlamingoBlockchainException

    '''

    def decorator(fcn):
        '''Decorator function

        Parameters
        ----------
        fcn : function
          Decorated method

        Returns
        -------
        function
          Method wrapper

        '''
        
        def method_wrapper(self, *args, **kwargs):
            '''Method wrapper

            Parameters
            ----------
            self : FlamingoBase
              Parent object of method
            args : list
              Positional arguments passed to decorated method
            kwargs : dict
              Keyword arguments passed to decorated method

            Raises
            ------
            FlamingoBlockchainException

            '''

            if self._blockchain.preserve_integrity:
                fcnnames = self._blockchain.ascend(stage)
                if len(fcnnames) > 0:
                    raise FlamingoBlockchainException(
                        'Modification of stage %d by rerunning "%s" '
                        'might break the blockchain and force you to '
                        'rerun "%s". If this is intentional, set '
                        'object.preserve_integrity=False. Good luck.' % \
                        (stage, fcn.__name__, '" and "'.join(fcnnames)))
                
            if self._blockchain.check_integrity(stage):
                fcn(self, *args, **kwargs)
                self.update_hash()
                self._blockchain.append(stage,
                                        fcn.__name__,
                                        self._blockchain.descend(stage),
                                        self._hash,
                                        time.time())
            else:
                breach = self._blockchain.breach
                raise FlamingoBlockchainException(
                    'Blockchain broken at stage %d, rerun "%s" '
                    'first.' % (breach['stage'], '" and/or "'.join(breach['fcnnames'])))
            
        return method_wrapper
    return decorator


class FlamingoHash(str):
    '''Flamingo hash implementation

    An immutable ``str`` object containing a 32 character MD5
    hash. The object can be instantiated by a 32 character string,
    another :class:`FlamingoHash` object or by data to be hashed. In
    any case the result will be a 32 character hash string. An empty
    hash will result in a string with 32 spaces.

    Raises
    ------
    ValueError

    Examples
    --------
    >>> FlamingoHash()
        '                                '
    >>> FlamingoHash('01234567890123456789012345678912')
        '01234567890123456789012345678912'
    >>> FlamingoHash(data='01234567890123456789012345678912')
        '33cfd831fbdd5d7888c682d48e8eb275'
    >>> FlamingoHash(data='Hash this!')
        '7a353955a222e854abfaacc61ab8f215'
    >>> FlamingoHash(FlamingoHash('01234567890123456789012345678912'))
        '01234567890123456789012345678912'
    >>> FlamingoHash(data=FlamingoHash('01234567890123456789012345678912'))
        '01234567890123456789012345678912'
    >>> FlamingoHash(FlamingoHash(data='01234567890123456789012345678912'))
        '33cfd831fbdd5d7888c682d48e8eb275'
    >>> FlamingoHash('0123456789')
        ValueError: Invalid hash
    >>> FlamingoHash(12345678901234567890123456789123)
        ValueError: Invalid hash
    >>> FlamingoHash(data=12345678901234567890123456789123)
        ValueError: Expected string data, got <class 'int'>.

    '''

    def __new__(cls, x=None, data=None):
        '''Construction

        Constructs 32 character hash string from 32 character string,
        FlamingoHash object or creates a MD5 hash from data. If no
        construction data is provides it creates a string with 32
        spaces.

        Parameters
        ----------
        x : 32 char str or FlamingoHash, optional
          Hash string or FlamingoHash object
        data : str, optional
          String to be hashed of FlamingoHash object

        Raises
        ------
        ValueError

        '''
        
        if x is not None:
            # construct from 32 character string
            if not (isinstance(x, str) or \
                    isinstance(x, six.string_types) or \
                    isinstance(x, six.text_type) or \
                    isinstance(x, six.binary_type) or \
                    isinstance(x, FlamingoHash)) or \
                len(x) != 32:
                raise ValueError('Invalid hash')
            return super(FlamingoHash, cls).__new__(cls, x)
        elif data is not None:
            # construct from hashable data
            if isinstance(data, FlamingoHash):
                return super(FlamingoHash, cls).__new__(cls, data)
            elif isinstance(data, str) or \
                 isinstance(data, six.string_types) or \
                 isinstance(data, six.text_type) or \
                 isinstance(data, six.binary_type):
                try:
                    # try hashing data directly
                    return super(FlamingoHash, cls).__new__(
                        cls, hashlib.md5(data).hexdigest())
                except:
                    pass

                # otherwise encode as UTF-8
                return super(FlamingoHash, cls).__new__(
                    cls, hashlib.md5(data.encode('utf-8')).hexdigest())
            else:
                raise ValueError('Expected string data, got %s.' % type(data))
        else:
            # construct 32 character empty string
            return super(FlamingoHash, cls).__new__(cls, ' ' * 32)


class FlamingoBlockchain(object):
    '''Flamingo blockchain implementation

    A blockchain logs the order in which specific methods are
    called. The order is logged by logging hashes representing the
    current object state just before and after calling a
    method. Methods are grouped in stages. By comparing the `after`
    hashes logged by all methods called in a single stage with the
    `before` hashes logged by a method in the next stage, the order of
    the calls can be reestablished. This order is called the
    blockchain. If a previously called method is called again, the
    method is pushed to the end of the blockchain as the `before` hash
    matches the last `after` hash.

    Each called method is logged with exactly one `after`
    hash. However, methods in a blockchain are grouped in stages. The
    group of `after` hashes of all methods in a single stage are used
    as `before` hashes for the next stage. Therefore, there can be
    multiple `before` hashes.

    An example of a blockchain is:

    ..code-block::

      Stage  Function  Hashes (before)                  Hashes (after)                  
      
      0      fcn1      None                             8a56f75c86875ccce5b4012e61ee166c
             fcn2      None                             a113c6a0214ac8a53ada7a0bebe207bc
      2      fcn3      a113c6a0214ac8a53ada7a0bebe207bc 6cef867cf5262928edde3e1ecaac5c54
                       8a56f75c86875ccce5b4012e61ee166c
      3      fcn4      6cef867cf5262928edde3e1ecaac5c54 cfc2016d32b06135a77c61c7b7fd6e7b

      Consistent chainage: True

    A blockchain can be used to ensure that specific methods are
    called in a prescribed right order. A method decorator can be used
    to register the prescribed call order, validate the blockchain and
    raise an exception if the blockchain is broken or about to
    break. An example of such decorator is implemented in
    :func:`workflow`.

    See Also
    --------
    workflow

    '''

    
    def __init__(self):
        '''Instantiation'''

        self.chain = {}
        self.breach = None
        self.preserve_integrity = True


    def ascend(self, stage):
        '''Collects all called methods after a given stage by ascending the blockchain

        Parameters
        ----------
        stage : int
          Stage where to start the ascend

        Returns
        -------
        fcnnames : list
          List with method names called in higher stages

        '''
        
        fcnnames = []
        ix = self.index(stage)
        logger.debug('Ascend blockchain from stage "%s".' % str(ix))
        if ix < len(self.stages):
            for s in self.stages[ix+1:]:
                fcnnames.extend(self.chain[s].keys())
        return fcnnames

        
    def descend(self, stage):
        '''Collects all `after` hashes before a given stage by descending the blockchain

        Parameters
        ----------
        stage : int
          Stage where to start the descend

        Returns
        -------
        set
          Set with `after` hashes logged in lower stages

        '''
        
        ix = self.index(stage)
        logger.debug('Descend blockchain from stage "%s".' % str(ix))
        if ix > 0:
            for s in self.stages[ix-1::-1]:
                return set([x[1] for x in self.chain[s].values()])
        return set([None])
            

    def check_integrity(self, stage):
        '''Check the integrity of the blockchain before a given stage

        If the blockchain is broken, the breach is registered in the
        ``breach`` attribute that can be used by error handlers.

        Parameters
        ----------
        stage : int
          Stage where to start the integrity check

        Returns
        -------
        bool
          Boolean indicating whether blockchain integrity is valid

        '''
        
        ix = self.index(stage)
        previous_hashes = set([None])
        for s in self.stages[:ix]:
            fcnnames = []
            hashes = [set(), set()]
            for fcnname, shackle in sorted(self.chain[s].items()):
                hashes[0].update(shackle[0])
                hashes[1].add(shackle[1])
                fcnnames.append(fcnname)
                
            if len(hashes[0].difference(previous_hashes)) == 0:
                previous_hashes = hashes[1]
                continue

            # if the missing hashes happens to match future hashes,
            # one or more previous action have been repeated without
            # affecting the results. this situation is not ideal, but
            # accepted.
            ixs = self.index(s)
            future_hashes = [s2[1]
                             for s1 in self.stages[ixs:]
                             for s2 in self.chain[s1].values()]
            for missing_hash in list(previous_hashes.difference(hashes[0])):
                if missing_hash not in future_hashes:
                    break
            else:
                previous_hashes = hashes[1]
                continue

            # chain is broken
            self.breach = dict(stage=s,
                               fcnnames=fcnnames)
            return False

        # chain is ok
        self.breach = {}
        return True
    
        
    def append(self, stage, fcnname, *args):
        '''Append a shackle to the blockchain

        Parameters
        ----------
        stage : int
          Stage where the function belongs to
        fcnname : str
          Method name that was called
        args : 2-tuple
          Tuple with `before` and `after` hash

        '''
        
        if stage not in self.stages:
            self.chain[stage] = {}
        self.chain[stage][fcnname] = args


    def index(self, stage, default=None):
        '''Returns index of a stage

        Returns the index of the stage in the blockchain or, if the
        stage doesn't exist the index that it would have if the stage
        was appended to the blockchain. Alternatively, a ``default``
        value can be provided that is returned if the requested stage
        does not exist.

        Parameters
        ----------
        stage : int
          Stage for which the index should be determined.
        default : int
          Alternative index in case requested stage does not exist.

        Returns
        -------
        int
          Index of requested stage.

        '''

        if stage in self.stages:
            return self.stages.index(stage)
        elif default is None:
            # determine future stage index
            return sorted(list(set(self.stages + [stage]))).index(stage)
        else:
            return default


    def print_chainage(self):
        '''Print current blockchain'''

        fmt = '%-10s %-20s %-32s %-32s'
        header = fmt % ('Stage', 'Function', 'Hashes (before)', 'Hashes (after)')
        
        print(header)
        print('-' * len(header))

        for stage, shackle in sorted(self.chain.items()):
            for fcn, hashes in sorted(list(shackle.items()), key=lambda x: x[1][2]):
                hashes0 = list(hashes[0])
                hashes1 = hashes[1]
                for i in range(len(hashes0)):
                    print(fmt % (stage, fcn, hashes0[i], hashes1))
                    fcn, hashes1, stage = '', '', ''
        
        print('')
        print('Consistent chainage: %s' % str(self.check_integrity(self.stages[-1])))


    @property
    def stages(self):
        '''List with logged stages in order'''
        return list(sorted(self.chain.keys()))


class FlamingoConfig(object):
    '''Flamingo configuration reader'''


    def __init__(self, filename_or_dict=None):

        self.config = {}
        self._file = ''
        self._path = ''
        
        self.load(filename_or_dict)


    def __call__(self, *args):
        cfg = self.config
        for arg in args:
            if isinstance(cfg, dict) and arg in cfg.keys():
                cfg = cfg[arg]
            else:
                return None
        return cfg


    def load(self, filename_or_dict):
        if isinstance(filename_or_dict, FlamingoConfig):
            self.config = filename_or_dict.config
            self._file = filename_or_dict._file
            self._path = filename_or_dict._path
        if isinstance(filename_or_dict, dict):
            self.config = filename_or_dict
            self._file = ''
            self._path = ''
        elif isinstance(filename_or_dict, str) or \
             isinstance(filename_or_dict, six.string_types) or \
             isinstance(filename_or_dict, six.text_type) or \
             isinstance(filename_or_dict, six.binary_type):
            with open(filename_or_dict, 'r') as fp:
                self.config = json.load(fp)
                self._file = os.path.abspath(filename_or_dict)
                self._path = os.path.split(self._file)[0]
                self.config = self.resolve_file_references()
                
            logger.info('Loaded configuration from %s.' % filename_or_dict)
                

    def resolve_file_references(self, obj=None):
        if obj is None:
            obj = self.config
        if isinstance(obj, (list, tuple, set)):
            return [self.resolve_file_references(o) for o in obj]
        elif isinstance(obj, (dict)):
            return {k:self.resolve_file_references(o) for k, o in obj.items()}
        elif isinstance(obj, str) or \
             isinstance(obj, six.string_types) or \
             isinstance(obj, six.text_type) or \
             isinstance(obj, six.binary_type):
            fpath = os.path.join(self._path, obj)
            try:
                return np.loadtxt(fpath)
            except:
                return obj
        else:
            return obj


class FlamingoBase(object):
    '''Base object for Flamingo data'''
    
    _type = 'Flamingo Object'
    _ext = '.fo'

    _exclude_copy = set()
    _exclude_hash = set(['_tstamp', '_hash', '_fromhash', '_description', 
                         '_path', '_ext', '_config', '_blockchain',
                         'preserve_integrity'])
    
    
    def __init__(self, filename_or_obj=None, config=None,
                 description=None, attributes=None,
                 exclude_attributes=[]):
        '''Initialization

        Parameters
        ----------
        filename_or_obj : str or FlamingoBase
            Flamingo object file or Flamingo object to be read.
        config : str
            Configuration file containing Flamingo toolbox settings.
        description : str
            User description of Flamingo object
        attributes : list, optional
            List of attributes to be read from a Flamingo object. If
            not given, all attributes are read.
        exclude_attributes : list, optional
            List of attributes to be excluded from reading from a
            Flamingo object. Has preference over ``attributes``.

        '''

        self._empty = True
        self._hash = FlamingoHash()
        self._blockchain = FlamingoBlockchain()
        self._tstamp = 0.
        self._description = description
        self._fromhash = ''
    
        self.load_config(config)
            
        if filename_or_obj:
            if isinstance(filename_or_obj, FlamingoBase):
                # copy attributes if input is a valid Flamingo object
                self.copy_attributes(filename_or_obj, 
                                     attributes=attributes,
                                     exclude_attributes=exclude_attributes)
            elif isinstance(filename_or_obj, str):
                # load data from file if input is a string
                self.load(filename_or_obj, 
                          attributes=attributes, 
                          exclude_attributes=exclude_attributes)
            else:
                raise TypeError('Expect str or FlamingoBase, '
                                '%s found' % type(filename_or_obj).__name__)
    

    def __repr__(self):
        s = '%s:\n' % self._type
        if self._empty:
            s += '  <empty>\n'
        else:
            if self._description is not None:
                s += '\n'
                s += '  %s\n' % self._description
                s += '\n'
            s += '  hash : %s\n' % self._hash
            s += '  time : %s\n' % time.strftime('%Y-%m-%d %H:%M:%S', 
                                                 time.gmtime(self._tstamp))
        return s


    def set_description(self, description):
        '''Set user description of object'''
        self._description = str(description)
        return self
    
    
    @workflow(stage=0)
    def load(self, filename, attributes=None, exclude_attributes=[]):
        '''Load object attributes from Flamingo object file'''
        
        exclude_attributes.extend(self._exclude_copy)
        FlamingoIO(filename).read(self, 
                                  attributes=attributes, 
                                  exclude_attributes=exclude_attributes)
        return self
                
                
    def dump(self, filename, attributes=None, exclude_attributes=[]):
        '''Dump object as Flamingo object file'''
        
        self.update_hash()
        exclude_attributes.extend(self._exclude_copy)
        filename = os.path.join(self._path, 
                                ''.join((os.path.splitext(filename)[0], self._ext)))
        FlamingoIO(filename).write(self, 
                                   attributes=attributes,
                                   exclude_attributes=exclude_attributes)
        return self

         
    @workflow(stage=0)
    def load_config(self, filename_or_dict):
        '''Load Flamingo configuration file'''

        self.config = FlamingoConfig(filename_or_dict)
        self._path = self.config._path
        
        return self


    @workflow(stage=0)
    def copy_attributes(self, obj, attributes=None, exclude_attributes=[]):
        '''Copy attributes from Flamingo object'''

        if self._type != obj._type:
            raise IOError('Invalid %s object, this seems to be '
                          'a %s object' % (self._type, obj._type))
            
        for name, attr in obj.iterate_attributes(attributes=attributes, 
                                                 exclude_attributes=exclude_attributes):
            try:
                setattr(self, name, attr)
                logger.debug('Copied attribute %s.' % name)
            except AttributeError:
                pass
            
        return self

            
    def iterate_attributes(self, attributes=None, exclude_attributes=[]):
        '''Iterate attributes'''

        for name in dir(self):
            if attributes is not None and name not in attributes:
                continue
            if name in exclude_attributes:
                continue
            if name.startswith('__'):
                continue
            attr = getattr(self, name)
            if hasattr(attr, '__call__'):
                continue
            yield name, attr
                
            
    def update_hash(self, attributes=None, exclude_attributes=[]):
        '''Compute hash from current object state

        Computes hash from object attributes. All attributes are first
        stringified and then concatenated. A MD5 hash is computed from
        th resulting string. Methods and aatributes listed in
        ``_exclude_copy`` and ``_exclude_hash`` are ignored.

        Parameters
        ----------
        
        '''

        self._empty = False
        self._tstamp = time.time()

        exclude_attributes.extend(self._exclude_copy)
        exclude_attributes.extend(self._exclude_hash)

        data = ''
        attrs = self._hash_attributes(attributes=attributes, 
                                      exclude_attributes=exclude_attributes)
        for name, attr in sorted(attrs.items()):
            data += '__%s_%s_' % (name.upper(), attr)
        self._hash = FlamingoHash(data=data)
        logger.debug('Computed hash in %0.1f s.' % (time.time() - self._tstamp))
        return self


    def compare_hash(self, obj=None, attributes=None, exclude_attributes=[]):
        '''Check reproducibility of object hash'''

        self._empty = False
        self._tstamp = time.time()

        exclude_attributes.extend(self._exclude_copy)
        exclude_attributes.extend(self._exclude_hash)

        not_reproducible = 0
        attrs1 = self._hash_attributes(attributes=attributes, 
                                       exclude_attributes=exclude_attributes)
        if obj is None:
            attrs2 = self._hash_attributes(attributes=attributes, 
                                           exclude_attributes=exclude_attributes)
        else:
            attrs2 = obj._hash_attributes(attributes=attributes, 
                                          exclude_attributes=exclude_attributes)

        for name in attrs1.keys():
            if attrs1[name] != attrs2[name]:
                logger.warn('Attribute hash not reproducible: %s' % name)
                not_reproducible += 1
        if not_reproducible == 0:
            logger.info('All attribute hashes were reproducible.')
            return True
        else:
            logger.info('%d attribute hashes were not reproducible' % not_reproducible)
            return False


    def _hash_attributes(self, attributes=None, exclude_attributes=[]):
        '''Convert attributes into hashes'''
        attrs = {}
        for name, attr in self.iterate_attributes(attributes=attributes, 
                                                  exclude_attributes=exclude_attributes):
            attrs[name] = FlamingoHash(data=self._stringify(attr))
        return attrs


    def _stringify(self, attr):
        '''Convert attribute value to hashable string'''

        if isinstance(attr, list):
            return '_'.join([self._stringify(a) for a in attr])
        elif isinstance(attr, dict):
            return self._stringify([self._stringify(k) + '=' + self._stringify(a)
                                    for k,v in sorted(attr.items())])
        elif isinstance(attr, set):
            return self._stringify(sorted(attr))
        elif isinstance(attr, np.ndarray):
            return attr.tostring()
        elif isinstance(attr, pd.DataFrame):
            df = copy.deepcopy(attr)
            if 'uuid' in attr.columns:
                df = df.drop(['uuid'], axis=1) # FIXME
            return df.as_matrix().tostring()
        elif isinstance(attr, FlamingoBase):
            return attr._hash
        elif isinstance(attr, FlamingoConfig):
            return self._stringify(attr.config)
        elif isinstance(attr, FlamingoHash):
            return attr
        else:
            return str(attr)
            #return pickle.dumps(attr) # slow

    
    @property
    def preserve_integrity(self):
        return self._blockchain.preserve_integrity


    @preserve_integrity.setter
    def preserve_integrity(self, x):
        self._blockchain.preserve_integrity = x
    
    
class FlamingoImage(FlamingoBase):

    
    _type = 'Flamingo Image Object'
    _ext = '.fio'
    
    
    def __init__(self, image_or_obj=None, config=None,
                 description=None, attributes=None,
                 exclude_attributes=[]):

        self._exclude_copy.update(['image', 'n_segments', 'n_features'])

        self._name = None
        self._image = None
        self._roi = None
        self._crop = [0, 0, 0, 0]
   
        self.channels = None
        self.segmentation = None
        self.segmentation_contours = None
        self.features = None
        self.features_statistics = None
        self.annotation = None
        self.prediction = None

        try:
            super(FlamingoImage, self).__init__(filename_or_obj=image_or_obj, 
                                                config=config,
                                                description=description,
                                                attributes=attributes,
                                                exclude_attributes=exclude_attributes)
        except (IOError, TypeError, zipfile.BadZipfile):
            self.imread(image_or_obj)
            
            
    def __repr__(self):
        s = super(FlamingoImage, self).__repr__()
        s += '  name : %s\n' % self._name
        return s


    @workflow(stage=0)
    def apply_roi(self):
        roi = self.config('classification', 'roi')
        if roi is not None:
            if not isinstance(roi, list):
                roi = [roi]
            
            roi_file = None
            for i, r in enumerate(roi):
                if 'filename_pattern' in r.keys():
                    # try unix style file pattern matching
                    if fnmatch.fnmatch(self._name, r['filename_pattern']):
                        roi_file = r['roi_file']
                        break
                    try:
                        # try regular expression matching
                        if re.match(r['filename_pattern'], self._name):
                            roi_file = r['roi_file']
                            break
                    except:
                        pass
                else:
                    # always match when no pattern exists
                    roi_file = r['roi_file']
                    break

            if roi_file is not None:
                logger.info('Applied ROI #%d.' % (i+1))
        return self


    @workflow(stage=0)
    def apply_crop(self):
        crop = self.config('image', 'crop')
        if crop is not None:
            if isinstance(crop, list):
                if len(crop) == 4:
                    self._crop = crop
            elif isinstance(crop, int):
                self._crop = [crop] * 4
            else:
                raise ValueError('Crop should be defined as integer or 4-item list.')

            logger.info('Applied crop %s.' % str(self._crop))
        return self


    @workflow(stage=1)
    def extract_channels(self):
        return self
    
    
    @workflow(stage=2)
    def create_segmentation(self):

        # collect segmentation settings
        kwargs = {}
        cfg = self.config('classification', 'segmentation')
        if isinstance(cfg, dict):
            kwargs.update(cfg)
        kwargs['roi'] = self._roi

        # perform segmentation
        self.segmentation, self.segmentation_contours = \
            fcl.segmentation.get_segmentation(self.image, **kwargs)

        logger.info('Created %d segments.' % self.n_segments)

        return self
    
    
    @workflow(stage=3)
    def extract_features(self):

        # collect feature extraction settings
        kwargs = {}
        cfg = self.config('classification', 'features')
        if isinstance(cfg, dict):
            kwargs.update(cfg)

        # perform feature extraction
        self.features = fcl.features.extract_blocks(
            self.image, self.segmentation, **kwargs)[0]

        # remove too large features
        self.features = fcl.features.postprocess.remove_large_features(
            self.features)

        # make features scale invariant
        self.features = fcl.features.postprocess.scale_features(
            self.image, self.features)

        # linearize features
        self.features = fcl.features.postprocess.linearize(
            self.features)

        # compute feature statistics
        self.features_statistics = fcl.features.postprocess.compute_feature_stats(
            self.features)

        logger.info('Extracted %d features.' % self.n_features)

        return self


    @workflow(stage=3)
    def add_annotation(self, annotation):
        annotation = np.asarray(annotation)
        n_classes = np.prod(annotation.shape)
        if self.n_segments == n_classes:
            self.annotation = annotation
            logger.info('Added annotation data.')
        else:
            raise ValueError('Number of elements in annotation matrix (%d) '
                             'do not match the number of elements in the '
                             'segmentation matrix (%d)' %
                             (n_classes, self.n_segments))
        return self
    
    
    def make_prediction(self):
        pass
    

    def get_feature_set(self, exclude_features=None, class_aggregation=None):

        X = self.features
        if exclude_features is not None:
            X = X.drop(exclude_features, axis=1)
        X = X.sort_index(axis=1).as_matrix()

        Y = self.annotation
        if class_aggregation is not None:
            Y = fcl.utils.aggregate_classes(Y, class_aggregation)

        return X, Y

    
    @workflow(stage=0)
    def imread(self, filename):
        self._path, self._name = os.path.split(filename)
        logger.info('Set working directory to %s.' % self._path)
        with open(filename, 'rb') as fp:
            self._image = fp.read()
            logger.info('Read image data from %s.' % filename)
        self.apply_roi()
        self.apply_crop()

    
    def imwrite(self, filename=None):
        if filename is None:
            filename = self._name
        with open(os.path.join(self._path, filename), 'wb') as fp:
            fp.write(self._image)
            logger.info('Wrote image data to %s.' % filename)


    def dump(self, filename=None):
        if filename is None:
            filename = self._name
        super(FlamingoImage, self).dump(filename=filename)


    @property
    def image(self):
        img_array = np.asarray(bytearray(self._image), dtype=np.uint8)
        img = cv2.cvtColor(cv2.imdecode(img_array, -1), cv2.COLOR_BGR2RGB)

        c = self._crop
        if c[0] > 0:
            img = img[c[0]:,:,...]
        if c[2] > 0:
            img = img[:-c[2],:,...]
        if c[1] > 0:
            img = img[:,c[1]:,...]
        if c[3] > 0:
            img = img[:,:-c[3],...]

        return img


    @property
    def n_segments(self):
        if self.segmentation is not None:
            return len(np.unique(self.segmentation))
        else:
            return None


    @property
    def n_features(self):
        if self.features is not None:
            return len(self.features.columns)
        else:
            return None


class FlamingoDataset(FlamingoBase):
    

    _type = 'Flamingo Dataset Object'
    _ext = '.fdo'
    

    def __init__(self, images_or_obj=None, config=None,
                 description=None, attributes=None,
                 exclude_attributes=[]):
        
        self._exclude_copy.update(['n_images', 'n_partitions', 'n_models'])

        self.images = []
        self.statistics = []
        self.partitions = []
        self.models = []
        
        try:
            super(FlamingoDataset, self).__init__(filename_or_obj=images_or_obj, 
                                                  config=config, 
                                                  description=description,
                                                  attributes=attributes, 
                                                  exclude_attributes=exclude_attributes)
        except (IOError, TypeError, zipfile.BadZipfile):
            self.add_images(images_or_obj)
            
        self.update_hash()
        
        
    def __repr__(self):
        s = super(FlamingoDataset, self).__repr__()
        s += '  images : %d\n' % len(self.images)
        s += '  partitions : %d\n' % len(self.partitions)
        s += '  models : %d\n' % len(self.models)
        return s

                
    @workflow(stage=0)
    def add_images(self, images, attributes=None, exclude_attributes=[]):
        if isinstance(images, list):
            for image in images:
                self._append(image,
                             attributes=attributes,
                             exclude_attributes=exclude_attributes)
        elif isinstance(images, str):
            for image in glob.glob(images):
                self._append(image,
                             attributes=attributes,
                             exclude_attributes=exclude_attributes)
        else:
            raise ValueError('Definition of images not understood')

        return self
            
            
    def _append(self, image, attributes=None, exclude_attributes=[]):
        obj = FlamingoImage(image, 
                            config=self.config,
                            attributes=attributes, 
                            exclude_attributes=exclude_attributes)
        self.images.append(obj)
        logger.info('Added image %s.' % image)


    @workflow(stage=1)
    def preprocess_classification(self):
        for img in self.images:
            if not img.features:
                if not img.segmentation:
                    img.create_segmentation()
                img.extract_features()
        return self
        

    @workflow(stage=2)
    def compute_statistics(self):
        self.statistics = fcl.features.postprocess.aggregate_feature_stats(
            [img.features_statistics for img in self.images])

        return self

    
    @workflow(stage=2)
    def create_partitions(self):
        partitions = self.config('classification', 'paritions')
        if isinstance(partitions, dict):
            if 'n' in partitions.keys():
                n = partitions.pop('n')
            else:
                n = 1
        else:
            # no partitions defined, assume single set
            n = 1
            partitions = dict(test = 0.,
                              validate = 0.,
                              train = 1.)

        self.partitions = []
        for i in range(n):
            self.partitions.append(
                FlamingoDatasetPartition(self, config=self.config).split(**partitions))

        logger.info('Created %d paritions.' % len(self.partitions))

        return self

    
    @workflow(stage=3)
    def generate_models(self, options, append=False):
        
        if not append:
            self.models = []
        if not isinstance(options, (list, tuple, set)):
            options = [options]
        for kwargs in options:
            model = FlamingoModel(config=self.config, **kwargs)
            self.models.append(model)
            logger.info('Created model "%s"')


    @workflow(stage=4)
    def train_models(self, exclude_features=None, class_aggregation=None):
        
        for model in self.models:
            for partition in self.partition:
                logger.info('Training model "%s" against partition '
                            '"%s"...' % (model._hash, partition._hash))
                model.train(self, partition,
                exclude_features=exclude_features,
                class_aggregation=class_aggregation)


    def get_feature_set(self, partition=None, exclude_features=None,
                        class_aggregation=None):

        features = []
        annotations = []
        for img in self.images:
            features.append(img.features)
            annotations.append(img.annotation)

        if partition is not None:
            ix = partition.train
        else:
            ix = slice(0,-1,1)

        X = pd.concat(self.features[ix])
        if exclude_features is not None:
            X = X.drop(exclude_features, axis=1)
        X = X.sort_index(axis=1).as_matrix()

        Y = np.concatenate(self.annotations[ix])
        if class_aggregation is not None:
            Y = fcl.utils.aggregate_classes(Y, class_aggregation)

        return X, Y

    
    @property
    def n_images(self):
        return len(self.images)

    
    @property
    def n_partitions(self):
        return len(self.partitions)

        
    @property
    def n_models(self):
        return len(self.models)


class FlamingoDatasetPartition(FlamingoBase):
    

    _type = 'Flamingo Dataset Partition Object'
    _ext = '.fpo'
    

    def __init__(self, dataset_or_obj=None, config=None,
                 description=None, attributes=None,
                 exclude_attributes=[]):
        
        self.dataset = None
        self.images = []
        self.train = []
        self.test = []
        self.validate = []

        try:
            super(FlamingoDatasetPartition, self).__init__(
                filename_or_obj=None, 
                config=config, 
                description=description,
                attributes=attributes, 
                exclude_attributes=exclude_attributes)
        except (IOError, TypeError, zipfile.BadZipfile):
            if isinstance(dataset_or_obj, FlamingoDataset):
                self.dataset = dataset_or_obj._hash
                self.images = [img._hash for img in dataset_or_obj.images]
            else:
                raise ValueError('Expected FlamingoDataset or FlamingoDatasetPartition, '
                                 'got %s' % type(dataset_or_obj))
            
        self.update_hash()
        
        
    def __repr__(self):
        s = super(FlamingoDatasetPartition, self).__repr__()
        s += '  train : %d\n' % len(self.train)
        s += '  test : %d\n' % len(self.test)
        s += '  validate : %d\n' % len(self.validate)
        return s


    def split(self, train=1., test=0., validate=0.):
        
        # normalize partitions
        f = train + test + validate
        train = train / f
        test = (test + validate) / f
        if test != 0:
            validate = validate / f / test

        # create partition
        ix = range(len(self.images))
        ix_train, ix_test = train_test_split(ix, test_size=test)
        ix_test, ix_validate = train_test_split(ix_test, test_size=validate)

        self.train = ix_train
        self.test = ix_test
        self.validate = ix_validate

        self.update_hash()

        return self


class FlamingoModel(FlamingoBase):

    
    _type = 'Flamingo Model Object'
    _ext = '.fmo'
    

    def __init__(self, model_type='LR', config=None,
                 description=None, attributes=None,
                 exclude_attributes=[], **kwargs):

        self.model = None
        self.dataset = None
        self.partition = None
        self.statistics = None
        self.exclude_features = None
        self.class_aggregation = None

        super(FlamingoModel, self).__init__(filename_or_obj=None, 
                                            config=config, 
                                            description=description,
                                            attributes=attributes, 
                                            exclude_attributes=exclude_attributes)

        self.initialize_model(model_type, **kwargs)

        self.update_hash()
                

    @workflow(stage=0)
    def initialize_model(self, model_type, **kwargs):
        if model_type == 'LR':
            self.model = sklearn.linear_model.LogisticRegression(**kwargs)
        else:
            raise ValueError('Unsupported model type: %s' % model_type)


    @workflow(stage=1)
    def train(self, dataset, partition=None, exclude_features=None,
              class_aggregation=None):

        self.exclude_features = exclude_features
        self.class_aggregation = class_aggregation

        if isinstance(dataset, FlamingoDataset):
            self.dataset = dataset._hash
            self.statistics = dataset.statistics

            X, Y = dataset.get_feature_set(partition=partition,
                                           exclude_features=extract_features,
                                           class_aggregation=class_aggregation)

            if self.statistics is not None:
                X = fcl.features.postprocess.normalize_features(X, self.statistics)
        else:
            raise ValueError('Expected FlamingoDataset, got %s.' % type(dataset))

        if isinstance(partition, FlamingoDatasetPartition):
            self.partition = partition._hash

        self.model.fit(X, Y)
        return self


    def predict(self, image):

        if isinstance(image, FlamingoImage):
            X = image.get_feature_set(exclude_features=self.extract_features,
                                      class_aggregation=self.class_aggregation)[0]

            if self.statistics is not None:
                X = fcl.features.postprocess.normalize_features(X, self.statistics)
        else:
            raise ValueError('Expected FlamingoImage, got %s.' % type(image))

        return self.model.predict(X)


class FlamingoIO(object):
    '''Class implements reading and writing of Flamingo objects from
    and to disk. Flamingo objects are stored as a compressed zip
    archive containing the object attributes. Each attribute is stored
    in a separate file with a name corresponding to the attribute name
    and contents corresponding to the attribute value.

    Two types of file formats are supported: JSON and
    pickle. Primitive attributes are stored as JSON, if
    possible. Primitive attributes are attributes that consist
    entirely from Python primitives. Hence, a list of Flamingo objects
    is not a primitive attribute. Non-primitive attributes, or
    attributes that somehow fail to be stored as JSON files are stored
    as pickle files. JSON and pickle files are stored in separate sub
    directories within the zip archive.

    Examples
    --------
    >>> obj = FlamingoIO('dataset.fdo').read()

    >>> obj = FlamingoDataset()
    >>> FlamingoIO('dataset.fdo').read(obj)

    >>> FlamingoIO('image.fio').write(obj)

    '''

    
    _version = 1.0

    
    def __init__(self, filename_or_fileobj, compression=zipfile.ZIP_DEFLATED):
        '''Constructor

        Parameters
        ----------
        filename_or_fileobj : str or file object
            Filename or fileobject that will be read/written
        compression : int
            Can be either ``zipfile.ZIP_STORED`` or
            ``zipfile.ZIP_DEFLATED``.

        '''
        
        self.filename_or_fileobj = filename_or_fileobj
        self.compression = compression
        
        
    def read(self, obj=None, attributes=None, exclude_attributes=[]):
        '''Read Flamingo object from file

        Parameters
        ----------
        obj : FlamingoBase, optional
            Object to be populated with attributes from file. if not
            given a Flamingo object of the type corresponding to the
            stored object is created. If given, type should match the
            stored object.
        attributes : list, optional
            List of attributes to be read. If not given, all
            attributes are read.
        exclude_attributes : list, optional
            List of attributes to be excluded from reading. Has
            preference over ``attributes``.
            
        Returns
        -------
        obj : FlamingoBase
            Flaming object with attributes populated from file.

        Raises
        ------
        TypeError
            If type of given Flamingo object does not match the type
            of the Flamingo object on disk, or if the type of the
            Flamingo object on disk cannot be determined.

        '''

        if obj is not None and not isinstance(obj, FlamingoBase):
            raise TypeError('Expected Flamingo object, got %s.' % type(obj))

        t0 = time.time()
        with zipfile.ZipFile(self.filename_or_fileobj, 'r') as fp:
            
            # get object version
            try:
                objversion = json.loads(fp.read('_version'))
            except KeyError:
                objversion = self._version
                logger.warn('Cannot read Flamingo object version, '
                            'assuming %s.' % self._version)

            # get object hash
            try:
                objhash = json.loads(fp.read('json/_hash'))
            except KeyError:
                objhash = ''
                logger.warn('Cannot read Flamingo object hash.')

            # get object type
            try:
                objtype = json.loads(fp.read('json/_type'))
            except KeyError:
                raise TypeError('Cannot determine Flamingo object type.')
            
            # get attribute hashes
            try:
                atthash = json.loads(fp.read('_hashes'))
            except KeyError:
                atthashes = {}

            # create object or check type
            if obj is None:
                obj = self.create_object(objtype)
            elif obj._type != objtype:
                raise TypeError('Expected %s, got %s.' % (obj._type, objtype))
                
            # copy attributes
            for fname in fp.namelist():
                fpath, name = os.path.split(fname)
                if fpath.startswith('collection' + os.path.sep):
                    fpath, name = os.path.split(fpath)

                if fpath == '':
                    continue

                if (attributes is not None and name not in attributes) or \
                        name in exclude_attributes:

                    # if attribute is not loaded, replace it with its
                    # hash to ensure the object hash remains constant
                    if name in atthash.keys():
                        try:
                            logger.debug('Replace %s by hash.' % name)
                            setattr(obj, name, FlamingoHash(atthash[name]))
                        except AttributeError:
                            # probably a dynamic property, ignore.
                            pass

                    continue

                try:
                    if fpath == 'hash':
                        logger.debug('Read hash for %s.' % name)
                        setattr(obj, name, FlamingoHash(fp.read(fname)))
                    elif fpath == 'json':
                        logger.debug('Read %s from JSON file.' % name)
                        setattr(obj, name, json.loads(fp.read(fname)))
                    elif fpath == 'pickle':
                        logger.debug('Read %s from pickle file.' % name)
                        setattr(obj, name, pickle.loads(fp.read(fname)))
                    elif fpath == 'collection':
                        attr = getattr(obj, name)
                        logger.debug('Read item %d of %s from collection' % 
                                     (len(attr)+1, name))
                        fo = io.BytesIO(fp.read(fname))
                        item = FlamingoIO(fo).read(
                            attributes=attributes, 
                            exclude_attributes=exclude_attributes)
                        attr.append(item)
                        setattr(obj, name, attr)
                        fo.close()
                    else:
                        logger.warn('Unknown attribute format: %s. Ignore.' % fpath)
                except AttributeError:
                    logger.warn('Attribute not writable or does not exist: %s. '
                                'Ignore.' % name)

            # store original hash
            setattr(obj, '_fromhash', objhash)

        logger.info('Read object from %s in %0.1f s.' % (self.filename_or_fileobj,
                                                         time.time() - t0))
                    
        return obj
            
        
    def write(self, obj, attributes=None, exclude_attributes=[]):
        '''Write Flamingo object to file

        Parameters
        ----------
        obj : FlamingoBase
            Object to be written to file.
        attributes : list, optional
            List of attributes to be written. If not given, all
            attributes are written.
        exclude_attributes : list, optional
            List of attributes to be excluded from writing. Has
            preference over ``attributes``.
            
        Raises
        ------
        TypeError
            If the given object is not a valid Flamingo object.

        '''

        if not isinstance(obj, FlamingoBase):
            raise TypeError('Expected Flamingo object, got %s.' % type(obj))

        t0 = time.time()
        with zipfile.ZipFile(self.filename_or_fileobj, 'w', self.compression) as fp:

            # store version
            fp.writestr('_version', json.dumps(self._version))

            # store attribute hashes
            atthash = obj._hash_attributes()
            fp.writestr('_hashes', json.dumps(atthash))

            # store attributes
            for name, attr in obj.iterate_attributes():
            
                if (attributes is not None and name not in attributes) or \
                        name in exclude_attributes:
                    logger.debug('Store hash for %s.' % name)

                    # if attribute is not written, replace it with its
                    # hash to ensure the object hash remains constant
                    fp.writestr(os.path.join('hash', name), atthash[name])

                    continue

                if self.is_primitive(attr):
                    # store primitives as json
                    try:
                        logger.debug('Store %s as JSON file.' % name)
                        fp.writestr(os.path.join('json', name), 
                                    json.dumps(attr, indent=4))
                    except:
                        logger.debug('Tried to write %s as JSON, but failed. '
                                     'Will store as pickle now.' % name)
                        pass
                    else:
                        continue
                elif self.is_collection(attr):
                    # store flamingo object collections as a series of
                    # Flamingo object files
                    for i, a in enumerate(attr):
                        logger.debug('Store item #%d from %s as uncompressed '
                                     'zip file.' % (i, name))
                        fo = io.BytesIO()
                        FlamingoIO(fo, compression=zipfile.ZIP_STORED).write(
                            a, attributes=attributes, 
                            exclude_attributes=exclude_attributes)
                        #nbytes = fo.seek(0, io.SEEK_END)
                        fo.seek(0, io.SEEK_SET)
                        fp.writestr(os.path.join('collection', name, '%04d' % i), 
                                    fo.getvalue()) 
                        fo.close()
                    continue

                # store as pickle otherwise
                logger.debug('Store %s as pickle file.' % name)
                fp.writestr(os.path.join('pickle', name), pickle.dumps(attr))

        logger.info('Wrote object to %s in %0.1f s.' % (self.filename_or_fileobj, 
                                                        time.time() - t0))


    def is_primitive(self, attr):
        '''Checks if attribute is primitive

        Recursively search the attribute structure for non-primitive
        elements. If any non-primitive element is found, returns
        False. Otherwise returns True.

        Parameters
        ----------
        attr : any
            Attribute value to be checked for non-primitive elements.

        Returns
        -------
        bool
            Boolean indicating whether attribute is primitive.

        '''

        if isinstance(attr, float) or \
           isinstance(attr, str) or \
           isinstance(attr, six.integer_types) or \
           isinstance(attr, six.string_types) or \
           isinstance(attr, six.text_type) or \
           isinstance(attr, six.binary_type) or \
           attr is None:
            return True
        elif isinstance(attr, (list, tuple, set)):
            return np.all([self.is_primitive(a) for a in attr])
        elif isinstance(attr, (dict)):
            return np.all([self.is_primitive(a) for a in attr.values()])
        else:
            return False


    def is_collection(self, attr):
        '''Checks if attribute is a collection of Flamingo objects

        Parameters
        ----------
        attr : any
            Attribute value to be checked for Flamingo objects.

        Returns
        -------
        bool
            Boolean indicating whether attribute is a collection of
            Flamingo objects.

        '''

        if isinstance(attr, list):
            if all([isinstance(a, FlamingoBase) for a in attr]):
                return True

        return False


    @staticmethod
    def create_object(objtype):
        '''Create Flamingo object of given type

        Parameters
        ----------
        objtype : str
            Flamingo object type to be created.

        Returns
        -------
        FlamingoBase
            Flamingo object of requested type

        Raises
        ------
        TypeError
            Raises error if requested type is not recognized

        '''

        if objtype == 'Flamingo Object':
            return FlamingoBase()
        elif objtype == 'Flamingo Image Object':
            return FlamingoImage()
        elif objtype == 'Flamingo Dataset Object':
            return FlamingoDataset()
        elif objtype == 'Flamingo Dataset Partition Object':
            return FlamingoDatasetPartition()
        elif objtype == 'Flamingo Model Object':
            return FlamingoModel()
        else:
            raise TypeError('Unknown Flamingo Object type: %s.' % objtype)
