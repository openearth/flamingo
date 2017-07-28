from __future__ import absolute_import

import os
import re
import sys
import docopt
import logging


# initialize log
logger = logging.getLogger(__name__)


def classify():
    '''flamingo-classify : train, score and apply image classification models

Usage:
    flamingo-classify preprocess <path> [options]
    flamingo-classify partition <path> [options]
    flamingo-classify train <path> [options]
    flamingo-classify score <path> [options]
    flamingo-classify regularize <path> [options]
    flamingo-classify predict <path> [options]
    
Positional arguments:
    path               location of image dataset
    
Options:
    -h, --help         show this help message and exit
    --config=FILE      configuration file to use instead of command line options
    --overwrite        overwrite existing files
    --verbose          print logging messages
    
Use flamingo-classify <action> --help for additional options.

    '''

    if len(sys.argv) > 1:
        if sys.argv[1] == 'preprocess':
            return classify_preprocess()
        elif sys.argv[1] == 'partition':
            return classify_partition()
        elif sys.argv[1] == 'train':
            return classify_train()
        elif sys.argv[1] == 'score':
            return classify_score()
        elif sys.argv[1] == 'regularize':
            return classify_regularize()
        elif sys.argv[1] == 'predict':
            return classify_predict()
        
    args = docopt.docopt(classify.__doc__)


def classify_preprocess():
    '''flamingo-classify : train, score and apply image classification models

Usage:
    flamingo-classify preprocess <path> [options]
    
Positional arguments:
    path               location of image dataset

Options:    
    -h, --help         show this help message and exit
    --segmentate       create segmentation of images
    --channels         include channel extraction
    --features         include feature extraction
    --extract          extract channels/features
    --update           update channels/features
    --normalize        normalize channels/features
    --config=FILE      configuration file to use instead of command line options
    --overwrite        overwrite existing files
    --verbose          print logging messages

    '''

    args = docopt.docopt(classify_preprocess.__doc__)
    set_logger(args['--verbose'])

    


def classify_partition():
    '''flamingo-classify : train, score and apply image classification models

Usage:
    flamingo-classify partition <path> [options]
    
Positional arguments:
    path               location of image dataset

Options:    
    -h, --help         show this help message and exit
    --n=N              number of partitions, use 0 for exhaustive [default: 5]
    --frac=FRAC        fraction of images used for testing [default: 0.25]
    --config=FILE      configuration file to use instead of command line options
    --overwrite        overwrite existing files
    --verbose          print logging messages

    '''

    args = docopt.docopt(classify_partition.__doc__)
    set_logger(args['--verbose'])


def classify_train():
    '''flamingo-classify : train, score and apply image classification models

Usage:
    flamingo-classify train <path> [options]
    
Positional arguments:
    path               location of image dataset

Options:    
    -h, --help         show this help message and exit
    --type=NAME        model type to train [default: LR]
    --partition=N      only train this partition
    --config=FILE      configuration file to use instead of command line options
    --overwrite        overwrite existing files
    --verbose          print logging messages

    '''

    args = docopt.docopt(classify_train.__doc__)
    set_logger(args['--verbose'])


def classify_score():
    '''flamingo-classify : train, score and apply image classification models

Usage:
    flamingo-classify score <path> [options]
    
Positional arguments:
    path               location of image dataset

Options:    
    -h, --help         show this help message and exit
    --model=NAME       name of model to be scored
    --config=FILE      configuration file to use instead of command line options
    --overwrite        overwrite existing files
    --verbose          print logging messages

    '''

    args = docopt.docopt(classify_score.__doc__)
    set_logger(args['--verbose'])


def classify_regularize():
    '''flamingo-classify : train, score and apply image classification models

Usage:
    flamingo-classify regularize <path> [options]
    
Positional arguments:
    path               location of image dataset

Options:
    -h, --help         show this help message and exit
    --C=C              regulariation coefficients to be evaluated [default: 0.001,0.01,0.1,1,10]
    --type=NAME        model type to train [default: LR]
    --partition=N      only train this partition
    --config=FILE      configuration file to use instead of command line options
    --overwrite        overwrite existing files
    --verbose          print logging messages

    '''

    args = docopt.docopt(classify_regularize.__doc__)
    set_logger(args['--verbose'])


def classify_predict():
    '''flamingo-classify : train, score and apply image classification models

Usage:
    flamingo-classify predict <path> [options]
    
Positional arguments:
    path               location of image dataset

Options:    
    -h, --help         show this help message and exit
    --model=NAME       name of model to be used for prediction
    --config=FILE      configuration file to use instead of command line options
    --overwrite        overwrite existing files
    --verbose          print logging messages

    '''

    args = docopt.docopt(classify_predict.__doc__)
    set_logger(args['--verbose'])

    
def rectify():
    pass


def calibrate():
    pass


def set_logger(verbosity_stream=logging.INFO, verbosity_file=logging.DEBUG):

    name = re.sub('[^a-zA-Z0-9]+', '_', os.path.split(os.getcwd())[1])
    
    # initialize file logger
    logging.basicConfig(level=int(verbosity_file),
                        format='%(asctime)-15s %(name)-8s %(levelname)-8s %(message)s',
                        filename='flamingo_%s.log' % name)
    
    # initialize console logger
    console = logging.StreamHandler()
    console.setLevel(int(verbosity_stream))
    console.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))
    logging.getLogger('').addHandler(console)
