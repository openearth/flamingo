import os
import json
import logging
import inspect
import ConfigParser

CLASSIFICATION_DEFAULTS = {
    'general'           : {'dataset' : '',
                           'model_type' : 'LR',
                           'model_dataset' : '',
                           'colorspace' : 'rgb',
                           'class_aggregation' : ''},
    'segmentation'      : {'enabled' : True,
                           'method' : 'slic',
                           'method_params' : {'n_segments' : 2000,
                                              'compactness' : 10},
                           'remove_disjoint' : True,
                           'extract_contours' : False},
    'channels'          : {'enabled' : True,
                           'methods' : ['gaussian',
                                        'sobel'],
                           'methods_params' : {'frequencies' : [0.05, 0.15, 0.25],
                                               'thetas' : [0.0, 0.785, 1.571, 2.356],
                                               'sigmas' : [1, 8, 15]}},
    'features'          : {'enabled' : True,
                           'feature_blocks' : 'all'},
    'relative_location' : {'enabled' : False,
                           'n' : 100,
                           'sigma' : 2},
    'partition'         : {'enabled' : True,
                           'n_partitions' : 5,
                           'frac_validation' : 0.0,
                           'frac_test' : 0.25,
                           'force_split' : False},
    'train'             : {'partitions' : 'all'},
    'score'             : {},
    'regularization'    : {'partition' : 0,
                           'C' : [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
}


def read_config(cfgfile, defaults=CLASSIFICATION_DEFAULTS):

    if not cfgfile or not os.path.exists(cfgfile):
        return None

    settings = defaults.copy()

    cfg = ConfigParser.RawConfigParser()
    cfg.read(cfgfile)

    for section, options in settings.iteritems():
        if cfg.has_section(section):
            for option, value in options.iteritems():
                if cfg.has_option(section, option):
                    if type(value) is int:
                        value = cfg.getint(section, option)
                    elif type(value) is float:
                        value = cfg.getfloat(section, option)
                    elif type(value) is bool:
                        value = cfg.getboolean(section, option)
                    elif type(value) in [list, dict]:
                        value = json.loads(cfg.get(section, option))
                    else:
                        value = cfg.get(section, option)

                    settings[section][option] = value

    settings = _parse_references(settings)

    return settings


def write_config(cfgfile, defaults=CLASSIFICATION_DEFAULTS):

    cfg = ConfigParser.RawConfigParser()

    for section, options in defaults.iteritems():
        cfg.add_section(section)
        for option, value in options.iteritems():
            if type(value) in [list, dict]:
                cfg.set(section, option, json.dumps(value))
            else:
                cfg.set(section, option, str(value))
    
    with open(cfgfile, 'wb') as fp:
        cfg.write(fp)


def parse_config(sections=[]):
    def wrapper(f):
        def parse(*args, **kwargs):
            if kwargs.has_key('cfg'):
                if 'cfg' in inspect.getargspec(f).args:
                    kwargs.update(get_function_args(f, kwargs['cfg'], sections))
                else:
                    name = getattr(f, '__name__')
                    logger = logging.getLogger(name)
                    logger.warn('Function does nog support config argument. Ignored. [%s]' % name)
                    del kwargs['cfg']
            return f(*args, **kwargs)
        return parse
    return wrapper


def get_function_args(fcn, cfg, sections=[]):

    args = {}

    if not cfg:
        return args

    if not type(sections) is list:
        sections = [sections]

    if not 'general' in sections:
        sections.append('general')

    names = inspect.getargspec(fcn).args
    for section in sections:
        if cfg[section].has_key('enabled'):
            if section in names:
                args[section] = cfg[section]['enabled']
        for arg in names:
            if cfg[section].has_key(arg):
                args[arg] = cfg[section][arg]

    return args


def _parse_references(settings):

    if settings:
        if settings['general'].has_key('class_aggregation'):
            fname = settings['general']['class_aggregation']
            if os.path.exists(fname):
                with open(fname, 'r') as fp:
                    settings['general']['class_aggregation'] = json.load(fp)
            else:
                settings['general']['class_aggregation'] = None

    return settings

