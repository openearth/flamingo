import os
import sys
import resource
from logging import *

LOG_FORMAT = '%(asctime)s [%(levelname)-5.5s]  %(message)s'

def init(path=os.getcwd(), filename='batch.log', level=DEBUG):
    fmt = Formatter(LOG_FORMAT)
    root = getLogger()
    root.setLevel(level)

    root.handlers = [] # reset all handlers

    fh = FileHandler(os.path.join(path, filename))
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return root

def raise_and_log(msg, exception=Exception):
    error(msg)
    raise exception(msg)

def memory_usage(id=''):
    if id == '':
        debug(_memory_usage())
    else:
        debug('%s [%s]' % (_memory_usage(), id))

def _memory_usage():
    return 'Memory usage: %10d %10d %10d' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                             resource.getrusage(resource.RUSAGE_SELF).ru_ixrss / 1024,
                                             resource.getrusage(resource.RUSAGE_SELF).ru_idrss / 1024)
