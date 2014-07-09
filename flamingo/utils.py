import logging
from functools import wraps

import psutil


def printinfo(f):
    """print before and after a function is called"""
    @wraps(f)
    def wrapper(*args, **kwds):
        name = getattr(f, '__name__')
        logger = logging.getLogger(name)
        logger.debug("available physical memory before call of {}: {}".format(
            name,
            psutil.virtual_memory()
        ))
        logger.debug("available swap memory before call of {}: {}".format(
            name,
            psutil.swap_memory()
        ))
        result = f(*args, **kwds)
        logger.debug("available physical memory after call of {}: {}".format(
            name,
            psutil.virtual_memory()
        ))
        logger.debug("available swap memory after call of {}: {}".format(
            name,
            psutil.swap_memory()
        ))

        return result
    return wrapper
