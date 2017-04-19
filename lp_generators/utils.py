''' Utilities for attaching data to instances, writing generated files,
using temporary files for performance calculations, and randomly seeding
processes. '''

import tempfile
import os
from contextlib import contextmanager, suppress
import sys
import functools
import random


@contextmanager
def temp_file_path(ext=''):
    ''' Context manager returning a unique temporary file path without
    actually creating the file. Deletes the file on exit of the context
    if it exists. '''
    path = tempfile.mktemp() + ext
    yield path
    with suppress(FileNotFoundError):
        os.remove(path)


def calculate_data(*calculators):
    ''' Wrap a function which generates instances, passing instances to
    calculation functions before returning. Results from the calculation
    functions are added to the instances data dictionary. '''
    def calculate_data_decorator(func):
        @functools.wraps(func)
        def calculate_data_fn(*args, **kwargs):
            instance = func(*args, **kwargs)
            if not hasattr(instance, 'data'):
                instance.data = dict()
            for calculator in calculators:
                instance.data.update(calculator(instance))
            return instance
        return calculate_data_fn
    return calculate_data_decorator


def write_instance(write_func, name_format):
    ''' Wrap a function which generates instances, passing the instance to a
    writer function before returning it. :name_format should use members of the
    :data dictionary of the instance to generate a unique name. '''
    def write_instance_decorator(func):
        @functools.wraps(func)
        def write_instance_fn(*args, **kwargs):
            # ensure the directory exists
            directory, _ = os.path.split(name_format)
            with suppress(FileExistsError):
                os.makedirs(directory)
            # generate, write, and pass the instance on
            instance = func(*args, **kwargs)
            write_func(instance, name_format.format(**instance.data))
            return instance
        return write_instance_fn
    return write_instance_decorator


def system_random_seeds(n, bits):
    ''' Generator for a list of system random seeds of the specified bit size. '''
    rand = random.SystemRandom()
    for _ in range(n):
        yield rand.getrandbits(bits)
