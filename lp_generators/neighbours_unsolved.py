''' Neighbourhood modifiers that return UnsolvedInstance objects.
Input instances must be able to return objective() and rhs() results to be
copied in this scheme. '''

import numpy as np

from .instance import UnsolvedInstance
from .neighbours_common import (
    _scale_vector_entry, _scale_lhs_entry, _remove_lhs_entry, _add_lhs_entry)


def copied_neighbour(func):
    ''' Intercept call to decorated function, copying the instance first.
    The wrapper calls :func on the instance, then returns the copy.
    Wrapped function can be sure the :instance argument in an UnsolvedInstance,
    with data stored as _lhs_matrix, _rhs, _objective. '''
    def copied_neighbour_fn(instance, random_state, *args, **kwargs):
        new_instance = UnsolvedInstance(
            lhs=np.copy(instance.lhs()),
            rhs=np.copy(instance.rhs()),
            objective=np.copy(instance.objective()))
        func(new_instance, random_state, *args, **kwargs)
        return new_instance
    return copied_neighbour_fn


@copied_neighbour
def scale_obj_entry(instance, random_state, count, mean, sigma):
    _scale_vector_entry(instance._objective, random_state, mean, sigma, dist='normal', count=count)


@copied_neighbour
def scale_rhs_entry(instance, random_state, count, mean, sigma):
    _scale_vector_entry(instance._rhs, random_state, mean, sigma, dist='normal', count=count)


@copied_neighbour
def remove_lhs_entry(instance, random_state, count):
    _remove_lhs_entry(instance._lhs_matrix, random_state, count=count)


@copied_neighbour
def add_lhs_entry(instance, random_state, count, mean, sigma):
    _add_lhs_entry(instance._lhs_matrix, random_state, mean, sigma, count=count)


@copied_neighbour
def scale_lhs_entry(instance, random_state, count, mean, sigma):
    _scale_lhs_entry(instance._lhs_matrix, random_state, mean, sigma, count=count)
