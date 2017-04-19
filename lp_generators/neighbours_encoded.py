''' Neighbourhood modifiers that return EncodedInstance objects.
Input instances must be able to return alpha() and beta() results to be
copied in this scheme. '''

import numpy as np

from .instance import EncodedInstance
from .neighbours_common import (
    _scale_vector_entry, _exchange_basis, _scale_lhs_entry,
    _remove_lhs_entry, _add_lhs_entry)


def copied_neighbour(func):
    ''' Intercept call to decorated function, copying the instance first.
    The wrapper calls :func on the instance, then returns the copy.
    Wrapped function can be sure the :instance argument in an EncodedInstance,
    with data stored as _lhs_matrix, _alpha, _beta. '''
    def copied_neighbour_fn(instance, random_state, *args, **kwargs):
        instance = EncodedInstance(
            lhs=np.copy(instance.lhs()),
            alpha=np.copy(instance.alpha()),
            beta=np.copy(instance.beta()))
        func(instance, random_state, *args, **kwargs)
        return instance
    return copied_neighbour_fn


@copied_neighbour
def exchange_basis(instance, random_state, count):
    _exchange_basis(instance._beta, random_state, count=count)


@copied_neighbour
def scale_optvalue(instance, random_state, count, mean, sigma):
    _scale_vector_entry(instance._alpha, random_state, mean, sigma, dist='lognormal', count=count)


@copied_neighbour
def remove_lhs_entry(instance, random_state, count):
    _remove_lhs_entry(instance._lhs_matrix, random_state, count=count)


@copied_neighbour
def add_lhs_entry(instance, random_state, count, mean, sigma):
    _add_lhs_entry(instance._lhs_matrix, random_state, mean, sigma, count=count)


@copied_neighbour
def scale_lhs_entry(instance, random_state, count, mean, sigma):
    _scale_lhs_entry(instance._lhs_matrix, random_state, mean, sigma, count=count)
