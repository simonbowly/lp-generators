''' Elementwise modifiers to instance data. Functions here take a matrix of
instance data and modify in place. Implementors at the instance level should
copy the data. '''

import numpy as np


def apply_repeat(func):
    ''' Add a count argument to the decorated modifier function to allow its
    operation to be applied repeatedly. '''
    def apply_repeat_fn(arr, random_state, *args, count, **kwargs):
        for _ in range(count):
            func(arr, random_state, *args, **kwargs)
    return apply_repeat_fn


@apply_repeat
def _exchange_basis(beta, random_state):
    ''' Exchange elements in a basis vector. '''
    incoming = random_state.choice(np.where(beta == 0)[0])
    outgoing = random_state.choice(np.where(beta == 1)[0])
    beta[incoming] = 1
    beta[outgoing] = 0


@apply_repeat
def _scale_vector_entry(vector, random_state, mean, sigma, dist):
    ''' Scale element in a one dimensional vector. '''
    scale_index = random_state.choice(vector.shape[0])
    if dist == 'normal':
        scale_value = random_state.normal(loc=mean, scale=sigma)
    elif dist == 'lognormal':
        scale_value = random_state.lognormal(mean=mean, sigma=sigma)
    else:
        raise ValueError('Vector entry scales only with normal or lognormal')
    vector[scale_index] = scale_value * vector[scale_index]


@apply_repeat
def _remove_lhs_entry(lhs, random_state):
    ''' Remove element from lhs matrix. '''
    nz_rows, nz_cols = np.where(lhs != 0)
    if len(nz_rows) == 0:
        return
    remove_index = random_state.choice(nz_rows.shape[0])
    lhs[nz_rows[remove_index], nz_cols[remove_index]] = 0


@apply_repeat
def _add_lhs_entry(lhs, random_state, mean, sigma):
    ''' Add an element to lhs matrix. '''
    zero_rows, zero_cols = np.where(lhs == 0)
    if len(zero_rows) == 0:
        return
    add_index = random_state.choice(zero_rows.shape[0])
    add_value = random_state.normal(loc=mean, scale=sigma)
    lhs[zero_rows[add_index], zero_cols[add_index]] = add_value


@apply_repeat
def _scale_lhs_entry(lhs, random_state, mean, sigma):
    ''' Scale an element of the constraint matrix. '''
    nz_rows, nz_cols = np.where(lhs != 0)
    if len(nz_rows) == 0:
        return lhs
    scale_index = random_state.choice(nz_rows.shape[0])
    scale_value = random_state.normal(loc=mean, scale=sigma)
    lhs[nz_rows[scale_index], nz_cols[scale_index]] = (
        scale_value * lhs[nz_rows[scale_index], nz_cols[scale_index]])
