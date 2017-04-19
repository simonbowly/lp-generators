''' Generates encoded primal-dual solutions for instances by generating
a basis vector and values for the resulting non-zero x, y, r, s values. '''

import numpy as np


def generate_beta(variables, constraints, basis_split, random_state):
    ''' Generate a vector specifying a random basis for an instance. '''
    primal_count = int(round(basis_split * min(variables, constraints)))
    slack_count = constraints - primal_count
    primals_in_basis = random_state.choice(
        np.arange(0, variables),
        size=primal_count, replace=False)
    slacks_in_basis = random_state.choice(
        np.arange(variables, variables + constraints),
        size=slack_count, replace=False)
    beta_vector = np.zeros(variables + constraints)
    basis = np.concatenate([primals_in_basis, slacks_in_basis])
    beta_vector[basis] = 1
    return beta_vector


def generate_alpha(variables, constraints, frac_violations, beta_param,
                   mean_primal, std_primal, mean_dual, std_dual, random_state):
    ''' Generate a non-negative vector of values for an optimal solution. '''
    # choosing which solution values will be non-integral
    num_violations = int(round(frac_violations * variables))
    ind_frac = random_state.choice(
        np.arange(0, variables), size=num_violations, replace=False)
    # fractional components
    frac_values = random_state.beta(
        a=beta_param, b=beta_param, size=num_violations)
    # subtract fractional components from a base vector of integers
    primal_alpha_vector = np.ceil(random_state.lognormal(
        mean=mean_primal, sigma=std_primal, size=variables)).astype(np.float)
    primal_alpha_vector[ind_frac] = primal_alpha_vector[ind_frac] - frac_values
    # slack values
    dual_alpha_vector = random_state.lognormal(
        mean=mean_dual, sigma=std_dual,
        size=constraints).astype(np.float)
    alpha_vector = np.concatenate([primal_alpha_vector, dual_alpha_vector])
    return alpha_vector
