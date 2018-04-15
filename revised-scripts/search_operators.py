''' Modificaion operators should pick from the same distribution as the generators. '''

import itertools
import numpy as np

from lp_generators.instance import UnsolvedInstance, SolvedInstance, Solution
from lp_generators.features import coeff_features, solution_features
from lp_generators.lhs_generators import generate_lhs
from lp_generators.solution_generators import generate_alpha


def generate_rhs(variables, constraints, rhs_mean, rhs_std, random_state):
    return random_state.normal(loc=rhs_mean, scale=rhs_std, size=constraints)


def generate_objective(variables, constraints, obj_mean, obj_std, random_state):
    return random_state.normal(loc=obj_mean, scale=obj_std, size=variables)


def lhs_row(random_state, ncols):
    lhs_params = dict(
        density=random_state.uniform(low=0.0, high=1.0),
        pv=random_state.uniform(low=0.0, high=1.0),
        pc=random_state.uniform(low=0.0, high=1.0),
        coeff_loc=random_state.uniform(low=-2.0, high=2.0),
        coeff_scale=random_state.uniform(low=0.1, high=1.0))
    lhs = generate_lhs(
        random_state=random_state, variables=ncols, constraints=2,
        **lhs_params).todense()
    return lhs[0]


def lhs_col(random_state, nrows):
    lhs_params = dict(
        density=random_state.uniform(low=0.0, high=1.0),
        pv=random_state.uniform(low=0.0, high=1.0),
        pc=random_state.uniform(low=0.0, high=1.0),
        coeff_loc=random_state.uniform(low=-2.0, high=2.0),
        coeff_scale=random_state.uniform(low=0.1, high=1.0))
    lhs = generate_lhs(
        random_state=random_state, variables=2, constraints=nrows,
        **lhs_params).todense()
    return lhs.transpose()[0].transpose()


def lp_random_row(random_state, ncols):
    ''' Create a random row A_j with rhs b_j. '''
    rhs_params = dict(
        rhs_mean=random_state.uniform(low=-100.0, high=100.0),
        rhs_std=random_state.uniform(low=1.0, high=30.0))
    rhs = generate_rhs(
        random_state=random_state, variables=ncols, constraints=1,
        **rhs_params)
    return lhs_row(random_state, ncols), rhs[0]


def lp_random_col(random_state, nrows):
    ''' Create a random column A_i with objective c_i. '''
    objective_params = dict(
        obj_mean=random_state.uniform(low=-100.0, high=100.0),
        obj_std=random_state.uniform(low=1.0, high=10.0))
    objective = generate_objective(
        random_state=random_state, variables=1, constraints=nrows,
        **objective_params)
    return lhs_col(random_state, nrows), objective[0]


def lp_column_neighbour(rstate, instance, n_replace):
    ''' Create a neighbour by replacing a number of columns A_i, c_i
    with new random columns. '''
    a, b, c = instance.lhs().copy(), instance.rhs().copy(), instance.objective().copy()
    inds = rstate.choice(instance.variables, n_replace, replace=False)
    for ind in inds:
        a_i, c_i = lp_random_col(rstate, instance.constraints)
        a[:, ind] = np.array([a_i]).transpose()
        c[ind] = c_i
    return UnsolvedInstance(lhs=a, rhs=b, objective=c)


def lp_row_neighbour(rstate, instance, n_replace):
    ''' Create a neighbour by replacing a number of rows A_j, b_j
    with new random rows. '''
    a, b, c = instance.lhs().copy(), instance.rhs().copy(), instance.objective().copy()
    inds = rstate.choice(instance.constraints, n_replace, replace=False)
    for ind in inds:
        a_j, b_j = lp_random_row(rstate, instance.variables)
        a[ind, :] = np.array([a_j])
        b[ind] = b_j
    return UnsolvedInstance(lhs=a, rhs=b, objective=c)


def solution_element(random_state):
    alpha_params = dict(
        frac_violations=random_state.uniform(low=0.0, high=1.0),
        beta_param=random_state.lognormal(mean=-0.2, sigma=1.8),
        mean_primal=0,
        std_primal=1,
        mean_dual=0,
        std_dual=1)
    alpha = generate_alpha(
        random_state=random_state,
        variables=1, constraints=0,
        **alpha_params)
    return alpha[0]


def sp_random_row(rstate, ncols):
    A_j = lhs_row(rstate, ncols)
    alpha = solution_element(rstate)
    if rstate.uniform(0, 1) < 0.5:
        y_j, s_j = alpha, 0
    else:
        y_j, s_j = 0, alpha
    return A_j, y_j, s_j


def sp_random_col(rstate, nrows):
    ''' Create a random column A_i with objective c_i. '''
    A_i = lhs_col(rstate, nrows)
    alpha = solution_element(rstate)
    if rstate.uniform(0, 1) < 0.5:
        x_i, r_i = alpha, 0
    else:
        x_i, r_i = 0, alpha
    return A_i, x_i, r_i


def encoded_column_neighbour(rstate, instance, n_replace):
    loc = rstate.uniform(1, 4)
    a, x, y, r, s = (
        instance.lhs().copy(),
        instance.solution().x.copy(),
        instance.solution().y.copy(),
        instance.solution().r.copy(),
        instance.solution().s.copy())
    inds = rstate.choice(instance.variables, n_replace, replace=False)
    for ind in inds:
        a_i, x_i, r_i = sp_random_col(rstate, instance.constraints)
        a[:, ind] = a_i
        x[ind] = x_i
        r[ind] = r_i
    basis = np.zeros(instance.variables + instance.constraints)
    return SolvedInstance(lhs=a, solution=Solution(x=x, y=y, r=r, s=s, basis=basis))


def encoded_row_neighbour(rstate, instance, n_replace):
    loc = rstate.uniform(1, 4)
    a, x, y, r, s = (
        instance.lhs().copy(),
        instance.solution().x.copy(),
        instance.solution().y.copy(),
        instance.solution().r.copy(),
        instance.solution().s.copy())
    inds = rstate.choice(instance.constraints, n_replace, replace=False)
    for ind in inds:
        a_j, y_j, s_j = sp_random_col(rstate, instance.variables)
        a[ind, :] = a_j.transpose()
        y[ind] = y_j
        s[ind] = s_j
    basis = np.zeros(instance.variables + instance.constraints)
    return SolvedInstance(lhs=a, solution=Solution(x=x, y=y, r=r, s=s, basis=basis))
