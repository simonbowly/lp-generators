''' Feature calculation functions for a canonical form LP instance. '''

import numpy as np

from .lp_ext import LPCy


def coeff_features(instance):
    ''' Features based on variable/constraint degree and coefficient
    value distributions. '''
    lhs = instance.lhs()
    nonzeros = lhs != 0
    rhs = instance.rhs()
    objective = instance.objective()
    result = dict(
        variables=int(instance.variables),
        constraints=int(instance.constraints),
        nonzeros=int(np.sum(nonzeros)),
        lhs_std=float(lhs[np.where(nonzeros)].std()),
        lhs_mean=float(lhs[np.where(nonzeros)].mean()),
        lhs_abs_mean=float(np.abs(lhs[np.where(nonzeros)]).mean()),
        rhs_std=float(rhs.std()),
        rhs_mean=float(rhs.mean()),
        obj_std=float(objective.std()),
        obj_mean=float(objective.mean()))
    result['coefficient_density'] = float(
        result['nonzeros'] / (instance.variables * instance.constraints))
    var_degree, cons_degree = degree_seq(lhs)
    result.update(
        cons_degree_min=int(cons_degree.min()),
        cons_degree_max=int(cons_degree.max()),
        var_degree_min=int(var_degree.min()),
        var_degree_max=int(var_degree.max()))
    result.update(
        rhs_mean_normed=result['rhs_mean'] / result['lhs_abs_mean'],
        obj_mean_normed=result['obj_mean'] / result['lhs_abs_mean'])
    return result


def solution_features(instance):
    ''' Solve the instance (using extension module) and retrieve solution
    data to calculate features of the LP relaxation solution. '''
    model = LPCy()
    model.construct_dense_canonical(
        instance.variables, instance.constraints,
        np.asarray(instance.lhs()),
        np.asarray(instance.rhs()),
        np.asarray(instance.objective()))
    model.solve()
    if (model.get_solution_status() != 0):
        return dict(solvable=False)
    # features specific to instances with a relaxation solution
    primals = model.get_solution_primals()
    fractional_components = np.abs(primals - np.round(primals))
    slacks = model.get_solution_slacks()
    return dict(
        solvable=True,
        binding_constraints=int(np.sum(np.abs(slacks < 10 ** -10))),
        fractional_primal=int(np.sum(fractional_components > 10 ** -10)),
        total_fractionality=float(np.sum(fractional_components)))


def degree_seq(lhs):
    ''' Return variable and constraint degree coefficients as numpy arrays. '''
    lhs = np.array(lhs)
    nonzeros = lhs != 0
    return nonzeros.sum(axis=0), nonzeros.sum(axis=1)
