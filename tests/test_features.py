
import pytest
import numpy as np

import lp_generators.features as features
from lp_generators.instance import UnsolvedInstance

#TODO test serialisable

@pytest.fixture
def unsolved_instance():
    return UnsolvedInstance(
        lhs=np.matrix([
            [ 0.42,  0.61,  0.06,  0.01,  0.49],
            [ 0.74,  0.12,  0.57,  0,     0.23],
            [ 0.78,  0.92,  0,     0.32,  0.64],
            [ 0.02,  0.67,  0.2 ,  0.45,  0.39]]),
        rhs=np.array([ 0.6093,  0.683 ,  1.2136,  0.9063]),
        objective=np.array([-0.6982,  0.3623, -0.479 ,  0.0235,  0.1651]))


def test_coeff_features(unsolved_instance):
    result = features.coeff_features(unsolved_instance)
    assert result['variables'] == 5
    assert result['constraints'] == 4
    assert result['nonzeros'] == 18
    assert abs(result['coefficient_density'] - 0.9) < 10 ** -5
    assert abs(result['lhs_std'] - 0.26874651878308059) < 10 ** -5
    assert abs(result['lhs_mean'] - 0.42444444444444446) < 10 **- 5
    assert abs(result['rhs_std'] - 0.23513981479111529) < 10 ** -5
    assert abs(result['rhs_mean'] - 0.85304999999999997) < 10 ** -5
    assert abs(result['obj_std'] - 0.39938589158857379) < 10 ** -5
    assert abs(result['obj_mean'] - -0.12525999999999998) < 10 ** -5
    assert result['cons_degree_min'] == 4
    assert result['cons_degree_max'] == 5
    assert result['var_degree_min'] == 3
    assert result['var_degree_max'] == 4


def test_degree_seq(unsolved_instance):
    var_degree, cons_degree = features.degree_seq(unsolved_instance.lhs())
    assert np.all(var_degree == [4, 4, 3, 3, 4])
    assert np.all(cons_degree == [5, 4, 4, 5])
