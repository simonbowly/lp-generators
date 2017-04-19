
import numpy as np

from lp_generators.instance import EncodedInstance


EQ_TOLERANCE = 10 ** -10


def assert_approx_equal(m1, m2):
    assert np.all(np.abs(m1 - m2) < EQ_TOLERANCE)


def random_encoded(variables, constraints):
    A = np.random.random((constraints, variables))
    alpha = np.random.random(variables + constraints)
    beta = np.zeros(variables + constraints)
    basis = np.random.choice(
        variables + constraints,
        size=constraints, replace=False)
    beta[basis] = 1
    return EncodedInstance(lhs=A, alpha=alpha, beta=beta)
