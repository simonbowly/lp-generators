
import pytest
import numpy as np

from lp_generators.instance import EncodedInstance, Solution, SolvedInstance, UnsolvedInstance
from .testing import assert_approx_equal


@pytest.fixture
def variables():
    return 5


@pytest.fixture
def constraints():
    return 4


@pytest.fixture
def lhs_matrix():
    return np.matrix([
        [ 0.42,  0.61,  0.06,  0.01,  0.49],
        [ 0.74,  0.12,  0.57,  0.23,  0.23],
        [ 0.78,  0.92,  0.67,  0.32,  0.64],
        [ 0.02,  0.67,  0.2 ,  0.45,  0.39]])


@pytest.fixture
def alpha_vector():
    return np.array([ 0.93,  0.99,  0.52,  0.54,  0.12,  0.55,  0.44,  0.13,  0.04])


@pytest.fixture
def beta_vector():
    return np.array([0, 1, 0, 1, 0, 0, 1, 1, 0])


@pytest.fixture
def rhs_vector():
    return np.array([ 0.6093,  0.683 ,  1.2136,  0.9063])


@pytest.fixture
def objective_vector():
    return np.array([-0.6982,  0.3623, -0.479 ,  0.0235,  0.1651])


@pytest.fixture
def solution(beta_vector):
    return Solution(
        x=np.array([   0,  0.99,     0,  0.54,     0]),
        r=np.array([0.93,     0,  0.52,     0,  0.12]),
        y=np.array([0.55,     0,     0,  0.04]),
        s=np.array([   0,  0.44,  0.13,     0]),
        basis=beta_vector)


@pytest.fixture(params=['encoded', 'solved', 'unsolved'])
def instance(request, lhs_matrix, alpha_vector, beta_vector, solution, rhs_vector, objective_vector):
    if request.param == 'encoded':
        return EncodedInstance(lhs=lhs_matrix, alpha=alpha_vector, beta=beta_vector)
    elif request.param == 'solved':
        return SolvedInstance(lhs=lhs_matrix, solution=solution)
    elif request.param == 'unsolved':
        return UnsolvedInstance(lhs=lhs_matrix, rhs=rhs_vector, objective=objective_vector)


def test_size(instance, variables, constraints):
    assert instance.variables == variables
    assert instance.constraints == constraints


def test_alpha(instance, alpha_vector):
    assert_approx_equal(instance.alpha(), alpha_vector)


def test_beta(instance, beta_vector):
    assert_approx_equal(instance.beta(), beta_vector)


def test_lhs(instance, lhs_matrix):
    assert_approx_equal(instance.lhs(), lhs_matrix)


def test_solution(instance, solution):
    result = instance.solution()
    assert_approx_equal(result.x, solution.x)
    assert_approx_equal(result.y, solution.y)
    assert_approx_equal(result.r, solution.r)
    assert_approx_equal(result.s, solution.s)
    assert np.all(result.basis == solution.basis)


def test_rhs(instance, rhs_vector):
    assert_approx_equal(instance.rhs(), rhs_vector)


def test_objective(instance, objective_vector):
    assert_approx_equal(instance.objective(), objective_vector)
