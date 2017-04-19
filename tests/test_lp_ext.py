
import os

import numpy as np
import pytest

from lp_generators.lp_ext import LPCy
from lp_generators.utils import temp_file_path


@pytest.fixture
def matrices():
    ''' n, m, A, b, c for canonical form primal problem '''
    return (
        5, 4,
        np.array([
            [1,0,2,0,1],
            [0,1,0,1,0],
            [1,-1,0,1,0],
            [0,0,-1,1,0]], dtype=np.float),
        np.array([1, 2, 3, 4], dtype=np.float),
        np.array([1, 2, 3, 4, 5], dtype=np.float))


@pytest.fixture
def model(matrices):
    model = LPCy()
    model.construct_dense_canonical(*matrices)
    return model


@pytest.fixture
def easy_model():
    model = LPCy()
    model.construct_dense_canonical(
        2, 2, np.array([[1, 3], [3, 1]], dtype=np.float),
        np.array([4, 4], dtype=np.float),
        np.array([1, 1], dtype=np.float))
    return model


def test_construct(matrices, model):
    n, m, A, b, c = matrices
    assert np.all(model.get_dense_lhs() == A)
    assert np.all(model.get_rhs() == b)
    assert np.all(model.get_obj() == c)


def test_solve(easy_model):
    easy_model.solve()
    assert easy_model.get_solution_status() == 0
    assert np.all(easy_model.get_solution_primals() == [1, 1])
    assert np.all(easy_model.get_solution_slacks() == [0, 0])
    assert np.all(easy_model.get_solution_reduced_costs() == [0, 0])
    assert np.all(easy_model.get_solution_duals() == [.25, .25])
    assert np.all(easy_model.get_solution_basis() == [1, 1, 0, 0])


def test_write_lp(easy_model):
    with temp_file_path('.mps.gz') as file_path:
        easy_model.write_mps(file_path)
        assert os.path.exists(file_path)


def test_write_ip(easy_model):
    with temp_file_path('.mps.gz') as file_path:
        easy_model.write_mps_ip(file_path)
        assert os.path.exists(file_path)
