
from unittest import mock

import pytest
import numpy as np

from lp_generators.instance import EncodedInstance
import lp_generators.neighbours_common as neighbours


@pytest.fixture
def rstate():
    rstate_mock = mock.MagicMock()
    return rstate_mock


@pytest.fixture
def lhs():
    return np.array([[1, 0, 1], [1, 1, 0]], dtype=np.float)


@pytest.fixture
def alpha():
    return np.array([1, 1, 1, 1, 1], dtype=np.float)


@pytest.fixture
def beta():
    return np.array([1, 0, 0, 1, 0])


@pytest.fixture
def lhs_empty():
    return np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float)


@pytest.fixture
def lhs_full():
    return np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float)


def test_basis_exchange(beta, rstate):
    rstate.choice.side_effect = [1, 0]
    result = np.copy(beta)
    neighbours._exchange_basis(result, rstate, count=1)
    assert rstate.choice.call_count == 2
    assert np.all(rstate.choice.call_args_list[0][0][0] == [1, 2, 4])
    assert np.all(rstate.choice.call_args_list[1][0][0] == [0, 3])
    assert np.all(result == [0, 1, 0, 1, 0])


@pytest.mark.parametrize('dist', ['normal', 'lognormal'])
def test_scale_vector_value(alpha, rstate, dist):
    rstate.choice.return_value = 3
    rstate.normal.return_value = -1.2
    rstate.lognormal.return_value = 1.5
    result = np.copy(alpha)
    neighbours._scale_vector_entry(
        result, rstate, mean='mean', sigma='sigma', dist=dist, count=1)
    rstate.choice.assert_called_once_with(5)
    if dist == 'normal':
        rstate.normal.assert_called_once_with(loc='mean', scale='sigma')
        assert np.all(result == [1, 1, 1, -1.2, 1])
    elif dist == 'lognormal':
        rstate.lognormal.assert_called_once_with(mean='mean', sigma='sigma')
        assert np.all(result == [1, 1, 1, 1.5, 1])
    else:
        raise ValueError()


def test_remove_lhs_entry(lhs, rstate):
    rstate.choice.return_value = 2
    result = np.copy(lhs)
    neighbours._remove_lhs_entry(result, rstate, count=1)
    rstate.choice.assert_called_once_with(4)
    assert np.all(result == [[1, 0, 1], [0, 1, 0]])


def test_add_lhs_entry(lhs, rstate):
    rstate.choice.return_value = 1
    rstate.normal.return_value = 1.5
    result = np.copy(lhs)
    neighbours._add_lhs_entry(result, rstate, 'mean', 'sigma', count=1)
    rstate.normal.assert_called_once_with(loc='mean', scale='sigma')
    assert np.all(result == [[1, 0, 1], [1, 1, 1.5]])


def test_scale_lhs_entry(lhs, rstate):
    rstate.choice.return_value = 1
    rstate.normal.return_value = 1.5
    result = np.copy(lhs)
    neighbours._scale_lhs_entry(result, rstate, 'mean', 'sigma', count=1)
    rstate.normal.assert_called_once_with(loc='mean', scale='sigma')
    assert np.all(result == [[1, 0, 1.5], [1, 1, 0]])


def test_empty_remove(lhs_empty):
    ''' Edge cases where the lhs matrix has no entries. '''
    result = np.copy(lhs_empty)
    neighbours._remove_lhs_entry(lhs_empty, np.random, count=1)
    assert np.all(result == lhs_empty)


def test_empty_scale(lhs_empty):
    ''' Edge cases where the lhs matrix has no entries. '''
    result = np.copy(lhs_empty)
    neighbours._scale_lhs_entry(lhs_empty, np.random, 'mean', 'sigma', count=1)
    assert np.all(result == lhs_empty)


def test_full_add(lhs_full):
    ''' Edge cases where the lhs matrix has all nonzero entries. '''
    result = np.copy(lhs_full)
    neighbours._add_lhs_entry(lhs_full, np.random, 'mean', 'sigma', count=1)
    assert np.all(result == lhs_full)


RANDOM_MESSAGE = 'Random-valued test failed. This is expected occasionally.'


@pytest.mark.parametrize('count', range(1, 10))
def test_repeat_scale_vector_entry(count):
    input_vec = np.random.random(1000)
    result_vec = np.copy(input_vec)
    neighbours._scale_vector_entry(result_vec, np.random, 0, 1, 'normal', count=count)
    assert np.sum(input_vec != result_vec) == count, RANDOM_MESSAGE


@pytest.mark.parametrize('count', range(1, 10))
def test_repeat_scale_lhs_entry(count):
    input_vec = np.random.random((100, 100))
    result_vec = np.copy(input_vec)
    neighbours._scale_lhs_entry(result_vec, np.random, 0, 1, count=count)
    assert np.sum(input_vec != result_vec) == count, RANDOM_MESSAGE


@pytest.mark.parametrize('count', range(1, 10))
def test_repeat_remove_lhs_entry(count):
    input_vec = np.array(np.random.random((100, 100)) > 0.5, dtype=np.float)
    result_vec = np.copy(input_vec)
    neighbours._remove_lhs_entry(result_vec, np.random, count=count)
    assert np.sum(input_vec != 0) - np.sum(result_vec != 0) == count, RANDOM_MESSAGE


@pytest.mark.parametrize('count', range(1, 10))
def test_repeat_remove_lhs_entry(count):
    input_vec = np.array(np.random.random((100, 100)) > 0.5, dtype=np.float)
    result_vec = np.copy(input_vec)
    neighbours._add_lhs_entry(result_vec, np.random, 0, 1, count=count)
    assert np.sum(result_vec != 0) - np.sum(input_vec != 0) == count, RANDOM_MESSAGE
