
import numpy as np
import pytest

from lp_generators.instance import EncodedInstance, UnsolvedInstance
from .testing import random_encoded, assert_approx_equal


def encode(unsolved_instance):
    return EncodedInstance(
        lhs=unsolved_instance.lhs(),
        alpha=unsolved_instance.alpha(),
        beta=unsolved_instance.beta())


@pytest.mark.parametrize('encoded_instance', [
    random_encoded(3, 5),
    random_encoded(5, 3),
    random_encoded(50, 30),
    random_encoded(30, 50),
    ])
def test_encode_decode(encoded_instance):

    # Decoding
    unsolved_instance = UnsolvedInstance(
        lhs=encoded_instance.lhs(),
        rhs=encoded_instance.rhs(),
        objective=encoded_instance.objective())
    assert_approx_equal(unsolved_instance.lhs(), encoded_instance.lhs())
    assert_approx_equal(unsolved_instance.alpha(), encoded_instance.alpha())
    assert np.all(unsolved_instance.beta() == encoded_instance.beta())

    # Encoding
    encoded_instance = encode(unsolved_instance)
    assert_approx_equal(encoded_instance.lhs(), unsolved_instance.lhs())
    assert_approx_equal(encoded_instance.rhs(), unsolved_instance.rhs())
    assert_approx_equal(
        encoded_instance.objective(), unsolved_instance.objective())


def test_decode_infeasible():
    unsolved_instance = UnsolvedInstance(
        lhs=np.array([[1.0, 1.0], [-1.0, -1.0]]),
        rhs=np.array([1.0, -2.0]),
        objective=np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        encoded_instance = encode(unsolved_instance)


def test_decode_unbounded():
    unsolved_instance = UnsolvedInstance(
        lhs=np.array([[-1.0, -1.0]]),
        rhs=np.array([-1.0]),
        objective=np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        encoded_instance = encode(unsolved_instance)
