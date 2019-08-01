
import os

import pytest

from lp_generators.instance import EncodedInstance, UnsolvedInstance
from lp_generators.writers import (
    write_mps, write_mps_ip, write_mps_mip,
    write_tar_encoded, read_tar_encoded,
    write_tar_lp, read_tar_lp)
from lp_generators.utils import temp_file_path
from .testing import random_encoded, assert_approx_equal


@pytest.mark.parametrize('instance', [
    random_encoded(3, 5),
    random_encoded(5, 3)])
def test_write_mps(instance):
    with temp_file_path('.mps.gz') as file_path:
        write_mps(instance, file_path)
        assert os.path.exists(file_path)


@pytest.mark.parametrize('instance', [
    random_encoded(3, 5),
    random_encoded(5, 3)])
def test_write_mps_ip(instance):
    with temp_file_path('.mps.gz') as file_path:
        write_mps_ip(instance, file_path)
        assert os.path.exists(file_path)


@pytest.mark.parametrize('instance,vtypes', [
    (random_encoded(3, 5), "ICI"),
    (random_encoded(5, 3), "IIIIC")])
def test_write_mps_mip(instance, vtypes):
    with temp_file_path('.mps.gz') as file_path:
        write_mps_mip(instance, file_path, vtypes)
        assert os.path.exists(file_path)


@pytest.mark.parametrize('instance', [
    random_encoded(3, 5),
    random_encoded(5, 3)])
def test_read_write_tar_encoded(instance):
    with temp_file_path() as file_path:
        write_tar_encoded(instance, file_path)
        assert os.path.exists(file_path)
        read_instance = read_tar_encoded(file_path)
    assert isinstance(read_instance, EncodedInstance)
    assert_approx_equal(instance.lhs(), read_instance.lhs())
    assert_approx_equal(instance.alpha(), read_instance.alpha())
    assert_approx_equal(instance.beta(), read_instance.beta())


@pytest.mark.parametrize('instance', [
    random_encoded(3, 5),
    random_encoded(5, 3)])
def test_read_write_tar_lp(instance):
    with temp_file_path() as file_path:
        write_tar_lp(instance, file_path)
        assert os.path.exists(file_path)
        read_instance = read_tar_lp(file_path)
    assert isinstance(read_instance, UnsolvedInstance)
    assert_approx_equal(instance.lhs(), read_instance.lhs())
    assert_approx_equal(instance.rhs(), read_instance.rhs())
    assert_approx_equal(instance.objective(), read_instance.objective())
