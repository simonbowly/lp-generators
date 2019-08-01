''' Write instance data in various formats. Supports MPS for external use of
generated instances and a tar format used internally to read and write
instance data for generation and search where required. '''

import io
import tarfile

import numpy as np

from .lp_ext import LPCy
from .instance import EncodedInstance, UnsolvedInstance


def write_mps(instance, file_name):
    ''' Write an LP instance to MPS format (using A, b, c). '''
    writer = LPCy()
    writer.construct_dense_canonical(
        instance.variables, instance.constraints,
        np.asarray(instance.lhs()),
        np.asarray(instance.rhs()),
        np.asarray(instance.objective()))
    writer.write_mps(file_name)


def write_mps_ip(instance, file_name):
    ''' Write an LP instance to MPS format (using A, b, c). '''
    writer = LPCy()
    writer.construct_dense_canonical(
        instance.variables, instance.constraints,
        np.asarray(instance.lhs()),
        np.asarray(instance.rhs()),
        np.asarray(instance.objective()))
    writer.write_mps_ip(file_name)


def write_mps_mip(instance, file_name, vtypes):
    ''' Write an LP instance to MPS format (using A, b, c). '''
    writer = LPCy()
    writer.construct_dense_canonical(
        instance.variables, instance.constraints,
        np.asarray(instance.lhs()),
        np.asarray(instance.rhs()),
        np.asarray(instance.objective()))
    writer.write_mps_mip(file_name, vtypes)


def save_matrix_to_tar(tarstore, matrix, name):
    ''' Helper function encodes matrix to bytes with numpy and adds to tarball. '''
    fp = io.BytesIO()
    np.save(fp, matrix, allow_pickle=False)
    info = tarfile.TarInfo(name=name)
    info.size = fp.tell()
    fp.seek(0)
    tarstore.addfile(tarinfo=info, fileobj=fp)


def extract_matrix_from_tar(tarstore, name):
    ''' Helper reads file object from tarball and decodes with numpy. '''
    if name in tarstore.getnames():
        fp = tarstore.extractfile(name)
        fpio = io.BytesIO(fp.read())
        return np.load(fpio, allow_pickle=False)
    return None


def write_tar_encoded(instance, filename):
    ''' Internal use format: write the encoded form matrices as a tarball. '''
    with tarfile.TarFile(filename, mode='w') as store:
        save_matrix_to_tar(store, instance.lhs(), 'canonical_lhs.npy')
        save_matrix_to_tar(store, instance.alpha(), 'canonical_alpha.npy')
        save_matrix_to_tar(store, instance.beta(), 'canonical_beta.npy')


def read_tar_encoded(filename):
    ''' Internal use format: read the encoded form matrices from a tarball. '''
    with tarfile.TarFile(filename, mode='r') as store:
        lhs = extract_matrix_from_tar(store, 'canonical_lhs.npy')
        alpha = extract_matrix_from_tar(store, 'canonical_alpha.npy')
        beta = extract_matrix_from_tar(store, 'canonical_beta.npy')
    return EncodedInstance(lhs=lhs, alpha=alpha, beta=beta)


def write_tar_lp(instance, filename):
    ''' Internal use format: write the encoded form matrices as a tarball. '''
    with tarfile.TarFile(filename, mode='w') as store:
        save_matrix_to_tar(store, instance.lhs(), 'canonical_lhs.npy')
        save_matrix_to_tar(store, instance.rhs(), 'canonical_rhs.npy')
        save_matrix_to_tar(store, instance.objective(), 'canonical_objective.npy')


def read_tar_lp(filename):
    ''' Internal use format: read the encoded form matrices from a tarball. '''
    with tarfile.TarFile(filename, mode='r') as store:
        lhs = extract_matrix_from_tar(store, 'canonical_lhs.npy')
        rhs = extract_matrix_from_tar(store, 'canonical_rhs.npy')
        objective = extract_matrix_from_tar(store, 'canonical_objective.npy')
    return UnsolvedInstance(lhs=lhs, rhs=rhs, objective=objective)
