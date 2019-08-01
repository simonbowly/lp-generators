# Cython interface to C++ class connecting the COIN-CLP callable library.

from libcpp.string cimport string
from cython.operator cimport dereference as deref

import numpy as np
cimport numpy as np


cdef extern from "lp.hpp":
    cdef cppclass LP:
        LP()
        void constructDenseCanonical(int, int, double*, double*, double*)
        void writeMps(string)
        void writeMpsIP(string)
        void writeMpsMIP(string, string)
        int getNumVariables()
        int getNumConstraints()
        int getNumLHSElements()
        void getLhsMatrixDense(double*)
        void getRhsVector(double*)
        void getObjVector(double*)
        void solve()
        int getSolutionStatus();
        void getSolutionPrimals(double*)
        void getSolutionSlacks(double*)
        void getSolutionDuals(double*)
        void getSolutionReducedCosts(double*)
        void getSolutionBasis(double*)


cdef class LPCy(object):
    cdef LP *wrapped

    def __cinit__(self):
        self.wrapped = new LP()

    def __dealloc__(self):
        del self.wrapped

    def construct_dense_canonical(self, variables, constraints, A, b, c):
        deref(self.wrapped).constructDenseCanonical(
            variables, constraints,
            contiguous_2d_handle(A),
            contiguous_1d_handle(b),
            contiguous_1d_handle(c))

    def write_mps(self, file_name):
        cdef string strfilename = file_name.encode('UTF-8')
        deref(self.wrapped).writeMps(strfilename)

    def write_mps_ip(self, file_name):
        cdef string strfilename = file_name.encode('UTF-8')
        deref(self.wrapped).writeMpsIP(strfilename)

    def write_mps_mip(self, file_name, vtypes):
        assert len(vtypes) == deref(self.wrapped).getNumVariables()
        cdef string strfilename = file_name.encode('UTF-8')
        cdef string strvtypes = vtypes.encode("UTF-8")
        deref(self.wrapped).writeMpsMIP(strfilename, strvtypes)

    def get_dense_lhs(self):
        variables = deref(self.wrapped).getNumVariables()
        constraints = deref(self.wrapped).getNumConstraints()
        result = np.zeros(shape=(constraints, variables))
        deref(self.wrapped).getLhsMatrixDense(contiguous_2d_handle(result))
        return result

    def get_rhs(self):
        constraints = deref(self.wrapped).getNumConstraints()
        result = np.zeros(shape=(constraints))
        deref(self.wrapped).getRhsVector(contiguous_1d_handle(result))
        return result

    def get_obj(self):
        variables = deref(self.wrapped).getNumVariables()
        result = np.zeros(shape=(variables))
        deref(self.wrapped).getObjVector(contiguous_1d_handle(result))
        return result

    def solve(self):
        deref(self.wrapped).solve()

    def get_solution_status(self):
        return deref(self.wrapped).getSolutionStatus()

    def get_solution_primals(self):
        variables = deref(self.wrapped).getNumVariables()
        result = np.zeros(shape=(variables))
        deref(self.wrapped).getSolutionPrimals(contiguous_1d_handle(result))
        return result

    def get_solution_slacks(self):
        constraints = deref(self.wrapped).getNumConstraints()
        result = np.zeros(shape=(constraints))
        deref(self.wrapped).getSolutionSlacks(contiguous_1d_handle(result))
        return result

    def get_solution_duals(self):
        constraints = deref(self.wrapped).getNumConstraints()
        result = np.zeros(shape=(constraints))
        deref(self.wrapped).getSolutionDuals(contiguous_1d_handle(result))
        return result

    def get_solution_reduced_costs(self):
        variables = deref(self.wrapped).getNumVariables()
        result = np.zeros(shape=(variables))
        deref(self.wrapped).getSolutionReducedCosts(contiguous_1d_handle(result))
        return result

    def get_solution_basis(self):
        elements = deref(self.wrapped).getNumVariables() + deref(self.wrapped).getNumConstraints()
        result = np.zeros(shape=(elements))
        deref(self.wrapped).getSolutionBasis(contiguous_1d_handle(result))
        return result


cdef double* contiguous_1d_handle(np.ndarray[np.double_t, ndim=1, mode='c'] py_array):
    ''' Return c handle for contiguous 1d numpy array. '''
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] np_buff = np.ascontiguousarray(
        py_array, dtype=np.double)
    cdef double* im_buff = <double*> py_array.data
    return im_buff


cdef double* contiguous_2d_handle(np.ndarray[np.double_t, ndim=2, mode='c'] py_array):
    ''' Return c handle for contiguous 2d numpy array. '''
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] np_buff = np.ascontiguousarray(
        py_array, dtype=np.double)
    cdef double* im_buff = <double*> py_array.data
    return im_buff
