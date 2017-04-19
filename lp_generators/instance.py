''' Storage methods for instances.

Incomplete classes providing common methods:
    Constructor: build rhs and objective from solution
    DenseLHS: store constraint left hand side as dense numpy matrix
    SolutionEncoder: build alpha/beta from solution

Complete classes implementing the entire interface:
    EncodedInstance: store as lhs, alpha, beta
    SolvedInstance: store as lhs, solution
    UnsolvedInstance: store as A, b, c

Note that UnsolvedInstance may not be decodable (if it does not have a
solution) so attempting to solve will throw a value error.

All complete classes implement the interface given by the abstract base
class LPInstance.
'''

import collections
from abc import ABC, abstractproperty

import numpy as np

from .lp_ext import LPCy


Solution = collections.namedtuple('Solution', ['x', 'y', 'r', 's', 'basis'])


class LPInstance(ABC):
    ''' All complete LP instance classes use this as a base. '''

    @abstractproperty
    def variables(self):
        ''' Number of variables (n). '''
        pass

    @abstractproperty
    def constraints(self):
        ''' Number of constraints (m). '''
        pass

    def lhs(self):
        ''' Returns an m x n matrix-like representation of constraints. '''
        pass

    def rhs(self):
        ''' Length m vector of constraint right hand sides. '''
        pass

    def objective(self):
        ''' Length n vector of objective coefficients. '''
        pass

    def alpha(self):
        ''' Length m+n vector of encoded solution values. '''
        pass

    def beta(self):
        ''' Length m+n vector encoding the optimal basis. '''
        pass

    def solution(self):
        ''' Solution object. '''
        pass


class Constructor(object):
    ''' Use the result of lhs() and solution() methods to construct an
    instance with the required optimal solution. '''

    def rhs(self):
        solution = self.solution()
        A = self.lhs()
        x = np.matrix(solution.x).transpose()
        s = np.matrix(solution.s).transpose()
        _rhs = A * x + s
        return np.asarray(_rhs.transpose(), dtype=np.float)[0]

    def objective(self):
        solution = self.solution()
        A = self.lhs()
        y = np.matrix(solution.y).transpose()
        r = np.matrix(solution.r).transpose()
        _obj = A.transpose() * y - r
        return np.asarray(_obj.transpose(), dtype=np.float)[0]


class SolutionEncoder(object):
    ''' Use the result of solution() to build alpha and beta vectors. '''

    def alpha(self):
        solution = self.solution()
        primal = np.concatenate([solution.x, solution.s])
        dual = np.concatenate([solution.r, solution.y])
        # should verify complementarity somewhere...?
        return primal + dual

    def beta(self):
        return self.solution().basis


class DenseLHS(object):
    ''' Store the left hand side of the constraints. The result of lhs() must
    be able to be transposed and matrix multiplied. '''

    def __init__(self, lhs, **kwargs):
        super().__init__(**kwargs)
        self._lhs_matrix = np.matrix(lhs, dtype=np.float)

    @property
    def variables(self):
        return self._lhs_matrix.shape[1]

    @property
    def constraints(self):
        return self._lhs_matrix.shape[0]

    def lhs(self):
        return self._lhs_matrix


class EncodedInstance(Constructor, DenseLHS, LPInstance):
    ''' Full instance class storing data as (A, alpha, beta). '''

    def __init__(self, alpha, beta, **kwargs):
        super().__init__(**kwargs)
        n, m = self.variables, self.constraints
        assert alpha.shape == (n + m, )
        assert np.all(alpha >= 0)
        assert beta.shape == (n + m, )
        assert np.sum(beta == 1) == m
        assert np.sum(beta == 0) == n
        self._alpha = np.array(alpha, dtype=np.float)
        self._beta = np.array(beta, dtype=np.float)

    def alpha(self):
        return self._alpha

    def beta(self):
        return self._beta

    def solution(self):
        # Extract primal variables and reduced costs (complete solution)
        n, m = self.variables, self.constraints
        x = self._beta[:n] * self._alpha[:n]
        r = (1 - self._beta[:n]) * self._alpha[:n]
        y = (1 - self._beta[n:]) * self._alpha[n:]
        s = self._beta[n:] * self._alpha[n:]
        return Solution(x=x, r=r, y=y, s=s, basis=self._beta)


class SolvedInstance(Constructor, SolutionEncoder, DenseLHS, LPInstance):
    ''' Full instance class storing data as (A, x, r, y, s). '''

    def __init__(self, solution, **kwargs):
        super().__init__(**kwargs)
        n, m = self.variables, self.constraints
        assert solution.x.shape == (n, )
        assert solution.r.shape == (n, )
        assert solution.y.shape == (m, )
        assert solution.s.shape == (m, )
        assert solution.basis.shape == (n + m, )        
        self._solution = solution

    def solution(self):
        return self._solution


class UnsolvedInstance(SolutionEncoder, DenseLHS, LPInstance):
    ''' Full instance class storing data as (A, b, c). The instance may or
    may not have a solution. '''

    def __init__(self, rhs, objective, **kwargs):
        super().__init__(**kwargs)
        assert rhs.shape == (self.constraints, )
        assert objective.shape == (self.variables, )
        self._rhs = rhs
        self._objective = objective

    def rhs(self):
        return self._rhs

    def objective(self):
        return self._objective

    def solution(self):
        model = LPCy()
        model.construct_dense_canonical(
            self.variables, self.constraints,
            self.lhs(), self.rhs(), self.objective())
        model.solve()

        if model.get_solution_status() != 0:
            raise ValueError('Instance could not be decoded as it could not be solved.')

        x = model.get_solution_primals()
        s = model.get_solution_slacks()
        y = model.get_solution_duals()
        r = model.get_solution_reduced_costs()
        basis = model.get_solution_basis()
        return Solution(x=x, s=s, y=y, r=r, basis=basis)
