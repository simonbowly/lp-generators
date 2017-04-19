''' Randomly generate lhs matrices by varying degree and coefficient
statistics. Main function to be called is generate_lhs.

This process is a bit of a bottleneck in generation, since it is implemented
in pure python and uses a quadratic algorithm to achieve the required
expected frequencies of edges. This is fine for small scale experimental
purposes, but further work needed (equivalent efficient algorithm and/or C++
implementation) before using this for larger instances.
'''

import itertools
import collections
import operator

import numpy as np
import scipy.sparse as sparsemat


def degree_dist(vertices, edges, max_degree, param, random_state):
    ''' Gives degree values for :vertices vertices.
    For each iteration, deterministic weights are given by the current vertex
    degree, random weights are chosen uniformly.
    The final weights for the next choice are weighted by :param on [0, 1],
    after being normalised by their sum.
    Vertices are excluded once they reach :max_degree, so :edges can be at
    most :vertices x :max_degree '''
    if (edges > vertices * max_degree):
        raise ValueError('edges > vertices * max_degree')
    degree = [0] * vertices
    indices = list(range(vertices))
    for _ in range(edges):
        deterministic_weights = np.array([degree[i] + 0.0001 for i in indices])
        random_weights = random_state.uniform(0, 1, len(indices))
        weights = (
            deterministic_weights / deterministic_weights.sum() * param +
            random_weights / random_weights.sum() * (1 - param))
        ind = random_state.choice(a=indices, p=weights)
        degree[ind] += 1
        if degree[ind] >= max_degree:
            indices.remove(ind)
    return degree


def expected_bipartite_degree(degree1, degree2, random_state):
    # Generates edges with probability d1 * d2 / sum(d1), asserting that
    # sum(d1) = sum(d2).
    # There is a more efficient way, right?
    if abs(sum(degree1) - sum(degree2)) > 10 ** -5:
        raise ValueError('You\'ve unbalanced the force!')
    rho = 1 / sum(degree1)
    for i, di in enumerate(degree1):
        for j, dj in enumerate(degree2):
            if random_state.uniform(0, 1) < (di * dj * rho):
                # print('yield', i, j)
                yield i, j


def generate_by_degree(n1, n2, density, p1, p2, random_state):
    ''' Join together two vertex distributions to create a bipartite graph. '''
    nedges = max(int(round(n1 * n2 * density)), 1)
    degree1 = degree_dist(n1, nedges, n2, p1, random_state)
    degree2 = degree_dist(n2, nedges, n1, p2, random_state)
    return expected_bipartite_degree(degree1, degree2, random_state)


def connect_remaining(n1, n2, edges, random_state):
    ''' Finds any isolated vertices in the bipartite graph and connects them. '''
    degree1 = collections.Counter(map(operator.itemgetter(0), edges))
    degree2 = collections.Counter(map(operator.itemgetter(1), edges))
    missing1 = [i for i in range(n1) if degree1[i] == 0]
    missing2 = [j for j in range(n2) if degree2[j] == 0]
    random_state.shuffle(missing1)
    random_state.shuffle(missing2)
    for v1, v2 in itertools.zip_longest(missing1, missing2):
        try:
            if v1 is None:
                v1 = random_state.choice(
                    [i for i in range(n1) if degree1[i] < n2])
            if v2 is None:
                v2 = random_state.choice(
                    [j for j in range(n2) if degree2[j] < n1])
        except ValueError:
            print(degree1)
            print(degree2)
            raise
        yield v1, v2
        degree1[v1] += 1
        degree2[v2] += 2


def generate_edges(n1, n2, density, p1, p2, random_state):
    ''' Generate edges using size and weight parameters. '''
    edges = set(generate_by_degree(n1, n2, density, p1, p2, random_state))
    edges.update(connect_remaining(n1, n2, edges, random_state))
    return edges


def generate_lhs(variables, constraints, density, pv, pc,
                        coeff_loc, coeff_scale, random_state):
    ''' Generate lhs constraint matrix using sparsity parameters and
    coefficient value distribution. '''
    ind_var, ind_cons = zip(*generate_edges(
        variables, constraints, density,
        pv, pc, random_state))
    data = random_state.normal(
        loc=coeff_loc, scale=coeff_scale, size=len(ind_var))
    return sparsemat.coo_matrix((data, (ind_cons, ind_var)))
