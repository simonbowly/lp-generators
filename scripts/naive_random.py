''' Generate a set of instances using the naive method. '''

import multiprocessing
import json

import numpy as np
from tqdm import tqdm

from lp_generators.lhs_generators import generate_lhs
from lp_generators.instance import UnsolvedInstance
from lp_generators.features import coeff_features, solution_features
from lp_generators.performance import clp_simplex_performance
from lp_generators.utils import calculate_data, write_instance
from lp_generators.writers import write_tar_lp

from seeds import cli_seeds


def generate_rhs(variables, constraints, rhs_mean, rhs_std, random_state):
    return random_state.normal(loc=rhs_mean, scale=rhs_std, size=constraints)


def generate_objective(variables, constraints, obj_mean, obj_std, random_state):
    return random_state.normal(loc=obj_mean, scale=obj_std, size=variables)


@calculate_data(coeff_features, solution_features, clp_simplex_performance)
@write_instance(write_tar_lp, 'data/naive_random/inst_{seed}.tar')
def generate(seed):
    ''' Creates a distribution of fixed size instances using the
    'naive' strategy:
        - loc/scale parameters for rhs and objective distributions are
            distributed uniformly (at the instance level)
        - rhs/objective values are chosen from uniform distributions
            using the chosen instance-level parameters
     '''

    random_state = np.random.RandomState(seed)

    size_params = dict(variables=50, constraints=50)
    rhs_params = dict(
        rhs_mean=random_state.uniform(low=-100.0, high=100.0),
        rhs_std=random_state.uniform(low=1.0, high=30.0))
    objective_params = dict(
        obj_mean=random_state.uniform(low=-100.0, high=100.0),
        obj_std=random_state.uniform(low=1.0, high=10.0))
    lhs_params = dict(
        density=random_state.uniform(low=0.0, high=1.0),
        pv=random_state.uniform(low=0.0, high=1.0),
        pc=random_state.uniform(low=0.0, high=1.0),
        coeff_loc=random_state.uniform(low=-2.0, high=2.0),
        coeff_scale=random_state.uniform(low=0.1, high=1.0))

    instance = UnsolvedInstance(
        lhs=generate_lhs(random_state=random_state, **size_params, **lhs_params).todense(),
        rhs=generate_rhs(random_state=random_state, **size_params, **rhs_params),
        objective=generate_objective(random_state=random_state, **size_params, **objective_params))
    instance.data = dict(seed=seed)
    return instance


@cli_seeds
def run(seed_values):
    ''' Generate the required number of instances and store feature results. '''
    pool = multiprocessing.Pool()
    mapper = pool.imap_unordered
    print('Generating fixed size naive random instances.')
    instances = tqdm(
        mapper(generate, seed_values),
        total=len(seed_values), smoothing=0)
    features = [
        instance.data
        for instance in instances
    ]
    with open('data/naive_random.json', 'w') as outfile:
        json.dump(features, outfile, indent=4, sort_keys=True)

run()
