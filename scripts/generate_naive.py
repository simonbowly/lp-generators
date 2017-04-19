''' Command line script which generates instances using a direct
representation of instance data. '''

import json

import numpy as np
from tqdm import tqdm

from lp_generators.lhs_generators import generate_lhs
from lp_generators.instance import UnsolvedInstance
from lp_generators.features import coeff_features, solution_features
from lp_generators.utils import calculate_data, write_instance
from lp_generators.writers import write_tar_lp

from common import cli_seeds


def generate_rhs(variables, constraints, rhs_mean, rhs_std, random_state):
    return random_state.normal(loc=rhs_mean, scale=rhs_std, size=constraints)


def generate_objective(variables, constraints, obj_mean, obj_std, random_state):
    return random_state.normal(loc=obj_mean, scale=obj_std, size=variables)


@calculate_data(coeff_features, solution_features)
@write_instance(write_tar_lp, 'data/generate_naive/inst_{seed}.tar')
def generate(seed):
    ''' Generator varying A, b, c directly '''

    random_state = np.random.RandomState(seed)

    size_params = dict(variables=50, constraints=50)
        # variables=random_state.randint(10, 100),
        # constraints=random_state.randint(10, 100))
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
    print('Generating fixed size naive random instances.')
    instances = (generate(seed) for seed in seed_values)
    features = [instance.data for instance in tqdm(instances, total=len(seed_values), smoothing=0)]
    with open('data/generate_naive.json', 'w') as outfile:
        json.dump(features, outfile, indent=4, sort_keys=True)

run()
