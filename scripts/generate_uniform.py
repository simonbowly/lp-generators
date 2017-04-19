''' Command line script which generates instances using the constructor
method and varying expected feature values uniformly. '''

import json

import numpy as np
from tqdm import tqdm

from lp_generators.lhs_generators import generate_lhs
from lp_generators.solution_generators import generate_alpha, generate_beta
from lp_generators.instance import EncodedInstance
from lp_generators.features import coeff_features, solution_features
from lp_generators.utils import calculate_data, write_instance
from lp_generators.writers import write_tar_encoded

from common import cli_seeds


@write_instance(write_tar_encoded, 'data/generate_uniform/inst_{seed}.tar')
@calculate_data(coeff_features, solution_features)
def generate(seed):
    ''' Generator distributing uniformly across parameters with fixed size.
    Feature values are attached to each instance by the calculate_data
    decorator. Instances are written to compressed format using the
    write_instance decorator so they can be loaded later as start points
    for search algorithms. '''

    random_state = np.random.RandomState(seed)

    size_params = dict(variables=50, constraints=50)
    beta_params = dict(
        basis_split=random_state.uniform(low=0.0, high=1.0))
    alpha_params = dict(
        frac_violations=random_state.uniform(low=0.0, high=1.0),
        beta_param=random_state.lognormal(mean=-0.2, sigma=1.8),
        mean_primal=0,
        std_primal=1,
        mean_dual=0,
        std_dual=1)
    lhs_params = dict(
        density=random_state.uniform(low=0.0, high=1.0),
        pv=random_state.uniform(low=0.0, high=1.0),
        pc=random_state.uniform(low=0.0, high=1.0),
        coeff_loc=random_state.uniform(low=-2.0, high=2.0),
        coeff_scale=random_state.uniform(low=0.1, high=1.0))

    instance = EncodedInstance(
        lhs=generate_lhs(random_state=random_state, **size_params, **lhs_params).todense(),
        alpha=generate_alpha(random_state=random_state, **size_params, **alpha_params),
        beta=generate_beta(random_state=random_state, **size_params, **beta_params))
    instance.data = dict(seed=seed)
    return instance


@cli_seeds
def run(seed_values):
    ''' Generate instances from the given seed values and store feature results. '''
    print('Generating fixed size constructed instances.')
    instances = (generate(seed) for seed in seed_values)
    features = [instance.data for instance in tqdm(instances, total=len(seed_values), smoothing=0)]
    with open('data/generate_uniform.json', 'w') as outfile:
        json.dump(features, outfile, indent=4, sort_keys=True)

run()
