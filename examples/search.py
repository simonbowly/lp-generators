''' Example script using search to make an instance harder to solve by
primal simplex. '''

import functools

import numpy as np
from tqdm import tqdm
import pandas as pd

import lp_generators.neighbours_encoded as neighbours_encoded
from lp_generators.performance import clp_simplex_performance
from lp_generators.writers import read_tar_encoded, write_mps
from lp_generators.utils import calculate_data
from lp_generators.search import local_search, write_steps


# Decorator/wrapper to calculate performance data on any instance
# as it is generated or read.
calc_wrapper = calculate_data(clp_simplex_performance)

# List of neighbourhood operators, function to make a random choice
# of operator to apply at each step.
NEIGHBOURS = [
    functools.partial(neighbours_encoded.exchange_basis, count=5),
    functools.partial(neighbours_encoded.scale_optvalue, mean=0, sigma=1, count=5),
    functools.partial(neighbours_encoded.remove_lhs_entry, count=10),
    functools.partial(neighbours_encoded.add_lhs_entry, mean=0, sigma=1, count=10),
    functools.partial(neighbours_encoded.scale_lhs_entry, mean=0, sigma=1, count=10),
    ]

@calc_wrapper
def neighbour(instance, random_state):
    func = random_state.choice(NEIGHBOURS)
    return func(instance, random_state)

# Loads an instance to start the search, calculating performance data.
load_start = calc_wrapper(read_tar_encoded)


@write_steps(write_mps, 'search/step_{step:04d}.mps.gz', new_only=True)
def search():
    ''' Search process as a generator/iterator. Returns step_info, instance
    at each step. This search maximises the number of iterations required for
    primal simplex to solve the instance. Writes any improved instance found
    to mps format.'''
    return local_search(
        objective=lambda instance: instance.data['clp_primal_iterations'],
        sense='max',
        neighbour=neighbour,
        start_instance=load_start('generated/inst_3072533601.tar'),
        steps=100,
        random_state=np.random.RandomState(1640241240))

# Run search, display improvement steps
data = pd.DataFrame([
    dict(**step_info, **instance.data) for step_info, instance
    in tqdm(search()) if step_info['search_update'] == 'improved'])
print(data[['search_step', 'clp_primal_iterations', 'clp_dual_iterations']])
