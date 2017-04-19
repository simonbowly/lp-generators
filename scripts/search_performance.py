''' Command line script which runs searches from existing instances in
encoded representation to find more difficult instances for the specified
solver. '''

import json
import functools
import itertools

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from lp_generators.features import coeff_features, solution_features
from lp_generators.performance import clp_simplex_performance, strbr_performance
from lp_generators.writers import read_tar_encoded
import lp_generators.neighbours_encoded as neighbours
from lp_generators.utils import calculate_data, system_random_seeds
from lp_generators.search import local_search


NEIGHBOURS = [
    functools.partial(neighbours.exchange_basis, count=5),
    functools.partial(neighbours.scale_optvalue, mean=0, sigma=1, count=5),
    functools.partial(neighbours.remove_lhs_entry, count=10),
    functools.partial(neighbours.add_lhs_entry, mean=0, sigma=1, count=10),
    functools.partial(neighbours.scale_lhs_entry, mean=0, sigma=1, count=10),
    ]


def neighbour(instance, random_state):
    func = random_state.choice(NEIGHBOURS)
    return func(instance, random_state)


@click.command()
@click.option('--system-seeds', default=10, type=int, help='Number of start seeds to use')
@click.option('--seed-file', default=None, type=click.Path(exists=True), help='JSON seed file')
@click.option('--steps', default=200, type=int, help='Number of search steps per run')
@click.option('--metric', default='primal', type=click.Choice(['primal', 'dual', 'barrier', 'reopt']))
def run(system_seeds, seed_file, steps, metric):

    # Calculations required
    if metric == 'primal':
        calc_wrapper = calculate_data(clp_simplex_performance, coeff_features)
        metric = 'clp_primal_iterations'
    elif metric == 'dual':
        calc_wrapper = calculate_data(clp_simplex_performance, coeff_features)
        metric = 'clp_dual_iterations'
    elif metric == 'barrier':
        calc_wrapper = calculate_data(clp_simplex_performance, coeff_features)
        metric = 'clp_barrier_flops'
    elif metric == 'reopt':
        calc_wrapper = calculate_data(strbr_performance, coeff_features)
        metric = 'strbr_percall'

    # Start instance loader
    load = calc_wrapper(read_tar_encoded)

    # Start and search seed values
    if seed_file:
        with open(seed_file) as infile:
            search_seeds, start_seeds = zip(*(
                (entry['seed'], entry['start']) for entry in json.load(infile)))
    else:
        search_seeds = list(system_random_seeds(n=system_seeds, bits=32))
        # Starting instance candidates: hardest instances in each size bracket
        df = pd.read_json('data/generate_varied_size.json').set_index('seed')
        start_seeds = pd.groupby(df[metric], pd.cut(df.variables, bins=10)).idxmax()
        # Cycle start points (so number of results = number of search seeds)
        start_seeds = itertools.cycle(start_seeds)

    for start_seed, seed in zip(start_seeds, search_seeds):
        print('\'{}\' performance space search from {} with seed {}'.format(metric, start_seed, seed))
        random_state = np.random.RandomState(seed)
        result_file = 'data/search_perform_{}_s{}_r{}.json'.format(metric, start_seed, seed)
        results = local_search(
            objective=lambda instance: instance.data[metric],
            sense='max',
            neighbour=calc_wrapper(neighbour),
            start_instance=load('data/generate_varied_size/inst_{}.tar'.format(start_seed)),
            steps=steps,
            random_state=random_state)
        features = [
            dict(**step_info, **instance.data) for step_info, instance in
            tqdm(results, total=steps, smoothing=0)]
        with open(result_file, 'w') as outfile:
            json.dump(features, outfile, indent=4, sort_keys=True)

run()
