
import itertools
import multiprocessing
import json

import click
import numpy as np
from tqdm import tqdm

from lp_generators.features import coeff_features, solution_features
from lp_generators.performance import clp_simplex_performance

from search_operators import lp_column_neighbour, lp_row_neighbour
from seeds import cli_seeds
from performance_search_common import condition, objective, start_instance, calculate_features


def generate_by_search(arg):
    seed, perf_field = arg
    results = []
    pass_condition = 0
    step_change = 0
    random_state = np.random.RandomState(seed)
    current_instance = start_instance(random_state, perf_field)
    current_features = calculate_features(current_instance)
    for step in range(10001):
        if (step % 100) == 0:
            results.append(dict(
                **coeff_features(current_instance),
                **solution_features(current_instance),
                **clp_simplex_performance(current_instance),
                pass_condition=pass_condition,
                step_change=step_change,
                step=step, seed=seed))
        if (step % 2) == 0:
            new_instance = lp_row_neighbour(random_state, current_instance, 5)
        else:
            new_instance = lp_column_neighbour(random_state, current_instance, 5)
        new_features = calculate_features(new_instance)
        if condition(new_features):
            pass_condition += 1
            if objective(new_features, perf_field) < objective(current_features, perf_field):
                step_change += 1
                current_instance = new_instance
                current_features = new_features
    return results


@cli_seeds
@click.option('--perf-field', type=str)
def run(seed_values, perf_field):
    assert perf_field is not None
    ''' Generate the required number of instances and store feature results. '''
    pool = multiprocessing.Pool()
    mapper = pool.imap_unordered
    print('Generating instances by naive search.')
    features = list(tqdm(
        mapper(generate_by_search, zip(seed_values, itertools.repeat(perf_field))),
        total=len(seed_values), smoothing=0))
    features = list(itertools.chain(*features))
    with open('data/naive_performance_search_{}.json'.format(perf_field), 'w') as outfile:
        json.dump(features, outfile, indent=4, sort_keys=True)

run()
