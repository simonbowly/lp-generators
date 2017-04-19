''' Command line script which runs searches from existing instances to target
points in feature space using either an encoded or direct representation. '''

import json
import collections
import functools
import random

import click
import numpy as np
from tqdm import tqdm

import lp_generators.neighbours_unsolved as neighbours_unsolved
import lp_generators.neighbours_encoded as neighbours_encoded
from lp_generators.features import coeff_features, solution_features
from lp_generators.writers import read_tar_lp, read_tar_encoded
from lp_generators.utils import calculate_data, system_random_seeds
from lp_generators.search import local_search


def target_point(**point):
    ''' Feature space target point objective function which also rejects
    infeasible instances. '''
    def _objective(instance):
        if not instance.data['solvable']:
            return 1e+20
        return sum(
            (instance.data[field] - value) ** 2
            for field, value in point.items())
    return _objective


NEIGHBOURS_UNSOLVED = [
    functools.partial(neighbours_unsolved.scale_obj_entry, mean=0, sigma=1, count=5),
    functools.partial(neighbours_unsolved.scale_rhs_entry, mean=0, sigma=1, count=5),
    functools.partial(neighbours_unsolved.remove_lhs_entry, count=10),
    functools.partial(neighbours_unsolved.add_lhs_entry, mean=0, sigma=1, count=10),
    functools.partial(neighbours_unsolved.scale_lhs_entry, mean=0, sigma=1, count=10),
    ]


NEIGHBOURS_ENCODED = [
    functools.partial(neighbours_encoded.exchange_basis, count=5),
    functools.partial(neighbours_encoded.scale_optvalue, mean=0, sigma=1, count=5),
    functools.partial(neighbours_encoded.remove_lhs_entry, count=10),
    functools.partial(neighbours_encoded.add_lhs_entry, mean=0, sigma=1, count=10),
    functools.partial(neighbours_encoded.scale_lhs_entry, mean=0, sigma=1, count=10),
    ]


@calculate_data(coeff_features, solution_features)
def neighbour_unsolved(instance, random_state):
    ''' Neighbours for unsolved instances with calculated feature data. '''
    func = random_state.choice(NEIGHBOURS_UNSOLVED)
    return func(instance, random_state)


@calculate_data(coeff_features, solution_features)
def neighbour_encoded(instance, random_state):
    ''' Neighbours for encoded instances with calculated feature data. '''
    func = random_state.choice(NEIGHBOURS_ENCODED)
    return func(instance, random_state)


@calculate_data(coeff_features, solution_features)
def load_start_unsolved(seed):
    ''' Load an instance as a starting point. '''
    file_name = 'data/generate_naive/inst_{}.tar'.format(seed)
    return read_tar_lp(file_name)


@calculate_data(coeff_features, solution_features)
def load_start_encoded(seed):
    ''' Load an instance as a starting point. '''
    file_name = 'data/generate_uniform/inst_{}.tar'.format(seed)
    return read_tar_encoded(file_name)


# Data for starting instance candidates (feasible/bounded only)
with open('data/generate_naive.json') as infile:
    NAIVE_DATA = [data for data in json.load(infile) if data['solvable']]

with open('data/generate_uniform.json') as infile:
    UNIFORM_DATA = [data for data in json.load(infile) if data['solvable']]

RANDGEN = random.SystemRandom()


def random_start_seed(space, feature_data):
    ''' Choose a random starting candidate appropriate to the objective.
    Different candidate set used for each objective. '''
    if space == 'mean':
        return RANDGEN.choice([
            data['seed'] for data in feature_data
            if data['rhs_mean_normed'] < 25 and data['obj_mean_normed'] > -25])
    elif space == 'frac':
        return RANDGEN.choice([
            data['seed'] for data in feature_data
            if data['binding_constraints'] > 20])


@click.command()
@click.option('--method', default='encoded', type=click.Choice(['encoded', 'direct']), help='Encoding')
@click.option('--space', default='mean', type=click.Choice(['mean', 'frac']), help='Feature space to search')
@click.option('--system-seeds', default=10, type=int, help='Number of random start seeds to use')
@click.option('--seed-file', default=None, type=click.Path(exists=True), help='JSON seed file')
@click.option('--steps', default=1000, type=int, help='Number of search steps per run')
def run(method, space, system_seeds, seed_file, steps):

    # Set objective for given feature space
    if space == 'mean':
        objective = target_point(rhs_mean_normed=-50, obj_mean_normed=50)
    elif space == 'frac':
        objective = target_point(binding_constraints=50, total_fractionality=25)

    # Set functions for the given method
    if method == 'encoded':
        neighbour = neighbour_encoded
        feature_data = UNIFORM_DATA
        load_start = load_start_encoded
    elif method == 'direct':
        neighbour = neighbour_unsolved
        feature_data = NAIVE_DATA
        load_start = load_start_unsolved

    # Starting instances and run seeds
    if seed_file:
        with open(seed_file) as infile:
            seed_values, start_seeds = zip(*(
                (entry['seed'], entry['start']) for entry in json.load(infile)))
    else:
        seed_values = list(system_random_seeds(system_seeds, bits=32))
        start_seeds = [random_start_seed(space, feature_data) for _ in seed_values]

    # Seeded search from given start points
    for seed, start_seed in zip(seed_values, start_seeds):
        print('{} \'{}\' space feature search with seed {} from start {}'.format(method, space, seed, start_seed))
        random_state = np.random.RandomState(seed)

        # Build search object and save results.
        result_file = 'data/search_{}_{}_s{}_r{}.json'.format(method, space, start_seed, seed)
        results = local_search(
            objective=objective,
            sense='min',
            neighbour=neighbour,
            start_instance=load_start(start_seed),
            steps=steps,
            random_state=random_state)
        features = [
            dict(**step_info, **instance.data) for step_info, instance in
            tqdm(results, total=steps, smoothing=0)]
        with open(result_file, 'w') as outfile:
            json.dump(features, outfile, indent=4, sort_keys=True)

run()
