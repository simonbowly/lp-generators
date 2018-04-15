
import json

import numpy as np

from lp_generators.features import coeff_features, solution_features
from lp_generators.performance import clp_simplex_performance
from lp_generators.writers import read_tar_encoded


with open('data/parameterised_random.json') as infile:
    naive_random_data = json.load(infile)


def condition(data):
    return data['solvable'] is True


def objective(data, perf_field):
    ''' default min objective -> this favours harder instances '''
    return data[perf_field] * -1


def _sample(rstate):
    while True:
        choice = rstate.choice(naive_random_data)
        if condition(choice):
            return choice


def start_instance(rstate, perf_field):
    seed = min((_sample(rstate) for _ in range(200)), key=lambda d: objective(d, perf_field))['seed']
    return read_tar_encoded(f'data/parameterised_random/inst_{seed}.tar')


def calculate_features(instance):
    return dict(
        **solution_features(instance),
        **clp_simplex_performance(instance))
