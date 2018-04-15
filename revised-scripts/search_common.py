
import json

import numpy as np

from lp_generators.writers import read_tar_lp


with open('data/naive_random.json') as infile:
    naive_random_data = json.load(infile)


def condition(data):
    return data['solvable'] is True


def objective(data):
    return (data['rhs_mean'] + 100) ** 2 + (data['obj_mean'] - 100) ** 2


def _sample(rstate):
    while True:
        choice = rstate.choice(naive_random_data)
        if condition(choice):
            return choice


def start_instance(rstate):
    seed = min((_sample(rstate) for _ in range(20)), key=objective)['seed']
    return read_tar_lp(f'data/naive_random/inst_{seed}.tar')
