''' Common utilites used by scripts in this directory. Not installed as part
of the lp_generators package. '''

import functools
import json

import click

from lp_generators.utils import system_random_seeds


def cli_seeds(func):
    ''' Wrap a function taking a list of seed values with a cli command.
    Resulting cli command accepts either a JSON seed file or a count of
    system random seeds to generate. '''

    @click.command()
    @click.option('--system-seeds', default=100, type=int, help='Number of system random seeds')
    @click.option('--seed-file', default=None, type=click.Path(exists=True), help='JSON seed file')
    @functools.wraps(func)
    def cli_seeds_fn(system_seeds, seed_file):
        if seed_file:
            with open(seed_file) as infile:
                seed_values = json.load(infile)
        else:
            seed_values = list(system_random_seeds(n=system_seeds, bits=32))
        func(seed_values)

    return cli_seeds_fn
