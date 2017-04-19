''' Performance testing which runs encode/decode/feature/performance
calculations repeatedly to evaluate resource use.

For example
    python -m cProfile -s tottime performance.py --vars 100 --cons 100 --count 100 encode > profile.out
produces profiling information for the time spent in various functions for encode/decode operations.
'''

import sys

import click
import numpy as np
from tqdm import tqdm

from lp_generators.lp_ext import LPCy
from lp_generators.features import coeff_features, solution_features
from lp_generators.instance import EncodedInstance, UnsolvedInstance


def random_lp(n, m):
    ''' Returns (A, b, c) '''
    return (
        np.random.random((m, n)),
        np.random.random(m),
        np.random.random(n))


def random_encoded(n, m):
    ''' Returns (A, alpha, beta) '''
    beta = np.zeros(m + n)
    beta[np.random.choice(m + n, size=m, replace=False)] = 1
    return (
        np.random.random((m, n)),
        np.random.random(m + n),
        beta)


@click.group()
@click.option('--vars', default=100)
@click.option('--cons', default=100)
@click.option('--count', default=100)
@click.option('--progress/--no-progress', default=True)
@click.pass_context
def memtest(ctx, **kwargs):
    ctx.obj.update(kwargs)


def repeat_throwaway(count):
    def repeat_throwaway_decorator(func):
        def run_repeat(progress=True):
            it = range(count)
            if progress:
                it = tqdm(it, total=count, desc='Progress')
            for _ in it:
                result = func()
                result = None
        return run_repeat
    return repeat_throwaway_decorator


@memtest.command()
@click.pass_context
def construct(ctx):
    ''' Construct instances in the extension. '''
    print('Creating and destroying some C++ LPs')
    n, m = ctx.obj['vars'], ctx.obj['cons']
    @repeat_throwaway(ctx.obj['count'])
    def run():
        lp = LPCy()
        lp.construct_dense_canonical(n, m, *random_lp(n, m))
        return lp
    run(ctx.obj['progress'])


@memtest.command()
@click.pass_context
def solve(ctx):
    ''' Construct and solve instances in the extension. '''
    print('Solving some LPs')
    n, m = ctx.obj['vars'], ctx.obj['cons']
    @repeat_throwaway(ctx.obj['count'])
    def run():
        lp = LPCy()
        lp.construct_dense_canonical(n, m, *random_lp(n, m))
        lp.solve()
        return lp
    run(ctx.obj['progress'])


@memtest.command()
@click.pass_context
def encode(ctx):
    ''' Encode/decode instances. '''
    print('Encoding some LPs')
    n, m = ctx.obj['vars'], ctx.obj['cons']
    @repeat_throwaway(ctx.obj['count'])
    def run():
        try:
            A, alpha, beta = random_encoded(n, m)
            instance = EncodedInstance(lhs=A, alpha=alpha, beta=beta)
            lp_instance = UnsolvedInstance(
                lhs=instance.lhs(), rhs=instance.rhs(),
                objective=instance.objective())
            encoded_instance = EncodedInstance(
                lhs=lp_instance.lhs(), alpha=lp_instance.alpha(),
                beta=lp_instance.beta())
        except AssertionError as e:
            print('Failed assert')
    run(ctx.obj['progress'])


@memtest.command()
@click.pass_context
def features(ctx):
    ''' Calculate features of instances. '''
    print('Calculating features of some LPs')
    n, m = ctx.obj['vars'], ctx.obj['cons']
    @repeat_throwaway(ctx.obj['count'])
    def run():
        A, b, c = random_lp(n, m)
        instance = UnsolvedInstance(lhs=A, rhs=b, objective=c)
        instance.data = dict(
            **coeff_features(instance),
            **solution_features(instance))
        return instance
    run(ctx.obj['progress'])


if __name__ == '__main__':
    memtest(obj=dict())
