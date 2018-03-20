''' Performance calculation functions. Calls SCIP and CLP solvers as
subprocesses, so both must be available on the system path. '''

import subprocess
import re

from .writers import write_mps, write_mps_ip
from .utils import temp_file_path


def clp_solve_file(file, method):
    ''' Solve with a clp method and return statistics. '''
    result = subprocess.run(
        ['clp', file, '-{}'.format(method)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout = result.stdout.decode('utf-8')
    regex = 'Optimal objective +([0-9e\-\.\+]+) +- +([0-9]+) +iterations +time +([0-9\.]+)'
    match = re.search(regex, stdout)
    if match is None:
        # There are iteration counts to check here.
        # This occurs in infeasible/unbounded cases.
        return dict(objective=None, iterations=-1, time=-1)
    result = dict(
        objective=float(match.group(1)),
        iterations=int(match.group(2)),
        time=float(match.group(3)))
    regex = 'flop count +([0-9]+)'
    match = re.search(regex, stdout)
    if match is not None:
        result['flops'] = int(match.group(1))
    return result


def scip_strongbranch_file(file):
    ''' Run SCIP and force all full strong branching.
    Terminate at the root node, return strong branching stats.
    Gives a measure of reoptimisation effort. '''
    result = subprocess.run(
            [
            'scip', '-c', 'read {}'.format(file),
            '-c', 'set limits nodes 1',
            '-c', 'set branching allfullstrong priority 1000000',
            '-c', 'opt',
            '-c', 'display statistics',
            '-c', 'quit'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout = result.stdout.decode('utf-8')
    regex = 'strong branching +: +([0-9\.]+) +([0-9]+) +([0-9]+) +([0-9\.]+)'
    match = re.search(regex, stdout)
    return dict(
        time=float(match.group(1)),
        calls=int(match.group(2)),
        iterations=int(match.group(3)),
        percall=float(match.group(4)))


def clp_simplex_performance(instance):
    ''' Write an instance as LP, report primal simplex results. '''
    with temp_file_path('.mps.gz') as file:
        write_mps(instance, file)
        primal_result = clp_solve_file(file, 'primalsimplex')
        dual_result = clp_solve_file(file, 'dualsimplex')
        barrier_result = clp_solve_file(file, 'barrier')
        if 'flops' not in barrier_result:
            barrier_result['flops'] = -1
    return dict(
        clp_primal_objective=primal_result['objective'],
        clp_primal_iterations=primal_result['iterations'],
        clp_primal_time=primal_result['time'],
        clp_dual_objective=dual_result['objective'],
        clp_dual_iterations=dual_result['iterations'],
        clp_dual_time=dual_result['time'],
        clp_barrier_objective=barrier_result['objective'],
        clp_barrier_iterations=barrier_result['iterations'],
        clp_barrier_time=barrier_result['time'],
        clp_barrier_flops=barrier_result['flops'])


def strbr_performance(instance):
    ''' Write an instance as pure IP, report strong branching results. '''
    with temp_file_path('.mps.gz') as file:
        # integrality conversion
        write_mps_ip(instance, file)
        result = scip_strongbranch_file(file)
    return dict(
        strbr_time=result['time'],
        strbr_calls=result['calls'],
        strbr_iterations=result['iterations'],
        strbr_percall=result['percall'])
