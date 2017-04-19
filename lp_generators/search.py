''' Instance space search function, which applies neighbourhood operators to
generate new instances to check for improvement against the given objective.
The search function returns a generator which iterates over step results. '''

import os
from contextlib import suppress
import functools


def local_search(objective, sense, neighbour, start_instance, steps, random_state):
    ''' Start from a given instance, generating a random neighbour at each step
    and accepting it if it improves the objective function for the given sense.
    Result is a generator, where each step yields a tuple step_info, instance.
    step info is a dict:
        search_step: step count
        search_objective: current objective function value
        search_update: 'improved' if the current step is new, 'reject_poor' otherwise
    instance is the current instance object at this step '''

    if sense == 'min':
        def accept_next(c_new, c_old):
            return c_new < c_old
    elif sense == 'max':
        def accept_next(c_new, c_old):
            return c_new > c_old
    else:
        raise ValueError('Sense must be max or min')

    # initial state
    instance = start_instance
    next_instance = start_instance
    c_old = 1e+20 if sense == 'min' else -1e+20
    is_new = True
    step_info = dict(step='start')

    for step in range(steps):

        # data and objective calculation
        c_new = objective(next_instance)

        # step update rule
        if accept_next(c_new, c_old):
            instance = next_instance
            c_old = c_new
            is_new = True
            state = 'improved'
        else:
            state = 'reject_poor'

        step_info = dict(
            search_step=step,
            search_objective=c_old,
            search_update=state)
        yield step_info, instance

        # next candidate
        next_instance = neighbour(instance, random_state)
        is_new = False


def write_steps(write_func, name_format, new_only):
    ''' Write the results of a search function, passing the current step
    count to name_format. Reads from step_info whether the instance is new
    or not, so the function can optionally write new instances only. '''
    def write_steps_decorator(func):
        @functools.wraps(func)
        def write_steps_fn(*args, **kwargs):
            # ensure the directory exists
            directory, _ = os.path.split(name_format)
            with suppress(FileExistsError):
                os.makedirs(directory)
            # write each instance as it is yielded, pass on
            for step_info, instance in func(*args, **kwargs):
                if new_only is False or step_info['search_update'] == 'improved':
                    write_func(instance, name_format.format(step=step_info['search_step']))
                yield step_info, instance
        return write_steps_fn
    return write_steps_decorator
