''' Produce result figures from generated data. '''

import warnings
import glob
from functools import partial
import contextlib
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set data and figure directories
DATA_DIR = 'data'
FIGURE_DIR = 'result_figures'

if not os.path.exists(DATA_DIR):
    raise ValueError('Data directory does not exist'.format(DATA_DIR))
with contextlib.suppress(FileExistsError):
    os.mkdir(FIGURE_DIR)

# Standardise plot styling
sns.set()
sns.set_style('white')

# Standardise axis names and bounds for fields and data sets
pretty_names = dict(
    rhs_mean='RHS Value Mean',
    rhs_mean_normed='RHS Value Mean (Normalised)',
    coefficient_density='Coefficient Density',
    cons_degree_max='Constraint Degree Max',
    var_degree_max='Variable Degree Max',
    obj_mean='Objective Coefficient Mean',
    obj_mean_normed='Objective Coefficient Mean (Normalised)',
    binding_constraints='Number of Binding Constraints',
    total_fractionality='Total Fractionality',
    strbr_percall='Iterations per branch',
    clp_primal_iterations='Primal Simplex Iterations',
    clp_dual_iterations='Dual Simplex Iterations',
    clp_barrier_flops='Barrier Floating Point Operations',
    variables='Variables',
    constraints='Constraints')

standard_bounds = dict(
    rhs_mean=(-150, 150),
    rhs_mean_normed=(-100, 100),
    obj_mean=(-150, 150),
    obj_mean_normed=(-100, 100),
    binding_constraints=(-5, 55),
    total_fractionality=(-2.5, 27.5))

relabel = dict(
    generated='Constructor Generator',
    naive='Simple Random',
    search_encoded='Encoding Search',
    search_lp='Direct Search')


def load_pattern(pattern, min_index=0):
    ''' Load and concat all data files matching the given glob :pattern '''
    search_data = []
    pattern = os.path.join(DATA_DIR, pattern)
    for file in glob.glob(pattern):
        try:
            with open(file) as infile:
                search_data.append(pd.read_json(infile))
        except:
            print('Failed to load {}'.format(file))
    result = pd.concat(search_data)
    return result[result.index > min_index]


def scatter(x, y, color, label, **kwargs):
    with contextlib.suppress(KeyError):
        label = relabel[label]
    ax = sns.regplot(x=x, y=y, label=label, color=sns.color_palette()[color], fit_reg=False, **kwargs)
    with contextlib.suppress(KeyError):
        ax.set_xlabel(pretty_names[x])
    with contextlib.suppress(KeyError):
        ax.set_ylabel(pretty_names[y])
    with contextlib.suppress(KeyError):
        ax.set_xbound(standard_bounds[x])
    with contextlib.suppress(KeyError):
        ax.set_ybound(standard_bounds[y])
    ax.legend()
    return ax


# Random distribution data
uniform_data = pd.read_json(os.path.join(DATA_DIR, 'generate_uniform.json'))
varied_size_data = pd.read_json(os.path.join(DATA_DIR, 'generate_varied_size.json'))
all_naive_data = pd.read_json(os.path.join(DATA_DIR, 'generate_naive.json'))
naive_data = all_naive_data[all_naive_data.solvable]

# Feature space search data
encoded_mean_search_data = load_pattern('search_encoded_mean_s*_r*.json')
encoded_frac_search_data = load_pattern('search_encoded_frac_s*_r*.json')
lp_frac_search_data = load_pattern('search_direct_frac_s*_r*.json')
lp_mean_search_data = load_pattern('search_direct_mean_s*_r*.json')

# Performance space search data
encoded_primal_search_data = load_pattern('search_perform_clp_primal_iterations_s*_r*.json')
encoded_dual_search_data = load_pattern('search_perform_clp_dual_iterations_s*_r*.json')
encoded_barrier_search_data = load_pattern('search_perform_clp_barrier_flops_s*_r*.json')
encoded_strbr_search_data = load_pattern('search_perform_strbr_percall_s*_r*.json')

# Generated features

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

plot = partial(scatter, x='rhs_mean_normed', y='obj_mean_normed', ax=ax1)
plot(data=uniform_data, label='generated', color=0)

plot = partial(scatter, x='binding_constraints', y='total_fractionality', ax=ax2)
plot(data=uniform_data, label='generated', color=0)

plot = partial(scatter, x='coefficient_density', y='var_degree_max', ax=ax3)
plot(data=uniform_data, label='generated', color=0)

plot = partial(scatter, x='cons_degree_max', y='var_degree_max', ax=ax4)
plot(data=uniform_data, label='generated', color=0)

fig1.savefig(os.path.join(FIGURE_DIR, 'coeff_generated.png'))
fig2.savefig(os.path.join(FIGURE_DIR, 'relaxation_generated.png'))
fig3.savefig(os.path.join(FIGURE_DIR, 'lhs_generated.png'))
fig4.savefig(os.path.join(FIGURE_DIR, 'degree_generated.png'))

# Comparison with naive/simple random generator.

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

plot = partial(scatter, x='rhs_mean_normed', y='obj_mean_normed', ax=ax1)
plot(data=uniform_data, label='generated', color=0)
plot(data=naive_data, label='naive', color=1)

plot = partial(scatter, x='binding_constraints', y='total_fractionality', ax=ax2)
plot(data=uniform_data, label='generated', color=0)
plot(data=naive_data, label='naive', color=1)

fig1.savefig(os.path.join(FIGURE_DIR, 'coeff_compare.png'))
fig2.savefig(os.path.join(FIGURE_DIR, 'relaxation_compare.png'))

# Feature space search processes in the naive and encoded spaces.
# Coefficient features

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
plot = partial(scatter, x='rhs_mean_normed', y='obj_mean_normed')

plot(data=uniform_data, label='generated', color=0, ax=ax1)
plot(data=encoded_mean_search_data, label='search_encoded', color=2, ax=ax1)

plot(data=naive_data, label='naive', color=1, ax=ax2)
plot(data=lp_mean_search_data, label='search_lp', color=3, ax=ax2)

fig1.savefig(os.path.join(FIGURE_DIR, 'coeff_search_encoded.png'))
fig2.savefig(os.path.join(FIGURE_DIR, 'coeff_search_naive.png'))

# Relaxation features

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
plot = partial(scatter, x='binding_constraints', y='total_fractionality')

plot(data=uniform_data, label='generated', color=0, ax=ax1)
plot(data=encoded_frac_search_data, label='search_encoded', color=2, ax=ax1)

plot(data=naive_data, label='naive', color=1, ax=ax2)
plot(data=lp_frac_search_data, label='search_lp', color=3, ax=ax2)

for ax in (ax1, ax2):
    ax.set_xbound([-5, 55])
    ax.set_ybound([-2.5, 27.5])

fig1.savefig(os.path.join(FIGURE_DIR, 'relaxation_search_encoded.png'))
fig2.savefig(os.path.join(FIGURE_DIR, 'relaxation_search_naive.png'))

# Algorithm performance on generated and searched instances of varying size
# Performance search targets in encoded space
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

plot = partial(scatter, x='variables', y='clp_primal_iterations', ax=ax1)
plot(data=varied_size_data, label='generated', color=0)
plot(data=encoded_primal_search_data, label='search_encoded', color=2)

plot = partial(scatter, x='variables', y='clp_dual_iterations', ax=ax2)
plot(data=varied_size_data, label='generated', color=0)
plot(data=encoded_dual_search_data, label='search_encoded', color=2)

plot = partial(scatter, x='variables', y='clp_barrier_flops', ax=ax3)
plot(data=varied_size_data, label='generated', color=0)
plot(data=encoded_barrier_search_data, label='search_encoded', color=2)
ax3.set_ybound([-20000, 400000])

plot = partial(scatter, x='variables', y='strbr_percall', ax=ax4)
plot(data=varied_size_data, label='generated', color=0)
plot(data=encoded_strbr_search_data, label='search_encoded', color=2)

fig1.savefig(os.path.join(FIGURE_DIR, 'performance_primal_search.png'))
fig2.savefig(os.path.join(FIGURE_DIR, 'performance_dual_search.png'))
fig3.savefig(os.path.join(FIGURE_DIR, 'performance_barrier_search.png'))
fig4.savefig(os.path.join(FIGURE_DIR, 'performance_strbr_search.png'))
