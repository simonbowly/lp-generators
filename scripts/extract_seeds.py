''' Extracts random seeds used in result generation for later repeatability.
In particular, by running:
    make -f MakeRandom.make
    python extract_seeds.py
    rm -r data/
    make
The second make command should recreate the exact data set of the first in data/
by using the extracted random seeds to start new generation and search runs. '''

import json
import glob
import re
import operator


def from_json(fin, fout):
    with open(fin) as infile:
        seed_values = [entry['seed'] for entry in json.load(infile)]
    with open(fout, 'w') as outfile:
        json.dump(sorted(seed_values), outfile, indent=0)


def from_pattern(pattern, fout):
    files = glob.glob(pattern)
    result = []
    for file in files:
        match = re.match('.*_s([0-9]+)_r([0-9]+)', file)
        result.append(dict(start=int(match.group(1)), seed=int(match.group(2))))
    with open(fout, 'w') as outfile:
        json.dump(
            sorted(result, key=operator.itemgetter('seed', 'start')),
            outfile, indent=0, sort_keys=True)


from_json('data/generate_uniform.json', 'seed_files/uniform.json')
from_json('data/generate_naive.json', 'seed_files/naive.json')
from_json('data/generate_varied_size.json', 'seed_files/varied_size.json')

from_pattern('data/search_encoded_frac*', 'seed_files/search_encoded_frac.json')
from_pattern('data/search_encoded_mean*', 'seed_files/search_encoded_mean.json')

from_pattern('data/search_direct_frac*', 'seed_files/search_direct_frac.json')
from_pattern('data/search_direct_mean*', 'seed_files/search_direct_mean.json')

from_pattern('data/search_perform_clp_primal_iterations*', 'seed_files/search_perform_primal.json')
from_pattern('data/search_perform_clp_dual_iterations*', 'seed_files/search_perform_dual.json')
from_pattern('data/search_perform_clp_barrier_flops*', 'seed_files/search_perform_barrier.json')
from_pattern('data/search_perform_strbr_percall*', 'seed_files/search_perform_reopt.json')
