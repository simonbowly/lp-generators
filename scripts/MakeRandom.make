# Recreate experimental results using a new set of system random seeds each time.
# Each of the 'generate' scripts produces one datafile, so their results are
# satisfied by running once. Repeated runs without deleting the data/ folder will
# produce extra search runs from new random seeds.

all: uniform varied_size naive encoded_frac encoded_mean naive_frac naive_mean primal_search dual_search barrier_search reopt_search

clean:
	rm -rf data/

data/generate_uniform.json:
	@mkdir -p data
	python generate_uniform.py --system-seeds 1000

data/generate_varied_size.json:
	@mkdir -p data
	python generate_varied_size.py --system-seeds 200

data/generate_naive.json:
	@mkdir -p data
	python generate_naive.py --system-seeds 1000

uniform: data/generate_uniform.json
varied_size: data/generate_varied_size.json
naive: data/generate_naive.json

encoded_frac: uniform
	python search_feature.py --method encoded --space frac --system-seeds 10 --steps 1000

encoded_mean: uniform
	python search_feature.py --method encoded --space mean --system-seeds 10 --steps 1000

naive_frac: naive
	python search_feature.py --method direct  --space frac --system-seeds 10 --steps 1000

naive_mean: naive
	python search_feature.py --method direct  --space mean --system-seeds 10 --steps 1000

primal_search: varied_size
	python search_performance.py --metric primal  --system-seeds 10 --steps 200

dual_search: varied_size
	python search_performance.py --metric dual    --system-seeds 10 --steps 200

barrier_search: varied_size
	python search_performance.py --metric barrier --system-seeds 10 --steps 200

reopt_search: varied_size
	python search_performance.py --metric reopt   --system-seeds 10 --steps 200
