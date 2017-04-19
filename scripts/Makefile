# Recreate exact experimental results using the seed information in seed_files/ directory.

all: uniform varied_size naive encoded_frac encoded_mean naive_frac naive_mean primal_search dual_search barrier_search reopt_search

clean:
	rm -rf data/

data/generate_uniform.json:
	@mkdir -p data
	python generate_uniform.py --seed-file seed_files/uniform.json

data/generate_varied_size.json:
	@mkdir -p data
	python generate_varied_size.py --seed-file seed_files/varied_size.json

data/generate_naive.json:
	@mkdir -p data
	python generate_naive.py --seed-file seed_files/naive.json

uniform: data/generate_uniform.json
varied_size: data/generate_varied_size.json
naive: data/generate_naive.json

encoded_frac: uniform
	python search_feature.py --method encoded --space frac --seed-file seed_files/search_encoded_frac.json --steps 1000

encoded_mean: uniform
	python search_feature.py --method encoded --space mean --seed-file seed_files/search_encoded_mean.json --steps 1000

naive_frac: naive
	python search_feature.py --method direct  --space frac --seed-file seed_files/search_direct_frac.json  --steps 1000

naive_mean: naive
	python search_feature.py --method direct  --space mean --seed-file seed_files/search_direct_mean.json  --steps 1000

primal_search: varied_size
	python search_performance.py --metric primal  --seed-file seed_files/search_perform_primal.json  --steps 200

dual_search: varied_size
	python search_performance.py --metric dual    --seed-file seed_files/search_perform_dual.json    --steps 200

barrier_search: varied_size
	python search_performance.py --metric barrier --seed-file seed_files/search_perform_barrier.json --steps 200

reopt_search: varied_size
	python search_performance.py --metric reopt   --seed-file seed_files/search_perform_reopt.json   --steps 200
