# Recreate exact experimental results using the seed information in seed_files/ directory.

all: naive parameterised \
	naive_search parameterised_search \
	naive_performance_search parameterised_performance_search

clean:
	rm -rf data/

data/naive_random.json:
	@mkdir -p data
	python naive_random.py --system-seeds 3000

data/parameterised_random.json:
	@mkdir -p data
	python parameterised_random.py --system-seeds 1000

data/naive_search.json: data/naive_random.json
	python naive_search.py --system-seeds 100

data/parameterised_search.json: data/naive_random.json
	python parameterised_search.py --system-seeds 100

data/naive_performance_search_clp_dual_iterations.json: data/naive_random.json
	python naive_performance_search.py --system-seeds 100 --perf-field clp_dual_iterations

data/naive_performance_search_clp_primal_iterations.json: data/naive_random.json
	python naive_performance_search.py --system-seeds 100 --perf-field clp_primal_iterations

data/naive_performance_search_clp_barrier_iterations.json: data/naive_random.json
	python naive_performance_search.py --system-seeds 100 --perf-field clp_barrier_iterations

data/naive_performance_search_clp_barrier_flops.json: data/naive_random.json
	python naive_performance_search.py --system-seeds 100 --perf-field clp_barrier_flops

data/parameterised_performance_search_clp_dual_iterations.json: data/naive_random.json
	python parameterised_performance_search.py --system-seeds 100 --perf-field clp_dual_iterations

data/parameterised_performance_search_clp_primal_iterations.json: data/naive_random.json
	python parameterised_performance_search.py --system-seeds 100 --perf-field clp_primal_iterations

data/parameterised_performance_search_clp_barrier_iterations.json: data/naive_random.json
	python parameterised_performance_search.py --system-seeds 100 --perf-field clp_barrier_iterations

data/parameterised_performance_search_clp_barrier_flops.json: data/naive_random.json
	python parameterised_performance_search.py --system-seeds 100 --perf-field clp_barrier_flops

naive: data/naive_random.json
parameterised: data/parameterised_random.json
naive_search: data/naive_search.json
parameterised_search: data/parameterised_search.json
naive_performance_search: \
	data/naive_performance_search_clp_primal_iterations.json \
	data/naive_performance_search_clp_dual_iterations.json \
	data/naive_performance_search_clp_barrier_iterations.json \
	data/naive_performance_search_clp_barrier_flops.json
parameterised_performance_search: \
	data/parameterised_performance_search_clp_primal_iterations.json \
	data/parameterised_performance_search_clp_dual_iterations.json \
	data/parameterised_performance_search_clp_barrier_iterations.json \
	data/parameterised_performance_search_clp_barrier_flops.json
