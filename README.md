
# About

This package is a collection of algorithms for generating random MIP instances by constructing relaxations with known solutions.
Scripts are included to generate random instance distributions and instance space search results.

# Install

This package is developed in Python 3.5.2.
Building the C++ extension requires numpy, cython, pkgconfig python packages, and the COIN-LP callable library.

Basic functionality is installed using:

    pip install .

Extra python requirements for generating results and analysis:

    pip install .[scripts,analysis]

Algorithm performance evaluation functions require clp and scip on the system path.

See INSTALL.md for detailed installation instructions.

# Examples

The examples/ directory contains sample scripts for setting up generation and search algorithms in instance space.

    python generate.py      # Generates instances using constructor approach
    python search.py        # Make generated instances harder to solve

# Running the Experiments

Scripts to reproduce experimental results are contained in the scripts/ directory.

Regenerate results using seed values stored in seed_files/ directory:

    make

Generate new results (with the same parameter distributions and search targets) using system random seeds:

    make -f MakeRandom.make

Generate figures from the result set:

    python analysis.py
