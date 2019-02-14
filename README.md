
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
To generate results using system random seeds, run `make` from this directory.
Figures showing feature and performance distributions of the generated instance sets can be produced using the Jupyter notebook `scripts/figures.ipynb`.
