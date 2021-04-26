
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

# Citing this work

Paper published in Mathematical Programming Computation (MPC) where we describe
the generator and investigate properties of the generated instances:

```
@Article{Bowly.Smith-Miles.ea_2020_Generation-techniques,
    author      = {Bowly, Simon and Smith-Miles, Kate and Baatar, Davaatseren
                  and Mittelmann, Hans},
    title       = {Generation techniques for linear programming instances
                  with controllable properties},
    journal     = {Mathematical Programming Computation},
    year        = 2020,
    volume      = 12,
    number      = 3,
    pages       = {389--415},
    month       = sep,
}
```

Direct citation of the version of the code used in the MPC paper:

```
@software{Bowly_2018_MIP-instance,
  author       = {Simon Bowly},
  title        = {simonbowly/lp-generators: v0.2-beta},
  month        = apr,
  year         = 2018,
  publisher    = {Zenodo},
  version      = {v0.2-beta},
  doi          = {10.5281/zenodo.1220448},
  url          = {https://doi.org/10.5281/zenodo.1220448}
}
```
