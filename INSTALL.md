
This package has been developed and tested in Python 3.5.2 on an Ubuntu 1604 LTS system.
The following is the easiest installation sequence for Ubuntu/Debian in a python virtul environment (may need sudo).

Install required system libraries:

    apt install pkg-config zlib1g-dev libbz2-dev coinor-clp coinor-libclp-dev coinor-libosi-dev

Install Python3, set up virtualenv:

    apt install python3-dev python-pip
    pip install virtualenvwrapper
    source /usr/local/bin/virtualenvwrapper.sh
    mkvirtualenv --python=$(which python3.5) lp_generators
    workon lp_generators

Installation requires numpy, cython and pkgconfig in the local python3 environment.
It is easiest to install these first via pip:

    pip install -r build-requirements.txt

Followed by installing this package:

    pip install .

And extra requirements for scripts and analysis:

    pip install .[scripts,analysis]

Running the scripts additionally requires CLP and SCIP executables available on the system path.
CLP is provided on Ubuntu as the coinor-clp package.

# Tests

Tests of the python package and C++ extension require pytest and mock, installable
from the module setup:

    pip install .[tests]

Running the tests:

    pytest tests/
