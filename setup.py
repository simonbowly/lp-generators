
import os
from setuptools import setup, find_packages, Extension

try:
    import numpy as np
    from Cython.Build import cythonize
    import pkgconfig
except ImportError:
    raise ImportError('Build process requires numpy, cython and pkconfig.')

if not pkgconfig.exists('osi-clp'):
    raise ImportError('Building extension requires osi-clp discoverable by pkg-config')

requirements = pkgconfig.parse('osi-clp')
requirements['include_dirs'].append(np.get_include())

package_requires = [
    'numpy',
    'scipy',
    ]

extras_require = {
    'tests': ['pytest', 'mock'],
    'scripts': ['click', 'pandas', 'tqdm'],
    'analysis': ['matplotlib', 'seaborn'],
    }

extensions = cythonize(Extension(
    'lp_generators_ext', language='c++',
    sources=['cpp/lp_generators_ext.pyx', 'cpp/lp.cpp'],
    extra_compile_args=['-std=c++11'],
    **requirements))

setup(
    name='lp_generators',
    version='0.1.0',
    description='Generation of MIP test cases by LP relaxation.',
    url='https://github.com/simonbowly/lp-generators',
    author='Simon Bowly',
    author_email='simon.bowly@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Mathematics'],
    packages=['lp_generators'],
    package_dir={'lp_generators': 'lp_generators'},
    install_requires=package_requires,
    extras_require=extras_require,
    ext_modules=extensions)
