# witss

This Python package lets you compute the Iterated Sums Signature of a time series in `numpy`. All
methods are implemented using `numba`, which leads to a large speed-up of computations through
compiling functions directly in Python.

Together with weighted iterated sums and iterated sums over different semirings, this repository
provides a toolbox for time series feature extraction. This approach has been already explored in
our publication of [FRUITS](https://github.com/irkri/fruits).

## Installation

The package is installable via pip:

    $ pip install witss

The project's dependecies are just `numpy` and `numba`.
For development or further testing, it is recommended to use [poetry](https://python-poetry.org/)
or install the development dependencies listed in [pyproject.toml](pyproject.toml) otherwise.

Importing the package for the first time could take a minute, as some functions are compiled by
`numba` and then cached for later use.

## Examples

Examples can be found in [/examples](/examples/). For running the examples, you also have to
install the development dependencies.
