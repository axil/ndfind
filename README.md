# ndfind

[![pypi](https://img.shields.io/pypi/v/ndfind.svg)](https://pypi.python.org/pypi/ndfind)
[![python](https://img.shields.io/pypi/pyversions/ndfind.svg)](https://pypi.org/project/ndfind/)
![pytest](https://github.com/axil/ndfind/actions/workflows/python-package.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/pypi/l/ndfind)](https://pypi.org/project/ndfind/)

A collection of three cython-optimized search functions for NumPy. When the required value is found,
they return immediately, without scanning the whole array. It can result in 1000x or larger speedups for 
huge arrays if the value is located close to the the beginning of the array.

## Installation: 

    pip install ndfind

## Contents

Basic operations:
- `find(a, v)`
- `first_above(a, v)`
- `first_nonzero(v)`

## Testing

Run `pytest` in the project root.