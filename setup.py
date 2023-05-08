#!/usr/bin/env python

import setuptools

import numpy as np
from Cython.Build import cythonize

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ndfind',
    version='0.4.1',
    author='Lev Maximov',
    author_email='lev.maximov@gmail.com',
    url='https://github.com/axil/ndfind',
    description='A collection of cython-optimized search functions for NumPy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7",
    ext_modules=cythonize(["ndfind/main.pyx"]),
    include_dirs=np.get_include(),
    install_requires=[
        'numpy>=1.24',
    ],
    packages=['ndfind'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',    
        'Programming Language :: Python :: 3.10',    
        'Programming Language :: Python :: 3.11',    
    ],
    license='MIT License',
    zip_safe=False,
    keywords=['find', 'first_above', 'first_nonzero', 'numpy', 'python'],
)
