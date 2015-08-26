#!/usr/bin/env python

# Sets the __version__ variable
exec(open('sandpile/_version.py').read())

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
    import numpy as np
    from Cython.Build import cythonize
except ImportError as e:
    if not e.args:
        e.args = ('',)
    err_message = 'NumPy and Cython are required for installation.'
    e.args = (e.args[0] + err_message,) + e.args[1:]
    raise


setup(
    name='sandpile',
    version=__version__,
    author='Stefan Pfenninger',
    author_email='stefan@pfenninger.org',
    description='Sandpile cascading failure model',
    packages=['sandpile'],
    install_requires=[
        "numpy >= 1.8.1",
        "scipy >= 0.13.2",
        "networkx >= 1.8.1",
        "pandas >= 0.13.1",
        "Cython >= 0.20.1"
    ],
    entry_points={
        'console_scripts': [
            'sandpile = sandpile.cluster:main',
        ]
    },
    include_dirs=[np.get_include()],
    ext_modules=cythonize('sandpile/core.pyx')
)
