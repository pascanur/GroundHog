#!/usr/bin/env python

import os
import setuptools

setuptools.setup(
    name='groundhog',
    version='0.1dev',
    packages=setuptools.find_packages(),
    description='Recurrent neural net tools built on top of Theano',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='BSD 3-clause',
    url='http://github.com/lisa-groundhog/GroundHog/',
    install_requires=['numpy',
                      'Theano',],
    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano',],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
)
