GroundHog by lisa-groundhog
===========================

**WARNING: Groundhog development is over.** Please consider using 
[Blocks](https://github.com/mila-udem/blocks) instead. For an example of machine translation using Blocks please see [Blocks-examples](https://github.com/mila-udem/blocks-examples) repository

GroundHog is a python framework on top of Theano
(http://deeplearning.net/software/theano/) that aims to provide a flexible, yet
efficient way of implementing complex recurrent neural network models. It
supports a variety of recurrent layers, such as DT-RNN, DOT-RNN, RNN with gated
hidden units and LSTM. Furthermore, it enables the flexible combination of
various layers, for instance, to build a neural translation model.

This is a version forked from the original GroundHog
(https://github.com/pascanur/GroundHog) developed by Razvan Pascanu, Caglar
Gulcehre and Kyunghyun Cho. This fork will be the version developed and
maintained by the members of the LISA Lab at the University of Montreal. The
main contributors and maintainers of this fork are currently Dzmitry Bahdanau
and Kyunghyun Cho.

Most of the library documentation is still work in progress, but check the files
containing Tut (in tutorials) for a quick tutorial on how to use the library.

The library is under the 3-clause BSD license, so it may be used for commercial
purposes. 


Installation
------------
To install Groundhog in a multi-user setting (such as the LISA lab)

``python setup.py develop --user``

For general installation, simply use

``python setup.py develop``

NOTE: This will install the development version of Theano, if Theano is not
currently installed.

Neural Machine Translation
--------------------------

See experiments/nmt/README.md

