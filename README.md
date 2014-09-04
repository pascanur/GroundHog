GroundHog by lisa-groundhog
===========================

GroundHog is a python framework on top of Theano
(http://deeplearning.net/softward/theano/) that aims to provide a flexible, yet
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
containing Tut (in scripts) for a quick tutorial on how to use the library.

The library is under the 3-clause BSD license, so it may be used for commercial
purposes. 


List of Experiments
===================

- experiments/rnnencdec/
  Neural machine translation

Neural Machine Translation
--------------------------

The folder experiments/rnnencdec contains the implementations of "RNN Encoder-Decoder"
and "RNN Search" translation models used for the paper
"Neural Machine Translation by Jointly Learning to Align and Translate" 
(http://arxiv.org/abs/1409.0473). The code is structured as follows:

- encdec.py contains the actual models code
- train.py is a script to train a new model or continue training an existing one
- sample.py can be used to sample translations from the model 
  (or to find the most probable translations)
- score.py is used to score sentences pairs, that is to compute log-likelihood 
  of a translation to be generated from a source sentence
- state.py contains prototype states. In this project a *state* means in fact a full
  specification of the model and training process, including architectural choices,
  layer sizes, training data and vocabularies. The prototype states in the state.py files
  are base configurations from which one can start to train a model of his/her choice.
  The --proto option of the train.py script can be used to start with a particular prototype.
  







