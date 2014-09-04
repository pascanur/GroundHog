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
and "RNN Search" translation models used for the paper [1]

####Code Structure

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
  The *--proto* option of the train.py script can be used to start with a particular prototype.
  
####Using training script

Simply running
```
train.py
```
would start training in the current directory. Building a model
and compiling might take some time, up to 20 minutes for us. When 
training starts, look for the following files:

- search_model.npz contains all the parameters
- search_state.pkl contains the state
- search_timing.npz contains useful training statistics

The model is saved every 20 minutes. 
If restarted, the training will resume from the last saved model. 

The default prototype state used is *prototype_search_state* that corresponds 
to the RNNsearch-50 model from [1].
The *--proto* options allows to choose a different prototype state,
for instance to train an RNNenc-30 from [1] run
```
train.py --proto=prototype_encdec_state
```
To change options from the state use the positional argument *changes*. For instance, to train
RNNenc-50 from [1] you can run
```
train.py --proto=prototype_encdec_state "prefix='encdec-50_',seqlen=50,sort_k_batches=20"
```
For explanations of the state options see comments in the state.py file. If a lot of changes to the prototype
state are required (for instance you want to train on another dataset), 
you might want put them in a file, e.g. german-data.py, 
if you want to train the same model to translate to German: 
```
dict(
    source=["parallel-corpus/en-de/parallel.en.shuf.h5"],
    ...
    word_indx_trgt="parallel-corpus/en-de/vocab.de.pkl"
)
```
and run
```
train.py --proto=a_prototype_state_of_choice --state german-data.py 
```

####Using sampling script
The typicall call is
```
sample.py --state your_state.pkl your_model.npz 
```
where your_state.pkl and your_model.npz are a state and a model respectively produced by the train.py script.


####Known Issues

- float32 is hardcoded in many places, which effectively means that you can only 
  use the code with floatX=float32 in your .theanorc or THEANO_FLAGS
- In order to sample from the RNNsearch model you have to set the theano option on_unused_input to 'warn' 
  value via either .theanorc or THEANO_FLAGS

####References
"Neural Machine Translation by Jointly Learning to Align and Translate" 
by Dzmitry Bahdanau, Kyung-Hyun Cho and Yoshua Bengio
(http://arxiv.org/abs/1409.0473).
