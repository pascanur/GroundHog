Neural Machine Translation
--------------------------

The folder experiments/nmt contains the implementations of RNNencdec and
RNNsearch translation models used for the paper [1,2]

####Code Structure

- encdec.py contains the actual models code
- train.py is a script to train a new model or continue training an existing one
- sample.py can be used to sample translations from the model (or to find the
  most probable translations)
- score.py is used to score sentences pairs, that is to compute log-likelihood
  of a translation to be generated from a source sentence
- state.py contains prototype states. In this project a *state* means in fact a
  full specification of the model and training process, including architectural
  choices, layer sizes, training data and vocabularies. The prototype states in
  the state.py files are base configurations from which one can start to train a
  model of his/her choice.  The *--proto* option of the train.py script can be
  used to start with a particular prototype.
- preprocess/*.py are the scripts used to preprocess a parallel corpus to obtain
  a dataset (see 'Data Preparation' below for more detail.)
- web-demo/ contains files for running a web-based demonstration of a trained
  neural machine translation model (see web-demo/README.me for more detail).

All the paths below are relative to experiments/nmt.
  
####Using training script

Simply running
```
train.py
```
would start training in the current directory. Building a model and compiling
might take some time, up to 20 minutes for us. When training starts, look for
the following files:

- search_model.npz contains all the parameters
- search_state.pkl contains the state
- search_timing.npz contains useful training statistics

The model is saved every 20 minutes.  If restarted, the training will resume
from the last saved model. 

The default prototype state used is *prototype_search_state* that corresponds to
the RNNsearch-50 model from [1].

The *--proto* options allows to choose a different prototype state, for instance
to train an RNNencdec-30 from [1] run
```
train.py --proto=prototype_encdec_state
```
To change options from the state use the positional argument *changes*. For
instance, to train RNNencdec-50 from [1] you can run
```
train.py --proto=prototype_encdec_state "prefix='encdec-50_',seqlen=50,sort_k_batches=20"
```
For explanations of the state options, see comments in the state.py file. If a
lot of changes to the prototype state are required (for instance you want to
train on another dataset), you might want put them in a file, e.g.
german-data.py, if you want to train the same model to translate to German: 
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
The typical call is
```
sample.py --beam-search --state your_state.pkl your_model.npz 
```
where your_state.pkl and your_model.npz are a state and a model respectively
produced by the train.py script.  A batch mode is also supported, see the
sample.py source code.

####Data Preparation

In short, you need the following files:
- source sentence file in HDF5 format
- target sentence file in HDF5 format
- source dictionary in a pickle file (word -> id)
- target dictionary in a pickle file (word -> id)
- source inverse dictionary in a pickle file (id -> word)
- target inverse dictionary in a pickle file (id -> word)

In experiments/nmt/preprocess, we provide scripts that we use to generate these
data files from a parallel corpus saved in .txt files. 

The data preparation scripts assume that the the parallel corpus has been
correctly tokenized already. In the case of English, for instance, you can
tokenize the text file using tokenizer.pl from Moses (or the one in web-demo):
```
perl tokenizer.perl -l en < bitext.en.txt > bitext.en.tok.txt
```

```
python preprocess.py -d vocab.en.pkl -v 30000 -b binarized_text.en.pkl -p *en.txt.gz
```
This will create a dictionary (vocab.en.pkl) of 30,000 most frequent words and a
pickle file (binarized_text.pkl) that contains a list of numpy arrays of which
each corresponds to each line in the text files. 
```
python invert-dict.py vocab.en.pkl ivocab.en.pkl
```
This will generate an inverse dictionary (id -> word).
```
python convert-pkl2hdf5.py binarized_text.en.pkl binarized_text.en.h5
```
This will convert the generated pickle file into an HDF5 format. 
```
python shuffle-hdf5.py binarized_text.en.h5 binarized_text.fr.h5 binarized_text.en.shuf.h5 binarized_text.fr.shuf.h5
```
Since it can be very expensive to shuffle the dataset each time you train a
model, we shuffle dataset in advance. However, note that we do keep the original
files for debugging purpose.

####Tests

Run
```
GHOG=/path/to/groundhog test/test.bash
```
to test sentence pairs scoring and translation generation. The script will start with creating 
a workspace directory and download test models there. You can keep using the same test workspace
and data.

####Known Issues

- float32 is hardcoded in many places, which effectively means that you can only 
  use the code with floatX=float32 in your .theanorc or THEANO_FLAGS
- In order to sample from the RNNsearch model you have to set the theano option on_unused_input to 'warn' 
  value via either .theanorc or THEANO_FLAGS

####References

Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. 
Neural Machine Translation by Jointly Learning to Align and Translate
http://arxiv.org/abs/1409.0473

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk and Yoshua Bengio.
Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.
EMNLP 2014. http://arxiv.org/abs/1406.1078

