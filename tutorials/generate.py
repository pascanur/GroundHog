#!/usr/bin/python2.7

"""
Script that parses the wikipedia dump, and generates the dataset in a nice
numpy format (i.e. in numpy.npz files).
Call :
    generate_word,py --help
"""
from collections import Counter
import ConfigParser
import argparse
import os
import time
import sys
import numpy


def construct_vocabulary(dataset, oov_rate, level):
    filename = os.path.join(dataset,  'train')
    fd = open(filename, 'rt')
    txt = fd.read()
    if level == 'words':
        txt = txt.replace('\n', ' \n ')
        txt = txt.replace('  ', ' ')
        txt = txt.split(' ')
        txt = [x for x in txt if x != '']
    # Order the words
    print ' .. sorting words'
    all_items = Counter(txt).items()
    no_end = [x for x in all_items if x[0] !='\n']
    freqs = [x for x in all_items if x[0] == '\n'] + \
            sorted(no_end,
                   key=lambda t: t[1],
                   reverse=True)
    print ' .. shrinking the vocabulary size'
    # Decide length
    all_freq = float(sum([x[1] for x in freqs]))
    up_to = len(freqs)
    oov = 0.
    remove_word = True
    while remove_word:
        up_to -= 1
        oov += float(freqs[up_to][1])
        if oov / all_freq > oov_rate:
            remove_word = False
    up_to += 1
    freqs = freqs[:up_to]
    words = [x[0] for x in freqs]
    return dict(zip(words, range(up_to))), [x[1]/all_freq for x in freqs],freqs


def grab_text(path, filename, vocab, oov_default, dtype, level):
    filename = os.path.join(path, filename)
    fd = open(filename, 'rt')
    txt = fd.read()
    if level == 'words':
        txt = txt.replace('\n', ' \n ')
        txt = txt.replace('  ', ' ')
        txt = txt.split(' ')
        txt = [x for x in txt if x != '']
        return numpy.asarray(
            [vocab.get(w, oov_default) for w in txt],
            dtype=dtype)
    else:
        return numpy.array(
            [vocab.get(w, oov_default) for w in txt],
            dtype=dtype)


def main(parser):
    o = parser.parse_args()
    dataset = o.path
    print 'Constructing the vocabulary ..'
    vocab, freqs, freq_wd = construct_vocabulary(dataset, o.oov_rate, o.level)
    vocab['<unk>'] = numpy.max(list(vocab.values()))+1
    
    oov_default = vocab["<unk>"]
    print "EOL", vocab["\n"]
    print 'Constructing train set'
    train = grab_text(dataset, 'train', vocab, oov_default, o.dtype, o.level)
    print 'Constructing valid set'
    valid = grab_text(dataset, 'valid', vocab, oov_default, o.dtype, o.level)
    print 'Constructing test set'
    test = grab_text(dataset, 'test', vocab, oov_default, o.dtype, o.level)
    print 'Saving data'

    if o.level == 'words':
        data = {'train_words': train, 'valid_words': valid, 'test_words': test, 'n_words': len(vocab)}
    else:
        data = {'train_chars': train, 'valid_chars': valid, 'test_chars': test, 'n_chars': len(vocab)}
    keys = {'oov': oov_default, 'freqs': numpy.array(freqs), 'vocabulary': vocab, 'freq_wd': freq_wd}
    all_keys = dict(keys.items() + data.items())
    
    numpy.savez(o.dest, **all_keys)
    inv_map = [None] * len(vocab.items())
    for k, v in vocab.items():
        inv_map[v] = k

    if o.level == 'words':
        numpy.savez(o.dest+"_dict", unique_words=inv_map)
    else:
        numpy.savez(o.dest+"_dict", unique_chars=inv_map)
    print '... Done'


def get_parser():
    usage = """
This script generates more numpy friendly format of the dataset from a text
file.  The script will save the entire file into a numpy .npz file. The file
will contain the following fields:

    'train' : array/matrix where each element (word or letter) is
              represented by an index from 0 to vocabulary size or the
              oov value (out of vocabulary). It is the training data.
    'test' : array where each element (word or letter) is represented by an
             index from 0 to vocabulary size or the oov value. This is the
             test value.
    'valid' : array where each element (word or letter) is represented by an
             index from 0 to vocabulary size or the oov value. This is the
             validation set.
    'oov' : The value representing the out of vocabulary word
    'vocab_size' : The size of the vocabulary (this number does not account
                   for oov

FIXME: The current script supports generating a .npz file with either character
sequences or word sequences only.
    """
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('path', 
            default="ntst",
            help=('path to the dataset files: you should have {path}/train, {path}/test and {path}/valid'))
    parser.add_argument('--dest',
                      help=('Where to save the processed dataset (i.e. '
                            'under what name and at what path). It will generate {dest}.npz and {dest}_dict.npz'),
                      default='tmp_data')
    parser.add_argument('--level',
                      help=('Processing level. Either `words` or `chars`. '
                            'If set to word, the result dataset has one '
                            'token per word, otherwise a token per letter'),
                      default='words')
    """
    parser.add_argument('--n-chains',
                      type=int,
                      help=('Number of parallel chains for the training '
                            'data. The way it works, is that it takes the '
                            'training set and divides it in `n_chains` that '
                            'should be processed in parallel by your model'),
                      default=1)
    """
    parser.add_argument('--oov-rate',
                      type=float,
                      help=('Defines dictionary size. If for example '
                            'oov_rate is set to 0.01 (meaning 10%) it means '
                            'that we can shrink our dictionary such that '
                            'remaining unrepresented words of the **train** '
                            'set is less then 10%. If set to 0, all words in '
                            'the training set will be added to the '
                            'dictionary'),
                      default=0.)
    parser.add_argument('--dtype',
                      help='dtype in which to store data',
                      default='int32')
    return parser

if __name__ == '__main__':
    main(get_parser())
