#!/usr/bin/env python
# coding=<utf-8>

import sys
import argparse
import cPickle
import traceback
import logging
import numpy
from itertools import izip, product
from experiments.rnnencdec import RNNEncoderDecoder, prototype_state, parse_input
from collections import defaultdict
import operator

logger = logging.getLogger(__name__)

def get_model():
    args = parse_args()

    state = prototype_state()
    if hasattr(args, 'state'):
        with open(args.state) as src:
            state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']), 
            format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word_src = cPickle.load(open(state['word_indx'],'rb'))
    indx_word_trgt = cPickle.load(open(state['word_indx_trgt'], 'rb'))
    return [lm_model, enc_dec, indx_word_src, indx_word_trgt, state]


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


class Sentence(object):
    def __init__(string, dictionary, state, indx_word):  
        self.string_rep = string 
        self.state = state 
        self.indx_word = indx_word
        self.list_rep = string.strip().lower().split() 
        self.dicti = dictionary
        self.binarized_rep = parse_input(state, indx_word, string) 
        self.n_word = len(self.list_rep)
        #self.phrases =  


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("changes",  nargs="?", help="Changes to state", default="")
    return parser.parse_args()


def comp_scores(s_word, t_word, model, max_phrase_length, batch_size):
    [lm_model, enc_dec, indx_word_src, indx_word_trgt, state] = model

    eol_src = state['null_sym_source']
    eol_trgt = state['null_sym_target']
    scorer = enc_dec.create_scorer(batch=True)

    src_seq = parse_input(state, indx_word_src, s_word)
    if src_seq[-1] == eol_src:
        src_seq = src_seq[:-1]

    trgt_seq = parse_input(state, indx_word_trgt, t_word)
    if trgt_seq[-1] == eol_trgt:
        trgt_seq = trgt_seq[:-1]

    n_s = len(src_seq)
    n_t = len(trgt_seq)

    #Load dictionaries
    idict_fr = cPickle.load(open(state['indx_word_target'],'r'))
    idict_en = cPickle.load(open(state['indx_word'],'r'))
    idict_fr[eol_trgt] = '<eol>'
    idict_en[eol_src] = '<eol>'
    
    #Create phrase list: .001s
    trgt_p_i = []
    src_p_i = []
    for i in xrange(n_t):
    #### numpy.arange(i, n_t)
        for j in numpy.arange(i, n_t):
            trgt_p_i.append([numpy.hstack((trgt_seq[i:j+1], numpy.asarray(eol_trgt))), [i,j]])
    for k in xrange(n_s):
        for l in numpy.arange(k, min(k+max_phrase_length, n_s)):
            src_p_i.append([numpy.hstack((src_seq[k:l+1], numpy.asarray(eol_src))),[k,l]])

    #Create paddded arrays: .13 s
    m = max([len(a[0]) for a in src_p_i])
    M = max([len(a[0]) for a in trgt_p_i])
    
    src_phrase = numpy.empty((m,))
    src_mask = numpy.empty((m,))
    trgt_phrase = numpy.empty((M,))
    trgt_mask = numpy.empty((M,))

    for trgt in trgt_p_i:
       for src in chunks(src_p_i, batch_size): #Fix this stupid recursion where trgt is in the inner loop :/
           i, j = trgt[1]
           padded_trgt_phrase = numpy.pad(numpy.tile(trgt[0], (len(src), 1)), ((0, 0),(0, M-len(trgt[0]))),
                                          mode="constant", constant_values=(0, eol_trgt))
           trgt_phrase = numpy.vstack((trgt_phrase, padded_trgt_phrase))
           padded_trgt_mask = numpy.pad(numpy.ones((len(src), len(trgt[0])), dtype="float32"), \
                                        ((0, 0),(0, M - len(trgt[0]))), mode="constant", constant_values=(0, 0))
           trgt_mask = numpy.vstack((trgt_mask, padded_trgt_mask))

           for p in src:
               padded_p = numpy.pad(p[0], (0, m-len(p[0])), mode='constant', \
                                    constant_values=(0, eol_src)).astype("int64")
               src_phrase = numpy.vstack((src_phrase, padded_p)) 
               padded_mask = numpy.pad(numpy.ones(len(p[0]), dtype="float32"), (0, m-len(p[0])), \
                                    mode="constant", constant_values=(0, 0)).astype("int64")
               src_mask = numpy.vstack((src_mask, padded_mask))

    src_phrase = src_phrase[1:].astype("int64")
    src_mask = src_mask[1:].astype("float32")
    trgt_phrase = trgt_phrase[1:].astype("int64")
    trgt_mask = trgt_mask[1:].astype("float32")

    #Compute score_index table: .001 s
    score_index = []
    for trgt_element in trgt_p_i:
        for src_element in src_p_i:
            i, j = trgt_element[1]
            k, l = src_element[1]
            score_index.append([i, j, k, l])
    score_index = numpy.asarray(score_index)


    #Compute nested score dictionary
    scores = 1e9 * numpy.ones((n_t, n_t))
    segment = {}

    score_dict = defaultdict(dict)

    for batch_idx in xrange(0, len(score_index), batch_size):
        source_phrase = src_phrase[batch_idx:batch_idx + batch_size].T
        target_phrase = trgt_phrase[batch_idx:batch_idx + batch_size].T
        source_mask = src_mask[batch_idx:batch_idx + batch_size].T
        target_mask = trgt_mask[batch_idx:batch_idx + batch_size].T
        can_scores = scorer(source_phrase, target_phrase, source_mask, target_mask)[0]

        for idx_batch, idx_pair in enumerate(numpy.arange(batch_idx, 
                                                          min(batch_idx+batch_size,len(score_index)))):
            i, j, k, l = score_index[idx_pair]
            score_dict[i, j][k, l] = can_scores[idx_batch]
     
    #Compute best scores
    for key in score_dict:
        segment[key], scores[key] = min(score_dict[key].iteritems(), key=operator.itemgetter(1)) 
           
    print segment, scores

    #Print stuff
    for key, v in segment.iteritems():
        i, j = key
        k, l = v 
        print  [idict_en[q] for q in src_seq[k:l+1]],  [idict_fr[q] for q in trgt_seq[i:j+1]], scores[i,j] 

    return scores, segment


def find_align(source, target, model, max_phrase_length, batch_size):
    [lm_model, enc_dec, indx_word_src, indx_word_trgt, state] = model
    s_word = source.strip().split()
    t_word = target.strip().split()
    scores, phrases = comp_scores(source, target, model, max_phrase_length, batch_size)
    n_s = len(s_word)
    n_t = len(t_word)

    prefix_score = 1e9 * numpy.ones(n_t+1)
    prefix_score[0] = 0
    best_choice = -1 * numpy.ones(n_t+1, dtype="int64")
    best_choice[0] = 0

    for i in xrange(1, n_t+1):
        for j in xrange(max(0, i-max_phrase_length), i): 
            cand_cost = prefix_score[j] + scores[j, i-1]  
            if cand_cost < prefix_score[i]:
                prefix_score[i] = cand_cost
                best_choice[i] = j
    
    phrase_ends = []
    i = n_t
    while i > 0:
        phrase_ends.append(i)
        i = best_choice[i]
    phrase_ends = list(reversed(phrase_ends))
    
    segmentation = []
    past_index = 0
    for i in phrase_ends:
       segmentation.append([past_index,i-1])
       past_index = i

    phrase_table = []

    for segment in segmentation:
            [i, j] = segment
            value = phrases[i, j]
            [k, l] = value
            S = s_word[k:l+1]
            T = t_word[i:j+1]
            print " ".join(S)
            print " ".join(T)
            phrase_table.append([S, T])

    return phrase_table

if 1:
    model = get_model()
    max_text_len = 5
    batch_size =  1024
    max_phrase_length = 5
    s_text = []
    t_text = []
    phrase_table = []
    with open("/data/lisatmp3/pougetj/dev08_11.en") as f:
        for i, line in enumerate(f):
            if i>= max_text_len:
                break
            s_text.append(line)
    with open("/data/lisatmp3/pougetj/dev08_11.fr") as g:
        for i, line in enumerate(g):
            if i>= max_text_len:
                break
            t_text.append(line)
    for source, target in izip(s_text, t_text):
        phrase_table.append(find_align(source, target, model, max_phrase_length, batch_size))

    cPickle.dump(phrase_table, open("phrase_table.pkl", "wb"))
    print phrase_table

    print "TO DO : fix padding with two loops, make  max_phrase_len apply to both, nice output to text file including some of the better phrase pairs not included in the segmentation, include the reverse model to have that score as well, use cache..."
