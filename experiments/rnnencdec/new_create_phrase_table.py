#!/usr/bin/env python
# coding=<utf-8>

import sys
import argparse
import cPickle
import traceback
import logging
import numpy
from itertools import izip, product
from experiments.rnnencdec import RNNEncoderDecoder, prototype_state, parse_input, create_padded_batch
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

    logging.basicConfig(level=logging.ERROR, 
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


def comp_scores(source_sentence, target_sentence, model, max_phrase_length, batch_size, without_segmentation=False):

    #Setting up comp_score function
    logger.debug("setting up comp_score function")
    [lm_model, enc_dec, indx_word_src, indx_word_trgt, state] = model

    eol_src = state['null_sym_source']
    eol_trgt = state['null_sym_target']
    scorer = enc_dec.create_scorer(batch=True)

    src_seq = parse_input(state, indx_word_src, source_sentence)
    if src_seq[-1] == eol_src:
        src_seq = src_seq[:-1]

    trgt_seq = parse_input(state, indx_word_trgt, target_sentence)
    if trgt_seq[-1] == eol_trgt:
        trgt_seq = trgt_seq[:-1]

    n_s = len(src_seq)
    n_t = len(trgt_seq)
   
    #Create phrase lists
    tiled_target_phrase_list = []
    tiled_source_phrase_list = []
    index_order_list = []
    for i in xrange(n_t):
        for j in numpy.arange(i, min(i+max_phrase_length, n_t)):
            for k in xrange(n_s):
                for l in numpy.arange(k, min(k+max_phrase_length, n_s)):
                    index_order_list.append([i, j, k, l])

    logger.debug("sorting list")
    index_order_list.sort(key=lambda (i, j, k, l): (j - i, l - k))
    
    logger.debug("creating phrase lists")
    for i, j, k, l in index_order_list:
        tiled_target_phrase_list.append(trgt_seq[i:j+1])
        tiled_source_phrase_list.append(src_seq[k:l+1])
    
    score_index = numpy.asarray(index_order_list)
    
    #Create paddded arrays
    #logger.debug("creating padded arrays")

    
    
    #Compute nested score dictionary
    logger.debug("computing nested score dictionary")
    
    print >>sys.stderr, "scoring ", len(index_order_list), " phrases"
    #print >>sys.stderr, "length src_phrase", len(src_phrase)
    #print >>sys.stderr, "length trgt_phrase", len(trgt_phrase)


    score_dict = defaultdict(dict)

    for batch_idx in xrange(0, len(score_index), batch_size):
        source_phrase, source_mask, target_phrase, target_mask = create_padded_batch(state, 
                                       [numpy.asarray([tiled_source_phrase_list[i] for i in xrange(batch_idx, min(batch_idx + batch_size, len(index_order_list)))])],
                                       [numpy.asarray([tiled_target_phrase_list[i] for i in xrange(batch_idx, min(batch_idx + batch_size, len(index_order_list)))])])
        print "len(source_phrase), print len(target_phrase)", len(source_phrase), len(target_phrase)
        logger.debug("scoring batch number {}".format(batch_idx)) 
        #source_phrase = src_phrase[:,batch_idx:batch_idx + batch_size]
        #target_phrase = trgt_phrase[:,batch_idx:batch_idx + batch_size]
        #source_mask = src_mask[:,batch_idx:batch_idx + batch_size]
        #target_mask = trgt_mask[:,batch_idx:batch_idx + batch_size]
        can_scores =  - scorer(source_phrase, target_phrase, source_mask, target_mask)[0]

        for idx_batch, idx_pair in enumerate(numpy.arange(batch_idx, 
                                                          min(batch_idx+batch_size,len(score_index)))):
            i, j, k, l = score_index[idx_pair]
            score_dict[i, j][k, l] = can_scores[idx_batch]

    #Compute best scores
    logger.debug("finding best scores in dictionary")

    scores = 1e9 * numpy.ones((n_t, n_t))
    segment = {}

    for key in score_dict:
        segment[key], scores[key] = min(score_dict[key].iteritems(), key=operator.itemgetter(1)) 

    #Print stuff
    if without_segmentation:
        S = source_sentence.strip().split()
        T = target_sentence.strip().split()
        for key, v in segment.iteritems():
            i, j = key
            k, l = v 
            print " ".join(S[k:l+1]), " ||| ", " ".join(T[i:j+1]), " ||| ", scores[i, j]

    return scores, segment


def find_align(source, target, model, max_phrase_length, batch_size):
    [lm_model, enc_dec, indx_word_src, indx_word_trgt, state] = model
    split_source_sentence = source.strip().split()
    split_target_sentence = target.strip().split()
    scores, phrases = comp_scores(source, target, model, max_phrase_length, batch_size)

    logger.debug("starting segmentation")

    n_s = len(split_source_sentence)
    n_t = len(split_target_sentence)

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

    logger.debug("end of segmentation, beginning print")

    for segment in segmentation:
            [i, j] = segment
            value = phrases[i, j]
            [k, l] = value
            S = split_source_sentence[k:l+1]
            T = split_target_sentence[i:j+1]
            print " ".join(S), " |||  ", " ".join(T), "||| ", scores[i, j]
            phrase_table.append([S, T])

    return phrase_table

def main_with_segmentation():
    model = get_model()
    max_text_len = 1000
    batch_size = 1024
    max_phrase_length = 5
    s_text = []
    t_text = []
    phrase_table = []
    logger.debug("selecting sentences from text")
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
    import time
    t0 = time.time()
    counter = 0
    for source, target in izip(s_text, t_text):
        phrase_table.append(find_align(source, target, model, max_phrase_length, batch_size))
        counter += 1
        t1 = time.time()
        print >>sys.stderr, "total time : ", t1 - t0
        print >>sys.stderr, "total sentences processed : ", counter
        print >>sys.stderr, "current size of phrase table : ", len(phrase_table)


def main_without_segmentation():
    model = get_model()
    max_text_len = 1000
    batch_size = 512 #1024
    max_phrase_length = 5
    s_text = []
    t_text = []
    phrase_table = []
    logger.debug("selecting sentences from text")
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
        if len(source.strip().split()) + len(target.strip().split()) <= 80:
            phrase_table.append(comp_scores(source, target, model, max_phrase_length, batch_size, True))


if __name__ == "__main__":
    main_with_segmentation()

