# coding=<utf-8>
#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import numpy

from experiments.rnnencdec import RNNEncoderDecoder, prototype_state, parse_input

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("changes",  nargs="?", help="Changes to state", default="")
    return parser.parse_args()


def comp_scores(s_word, t_word, model):
    [lm_model, enc_dec, indx_word_src, indx_word_trgt, state] = model

    eol_src = state['null_sym_source']
    eol_trgt = state['null_sym_target']
    scorer = enc_dec.create_scorer(batch=False)

    src_seq = parse_input(state, indx_word_src, s_word, raise_unk=True)

    if src_seq[-1] == eol_src:
        src_seq = src_seq[:-1]
    trgt_seq = parse_input(state, indx_word_trgt, t_word, raise_unk=True)

    if trgt_seq[-1] == eol_trgt:
        trgt_seq = trgt_seq[:-1]

    n_s = len(src_seq)
    n_t = len(trgt_seq)

    scores = 1e9 * numpy.ones((n_t, n_t))
    segment = {}

    idict_fr = cPickle.load(open(state['indx_word_target'],'r'))
    idict_en = cPickle.load(open(state['indx_word'],'r'))
    idict_fr[eol_trgt] = '<eol>'
    idict_en[eol_src] = '<eol>'

    for i in xrange(n_t):
         for j in numpy.arange(i,n_t):
            for k in xrange(n_s):
                for l in numpy.arange(k, n_s):     
                   src_p = numpy.hstack((src_seq[k:l+1], numpy.asarray(eol_src)))
                   trgt_p = numpy.hstack((trgt_seq[i:j+1], numpy.asarray(eol_trgt)))
                   can_score = scorer(src_p, trgt_p)[0]
                   if can_score < scores[i, j]:
                       scores[i, j] = can_score
                       segment[i, j] = [k, l]

    for key, v in segment.iteritems():
        i, j = key
        k, l = v 
        print  [idict_en[q] for q in src_seq[k:l+1]],  [idict_fr[q] for q in trgt_seq[i:j+1]], scores[i,j] 

    return scores, segment


def find_align(source, target, model, max_phrase_length):
    [lm_model, enc_dec, indx_word_src, indx_word_trgt, state] = model
    s_word = source.strip().split()
    t_word = target.strip().split()
    scores, phrases = comp_scores(source, target, model)
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
       #segmentation.append([t_word[past_index:i]]) 
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
     source = u"Paris is the capital of France ."
     target = u"la capitale de la France est Paris ."
     #comp_scores(source, target, model)
     find_align(source, target, model, 7)
