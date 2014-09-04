#!/usr/bin/env python
# coding=<utf-8>
from __future__ import print_function

import os
import sys
import argparse
import cPickle
import traceback
import logging
import numpy
import time
from itertools import izip, product
from experiments.nmt import RNNEncoderDecoder, prototype_state, parse_input, create_padded_batch
from experiments.nmt.sample import BeamSearch
from experiments.nmt.sample import sample as sample_func
from collections import defaultdict
import operator

cache = dict()

def cached_sample_func(model, phrase, n_samples, sampler, beam_search):
    global cache
    k = (tuple(phrase), n_samples)
    if not k in cache:
        cache[k] = sample_func(model, phrase, n_samples, sampler, beam_search)
    return cache[k]

logger = logging.getLogger(__name__)

def get_models():
    args = parse_args()

    state_en2fr = prototype_state()
    if hasattr(args, 'state_en2fr'):
        with open(args.state_en2fr) as src:
            state_en2fr.update(cPickle.load(src))
    state_en2fr.update(eval("dict({})".format(args.changes)))

    state_fr2en = prototype_state()
    if hasattr(args, 'state_fr2en') and args.state_fr2en is not None:
        with open(args.state_fr2en) as src:
            state_fr2en.update(cPickle.load(src))
    state_fr2en.update(eval("dict({})".format(args.changes)))

    rng = numpy.random.RandomState(state_en2fr['seed'])
    enc_dec_en_2_fr = RNNEncoderDecoder(state_en2fr, rng, skip_init=True)
    enc_dec_en_2_fr.build()
    lm_model_en_2_fr = enc_dec_en_2_fr.create_lm_model()
    lm_model_en_2_fr.load(args.model_path_en2fr)
    indx_word_src = cPickle.load(open(state_en2fr['word_indx'],'rb'))
    indx_word_trgt = cPickle.load(open(state_en2fr['word_indx_trgt'], 'rb'))

    if hasattr(args, 'state_fr2en') and args.state_fr2en is not None:
        rng = numpy.random.RandomState(state_fr2en['seed'])
        enc_dec_fr_2_en = RNNEncoderDecoder(state_fr2en, rng, skip_init=True)
        enc_dec_fr_2_en.build()
        lm_model_fr_2_en = enc_dec_fr_2_en.create_lm_model()
        lm_model_fr_2_en.load(args.model_path_fr2en)

        return [lm_model_en_2_fr, enc_dec_en_2_fr, indx_word_src, indx_word_trgt, state_en2fr, \
            lm_model_fr_2_en, enc_dec_fr_2_en, state_fr2en]
    else:
        return [lm_model_en_2_fr, enc_dec_en_2_fr, indx_word_src, indx_word_trgt, state_en2fr,\
                None, None, None]

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="res", help="Base file name")
    parser.add_argument("--source", help="Source sentences")
    parser.add_argument("--from-scratch", action="store_true", default=False,
        help="Start from scratch")
    parser.add_argument("--n-samples", type=int, default=10,
        help="Beam size")
    parser.add_argument("--state_en2fr", help="State to use: en to fr")
    parser.add_argument("--model_path_en2fr", help="Path to the model: en to fr")
    parser.add_argument("--state_fr2en", help="State to use : fr to en")
    parser.add_argument("--model_path_fr2en", help="Path to the model : fr to en")
    parser.add_argument("--copy_UNK_words", help="segment around UNK words and copy them",
                        action='store_true')
    parser.add_argument("--normalize", help="normalize score for phrase length",
                        action='store_true')
    parser.add_argument("--reverse_score", help="use a fr2en model to add to the score",
                        action='store_true')
    parser.add_argument("--add_period", help="Add a period at the end of each phrase",
                        action='store_true')
    parser.add_argument("--changes",  nargs="?", help="Changes to state", default="")
    parser.add_argument("--old_begin", type=int, default=0, help="first line to start translating")
    parser.add_argument("--end", type=int, default=10, help="last line to translate (included)")
    return parser.parse_args()


def process_sentence(source_sentence, model, max_phrase_length, n_samples,
                     copy_UNK_words, add_period, normalize, reverse_score):

    #Setting up comp_score function
    logger.debug("setting up comp_score function")
    [lm_model, enc_dec, indx_word_src, indx_word_trgt, state, \
            lm_model_fr_2_en, enc_dec_fr_2_en, state_fr2en] = model

    eol_src = state['null_sym_source']
    src_seq = parse_input(state, indx_word_src, source_sentence)
    if src_seq[-1] == eol_src:
        src_seq = src_seq[:-1]
    n_s = len(src_seq)

    #Create sorted phrase lists
    tiled_source_phrase_list = []
    index_order_list = []
    for i in xrange(n_s):
        for j in numpy.arange(i, min(i+max_phrase_length, n_s)):
                    index_order_list.append([i, j])

    logger.debug("sorting list")
    index_order_list.sort(key=lambda (i, j): (j - i))

    logger.debug("creating phrase lists")
    if add_period:
        period_src = indx_word_src['.']
        for i, j in index_order_list:
            tiled_source_phrase_list.append(numpy.hstack((src_seq[i:j+1], period_src, eol_src)))
    else:
        for i, j in index_order_list:
            tiled_source_phrase_list.append(numpy.hstack((src_seq[i:j+1], eol_src)))


    #Compute nested score dictionary
    logger.debug("computing nested score dictionary")
    score_dict = {}
    trans = {}

    for phrase_idx in xrange(0, len(index_order_list)):
        logger.debug("{0} out of {1}".format(phrase_idx, len(index_order_list)))
        i, j = index_order_list[phrase_idx]
        logger.debug("Translating phrase : {}".format(" ".join(source_sentence.strip().split()[i:j+1])))

        if copy_UNK_words == True:
            phrase_to_translate = tiled_source_phrase_list[phrase_idx]
            n_UNK_words = numpy.sum([word == 1 for word in phrase_to_translate])
            if n_UNK_words >= 1 and n_UNK_words == len(phrase_to_translate) - 1:
                suggested_translation = " ".join(source_sentence.strip().split()[i:j+1])
                trans[i, j] = suggested_translation
                score = .0001
                score_dict[i, j] = score
            if n_UNK_words >= 1 and n_UNK_words != len(phrase_to_translate) -1:
                suggested_translation = "WILL NOT BE USED"
                trans[i, j] = suggested_translation
                score = 1e9
                score_dict[i, j] = score
            if n_UNK_words == 0:
                suggested_translation, score = sample_targets(
                                                    input_phrase= \
                                                           tiled_source_phrase_list[phrase_idx],
                                                    model=model,
                                                    n_samples=n_samples,
                                                    reverse_score=reverse_score,
                                                    normalize=normalize
                )
                trans[i, j] = suggested_translation
                score_dict[i, j] = score

        else:
            phrase_to_translate = tiled_source_phrase_list[phrase_idx]
            suggested_translation, score = sample_targets(
                                                input_phrase=phrase_to_translate,
                                                model=model, n_samples=n_samples,
                                                reverse_score=reverse_score,
                                                normalize=normalize
            )
            trans[i, j] = suggested_translation
            score_dict[i, j] = score

    #Remove the period at the end if not last word
    #Lower case first word if not first word
    if add_period:
       for phrase_idx in xrange(0, len(index_order_list)):
           i, j = index_order_list[phrase_idx]
           if i != 0:
               trans[i, j] = " ".join([trans[i,j][0].lower()] + [trans[i,j][1:]])
           if j != len(src_seq) - 1:
               last_word = trans[i,j].strip().split()[-1]
               if last_word == '.':
                   trans[i,j] = " ".join(trans[i,j].strip().split()[:-1])


    #Translation of full sentence without segmentation
    logger.debug("Translating full sentence")
    phrase_to_translate = numpy.hstack((src_seq, eol_src))
    full_translation, __ = sample_targets(input_phrase=phrase_to_translate,
                                          model=model,
                                          n_samples=n_samples,
                                          reverse_score=reverse_score,
                                          normalize=normalize
    )
    logger.debug("Translation output:".format(full_translation))

    return trans, score_dict, full_translation


def sample_targets(input_phrase, model, n_samples, reverse_score, normalize):

    [lm_model, enc_dec, indx_word_src, indx_word_trgt, state, \
            lm_model_fr_2_en, enc_dec_fr_2_en, state_fr2en] = model

    beam_search = BeamSearch(enc_dec)
    beam_search.compile()
    sampler = enc_dec.create_sampler(many_samples=True)

    #sample_func can take argument : normalize (bool)
    trans, scores, trans_bin = cached_sample_func(lm_model, input_phrase, n_samples,
                                           sampler=sampler, beam_search=beam_search)

    #Reordering scores-trans
    #Warning : selection of phrases to rescore is hard-coded
    trans = [tra for (sco, tra) in sorted(zip(scores, trans))][0:10]
    trans_bin = [tra_bin for (sco, tra_bin) in sorted(zip(scores, trans_bin))][0:10]
    scores = sorted(scores)[0:10]

    #Reverse scoring of selected phrases
    if reverse_score:
        reverse_scorer = enc_dec_fr_2_en.create_scorer(batch=True)

        source_phrases_to_reverse_score = []
        target_phrases_to_reverse_score = []
        for tra_bin in trans_bin:
            source_phrases_to_reverse_score.append(input_phrase)
            target_phrases_to_reverse_score.append(tra_bin)

        state_fr2en['seqlen'] = 1000
        x, x_mask, y, y_mask = create_padded_batch(
                                    state_fr2en,
                                    [numpy.asarray(target_phrases_to_reverse_score)],
                                    [numpy.asarray(source_phrases_to_reverse_score)])

        reverse_scores = - reverse_scorer(numpy.atleast_2d(x), numpy.atleast_2d(y),
                                          numpy.atleast_2d(x_mask),
                                          numpy.atleast_2d(y_mask))[0]

        for index in xrange(len(scores)):
            scores[index] = (scores[index] + reverse_scores[index]) / 2.

    else:
        for index in xrange(len(scores)):
            scores[index] = scores[index]

    trans = trans[numpy.argmin(scores)]
    score = numpy.min(scores)

    if normalize == False:
        final_score = score
    else:
        final_score = score / numpy.log(len(input_phrase) + 1)

    return trans, final_score


def find_align(source, model, max_phrase_length, n_samples,
        f_trans, f_total,
        normalize=False, copy_UNK_words=False,
        add_period=False, reverse_score=False):

    [lm_model, enc_dec, indx_word_src, indx_word_trgt, state, \
            lm_model_fr_2_en, enc_dec_fr_2_en, state_fr2en] = model

    split_source_sentence = source.strip().split()
    n_s = len(split_source_sentence)

    #Sampling and computing scores : bottleneck
    phrases, scores, full_translation = process_sentence(source_sentence=source, model=model,
                                            max_phrase_length=max_phrase_length,
                                            n_samples=n_samples,
                                            normalize=normalize,
                                            copy_UNK_words=copy_UNK_words,
                                            add_period=add_period,
                                            reverse_score=reverse_score)

    #Starting segmentation
    logger.debug("starting segmentation")

    prefix_score = 1e9 * numpy.ones(n_s+1)
    prefix_score[0] = 0
    best_choice = -1 * numpy.ones(n_s+1, dtype="int64")
    best_choice[0] = 0

    for i in xrange(1, n_s+1):
        for j in xrange(max(0, i-max_phrase_length), i):
            cand_cost = prefix_score[j] + scores[j, i-1]
            if cand_cost < prefix_score[i]:
                prefix_score[i] = cand_cost
                best_choice[i] = j

    phrase_ends = []
    i = n_s
    while i > 0:
        phrase_ends.append(i)
        i = best_choice[i]
    phrase_ends = list(reversed(phrase_ends))

    segmentation = []
    past_index = 0
    for i in phrase_ends:
       segmentation.append([past_index,i-1])
       past_index = i

    logger.debug("end of segmentation")

    T = []
    segmented_target = []
    segmented_source = []
    for segment in segmentation:
            [i, j] = segment
            value = phrases[i, j]
            T.append(value)
            segmented_target.append([phrases[i,j]])
            segmented_source.append([split_source_sentence[k] for k in xrange(i, j+1)])

    print("Input sentence: ", source, file=f_total)
    print("Segmentation: ", segmented_source, file=f_total)
    print("Translation with segmentation: {}".format(" ".join(T)), file=f_total)
    print("Translation with segmentation: {}".format(segmented_target), file=f_total)
    print("Translation without segmentation: {}".format(full_translation), file=f_total)
    print(" ".join(T), file=f_trans)
    f_total.flush()
    f_trans.flush()

def main_with_segmentation(begin, end, n_samples,
        source, console,
        options, outputs):
    model = get_models()
    max_phrase_length = 20

    s_text = []

    logger.debug("begin: {}, end: {}".format(begin, end))
    for i, line in enumerate(source):
        if i> end:
            break
        if i < begin:
            pass
        else:
            s_text.append(line)
    logger.debug("Read {} sentences".format(len(s_text)))

    t0 = time.time()
    old_t1 = time.time()
    counter_processed = 0
    counter_total = 0

    logger.debug("Translating with beam size {}".format(n_samples))
    for source in s_text:
        global cache
        cache = dict()

        for opts, (f_trans, f_total) in zip(options, outputs):
            find_align(source, model, max_phrase_length, n_samples,
                    f_trans, f_total,
                    **opts)

        counter_total += 1
        counter_processed += 1
        t1 = time.time()
        logger.debug("total time last sentence : {}".format(t1 - old_t1))
        old_t1 = t1
        logger.debug("total time : {}".format(t1 - t0))
        logger.debug("sentence processed : {}".format(counter_total + begin))
        logger.debug("total sentences processed : {}".format(counter_processed))
        print(counter_total + begin - 1, file=console)
        console.flush()


def main():
    logging.basicConfig(level=logging.DEBUG,
            format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    args = parse_args()
    assert args.state_en2fr
    assert args.model_path_en2fr
    assert args.state_fr2en
    assert args.model_path_fr2en
    assert args.source
    n_samples = args.n_samples
    base_file_name = args.base
    old_begin = args.old_begin
    end = args.end

    base_file_name += "_{}_{}".format(old_begin, end)

    console_file_name = base_file_name + "_cons.txt"
    just_translation_file_name = base_file_name + "_trans_{}.txt"
    total_file_name = base_file_name + "_total_{}.txt"

    # Decide on the start sentence
    if os.path.isfile(console_file_name):
        with open(console_file_name) as f:
            line = None
            for line in f:
                pass
            if line is not None and not args.from_scratch:
                last_sentence_number = int(line.strip())
                if last_sentence_number >= end:
                    sys.exit("job finished, end is {0}, begin is {0}".format(
                             end, last_sentence_number))
                else:
                    begin = last_sentence_number + 1
            else:
                begin = old_begin
    else:
        begin = old_begin

    source = open(args.source, 'r')
    console = open(console_file_name, "a")

    modes = ['default', 'rev', 'norm', 'normrev']
    options = [dict(),
            dict(reverse_score=True),
            dict(normalize=True),
            dict(normalize=True, reverse_score=True)]
    outputs = [(open(just_translation_file_name.format(k), 'a'),
            open(total_file_name.format(k), 'a'))
        for k in modes]

    main_with_segmentation(begin, end, n_samples,
            source, console,
            options, outputs)

if __name__ == "__main__":
    main()
