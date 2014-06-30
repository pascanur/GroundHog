#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging

import numpy

from experiments.rnnencdec import RNNEncoderDecoder, prototype_state, parse_input

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("changes",  nargs="?", help="Changes to state", default="")
    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_state()
    if hasattr(args, 'state'):
        with open(args.state) as src:
            state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = cPickle.load(open(state['word_indx'],'rb'))

    while True:
        try:
            seqin = raw_input('Input Sequence: ')
            n_samples = int(raw_input('How many samples? '))
            alpha = float(raw_input('Inverse Temperature? '))
            seq = parse_input(state, indx_word, seqin)
        except Exception:
            print "Exception while parsing your input:"
            traceback.print_exc()
            continue
        sentences = []
        all_probs = []
        sum_log_probs = []
        for sidx in xrange(n_samples):
            logger.debug("Sample {}".format(sidx))
            [values, probs] = lm_model.sample_fn(3 * len(seq) - 3, alpha, seq)
            sen = []
            for k in xrange(values.shape[0]):
                if lm_model.word_indxs[values[k]] == '<eol>':
                    break
                sen.append(lm_model.word_indxs[values[k]])
            sentences.append(" ".join(sen))
            if not state['sample_all_probs']:
                probs = numpy.array(probs[:len(sen) + 1])
            all_probs.append(numpy.exp(-probs))
            sum_log_probs.append(-numpy.sum(probs))
        sprobs = numpy.argsort(sum_log_probs)
        for pidx in sprobs:
            print "{}: {} {} {}".format(pidx, -sum_log_probs[pidx], all_probs[pidx], sentences[pidx])
        print

if __name__ == "__main__":
    main()
