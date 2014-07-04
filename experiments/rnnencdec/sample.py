#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
from pprint import pprint

import numpy

import experiments.rnnencdec
from experiments.rnnencdec import RNNEncoderDecoder, parse_input

logger = logging.getLogger(__name__)

class BeamSearch(object):

    def __init__(self, enc_dec):
        self.enc_dec = enc_dec

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, seq, n_samples):
        c = self.comp_repr(seq)[0]
        states = map(lambda x : x[None, :], self.comp_init_states(c))

        num_levels = len(states)

        fin_trans = []
        fin_costs = []

        trans = [[]]
        costs = [0.0]

        for k in range(3 * len(seq)):
            if n_samples == 0:
                break
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t : t[-1], trans))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))
            probs = self.comp_next_probs(c, last_words, *states)[0]

            n_choices = beam_size * n_samples
            choices = numpy.zeros((n_choices, 2), dtype="int64")
            choice_costs = numpy.zeros(n_choices)

            for i in range(beam_size):
                best_words = numpy.argsort(probs[i])[-n_samples:]
                rrange = slice(n_samples * i, n_samples * (i + 1))
                choices[rrange, 0] = i
                choices[rrange, 1] = best_words
                choice_costs[rrange] =\
                        costs[i] - numpy.log(probs[i, best_words])

            best_choices = numpy.argsort(choice_costs)[:n_samples]

            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_states = [numpy.zeros((n_samples, c.shape[0]), dtype="float32") for level
                    in range(num_levels)]
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, j in enumerate(best_choices):
                orig_idx = choices[j, 0]
                next_word = choices[j, 1]
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = choice_costs[j]
                for level in range(num_levels):
                    new_states[level][i] = states[level][orig_idx]
                inputs[i] = next_word
            new_states = self.comp_next_states(c, inputs, *new_states)

            trans = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
            states = map(lambda x : x[indices], new_states)

        return fin_trans, fin_costs

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--state-fn", help="Initialization function for state", default="prototype_state")
    parser.add_argument("--beam-search", help="Do beam search instead of sampling",
            action="store_true", default=False)
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("changes",  nargs="?", help="Changes to state", default="")
    return parser.parse_args()

def main():
    args = parse_args()

    state = getattr(experiments.rnnencdec, args.state_fn)()
    if hasattr(args, 'state') and args.state:
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

    if args.beam_search:
        beam_search = BeamSearch(enc_dec)
        beam_search.compile()
    else:
        sampler = enc_dec.create_sampler(many_samples=True)

    while True:
        try:
            seqin = raw_input('Input Sequence: ')
            n_samples = int(raw_input('How many samples? '))
            if not args.beam_search:
                alpha = float(raw_input('Inverse Temperature? '))
            seq = parse_input(state, indx_word, seqin)
        except Exception:
            print "Exception while parsing your input:"
            traceback.print_exc()
            continue

        if args.beam_search:
            trans, costs = beam_search.search(seq, n_samples)
            for i in numpy.argsort(costs):
                sen = indices_to_words(lm_model.word_indxs, trans[i])
                print "{}: {}".format(costs[i], " ".join(sen))
        else:
            sentences = []
            all_probs = []
            sum_log_probs = []

            values, cond_probs = sampler(n_samples, 3 * (len(seq) - 1), alpha, seq)
            for sidx in xrange(n_samples):
                sen = []
                for k in xrange(values.shape[0]):
                    if lm_model.word_indxs[values[k, sidx]] == '<eol>':
                        break
                    sen.append(lm_model.word_indxs[values[k, sidx]])
                sentences.append(" ".join(sen))
                probs = cond_probs[:, sidx]
                probs = numpy.array(cond_probs[:len(sen) + 1, sidx])
                all_probs.append(numpy.exp(-probs))
                sum_log_probs.append(-numpy.sum(probs))
            sprobs = numpy.argsort(sum_log_probs)
            for pidx in sprobs:
                print "{}: {} {} {}".format(pidx, -sum_log_probs[pidx], all_probs[pidx], sentences[pidx])
            print

if __name__ == "__main__":
    main()
