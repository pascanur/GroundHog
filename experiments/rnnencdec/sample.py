#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging

import numpy

import experiments.rnnencdec
from experiments.rnnencdec import RNNEncoderDecoder, parse_input

logger = logging.getLogger(__name__)

class BeamSearch(object):

    def __init__(self, enc_dec):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['unk_sym_target']

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, seq, n_samples, ignore_unk=False, minlen=1):
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
            if ignore_unk:
                probs[:,self.unk_id] = -numpy.inf
            if k < minlen:
                probs[:,self.eos_id] = -numpy.inf

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

def sample(lm_model, seq, n_samples,
        sampler=None, beam_search=None,
        normalize=False, alpha=1, verbose=False):
    if beam_search:
        sentences = []
        trans, costs = beam_search.search(seq, n_samples,
                ignore_unk=True, minlen=len(seq) / 2)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in numpy.argsort(costs):
            sen = indices_to_words(lm_model.word_indxs, trans[i])
            sentences.append(" ".join(sen))
            if verbose:
                print "{}: {}".format(costs[i], sentences[-1])
    elif sampler:
        sentences = []
        all_probs = []
        costs = []

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
            costs.append(-numpy.sum(probs))
        if normalize:
            counts = [len(s.strip().split(" ")) for s in sentences]
            costs = [co / cn for co, cn in zip(costs, counts)]
        sprobs = numpy.argsort(costs)
        if verbose:
            for pidx in sprobs:
                print "{}: {} {} {}".format(pidx, -costs[pidx], all_probs[pidx], sentences[pidx])
            print
    else:
        raise Exception("I don't know what to do")

    return sentences, costs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--state-fn", help="Initialization function for state", default="prototype_state")
    parser.add_argument("--beam-search", help="Beam size, turns on beam-search", type=int)
    parser.add_argument("--source", help="File of source sentences", default="")
    parser.add_argument("--trans", help="File to save translations in", default="")
    parser.add_argument("--normalize", help="Normalize log-prob with the word count", action="store_true", default=False)
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
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = cPickle.load(open(state['word_indx'],'rb'))

    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_dec)
        beam_search.compile()
    else:
        sampler = enc_dec.create_sampler(many_samples=True)

    idict_src = cPickle.load(open(state['indx_word'],'r'))

    if args.source != "" and args.trans != "":
        # Actually only beam search is currently supported here
        assert beam_search

        fsrc = open(args.source, 'r')
        ftrans = open(args.trans, 'w')

        n_samples = args.beam_search
        total_cost = 0.0
        logging.debug("Beam size: {}".format(n_samples))
        for line in fsrc:
            seqin = line.strip()
            seq,parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
            print "Parsed Input:", parsed_in
            trans, costs = sample(lm_model, seq, n_samples, sampler=sampler,
                    beam_search=beam_search, normalize=args.normalize)
            best = numpy.argmin(costs)
            print >>ftrans, trans[best]
            print "Translation:", trans[best]
            total_cost += costs[best]
        print "Total cost of the translations: {}".format(total_cost)


        fsrc.close()
        ftrans.close()
    else:
        while True:
            try:
                seqin = raw_input('Input Sequence: ')
                n_samples = int(raw_input('How many samples? '))
                alpha = None
                if not args.beam_search:
                    alpha = float(raw_input('Inverse Temperature? '))
                seq,parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
                print "Parsed Input:", parsed_in
            except Exception:
                print "Exception while parsing your input:"
                traceback.print_exc()
                continue

            trans, costs = sample(lm_model, seq, n_samples, sampler=sampler,
                    beam_search=beam_search, normalize=args.normalize,
                    alpha=alpha, verbose=True)

if __name__ == "__main__":
    main()
