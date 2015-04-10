#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import numpy

from experiments.nmt import\
        RNNEncoderDecoder,\
        parse_input,\
        get_batch_iterator,\
        prototype_state

logger = logging.getLogger(__name__)

class BatchTxtIterator(object):

    def __init__(self, state, txt, indx,  batch_size, raise_unk, unk_sym=-1, null_sym=-1):
        self.__dict__.update(locals())
        self.__dict__.pop('self')

    def start(self):
        self.txt_file = open(self.txt)

    def _pack(self, seqs):
        num = len(seqs)
        max_len = max(map(len, seqs))
        x = numpy.zeros((num, max_len), dtype="int64")
        x_mask = numpy.zeros((num, max_len), dtype="float32")
        for i, seq in enumerate(seqs):
            x[i, :len(seq)] = seq
            x_mask[i, :len(seq)] = 1.0
        return x.T, x_mask.T

    def __iter__(self):
        return self

    def next(self):
        seqs = []
        try:
            while len(seqs) < self.batch_size:
                line = next(self.txt_file).strip()
                seq, _ = parse_input(self.state, self.indx, line, raise_unk=self.raise_unk,
                                     unk_sym=self.unk_sym, null_sym=self.null_sym)
                seqs.append(seq)
            return self._pack(seqs)
        except StopIteration:
            if not seqs:
                raise StopIteration()
            return self._pack(seqs)

class BatchBiTxtIterator(object):

    def __init__(self, state, src, indx_src, trg, indx_trg, batch_size, raise_unk):
        self.__dict__.update(locals())
        self.__dict__.pop('self')
        self.src_iter = BatchTxtIterator(state, src, indx_src, batch_size, raise_unk, 
                                         unk_sym=state['unk_sym_source'], null_sym=state['null_sym_source'])
        self.trg_iter = BatchTxtIterator(state, trg, indx_trg, batch_size, raise_unk, 
                                         unk_sym=state['unk_sym_target'], null_sym=state['null_sym_target'])

    def start(self):
        self.src_iter.start()
        self.trg_iter.start()

    def __iter__(self):
        return self

    def next(self):
        x, x_mask = next(self.src_iter)
        y, y_mask = next(self.trg_iter)
        assert x.shape[1] == y.shape[1]
        return dict(x=x, x_mask=x_mask, y=y, y_mask=y_mask)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, help="State to use")

    # Paths
    parser.add_argument("--src", help="Source phrases")
    parser.add_argument("--trg", help="Target phrases")
    parser.add_argument("--scores", default=None, help="Save scores to")
    parser.add_argument("model_path", help="Path to the model")

    # Options
    parser.add_argument("--print-probs", default=False, action="store_true",
            help="Print probs instead of log probs")
    parser.add_argument("--allow-unk", default=False, action="store_true",
            help="Allow unknown words in the input")
    parser.add_argument("--mode", default="interact",
            help="Processing mode, one of 'batch', 'txt', 'interact'")
    parser.add_argument("--n-batches", default=-1, type=int,
            help="Score only first n batches")
    parser.add_argument("--verbose", default=False, action="store_true",
            help="Print more stuff")
    parser.add_argument("--y-noise",  type=float,
            help="Probability for a word to be replaced by a random word")

    # Additional arguments
    parser.add_argument("changes",  nargs="?", help="Changes to state", default="")

    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    state['sort_k_batches'] = 1
    state['shuffle'] = False
    state['use_infinite_loop'] = False
    state['force_enc_repr_cpu'] = False

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True, compute_alignment=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)

    indx_word_src = cPickle.load(open(state['word_indx'],'rb'))
    indx_word_trgt = cPickle.load(open(state['word_indx_trgt'], 'rb'))

    if args.mode == "batch":
        data_given = args.src or args.trg
        txt = data_given and not (args.src.endswith(".h5") and args.trg.endswith(".h5"))
        if data_given and not txt:
            state['source'] = [args.src]
            state['target'] = [args.trg]
        if not data_given and not txt:
            logger.info("Using the training data")
        if txt:
            data_iter = BatchBiTxtIterator(state,
                    args.src, indx_word_src, args.trg, indx_word_trgt,
                    state['bs'], raise_unk=not args.allow_unk)
            data_iter.start()
        else:
            data_iter = get_batch_iterator(state)
            data_iter.start(0)

        score_file = open(args.scores, "w") if args.scores else sys.stdout

        scorer = enc_dec.create_scorer(batch=True)

        count = 0
        n_samples = 0
        logger.info('Scoring phrases')
        for i, batch in enumerate(data_iter):
            if batch == None:
                continue
            if args.n_batches >= 0 and i == args.n_batches:
                break

            if args.y_noise:
                y = batch['y']
                random_words = numpy.random.randint(0, 100, y.shape).astype("int64")
                change_mask = numpy.random.binomial(1, args.y_noise, y.shape).astype("int64")
                y = change_mask * random_words + (1 - change_mask) * y
                batch['y'] = y

            st = time.time()
            [scores] = scorer(batch['x'], batch['y'],
                    batch['x_mask'], batch['y_mask'])
            if args.print_probs:
                scores = numpy.exp(scores)
            up_time = time.time() - st
            for s in scores:
                print >>score_file, "{:.5e}".format(float(s))

            n_samples += batch['x'].shape[1]
            count += 1

            if count % 100 == 0:
                score_file.flush()
                logger.debug("Scores flushed")
            logger.debug("{} batches, {} samples, {} per sample; example scores: {}".format(
                count, n_samples, up_time/scores.shape[0], scores[:5]))

        logger.info("Done")
        score_file.flush()
    elif args.mode == "interact":
        scorer = enc_dec.create_scorer()
        while True:
            try:
                compute_probs = enc_dec.create_probs_computer()
                src_line = raw_input('Source sequence: ')
                trgt_line = raw_input('Target sequence: ')
                src_seq = parse_input(state, indx_word_src, src_line, raise_unk=not args.allow_unk, 
                                      unk_sym=state['unk_sym_source'], null_sym=state['null_sym_source'])
                trgt_seq = parse_input(state, indx_word_trgt, trgt_line, raise_unk=not args.allow_unk,
                                       unk_sym=state['unk_sym_target'], null_sym=state['null_sym_target'])
                print "Binarized source: ", src_seq
                print "Binarized target: ", trgt_seq
                probs = compute_probs(src_seq, trgt_seq)
                print "Probs: {}, cost: {}".format(probs, -numpy.sum(numpy.log(probs)))
            except Exception:
                traceback.print_exc()
    elif args.mode == "txt":
        assert args.src and args.trg
        scorer = enc_dec.create_scorer()
        src_file = open(args.src, "r")
        trg_file = open(args.trg, "r")
        compute_probs = enc_dec.create_probs_computer(return_alignment=True)
        try:
            numpy.set_printoptions(precision=3, linewidth=150, suppress=True)
            i = 0
            while True:
                src_line = next(src_file).strip()
                trgt_line = next(trg_file).strip()
                src_seq, src_words = parse_input(state,
                        indx_word_src, src_line, raise_unk=not args.allow_unk,
                        unk_sym=state['unk_sym_source'], null_sym=state['null_sym_source'])
                trgt_seq, trgt_words = parse_input(state,
                        indx_word_trgt, trgt_line, raise_unk=not args.allow_unk,
                        unk_sym=state['unk_sym_target'], null_sym=state['null_sym_target'])
                probs, alignment = compute_probs(src_seq, trgt_seq)
                if args.verbose:
                    print "Probs: ", probs.flatten()
                    if alignment.ndim == 3:
                        print "Alignment:".ljust(20), src_line, "<eos>"
                        for i, word in enumerate(trgt_words):
                            print "{}{}".format(word.ljust(20), alignment[i, :, 0])
                        print "Generated by:"
                        for i, word in enumerate(trgt_words):
                            j = numpy.argmax(alignment[i, :, 0])
                            print "{} <--- {}".format(word,
                                    src_words[j] if j < len(src_words) else "<eos>")
                i += 1
                if i % 100 == 0:
                    sys.stdout.flush()
                    logger.debug(i)
                print -numpy.sum(numpy.log(probs))
        except StopIteration:
            pass
    else:
        raise Exception("Unknown mode {}".format(args.mode))

if __name__ == "__main__":
    main()
