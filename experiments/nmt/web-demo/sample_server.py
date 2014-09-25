#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys
import os

import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_state,\
    parse_input
from experiments.nmt.sample import sample, BeamSearch

import BaseHTTPServer
from subprocess import Popen, PIPE
import urllib

logger = logging.getLogger(__name__)

class MTReqHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_GET(self):
        print 'header:'
        print self.headers

        print 'path:'
        print self.path

        args = self.path.split('?')[1]
        args = args.split('&')
        source_sentence = None
        ignore_unk = False
        beamwidth = 10
        for aa in args:
            cc = aa.split('=')
            if cc[0] == 'source':
                source_sentence = cc[1]
            if cc[0] == 'ignore_unk':
                ignore_unk = int(cc[1])
            if cc[0] == 'beamwidth':
                beamwidth = int(cc[1])

        if source_sentence == None:
            self.send_response(400)
            return

        source_sentence = urllib.unquote_plus(source_sentence)
        print 'source: ', source_sentence

        translation, unknown_words = self.server.sampler.sample(
                source_sentence, ignore_unk=ignore_unk, beamwidth=beamwidth)

        response = urllib.quote(translation+'\n'+','.join(unknown_words))

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(response)

class Sampler:
    def __init__(self, state, lm_model, indx_word, idict_src, beam_search, tokenizer_cmd=None, detokenizer_cmd=None):
        self.state = state
        self.lm_model = lm_model
        self.indx_word = indx_word
        self.idict_src = idict_src
        self.beam_search = beam_search
        self.tokenizer_cmd = tokenizer_cmd
        self.detokenizer_cmd = detokenizer_cmd

    def sample(self, sentence, ignore_unk=False, beamwidth=10):
        if self.tokenizer_cmd:
            tokenizer=Popen(self.tokenizer_cmd, stdin=PIPE, stdout=PIPE)
            sentence, _ = tokenizer.communicate(sentence)
        seq, parsed_in = parse_input(self.state, self.indx_word, sentence, idx2word=self.idict_src)
        # Sample a translation and detokenize it
        trans, cost, _ = sample(self.lm_model, seq, beamwidth,
                beam_search=self.beam_search, normalize=True,
                ignore_unk=ignore_unk)
        if self.detokenizer_cmd:
            detokenizer=Popen(self.detokenizer_cmd, stdin=PIPE, stdout=PIPE)
            detokenized_sentence, _ = detokenizer.communicate(trans[0])
        else:
            detokenized_sentence = trans[0]

        unknown_words = [word for word, index
                         in zip(sentence.split(), seq)
                         if index == 1]
        return detokenized_sentence, unknown_words

def parse_args():
    parser = argparse.ArgumentParser(
            "Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--port", help="Port to use", type=int, default=8888)
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--beam-search",
            action="store_true", help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size",
            type=int, help="Beam size", default=5)
    parser.add_argument("--ignore-unk",
            default=False, action="store_true",
            help="Ignore unknown words")
    parser.add_argument("--source",
            help="File of source sentences")
    parser.add_argument("--trans",
            help="File to save translations in")
    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("model_path",
            help="Path to the model")
    parser.add_argument("changes",
            nargs="?", default="",
            help="Changes to state")
    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    server_address = ('', args.port)
    httpd = BaseHTTPServer.HTTPServer(server_address, MTReqHandler)

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

    tokenizer_cmd = [os.getcwd()+'/tokenizer.perl', '-l', 'en', '-q', '-']
    detokenizer_cmd = [os.getcwd()+'/detokenizer.perl', '-l', 'fr', '-q', '-']
    sampler = Sampler(state, lm_model, indx_word, idict_src, beam_search = beam_search,
            tokenizer_cmd=tokenizer_cmd, detokenizer_cmd=detokenizer_cmd)
    httpd.sampler = sampler

    print 'Server starting..'
    httpd.serve_forever()

    '''
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

        sample(lm_model, seq, n_samples, sampler=sampler,
                beam_search=beam_search,
                ignore_unk=args.ignore_unk, normalize=args.normalize,
                alpha=alpha, verbose=True)
    '''

if __name__ == "__main__":
    main()
