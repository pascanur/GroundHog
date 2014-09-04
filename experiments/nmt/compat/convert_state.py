#!/usr/bin/env python

import argparse
import numpy
import cPickle

def separate_enc_dec_rec_layers(state):
    state['enc_rec_layer'] = state['rec_layer']
    state['enc_rec_gating'] = state['rec_gating']
    state['enc_rec_reseting'] = state['rec_reseting']
    state['enc_rec_gater'] = state['rec_gater']
    state['enc_rec_reseter'] = state['rec_reseter']

    state['dec_rec_layer'] = state['rec_layer']
    state['dec_rec_gating'] = state['rec_gating']
    state['dec_rec_reseting'] = state['rec_reseting']
    state['dec_rec_gater'] = state['rec_gater']
    state['dec_rec_reseter'] = state['rec_reseter']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv-fn", help="Conversion function")
    parser.add_argument("--changes", help="Changes to the state")
    parser.add_argument("src", help="State to convert")
    parser.add_argument("dst",  nargs="?", help="Destination to save")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.src, 'r') as src:
        state = cPickle.load(src)
    state.update(eval("dict({})".format(args.changes)))

    if args.conv_fn:
        eval(args.conv_fn)(state)

    if not args.dst:
        args.dst = args.src
    with open(args.dst, 'w') as dst:
        cPickle.dump(state, dst)

if __name__ == "__main__":
    main()

