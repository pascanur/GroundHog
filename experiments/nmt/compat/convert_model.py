#!/usr/bin/env python

import argparse
import numpy

def merge_state_projections(model):
    model['W_0_dec_dec_inputter_0'] = numpy.vstack([
        model['W_0_dec_dec_inputter_0'], model['W_0_dec_back_dec_inputter_0']])
    model['W_0_dec_dec_reseter_0'] = numpy.vstack([
        model['W_0_dec_dec_reseter_0'], model['W_0_dec_back_dec_reseter_0']])
    model['W_0_dec_dec_updater_0'] = numpy.vstack([
        model['W_0_dec_dec_updater_0'], model['W_0_dec_back_dec_updater_0']])
    model['W_0_dec_repr_readout'] = numpy.vstack([
        model['W_0_dec_repr_readout'], model['W_0_dec_back_repr_readout']])
    del model['W_0_dec_back_dec_inputter_0']
    del model['W_0_dec_back_dec_reseter_0']
    del model['W_0_dec_back_dec_updater_0']
    del model['W_0_dec_back_repr_readout']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv-fn", help="Conversion function")
    parser.add_argument("input", help="Model to convert")
    parser.add_argument("output", nargs="?", help="Destintation to save")
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.output:
        args.output = args.input

    with open(args.input, 'r') as src:
        model = dict(numpy.load(src).items())
    eval(args.conv_fn)(model)

    numpy.savez(args.output, **model)

if __name__ == "__main__":
    main()
