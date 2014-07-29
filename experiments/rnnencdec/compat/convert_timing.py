#!/usr/bin/env python

import argparse
import numpy

def rename_costs(timing):
    timing['log2_p_expl'] = timing['cost2_p_expl']
    timing['log2_p_word'] = timing['cost2_p_word']
    del timing['cost2_p_expl']
    del timing['cost2_p_word']
    return timing

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv-fn", help="Conversion function")
    parser.add_argument("input", help="Timings to convert")
    parser.add_argument("output", nargs="?", help="Destintation to save")
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.output:
        args.output = args.input

    with open(args.input, 'r') as src:
        timing = dict(numpy.load(src).items())
    eval(args.conv_fn)(timing)

    numpy.savez(args.output, **timing)

if __name__ == "__main__":
    main()
