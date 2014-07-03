#!/usr/bin/env python

import numpy
import pandas
import argparse
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start from this iteration")
    parser.add_argument("--finish", type=int, default=10 ** 9, help="Finish with that iteration")
    parser.add_argument("--window", type=int, default=100, help="Window width")
    parser.add_argument("timing_path", help="Path to timing file")
    parser.add_argument("plot_path", help="Path to save plot")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    tm = numpy.load(args.timing_path)
    num_steps = min(tm['step'], args.finish)
    df = pandas.DataFrame({k : tm[k] for k in ['traincost', 'time_step']})[args.start:num_steps]
    one_step = df['time_step'].median() / 3600.0
    df.index = (args.start + numpy.arange(0, df.index.shape[0])) * one_step
    pandas.rolling_mean(df['traincost'], args.window).plot()

    pyplot.savefig(args.plot_path)
