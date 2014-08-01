#!/usr/bin/env python

import numpy
import pandas
import argparse
import matplotlib
import logging
matplotlib.use("Agg")
from matplotlib import pyplot

logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start from this iteration")
    parser.add_argument("--finish", type=int, default=10 ** 9, help="Finish with that iteration")
    parser.add_argument("--window", type=int, default=100, help="Window width")
    parser.add_argument("--hours", action="store_true", default=False, help="Display time on X-axis")
    parser.add_argument("--legend", default=None, help="Legend to use in plot")
    parser.add_argument("--y", default="log2_p_expl", help="What to plot")
    parser.add_argument("timings", nargs="+", help="Path to timing files")
    parser.add_argument("plot_path", help="Path to save plot")
    return parser.parse_args()

def load_timings(path, args, y):
    logging.debug("Loading timings from {}".format(path))
    tm = numpy.load(path)
    num_steps = min(tm['step'], args.finish)
    df = pandas.DataFrame({k : tm[k] for k in [y, 'time_step']})[args.start:num_steps]
    one_step = df['time_step'].median() / 3600.0
    logging.debug("Median time for one step is {} hours".format(one_step))
    if args.hours:
        df.index = (args.start + numpy.arange(0, df.index.shape[0])) * one_step
    return pandas.rolling_mean(df, args.window).iloc[args.window:]

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    args.y = args.y.split(',')
    if len(args.y) < 2:
        args.y = [args.y[0]] * len(args.timings)
    datas = [load_timings(path, args, y) for path,y in zip(args.timings,args.y)]
    for path, y, data in zip(args.timings, args.y, datas):
        pyplot.plot(data.index, data[y])
        print "Average {} is {} after {} {} for {}".format(
                y, data[y].iloc[-1],
                data.index[-1], "hours" if args.hours else "iterations", path)

    pyplot.xlabel("hours" if args.hours else "iterations")
    pyplot.ylabel("log_2 likelihood")
    pyplot.legend(args.legend.split(",") if args.legend else range(len(datas)))
    pyplot.savefig(args.plot_path)
