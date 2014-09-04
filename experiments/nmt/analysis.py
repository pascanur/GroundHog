import logging
import numpy
import pandas
from collections import Counter

logger = logging.getLogger(__name__)

def load_timings(path, y="cost2_p_expl", start=0, finish=3000000, window=100, hours=False):
    logging.debug("Loading timings from {}".format(path))
    tm = numpy.load(path)
    num_steps = min(tm['step'], finish)
    df = pandas.DataFrame({k : tm[k] for k in [y, 'time_step']})[start:num_steps]
    one_step = df['time_step'][-window:].median() / 3600.0
    print "Median time for one step is {} hours".format(one_step)
    if hours:
        df.index = (start + numpy.arange(0, df.index.shape[0])) * one_step
    return pandas.rolling_mean(df, window).iloc[window:]

def show_timings(axes, timings, legend, y="cost2_p_expl", hours=False):
    for data, name in zip(timings, legend):
        axes.plot(data.index, data[y])
        print "Average {} is {} after {} {} for {}".format(
                y, data[y].iloc[-1],
                data.index[-1], "hours" if hours else "iterations", name)
    #axes.set_ylim(0, 20)
    axes.set_xlabel("hours" if hours else "iterations")
    axes.set_ylabel("log_2 likelihood")
    axes.legend(legend, loc='best')

def load_n_show_timings(axes, timings, legend, **kwargs):
    datas = [load_timings(path, **kwargs) for path in timings]
    show_args = set(kwargs.keys()).intersection(['y', 'hours'])
    show_timings(axes, datas, legend,
            **{k : kwargs[k] for k in show_args})

def bleu_stats(hypothesis, reference):
    yield len(hypothesis)
    yield len(reference)
    for n in xrange(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i + n]) for i in xrange(len(hypothesis) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i:i + n]) for i in xrange(len(reference) + 1 - n)])
        yield sum((s_ngrams & r_ngrams).values())
        yield max(len(hypothesis) + 1 - n, 0)

def bleu(stats):
    stats = numpy.atleast_2d(numpy.asarray(stats))[:, :10].sum(axis=0)
    if not all(stats):
        return 0
    c, r = stats[:2]
    log_bleu_prec = sum([numpy.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4.
    return numpy.exp(min(0, 1 - float(r) / c) + log_bleu_prec) * 100

def smoothed_bleu(stats):
    c, r = stats[:2]
    log_bleu_prec = sum([numpy.log((1 + float(x)) / (1 + y)) for x, y in zip(stats[2::2], stats[3::2])]) / 4.
    return numpy.exp(min(0, 1 - float(r) / c) + log_bleu_prec) * 100


