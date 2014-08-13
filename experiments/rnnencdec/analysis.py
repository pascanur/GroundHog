import logging
import numpy
import pandas

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
    axes.legend(legend)

def load_n_show_timings(axes, timings, legend, **kwargs):
    datas = [load_timings(path, **kwargs) for path in timings]
    show_args = set(kwargs.keys()).intersection(['y', 'hours'])
    show_timings(axes, datas, legend,
            **{k : kwargs[k] for k in show_args})
