"""
Stochastic Gradient Descent with momentum.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("KyungHyun Cho "
               "Razvan Pascanu "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import time

import theano
import theano.tensor as TT
from theano import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import print_time, print_mem, const


class SGD(object):
    """
    Stochastic gradient descent class
    """
    def __init__(self,
                 model,
                 state,
                 data):
        """
        :type model: groundhog model class
        :param model: class depicting the model to be optimized

        :type state: dictionary or jobman DD object
        :param state: dictionary containing various hyper-parameters. The
            class will write into this dictionary updates like the current
            training error and so on

        :type data: groundhog dataset object
        :param data: data iterator over which training is done
        """

        #####################################
        # Step 0. Constructs shared variables
        #####################################
        bs = state['bs']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])
        srng = RandomStreams(self.rng.randint(213))
        self.gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name)
                   for p in model.params]
        self.step = 0
        self.bs = bs
        self.state = state
        self.data = data
        self.step_timer = time.time()
        self.gdata = [theano.shared(numpy.zeros( (2,)*x.ndim,
                                                dtype=x.dtype),
                                    name=x.name) for x in model.inputs]

	if 'profile' not in self.state:
            self.state['profile'] = 0

        ###################################
        # Step 1. Compile training function
        ###################################
        print 'Constructing grad function'
        loc_data = self.gdata
        lr = TT.scalar('lr')
        self.prop_exprs = [x[1] for x in model.properties]
        self.prop_names = [x[0] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]
        rval = theano.clone(model.param_grads + self.update_rules + \
                            self.prop_exprs + [model.train_cost],
                            replace=zip(model.inputs, loc_data))
        nparams = len(model.params)
        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]



        norm_gs = sum(TT.sum(x**2)
            for x,p in zip(gs,
                           self.model.params)
                      if p not in self.model.exclude_params_for_norm)
        if 'cutoff' in state and state['cutoff'] > 0:
            c = numpy.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_gs))
            _gs = []
            for g,p in zip(gs,self.model.params):
                if p not in self.model.exclude_params_for_norm:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       TT.switch(notfinite, numpy.float32(.1)*p,
                           tmpg))
                else:
                    _gs.append(g)
            gs = _gs

        store_gs = [(s,g) for s,g in zip(self.gs, gs)]
        updates = store_gs + [(s[0], r) for s,r in zip(model.updates, rules)]
        print 'Compiling grad function'
        st = time.time()
        self.train_fn = theano.function(
            [], outs, name='train_function',
            updates = updates,
            givens = zip(model.inputs, loc_data),
            profile=self.state['profile'])
        print 'took', time.time() - st


        self.lr = numpy.float32(state['lr'])
        new_params = [p - s*lr*g for s, p, g in zip(model.params_grad_scale, model.params, self.gs)]
        self.update_fn = theano.function(
            [lr], [], name='update_function',
            allow_input_downcast=True,
            updates = zip(model.params, new_params),
            profile=self.state['profile'])

        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + \
                ['cost',
                 'time_step',
                 'whole_time',
                  'lr']


    def __call__(self):
        batch = self.data.next()
        # Perturb the data (! and the model)
        if isinstance(batch, dict):
            batch = self.model.perturb(**batch)
        else:
            batch = self.model.perturb(*batch)
        # Load the dataset into GPU
        # Note: not the most efficient approach in general, as it involves
        # each batch is copied individually on gpu
        if isinstance(batch, dict):
            for gdata in self.gdata:
                gdata.set_value(batch[gdata.name], borrow=True)
        else:
            for gdata, data in zip(self.gdata, batch):
                gdata.set_value(data, borrow=True)
        # Run the trianing function
        g_st = time.time()
        rvals = self.train_fn()
        for schedule in self.schedules:
            schedule(self, rvals[-1])
        self.update_fn(self.lr)
        g_ed = time.time()
        self.state['lr'] = float(self.lr)
        cost = rvals[-1]
        self.old_cost = cost
        whole_time = time.time() - self.step_timer
        if self.step % self.state['trainFreq'] == 0:
            msg = '.. iter %4d cost %.3f'
            vals = [self.step, cost]
            for dx, prop in enumerate(self.prop_names):
                msg += ' '+prop+' %.2e'
                vals += [float(numpy.array(rvals[dx]))]
            msg += ' step time %s whole time %s lr %.2e'
            vals += [print_time(g_ed - g_st),
                     print_time(time.time() - self.step_timer),
                     float(self.lr)]
            print msg % tuple(vals)
        self.step += 1
        ret = dict([('cost', float(cost)),
                       ('lr', float(self.lr)),
                       ('time_step', float(g_ed - g_st)),
                       ('whole_time', float(whole_time))]+zip(self.prop_names, rvals))
        return ret
