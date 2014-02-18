"""
Recurrent layers.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import copy
import theano
import theano.tensor as TT
# Nicer interface of scan
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog import utils
from groundhog.utils import sample_weights, \
        sample_weights_classic,\
        init_bias, \
        constant_shape, \
        powerup, \
        maxout, \
        sample_power

from basic import Layer

class RecurrentMultiLayer(Layer):
    """
    Constructs a recurrent layer whose transition from h_tm1 to h_t is given
    by an MLP or logistic regression. In our ICLR submission this is a
    DT-RNN model.
    """
    def __init__(self,
                 rng,
                 n_hids=[500,500],
                 lp_units = False,
                 init_power = sample_power,
                 power_scale=.001,
                 activation = [TT.tanh, TT.tanh],
                 scale=.01,
                 sparsity = -1,
                 activ_noise=0.,
                 weight_noise=False,
                 dropout = 1.,
                 init_fn='sample_weights',
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 grad_scale = 1.,
                 profile = 0,
                 name=None):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: list of ints or list of pairs of ints
        :param n_hids: Number of hidden units on each layer of the MLP; if a
            pair, the first entry is the projection of the layer below, the
            second value is the size of the layer after applying the
            activation function

        :type activation: string/function or list of
        :param activation: Activation function for the embedding layers. If
            a list it needs to have a value for each layer. If not, the same
            activation will be applied to all layers

        :type scale: float or list of
        :param scale: depending on the initialization function, it can be
            the standard deviation of the Gaussian from which the weights
            are sampled or the largest singular value. If a single value it
            will be used for each layer, otherwise it has to have one value
            for each layer

        :type sparsity: int or list of
        :param sparsity: if a single value, it will be used for each layer,
            otherwise it has to be a list with as many values as layers. If
            negative, it means the weight matrix is dense. Otherwise it
            means this many randomly selected input units are connected to
            an output unit


        :type weight_noise: bool
        :param weight_noise: If true, the model is used with weight noise
            (and the right shared variable are constructed, to keep track of the
            noise)

        :type dropout: float
        :param dropout: the probability with which hidden units are dropped
            from the hidden layer. If set to 1, dropout is not used

        :type init_fn: string or function
        :param init_fn: function used to initialize the weights of the
            layer. We recommend using either `sample_weights_classic` or
            `sample_weights` defined in the utils

        :type bias_fn: string or function
        :param bias_fn: function used to initialize the biases. We recommend
            using `init_bias` defined in the utils

        :type bias_scale: float
        :param bias_scale: argument passed to `bias_fn`, depicting the scale
            of the initial bias

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.

        """
        self.grad_scale = grad_scale

        if isinstance(n_hids, (str, unicode)):
            n_hids = eval(n_hids)

        if isinstance(lp_units, (str, unicode)):
            lp_units = eval(lp_units)

        if isinstance(init_power, (str, unicode)):
            init_power = eval(init_power)

        if isinstance(power_scale, (str, unicode)):
            power_scale = eval(power_scale)

        if isinstance(activation, (str, unicode)):
            activation = eval(activation)

        if isinstance(scale, (str, unicode)):
            scale= eval(scale)

        if isinstance(sparsity, (str, unicode)):
            sparsity = eval(sparsity)

        if isinstance(activ_noise, (str, unicode)):
            activ_nosie = eval(activ_noise)

        if isinstance(weight_noise, (str, unicode)):
            weight_noise = eval(weight_noise)

        if isinstance(dropout, (str, unicode)):
            dropout = eval(dropout)

        if isinstance(init_fn, (str, unicode)):
            init_fn = eval(init_fn)

        if isinstance(bias_fn, (str, unicode)):
            bias_fn = eval(bias_fn)

        if isinstance(bias_scale, (str, unicode)):
            bias_scale = eval(bias_scale)

        if isinstance(grad_scale, (str, unicode)):
            grad_scale = eval(grad_scale)

        if type(n_hids) not in (list, tuple):
            n_hids = [n_hids]
        n_layers = len(n_hids)
        if type(scale) not in (list, tuple):
            scale = [scale] * n_layers
        if type(sparsity) not in (list, tuple):
            sparsity = [sparsity] * n_layers
        for idx, sp in enumerate(sparsity):
            if sp < 0:
                sparsity[idx] = n_hids[idx]
                if isinstance(sparsity[idx], (list, tuple)):
                    sparsity[idx] = sparsity[idx][0]
        if type(activation) not in (list, tuple):
            activation = [activation] * n_layers
        if type(bias_scale) not in (list, tuple):
            bias_scale = [bias_scale] * (n_layers-1)
        if type(bias_fn) not in (list, tuple):
            bias_fn = [bias_fn] * (n_layers-1)
        if type(init_fn) not in (list, tuple):
            init_fn = [init_fn] * n_layers
        if type(lp_units) not in (list, tuple):
            lp_units = [lp_units] * n_layers
        if type(init_power) not in (list, tuple):
            init_power = [init_power] * n_layers
        if type(power_scale) not in (list, tuple):
            power_scale = [power_scale] * n_layers

        for dx in xrange(n_layers):
            if dx < n_layers-1:
                if type(bias_fn[dx]) is str or type(bias_fn[dx]) is unicode:
                    bias_fn[dx] = eval(bias_fn[dx])
            if type(init_fn[dx]) is str or type(init_fn[dx]) is unicode:
                init_fn[dx] = eval(init_fn[dx])
            if type(init_power[dx]) is str or type(init_power[dx]) is unicode:
                init_power[dx] = eval(init_power[dx])
            if type(activation[dx]) is str or type(activation[dx]) is unicode:
                activation[dx] = eval(activation[dx])
        self.lp_units = lp_units
        self.init_power = init_power
        self.power_scale = power_scale
        self.scale = scale
        self.n_layers = n_layers
        self.sparsity = sparsity
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.weight_noise = weight_noise
        self.activ_noise = activ_noise
        self.profile = profile
        self.dropout = dropout
        assert rng is not None, "random number generator should not be empty!"
        n_in = n_hids[0]
        n_out = n_hids[-1]
        if type(n_in) in (list, tuple):
            n_in = n_in[0]
        if type(n_out) in (list, tuple):
            n_out = n_out[1]
        super(RecurrentMultiLayer, self).__init__(n_in,
                                                 n_out,
                                                 rng,
                                                 name)

        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hhs = []
        self.b_hhs = []
        self.lp_power = []
        for dx in xrange(self.n_layers):
            n_in = self.n_hids[(dx-1)%self.n_layers]
            if type(n_in) in (list, tuple):
                n_in = n_in[1]
            n_out = self.n_hids[dx]
            if type(n_out) in (list, tuple):
                n_out = n_out[0]
            W_hh = self.init_fn[dx](n_in,
                                    n_out,
                                    self.sparsity[dx],
                                    self.scale[dx],
                                    rng=self.rng)
            self.W_hhs.append(theano.shared(value=W_hh, name="W%d_%s" %
                                       (dx,self.name)))
            if self.lp_units[dx]:
                p = theano.shared(self.init_power[dx](self.n_hids[dx][1],
                                                      self.power_scale[dx],
                                                      self.rng),
                                  name="p%d_%s"%(dx, self.name))

                self.lp_power.append(p)
            else:
                self.lp_power.append(None)
            if dx > 0:
                self.b_hhs.append(theano.shared(
                    self.bias_fn[dx-1](n_out,
                                       self.bias_scale[dx-1],
                                       self.rng),
                    name='b%d_%s' %(dx, self.name)))
        self.params = [x for x in self.W_hhs] +\
                [x for x in self.b_hhs] + \
                [x for x in self.lp_power if x is not None]
        self.params_grad_scale = [self.grad_scale for x in self.params]
        if self.weight_noise:
            # We do not use weight noise on the exponent of lp units !?
            self.nW_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_hhs]
            self.nb_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.b_hhs]
            self.noise_params = [x for x in self.nW_hhs] + [x for x in self.nb_hhs]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]


    def step_fprop(self,
                   state_below,
                   mask=None,
                   dpmask=None,
                   state_before=None,
                   use_noise=True,
                   no_noise_bias=False):
        """
        Constructs the computational graph of a single step of the recurrent
        layer.

        :type state_below: theano variable
        :param state_below: the input to the layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a
            minibatch

        :type state_before: theano variable
        :param state_before: the previous value of the hidden state of the
            layer

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """
        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hhs = [(x+y) for x, y in zip(self.W_hhs, self.nW_hhs)]
            if not no_noise_bias:
                b_hhs = [(x+y) for x, y in zip(self.b_hhs, self.nb_hhs)]
            else:
                b_hhs = self.b_hhs
        else:
            W_hhs = self.W_hhs
            b_hhs = self.b_hhs
        preactiv = TT.dot(state_before, W_hhs[0]) +state_below
        if self.lp_units[0]:
            # We are dealing with an lp unit that has additional arguments
            h = self.activation[0](preactiv, self.lp_power[0],
                                   self.n_hids[0])
        elif isinstance(self.n_hids[0], (list, tuple)):
            h = self.activation[0](preactiv, self.n_hids[0])
        else:
            h = self.activation[0](preactiv)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout

        rval +=[h]
        for dx in xrange(1, self.n_layers):
            preactiv = TT.dot(h, W_hhs[dx]) + b_hhs[dx-1]
            if self.lp_units[dx]:
                h = self.activation[dx](preactiv, self.lp_power[dx],
                                        self.n_hids[dx])
            elif isinstance(self.n_hids[dx], (list, tuple)):
                h = self.activation[dx](preactiv, self.n_hids[dx])
            else:
                h = self.activation[dx](preactiv)

            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
            if self.dropout < 1.:
                if use_noise:
                    if h.ndim == 2:
                        h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                        dpidx = dpidx + h.shape[1]
                    else:
                        h = h * dpmask[dpidx:dpidx+h.shape[0]]
                        dpidx = dpidx + h.shape[0]
                else:
                    h = h * self.dropout
            rval += [h]
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        rval[-1] = h
        return rval

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              n_steps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias = False):
        """
        Evaluates the forward through a recurrent layer

        :type state_below: theano variable
        :param state_below: the input of the recurrent layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a
            minibatch

        :type init_state: theano variable or None
        :param init_state: initial state for the hidden layer

        :type n_steps: None or int or theano scalar
        :param n_steps: Number of steps the recurrent netowrk does

        :type batch_size: int
        :param batch_size: the size of the minibatch over which scan runs

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type truncate_gradient: int
        :param truncate_gradient: If negative, no truncation is used,
            otherwise truncated BPTT is used, where you go backwards only this
            amount of steps

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """


        if theano.config.floatX=='float32':
            floatX = numpy.float32
        else:
            floatX = numpy.float64
        if n_steps is None:
            n_steps = state_below.shape[0]
            if batch_size and batch_size != 1:
                n_steps = n_steps / batch_size
        if batch_size is None and state_below.ndim == 3:
            batch_size = state_below.shape[1]
        if state_below.ndim == 2 and \
           (not isinstance(batch_size,int) or batch_size > 1):
            state_below = state_below.reshape((n_steps, batch_size, self.nin))


        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.nhid)
            else:
                init_state = TT.alloc(floatX(0), self.nhid)

        if mask:
            inps = [state_below, mask]
            fn = lambda x,y,z : self.step_fprop(x,y,None, z, use_noise=use_noise,
                                               no_noise_bias=no_noise_bias)
        else:
            inps = [state_below]
            fn = lambda tx, ty: self.step_fprop(tx, None, None, ty,
                                                use_noise=use_noise,
                                                no_noise_bias=no_noise_bias)

        if self.dropout < 1. and use_noise:
            # build dropout mask outside scan
            allhid = 0
            for hid in self.n_hids:
                if isinstance(hid, (list, tuple)):
                    allhid += hid[0]
                else:
                    allhid += hid

            shape = state_below.shape
            if state_below.ndim == 3:
                alldpmask = self.trng.binomial(
                        (n_steps, batch_size, allhid),
                        n = 1, p = self.dropout, dtype=state_below.dtype)
            else:
                alldpmask = self.trng.binomial(
                        (n_steps, allhid),
                        n = 1, p = self.dropout, dtype=state_below.dtype)
            inps.append(alldpmask)
            if mask:
                fn = lambda x,y,z,u : self.step_fprop(x,y,z,u,use_noise=use_noise)
            else:
                fn = lambda tx, ty, tu: self.step_fprop(tx,None,ty,tu,
                                                use_noise=use_noise)

        rval, updates = theano.scan(fn,
                        sequences = inps,
                        outputs_info = [None]*(self.n_layers-1) +
                                    [init_state],
                        name='layer_%s'%self.name,
                        profile=self.profile,
                        truncate_gradient = truncate_gradient,
                        n_steps = n_steps)
        if not isinstance(rval,(list, tuple)):
            rval = [rval]
        new_h = rval[-1]
        self.out = rval[-1]
        self.rval = rval
        self.updates =updates

        return self.out


class RecurrentMultiLayerInp(RecurrentMultiLayer):
    """
    Similar to the RecurrentMultiLayer, with the exception that the input is
    fed into the top layer of the MLP (rather than being an input to the
    MLP).
    """
    def _init_params(self):
        self.W_hhs = []
        self.b_hhs = []
        for dx in xrange(self.n_layers):
            n_in = self.n_hids[(dx-1)%self.n_layers]
            if type(n_in) in (list, tuple):
                n_in = n_in[1]
            n_out = self.n_hids[dx]
            if type(n_out) in (list, tuple):
                n_out = n_out[0]
            W_hh = self.init_fn[dx](n_in,
                                    n_out,
                                    self.sparsity[dx],
                                    self.scale[dx],
                                    rng=self.rng)
            self.W_hhs.append(theano.shared(value=W_hh, name="W%d_%s" %
                                       (dx,self.name)))
            if self.lp_units[dx]:
                p = theano.shared(self.init_power[dx](self.n_hids[dx][1],
                                                      self.power_scale[dx],
                                                      self.rng),
                                  name="p%d_%s"%(dx, self.name))

                self.lp_power.append(p)
            else:
                self.lp_power.append(None)
            if dx < self.n_layers-1:
                self.b_hhs.append(theano.shared(
                    self.bias_fn[dx](self.n_hids[dx],
                                       self.bias_scale[dx],
                                       self.rng),
                    name='b%d_%s' %(dx, self.name)))
        self.params = [x for x in self.W_hhs] +\
                [x for x in self.b_hhs] + \
                [x for x in self.lp_power if x is not None]
        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]
        if self.weight_noise:
            self.nW_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_hhs]
            self.nb_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.b_hhs]
            self.noise_params = [x for x in self.nW_hhs] + [x for x in self.nb_hhs]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]


    def step_fprop(self,
                   state_below,
                   mask=None,
                   dpmask=None,
                   state_before=None,
                   no_noise_bias=False,
                   use_noise=True):
        """
        See parent class
        """
        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hhs = [(x+y) for x, y in zip(self.W_hhs,self.nW_hss)]
            if not no_noise_bias and self.weight_noise:
                b_hhs = [(x+y) for x, y in zip(self.b_hhs,self.nb_hhs)]
            else:
                b_hhs = self.b_hhs
        else:
            W_hhs = self.W_hhs
            b_hhs = self.b_hhs
        preactiv = TT.dot(state_before, W_hhs[0]) + b_hhs[0]
        if self.lp_units[0]:
            # We are dealing with an lp unit that has additional arguments
            h = self.activation[0](preactiv, self.lp_power[0],
                                   self.n_hids[0])
        elif isinstance(self.n_hids[0], (list, tuple)):
            h = self.activation[0](preactiv, self.n_hids[0])
        else:
            h = self.activation[0](preactiv)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)

        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout

        rval += [h]
        for dx in xrange(1, self.n_layers-1):
            preactiv = TT.dot(h, W_hhs[dx]) + b_hhs[dx]
            if self.lp_units[dx]:
                h = self.activation[dx](preactiv, self.lp_power[dx],
                                        self.n_hids[dx])
            elif isinstance(self.n_hids[dx], (list, tuple)):
                h = self.activation[dx](preactiv, self.n_hids[dx])
            else:
                h = self.activation[dx](preactiv)

            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
            if self.dropout < 1.:
                if use_noise:
                    if h.ndim == 2:
                        h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                        dpidx = dpidx + h.shape[1]
                    else:
                        h = h * dpmask[dpidx:dpidx+h.shape[0]]
                        dpidx = dpidx + h.shape[0]
                else:
                    h = h * self.dropout
            rval += [h]

        preactiv = TT.dot(state_before, W_hhs[-1]) + state_below
        if self.lp_units[-1]:
            # We are dealing with an lp unit that has additional arguments
            h = self.activation[-1](preactiv, self.lp_power[-1],
                                   self.n_hids[-1])
        elif isinstance(self.n_hids[-1], (list, tuple)):
            h = self.activation[-1](preactiv, self.n_hids[-1])
        else:
            h = self.activation[-1](preactiv)
        #h = self.activation[-1](TT.dot(h, W_hhs[-1]) + state_below)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                    dpidx = dpidx + h.shape[1]
                else:
                    h = h * dpmask[dpidx:dpidx+h.shape[0]]
                    dpidx = dpidx + h.shape[0]
            else:
                h = h * self.dropout
        rval += [h]
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        rval[-1] = h
        return rval

class RecurrentMultiLayerShortPath(RecurrentMultiLayer):
    """
    A similar layer to RecurrentMultiLayer (the DT-RNN), with the difference
    that we have shortcut connections in the MLP representing the transition
    from previous hidden state to the next
    """
    def _init_params(self):
        self.W_hhs = []
        self.b_hhs = []
        self.W_shortp = []
        for dx in xrange(self.n_layers):
            W_hh = self.init_fn[dx](self.n_hids[(dx-1)%self.n_layers],
                                    self.n_hids[dx],
                                    self.sparsity[dx],
                                    self.scale[dx],
                                    rng=self.rng)
            self.W_hhs.append(theano.shared(value=W_hh, name="W%d_%s" %
                                       (dx,self.name)))

            if dx > 0:
                W_shp = self.init_fn[dx](self.n_hids[self.n_layers-1],
                                         self.n_hids[dx],
                                         self.sparsity[dx],
                                         self.scale[dx],
                                         rng=self.rng)
                self.W_shortp.append(theano.shared(value=W_shp,
                                                   name='W_s%d_%s'%(dx,self.name)))
                self.b_hhs.append(theano.shared(
                    self.bias_fn[dx-1](self.n_hids[dx],
                                       self.bias_scale[dx-1],
                                       self.rng),
                    name='b%d_%s' %(dx, self.name)))
        self.params = [x for x in self.W_hhs] + [x for x in self.b_hhs] +\
                [x for x in self.W_shortp]
        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]
        if self.weight_noise:
            self.nW_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_hhs]
            self.nb_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.b_hhs]
            self.nW_shortp = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_shortp]

            self.noise_params = [x for x in self.nW_hhs] + [x for x in self.nb_hhs] + [x for x in self.nW_shortp]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape) for x in self.noise_params]



    def step_fprop(self,
                   state_below,
                   mask=None,
                   dpmask=None,
                   state_before=None,
                   no_noise_bias=False,
                   use_noise=True):
        """
        See parent class
        """
        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hhs = [(x+y) for x, y in zip(self.W_hhs,self.nW_hhs)]
            if not no_noise_bias:
                b_hhs = [(x+y) for x, y in zip(self.b_hhs,self.nb_hhs)]
            else:
                b_hhs = self.b_hhs
            W_shp = [(x+y) for x, y in zip(self.W_shortp,self.nW_shortp)]
        else:
            W_hhs = self.W_hhs
            b_hhs = self.b_hhs
            W_shp = self.W_shortp
        h = self.activation[0](TT.dot(state_before,
                                      W_hhs[0])+state_below)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout
        rval += [h]
        for dx in xrange(1, self.n_layers):
            h = self.activation[dx](TT.dot(h,
                                           W_hhs[dx])+
                                    TT.dot(state_before,
                                           W_shp[dx-1])+b_hhs[dx-1])
            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
            if self.dropout < 1.:
                if use_noise:
                    if h.ndim == 2:
                        h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                        dpidx = dpidx + h.shape[1]
                    else:
                        h = h * dpmask[dpidx:dpidx+h.shape[0]]
                        dpidx = dpidx + h.shape[0]
                else:
                    h = h * self.dropout
            rval += [h]

        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        rval[-1] = h
        return rval

class RecurrentMultiLayerShortPathInp(RecurrentMultiLayer):
    """
    Similar to the RecurrentMultiLayerShortPath class, just that the input
    is fed into the last layer of the MLP (similar to
    RecurrentMultiLayerInp).
    """

    def _init_params(self):
        self.W_hhs = []
        self.b_hhs = []
        self.W_shortp = []
        for dx in xrange(self.n_layers):
            W_hh = self.init_fn[dx](self.n_hids[(dx-1)%self.n_layers],
                                        self.n_hids[dx],
                                        self.sparsity[dx],
                                        self.scale[dx],
                                        rng=self.rng)
            self.W_hhs.append(theano.shared(value=W_hh, name="W%d_%s" %
                                       (dx,self.name)))

            if dx > 0:
                W_shp = self.init_fn[dx](self.n_hids[self.n_layers-1],
                                         self.n_hids[dx],
                                         self.sparsity[dx],
                                         self.scale[dx],
                                         rng=self.rng)
                self.W_shortp.append(theano.shared(value=W_shp,
                                               name='W_s%d_%s'%(dx,self.name)))
            if dx < self.n_layers-1:
                self.b_hhs.append(theano.shared(
                    self.bias_fn[dx](self.n_hids[dx],
                                       self.bias_scale[dx],
                                       self.rng),
                    name='b%d_%s' %(dx, self.name)))
        self.params = [x for x in self.W_hhs] + [x for x in self.b_hhs] +\
                [x for x in self.W_shortp]
        self.restricted_params = [x for x in self.params]
        self.params_grad_scale = [self.grad_scale for x in self.params]
        if self.weight_noise:
            self.nW_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_hhs]
            self.nb_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.b_hhs]
            self.nW_shortp = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_shortp]

            self.noise_params = [x for x in self.nW_hhs] + [x for x in self.nb_hhs] + [x for x in self.nW_shortp]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape) for x in self.noise_params]



    def step_fprop(self,
                   state_below,
                   mask=None,
                   dpmask=None,
                   state_before=None,
                   no_noise_bias=False,
                   use_noise=True):
        """
        See parent class
        """
        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hhs = [(x+y) for x, y in zip(self.W_hhs, self.nW_hhs)]
            if not no_noise_bias:
                b_hhs = [(x+y) for x, y in zip(self.b_hhs, self.nb_hhs)]
            else:
                b_hhs = self.b_hhs
            W_shp = [(x+y) for x, y in zip(self.W_shortp, self.nW_shortp)]
        else:
            W_hhs = self.W_hhs
            b_hhs = self.b_hhs
            W_shp = self.W_shortp
        h = self.activation[0](TT.dot(state_before,
                                      W_hhs[0])+b_hhs[0])
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout
        rval += [h]
        for dx in xrange(1, self.n_layers-1):
            h = self.activation[dx](TT.dot(h,
                                           W_hhs[dx])+
                                    TT.dot(state_before,
                                           W_shp[dx-1])+b_hhs[dx])
            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
            if self.dropout < 1.:
                if use_noise:
                    if h.ndim == 2:
                        h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                        dpidx = dpidx + h.shape[1]
                    else:
                        h = h * dpmask[dpidx:dpidx+h.shape[0]]
                        dpidx = dpidx + h.shape[0]
                else:
                    h = h * self.dropout
            rval += [h]

        h = self.activation[-1](TT.dot(h, W_hhs[-1]) +
                                TT.dot(state_before, W_shp[-1])+state_below)
        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout

        rval +=[h]
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        rval += [h]
        return rval

class RecurrentMultiLayerShortPathInpAll(RecurrentMultiLayer):
    """
    Similar to RecurrentMultiLayerShortPathInp class, just that the input is
    fed to all layers of the MLP depicting the deep transition between h_tm1
    to h_t.
    """
    def _init_params(self):
        self.W_hhs = []
        self.W_shortp = []
        self.lp_power = []
        for dx in xrange(self.n_layers):
            n_in = self.n_hids[(dx-1)%self.n_layers]
            if type(n_in) in (list, tuple):
                n_in = n_in[1]
            n_out = self.n_hids[dx]
            if type(n_out) in (list, tuple):
                n_out = n_out[0]
            W_hh = self.init_fn[dx](n_in,
                                    n_out,
                                    self.sparsity[dx],
                                    self.scale[dx],
                                    rng=self.rng)
            self.W_hhs.append(theano.shared(value=W_hh, name="W%d_%s" %
                                       (dx,self.name)))

            if self.lp_units[dx]:
                p = theano.shared(self.init_power[dx](self.n_hids[dx][1],
                                                      self.power_scale[dx],
                                                      self.rng),
                                  name="p%d_%s"%(dx, self.name))

                self.lp_power.append(p)
            else:
                self.lp_power.append(None)
            if dx > 0:
                n_in = self.n_hids[self.n_layers-1]
                if type(n_in) in (list, tuple):
                    n_in = n_in[1]
                n_out = self.n_hids[dx]
                if type(n_out) in (list, tuple):
                    n_out = n_out[0]
                W_shp = self.init_fn[dx](n_in,
                                         n_out,
                                         self.sparsity[dx],
                                         self.scale[dx],
                                         rng=self.rng)
                self.W_shortp.append(theano.shared(value=W_shp,
                                               name='W_s%d_%s'%(dx,self.name)))
        self.params = [x for x in self.W_hhs] +\
                [x for x in self.W_shortp] +\
                [x for x in self.lp_power if x is not None]

        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]

        if self.weight_noise:
            self.nW_hhs = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_hhs]
            self.nW_shortp = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_shortp]

            self.noise_params = [x for x in self.nW_hhs] + [x for x in self.nW_shortp]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape) for x in self.noise_params]


    def step_fprop(self,
                   state_below,
                   mask=None,
                   dpmask=None,
                   state_before=None,
                   no_noise_bias=False,
                   use_noise=True):
        """
        See parent class
        """
        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hhs = [(x+y) for x, y in zip(self.W_hhs,self.nW_hhs)]
            W_shp = [(x+y) for x, y in zip(self.W_shortp,self.nW_shortp)]
        else:
            W_hhs = self.W_hhs
            W_shp = self.W_shortp
        def slice_state_below(dx, sb = state_below):
            st = 0
            for p in xrange(dx):
                val = self.n_hids[p]
                if isinstance(val, (list, tuple)):
                     st += val[0]
                else:
                    st += val
            val = self.n_hids[dx]
            if isinstance(val, (list, tuple)):
                ed = st + val[0]
            else:
                ed = st + val
            if sb.ndim == 1:
                return sb[st:ed]
            else:
                return sb[:,st:ed]


        preactiv = TT.dot(state_before, W_hhs[0]) + slice_state_below(0)

        if self.lp_units[0]:
            # We are dealing with an lp unit that has additional arguments
            h = self.activation[0](preactiv, self.lp_power[0],
                                   self.n_hids[0])
        elif isinstance(self.n_hids[0], (list, tuple)):
            h = self.activation[0](preactiv, self.n_hids[0])
        else:
            h = self.activation[0](preactiv)

        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)

        if self.dropout < 1.:
            if use_noise:
                if h.ndim == 2:
                    h = h * dpmask[:,:h.shape[1]]
                    dpidx = h.shape[1]
                else:
                    h = h * dpmask[:h.shape[0]]
                    dpidx = h.shape[0]
            else:
                h = h * self.dropout

        rval += [h]
        for dx in xrange(1, self.n_layers):
            preactiv = TT.dot(h, W_hhs[dx]) + \
                    TT.dot(state_before, W_shp[dx-1]) + \
                    slice_state_below(dx)


            if self.lp_units[dx]:
                h = self.activation[dx](preactiv, self.lp_power[dx],
                                        self.n_hids[dx])
            elif isinstance(self.n_hids[dx], (list, tuple)):
                h = self.activation[dx](preactiv, self.n_hids[dx])
            else:
                h = self.activation[dx](preactiv)

            if self.activ_noise and use_noise:
                h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
            if self.dropout < 1.:
                if use_noise:
                    if h.ndim == 2:
                        h = h * dpmask[:,dpidx:dpidx+h.shape[1]]
                        dpidx = dpidx + h.shape[1]
                    else:
                        h = h * dpmask[dpidx:dpidx+h.shape[0]]
                        dpidx = dpidx + h.shape[0]
                else:
                    h = h * self.dropout
            rval += [h]

        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        rval[-1] = h
        return rval


