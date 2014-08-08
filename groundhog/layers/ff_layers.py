"""
Feedforward layers.


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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog import utils
from groundhog.utils import sample_weights, \
            sample_weights_classic,\
            init_bias, \
            constant_shape, \
            sample_zeros
from basic import Layer


class MultiLayer(Layer):
    """
    Implementing a standard feed forward MLP
    """
    def __init__(self,
                 rng,
                 n_in,
                 n_hids=[500,500],
                 activation='TT.tanh',
                 scale=0.01,
                 sparsity=-1,
                 rank_n_approx=0,
                 rank_n_activ='lambda x: x',
                 weight_noise=False,
                 dropout = 1.,
                 init_fn='sample_weights_classic',
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 learn_bias = True,
                 grad_scale = 1.,
                 name=None):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: list of ints
        :param n_hids: Number of hidden units on each layer of the MLP

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

        :type rank_n_approx: int
        :param rank_n_approx: It applies to the first layer only. If
            positive and larger than 0, the first weight matrix is
            factorized into two matrices. The first one goes from input to
            `rank_n_approx` hidden units, the second from `rank_n_approx` to
            the number of units on the second layer

        :type rank_n_activ: string or function
        :param rank_n_activ: Function that is applied on on the intermediary
            layer formed from factorizing the first weight matrix (Q: do we
            need this?)

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

        :type learn_bias: bool
        :param learn_bias: flag, saying if we should learn the bias or keep
            it constant


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

        assert rank_n_approx >= 0, "Please enter a valid rank_n_approx"
        self.rank_n_approx = rank_n_approx

        if isinstance(rank_n_activ,  (str, unicode)):
            rank_n_activ = eval(rank_n_activ)
        self.rank_n_activ = rank_n_activ
        if type(n_hids) not in (list, tuple):
            n_hids = [n_hids]
        n_layers = len(n_hids)
        self.n_layers = n_layers
        if type(scale) not in (list, tuple):
            scale = [scale] * n_layers
        if type(sparsity) not in (list, tuple):
            sparsity = [sparsity] * n_layers
        for idx, sp in enumerate(sparsity):
            if sp < 0: sparsity[idx] = n_hids[idx]
        if type(activation) not in (list, tuple):
            activation = [activation] * n_layers
        if type(bias_scale) not in (list, tuple):
            bias_scale = [bias_scale] * n_layers
        if bias_fn not in (list, tuple):
            bias_fn = [bias_fn] * n_layers
        if init_fn not in (list, tuple):
            init_fn = [init_fn] * n_layers

        for dx in xrange(n_layers):
            if isinstance(bias_fn[dx],  (str, unicode)):
                bias_fn[dx] = eval(bias_fn[dx])
            if isinstance(init_fn[dx], (str, unicode)):
                init_fn[dx] = eval(init_fn[dx])
            if isinstance(activation[dx], (str, unicode)):
                activation[dx] = eval(activation[dx])
        super(MultiLayer, self).__init__(n_in, n_hids[-1], rng, name)
        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.activation = activation
        self.scale = scale
        self.sparsity = sparsity
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self._grad_scale = grad_scale
        self.weight_noise = weight_noise
        self.dropout = dropout
        self.n_hids = n_hids
        self.learn_bias = learn_bias
        self._init_params()

    def _init_params(self):
        """
        Initialize the parameters of the layer, either by using sparse initialization or small
        isotropic noise.
        """
        self.W_ems = []
        self.b_ems = []
        if self.rank_n_approx:
            W_em1 = self.init_fn[0](self.n_in,
                                 self.rank_n_approx,
                                 self.sparsity[0],
                                 self.scale[0],
                                 self.rng)
            W_em2 = self.init_fn[0](self.rank_n_approx,
                                 self.n_hids[0],
                                 self.sparsity[0],
                                 self.scale[0],
                                 self.rng)
            self.W_em1 = theano.shared(W_em1,
                                       name='W1_0_%s'%self.name)
            self.W_em2 = theano.shared(W_em2,
                                       name='W2_0_%s'%self.name)
            self.W_ems = [self.W_em1, self.W_em2]

        else:
            W_em = self.init_fn[0](self.n_in,
                                self.n_hids[0],
                                self.sparsity[0],
                                self.scale[0],
                                self.rng)
            self.W_em = theano.shared(W_em,
                                      name='W_0_%s'%self.name)
            self.W_ems = [self.W_em]

        self.b_em = theano.shared(
            self.bias_fn[0](self.n_hids[0], self.bias_scale[0],self.rng),
            name='b_0_%s'%self.name)
        self.b_ems = [self.b_em]

        for dx in xrange(1, self.n_layers):
            W_em = self.init_fn[dx](self.n_hids[dx-1] / self.pieces[dx],
                                self.n_hids[dx],
                                self.sparsity[dx],
                                self.scale[dx],
                                self.rng)
            W_em = theano.shared(W_em,
                                      name='W_%d_%s'%(dx,self.name))
            self.W_ems += [W_em]

            b_em = theano.shared(
                self.bias_fn[dx](self.n_hids[dx], self.bias_scale[dx],self.rng),
                name='b_%d_%s'%(dx,self.name))
            self.b_ems += [b_em]

        self.params = [x for x in self.W_ems]

        if self.learn_bias and self.learn_bias!='last':
            self.params = [x for x in self.W_ems] + [x for x in self.b_ems]
        elif self.learn_bias == 'last':
            self.params = [x for x in self.W_ems] + [x for x in
                                                     self.b_ems][:-1]
        self.params_grad_scale = [self._grad_scale for x in self.params]
        if self.weight_noise:
            self.nW_ems = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.W_ems]
            self.nb_ems = [theano.shared(x.get_value()*0, name='noise_'+x.name) for x in self.b_ems]

            self.noise_params = [x for x in self.nW_ems] + [x for x in self.nb_ems]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]


    def fprop(self, state_below, use_noise=True, no_noise_bias=False,
            first_only = False):
        """
        Constructs the computational graph of this layer.
        If the input is ints, we assume is an index, otherwise we assume is
        a set of floats.
        """
        if self.weight_noise and use_noise and self.noise_params:
            W_ems = [(x+y) for x, y in zip(self.W_ems, self.nW_ems)]
            if not no_noise_bias:
                b_ems = [(x+y) for x, y in zip(self.b_ems, self.nb_ems)]
            else:
                b_ems = self.b_ems
        else:
            W_ems = self.W_ems
            b_ems = self.b_ems
        if self.rank_n_approx:
            if first_only:
                emb_val = self.rank_n_activ(utils.dot(state_below, W_ems[0]))
                self.out = emb_val
                return emb_val
            emb_val = TT.dot(
                    self.rank_n_activ(utils.dot(state_below, W_ems[0])),
                    W_ems[1])
            if b_ems:
                emb_val += b_ems[0]
            st_pos = 1
        else:
            emb_val = utils.dot(state_below, W_ems[0])
            if b_ems:
                emb_val += b_ems[0]
            st_pos = 0


        emb_val = self.activation[0](emb_val)

        if self.dropout < 1.:
            if use_noise:
                emb_val = emb_val * self.trng.binomial(emb_val.shape, n=1, p=self.dropout, dtype=emb_val.dtype)
            else:
                emb_val = emb_val * self.dropout
        for dx in xrange(1, self.n_layers):
            emb_val = utils.dot(emb_val, W_ems[st_pos+dx])
            if b_ems:
                emb_val = self.activation[dx](emb_val+ b_ems[dx])
            else:
                emb_val = self.activation[dx](emb_val)

            if self.dropout < 1.:
                if use_noise:
                    emb_val = emb_val * self.trng.binomial(emb_val.shape, n=1, p=self.dropout, dtype=emb_val.dtype)
                else:
                    emb_val = emb_val * self.dropout
        self.out = emb_val
        return emb_val

class LastState(Layer):
    """
    This layer is used to construct the embedding of the encoder by taking
    the last state of the recurrent model
    """
    def __init__(self, ntimes = False, n = TT.constant(0)):
        """
        :type ntimes: bool
        :param ntimes: If the last state needs to be repeated `n` times

        :type n: int, theano constant, None
        :param n: how many times the last state is repeated
        """
        self.ntimes = ntimes
        self.n = n
        super(LastState, self).__init__(0, 0, None)

    def fprop(self, all_states):
        if self.ntimes:
            stateshape0 = all_states.shape[0]
            shape0 = TT.switch(TT.gt(self.n, 0), self.n, all_states.shape[0])

            single_frame = TT.shape_padleft(all_states[stateshape0-1])
            mask = TT.alloc(numpy.float32(1), shape0, *[1 for k in xrange(all_states.ndim-1)])
            rval = single_frame * mask
            self.out = rval
            return rval

        single_frame = all_states[all_states.shape[0]-1]
        self.out = single_frame
        return single_frame

last = LastState()
last_ntimes = LastState(ntimes=True)

class GaussianNoise(Layer):
    """
    This layer is used to construct the embedding of the encoder by taking
    the last state of the recurrent model
    """
    def __init__(self, rng, std = 0.1, ndim=0, avg =0, shape_fn=None):
        """
        """
        assert rng is not None, "random number generator should not be empty!"
        super(GaussianNoise, self).__init__(0, 0, rng)

        self.std = scale
        self.avg = self.avg
        self.ndim = ndim
        self.shape_fn = shape_fn
        if self.shape_fn:
            # Name is not important as it is not a parameter of the model
            self.noise_term = theano.shared(numpy.zeros((2,)*ndim,
                                                    dtype=theano.config.floatX),
                                        name='ndata')
            self.noise_params += [self.noise_term]
            self.noise_params_shape_fn += [shape_fn]
        self.trng = RandomStreams(rng.randint(1e5))

    def fprop(self, x):
        self.out = x
        if self.scale:
            if self.shape_fn:
                self.out += self.noise_term
            else:
                self.out += self.trng.normal(self.out.shape, std=self.std,
                                             avg = self.avg,
                                        dtype=self.out.dtype)
        return self.out

class BinaryOp(Layer):
    """
    This layer is used to construct the embedding of the encoder by taking
    the last state of the recurrent model
    """
    def __init__(self, op = 'lambda x,y: x+y', name=None):
        if type(op) is str:
            op = eval(op)
        self.op = op
        super(BinaryOp, self).__init__(0, 0, None, name)

    def fprop(self, x, y):
        self.out = self.op(x, y)
        return self.out

class DropOp(Layer):
    """
    This layers randomly drops elements of the input by multiplying with a
    mask sampled from a binomial distribution
    """
    def __init__(self, rng = None, name=None, dropout=1.):
        super(DropOp, self).__init__(0, 0, None, name)
        self.dropout = dropout
        if dropout < 1.:
            self.trng = RandomStreams(rng.randint(1e5))

    def fprop(self, state_below, use_noise = True):
        self.out = state_below
        if self.dropout < 1.:
            if use_noise:
                self.out = self.out * self.trng.binomial(self.out.shape,
                                                         n=1,
                                                         p=self.dropout,
                                                         dtype=self.out.dtype)
            else:
                self.out = self.out * self.dropout
        return self.out

class UnaryOp(Layer):
    """
    This layer is used to construct an embedding of the encoder by doing a
    max pooling over the hidden state
    """
    def __init__(self, activation = 'lambda x: x', name=None):
        if type(activation) is str:
            activation = eval(activation)
        self.activation = activation
        super(UnaryOp, self).__init__(0, 0, None, name)

    def fprop(self, state_below):
        self.out = self.activation(state_below)
        return self.out

tanh = UnaryOp('lambda x: TT.tanh(x)')
sigmoid = UnaryOp('lambda x: TT.nnet.sigmoid(x)')
rectifier = UnaryOp('lambda x: x*(x>0)')
hard_sigmoid = UnaryOp('lambda x: x*(x>0)*(x<1)')
hard_tanh = UnaryOp('lambda x: x*(x>-1)*(x<1)')


class Shift(Layer):
    """
    This layer is used to construct the embedding of the encoder by taking
    the last state of the recurrent model
    """
    def __init__(self, n=1, name=None):
        self.n = n
        super(Shift, self).__init__(0, 0, None, name)

    def fprop(self, var):
        rval = TT.zeros_like(var)
        if self.n >0:
            rval = TT.set_subtensor(rval[self.n:], var[:-self.n])
        elif self.n<0:
            rval = TT.set_subtensor(rval[:self.n], var[-self.n:])
        self.out = rval
        return rval

class MinPooling(Layer):
    """
    This layer is used to construct an embedding of the encoder by doing a
    max pooling over the hidden state
    """
    def __init__(self, ntimes=False, name=None):
        self.ntimes = ntimes
        super(MinPooling, self).__init__(0, 0, None, name)

    def fprop(self, all_states):
        shape0 = all_states.shape[0]
        single_frame = all_states.min(0)

        if self.ntimes:
            single_frame = TT.shape_padleft(all_states.max(0))
            mask = TT.alloc(numpy.float32(1),
                        shape0, *[1 for k in xrange(all_states.ndim-1)])
            rval = single_frame * mask
            self.out = rval
            return rval
        self.out = single_frame
        return single_frame

minpool = MinPooling()
minpool_ntimes = MinPooling(ntimes=True)

class MaxPooling(Layer):
    """
    This layer is used to construct an embedding of the encoder by doing a
    max pooling over the hidden state
    """
    def __init__(self, ntimes=False, name=None):
        self.ntimes = ntimes
        super(MaxPooling, self).__init__(0, 0, None, name)

    def fprop(self, all_states):
        shape0 = all_states.shape[0]
        single_frame = all_states.max(0)

        if self.ntimes:
            single_frame = TT.shape_padleft(all_states.max(0))
            mask = TT.alloc(numpy.float32(1),
                        shape0, *[1 for k in xrange(all_states.ndim-1)])
            rval = single_frame * mask
            self.out = rval
            return rval
        self.out = single_frame
        return single_frame

maxpool = MaxPooling()
maxpool_ntimes = MaxPooling(ntimes=True)

class Concatenate(Layer):

    def __init__(self, axis):
        self.axis = axis
        Layer.__init__(self, 0, 0, None)

    def fprop(self, *args):
        self.out = TT.concatenate(args, axis=self.axis)
        return self.out
