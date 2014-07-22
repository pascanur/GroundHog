"""
Recursive Convolutional layers.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("KyungHyun Cho ")
__contact__ = "Kyunghyun Cho <kyunghyun.cho@umontreal.ca>"

import numpy
import copy
import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog import utils
from groundhog.utils import sample_weights, \
        sample_weights_classic,\
        sample_weights_orth, \
        init_bias, \
        constant_shape, \
        sample_zeros
from basic import Layer


class RecursiveConvolutionalLayer(Layer):
    """
        (Binary) Recursive Convolutional Layer
    """
    def __init__(self, rng,
                 n_hids=500,
                 scale=.01,
                 sparsity = -1,
                 activation = TT.tanh,
                 activ_noise=0.,
                 weight_noise=False,
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 dropout = 1.,
                 init_fn='sample_weights',
                 kind_reg = None,
                 grad_scale = 1.,
                 profile = 0,
                 gating = False, # NOT USED
                 reseting = False, # NOT USED
                 gater_activation = TT.nnet.sigmoid, # NOT USED
                 reseter_activation = TT.nnet.sigmoid, # NOT USED
                 name=None):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: int
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

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)

        self.scale = scale
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

        super(RecursiveConvolutionalLayer, self).__init__(self.n_hids,
                self.n_hids, rng, name)

        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.params = []
        self._init_params()

    def _init_params(self):
        # Left weight matrix
        self.W_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="W_%s"%self.name)
        self.params = [self.W_hh]
        # Right weight matrix
        self.U_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="U_%s"%self.name)
        self.params += [self.U_hh]
        # Bias
        self.b_hh = theano.shared(
            self.bias_fn(self.n_hids,
                self.bias_scale,
                self.rng),
            name='b_%s' %self.name)
        self.params += [self.b_hh]
        # gaters
        self.GW_hh = theano.shared(
                numpy.float32(0.01 * self.rng.randn(self.n_hids, 3)),
                name="GW_%s"%self.name)
        self.params += [self.GW_hh]
        self.GU_hh = theano.shared(
                numpy.float32(0.01 * self.rng.randn(self.n_hids, 3)),
                name="GU_%s"%self.name)
        self.params += [self.GU_hh]
        self.Gb_hh = theano.shared(
            self.bias_fn(3,
                self.bias_scale,
                self.rng),
            name='Gb_%s' %self.name)
        self.params += [self.Gb_hh]

        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]
        if self.weight_noise:
            self.nW_hh = theano.shared(self.W_hh.get_value()*0, name='noise_'+self.W_hh.name)
            self.nU_hh = theano.shared(self.U_hh.get_value()*0, name='noise_'+self.U_hh.name)
            self.nb_hh = theano.shared(self.b_hh.get_value()*0, name='noise_'+self.b_hh.name)
            self.noise_params = [self.nW_hh,self.nU_hh,self.nb_hh]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]

    def step_fprop(self, mask_t, prev_level, return_gates = False):
        if self.weight_noise and use_noise and self.noise_params:
            W_hh = self.W_hh + self.nW_hh
            U_hh = self.U_hh + self.nU_hh
            b_hh = self.b_hh + self.nb_hh
        else:
            W_hh = self.W_hh
            U_hh = self.U_hh
            b_hh = self.b_hh
        GW_hh = self.GW_hh
        GU_hh = self.GU_hh
        Gb_hh = self.Gb_hh

        if prev_level.ndim == 3:
            b_hh = b_hh.dimshuffle('x','x',0)
        else:
            b_hh = b_hh.dimshuffle('x',0)
        lower_level = prev_level

        prev_shifted = TT.zeros_like(prev_level)
        prev_shifted = TT.set_subtensor(prev_shifted[1:], prev_level[:-1])
        lower_shifted = prev_shifted

        prev_shifted = TT.dot(prev_shifted, U_hh)
        prev_level = TT.dot(prev_level, W_hh)
        new_act = self.activation(prev_level + prev_shifted + b_hh)

        gater = TT.dot(lower_shifted, GU_hh) + \
                TT.dot(lower_level, GW_hh) + Gb_hh
        if prev_level.ndim == 3:
            gater_shape = gater.shape
            gater = gater.reshape((gater_shape[0] * gater_shape[1], 3))
        gater = TT.nnet.softmax(gater)
        if prev_level.ndim == 3:
            gater = gater.reshape((gater_shape[0], gater_shape[1], 3))

        if prev_level.ndim == 3:
            gater_new = gater[:,:,0].dimshuffle(0,1,'x')
            gater_left = gater[:,:,1].dimshuffle(0,1,'x')
            gater_right = gater[:,:,2].dimshuffle(0,1,'x')
        else:
            gater_new = gater[:,0].dimshuffle(0,'x')
            gater_left = gater[:,1].dimshuffle(0,'x')
            gater_right = gater[:,2].dimshuffle(0,'x')

        act = new_act * gater_new + \
                lower_shifted * gater_left + \
                lower_level * gater_right

        if mask_t:
            if prev_level.ndim == 3:
                mask_t = mask_t.dimshuffle('x',0,'x')
            else:
                mask_t = mask_t.dimshuffle('x', 0)
            new_level = TT.switch(mask_t, act, lower_level)
        else:
            new_level = act

        if return_gates:
            return new_level, gater

        return new_level

    def fprop(self,
              state_below,
              mask=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias = False,
              **kwargs
             ):

        if theano.config.floatX=='float32':
            floatX = numpy.float32
        else:
            floatX = numpy.float64
        if nsteps is None:
            nsteps = state_below.shape[0]
            if batch_size and batch_size != 1:
                nsteps = nsteps / batch_size
        if batch_size is None and state_below.ndim == 3:
            batch_size = state_below.shape[1]
        if state_below.ndim == 2 and \
           (not isinstance(batch_size,int) or batch_size > 1):
            state_below = state_below.reshape((nsteps, batch_size, self.n_in))
        if mask == None:
            mask = TT.alloc(1., nsteps, 1)

        rval = []

        rval, updates = theano.scan(self.step_fprop,
                        sequences = [mask[1:]],
                        outputs_info = [state_below],
                        name='layer_%s'%self.name,
                        profile=self.profile,
                        n_steps = nsteps-1)

        seqlens = TT.cast(mask.sum(axis=0), 'int64')-1
        roots = rval[-1]

        if state_below.ndim == 3:
            def _grab_root(seqlen,one_sample,prev_sample):
                return one_sample[seqlen]

            roots, updates = theano.scan(_grab_root,
                    sequences = [seqlens, roots.dimshuffle(1,0,2)],
                    outputs_info = [TT.alloc(0., self.n_hids)],
                    name='grab_root_%s'%self.name,
                    profile=self.profile)
            roots = roots.dimshuffle('x', 0, 1)
        else:
            roots = roots[seqlens] # there should be only one, so it's fine.

        # Note that roots has only a single timestep
        new_h = roots
        self.out = roots
        self.rval = roots
        self.updates =updates

        return self.out


