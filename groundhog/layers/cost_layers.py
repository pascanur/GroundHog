"""
Cost layers.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import copy
import logging
import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog import utils
from groundhog.utils import sample_weights, sample_weights_classic,\
    init_bias, constant_shape, sample_zeros

from basic import Layer

logger = logging.getLogger(__name__)

class CostLayer(Layer):
    """
    Base class for all cost layers
    """
    def __init__(self, rng,
                 n_in,
                 n_out,
                 scale,
                 sparsity,
                 rank_n_approx=0,
                 rank_n_activ='lambda x: x',
                 weight_noise=False,
                 init_fn='sample_weights_classic',
                 bias_fn='init_bias',
                 bias_scale=0.,
                 sum_over_time=True,
                 additional_inputs=None,
                 grad_scale=1.,
                 use_nce=False,
                 name=None):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator used to sample weights

        :type n_in: int
        :param n_in: number of input units

        :type n_out: int
        :param n_out: number of output units

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
            (and the right shared variable are constructed, to keep track
            of the noise)

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

        :type sum_over_time: bool
        :param sum_over_time: flag, stating if, when computing the cost, we
            should take the sum over time, or the mean. If you have variable
            length sequences, please take the sum over time

        :type additional_inputs: None or list of ints
        :param additional_inputs: dimensionality of each additional input

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type use_nce: bool
        :param use_nce: flag, if true, do not use MLE, but NCE-like cost

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.
        """
        self.grad_scale = grad_scale
        assert rank_n_approx >= 0, "Please enter a valid rank_n_approx"
        self.rank_n_approx = rank_n_approx
        if type(rank_n_activ) is str:
            rank_n_activ = eval(rank_n_activ)
        self.rank_n_activ = rank_n_activ
        super(CostLayer, self).__init__(n_in, n_out, rng, name)
        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.scale = scale
        if isinstance(bias_fn, str):
            self.bias_fn = eval(bias_fn)
        else:
            self.bias_fn = bias_fn
        self.bias_scale = bias_scale
        self.sum_over_time = sum_over_time
        self.weight_noise = weight_noise
        self.sparsity = sparsity
        if self.sparsity < 0:
            self.sparsity = n_out
        if type(init_fn) is str:
            init_fn = eval(init_fn)
        self.init_fn = init_fn
        self.additional_inputs = additional_inputs
        self.use_nce = use_nce
        self._init_params()

    def _init_params(self):
        """
        Initialize the parameters of the layer, either by using sparse
        initialization or small isotropic noise.
        """
        if self.rank_n_approx:
            W_em1 = self.init_fn(self.n_in,
                                 self.rank_n_approx,
                                 self.sparsity,
                                 self.scale,
                                 self.rng)
            W_em2 = self.init_fn(self.rank_n_approx,
                                 self.n_out,
                                 self.sparsity,
                                 self.scale,
                                 self.rng)
            self.W_em1 = theano.shared(W_em1,
                                       name='W1_%s' % self.name)
            self.W_em2 = theano.shared(W_em2,
                                       name='W2_%s' % self.name)
            self.b_em = theano.shared(
                self.bias_fn(self.n_out, self.bias_scale, self.rng),
                name='b_%s' % self.name)
            self.params += [self.W_em1, self.W_em2, self.b_em]

            if self.weight_noise:
                self.nW_em1 = theano.shared(W_em1*0.,
                                            name='noise_W1_%s' % self.name)
                self.nW_em2 = theano.shared(W_em*0.,
                                            name='noise_W2_%s' % self.name)
                self.nb_em = theano.shared(b_em*0.,
                                           name='noise_b_%s' % self.name)
                self.noise_params = [self.nW_em1, self.nW_em2, self.nb_em]
                self.noise_params_shape_fn = [
                    constant_shape(x.get_value().shape)
                    for x in self.noise_params]

        else:
            W_em = self.init_fn(self.n_in,
                                self.n_out,
                                self.sparsity,
                                self.scale,
                                self.rng)
            self.W_em = theano.shared(W_em,
                                      name='W_%s' % self.name)
            self.b_em = theano.shared(
                self.bias_fn(self.n_out, self.bias_scale, self.rng),
                name='b_%s' % self.name)

            self.params += [self.W_em, self.b_em]
            if self.weight_noise:
                self.nW_em = theano.shared(W_em*0.,
                                           name='noise_W_%s' % self.name)
                self.nb_em = theano.shared(
                    numpy.zeros((self.n_out,), dtype=theano.config.floatX),
                    name='noise_b_%s' % self.name)
                self.noise_params = [self.nW_em, self.nb_em]
                self.noise_params_shape_fn = [
                    constant_shape(x.get_value().shape)
                    for x in self.noise_params]
        self.additional_weights = []
        self.noise_additional_weights = []
        if self.additional_inputs:
            for pos, size in enumerate(self.additional_inputs):
                W_add = self.init_fn(size,
                                    self.n_out,
                                    self.sparsity,
                                    self.scale,
                                    self.rng)
                self.additional_weights += [theano.shared(W_add,
                                  name='W_add%d_%s'%(pos, self.name))]
                if self.weight_noise:
                    self.noise_additional_weights += [
                        theano.shared(W_add*0.,
                                      name='noise_W_add%d_%s'%(pos, self.name))]
        self.params = self.params + self.additional_weights
        self.noise_params += self.noise_additional_weights
        self.noise_params_shape_fn += [
            constant_shape(x.get_value().shape)
            for x in self.noise_additional_weights]

        self.params_grad_scale = [self.grad_scale for x in self.params]

    def compute_sample(self, state_below, temp=1, use_noise=False):
        """
        Constructs the theano expression that samples from the output layer.

        :type state_below: tensor or layer
        :param state_below: The theano expression (or groundhog layer)
            representing the input of the cost layer

        :type temp: float or tensor scalar
        :param temp: scalar representing the temperature that should be used
            when sampling from the output distribution

        :type use_noise: bool
        :param use_noise: flag. If true, noise is used when computing the
            output of the model
        """
        raise NotImplemented

    def get_cost(self,
                 state_below,
                 target=None,
                 mask=None,
                 temp=1,
                 reg=None,
                 scale=None,
                 sum_over_time=None,
                 use_noise=True,
                 additional_inputs=None,
                 no_noise_bias=False):
        """
        Computes the expression of the cost of the model (given the type of
        layer used).

        :type state_below: tensor or layer
        :param state_below: The theano expression (or groundhog layer)
            representing the input of the cost layer

        :type target: tensor or layer
        :param target: The theano expression (or groundhog layer)
            representing the target (used to evaluate the prediction of the
            output layer)

        :type mask: None or mask or layer
        :param mask: Mask, depicting which of the predictions should be
            ignored (e.g. due to them resulting from padding a sequence
            with 0s)

        :type temp: float or tensor scalar
        :param temp: scalar representing the temperature that should be used
            when sampling from the output distribution

        :type reg: None or layer or theano scalar expression
        :param reg: additional regularization term that should be added to
            the cost

        :type scale: float or None or theano scalar
        :param scale: scaling factor with which the cost is multiplied

        :type sum_over_time: bool or None
        :param sum_over_time: this flag overwrites the value given to this
            property in the constructor of the class

        :type use_noise: bool
        :param use_noise: flag. If true, noise is used when computing the
            output of the model

        :type additional_inputs: list theano variable or layers
        :param additional_inputs: list of theano variables or layers
            representing the additional inputs

        :type no_noise_bias: bool
        :param no_noise_bias: flag, stating if weight noise should be added
            to the bias as well, or only to the weights
        """
        raise NotImplemented

    def get_grads(self,
                  state_below,
                  target=None,
                  mask=None,
                  temp=1,
                  reg=None,
                  scale=None,
                  additional_gradients=None,
                  sum_over_time=None,
                  use_noise=True,
                  additional_inputs=None,
                  no_noise_bias=False):
        """
        Computes the expression of the gradients of the cost with respect to
        all parameters of the model.

        :type state_below: tensor or layer
        :param state_below: The theano expression (or groundhog layer)
            representing the input of the cost layer

        :type target: tensor or layer
        :param target: The theano expression (or groundhog layer)
            representing the target (used to evaluate the prediction of the
            output layer)

        :type mask: None or mask or layer
        :param mask: Mask, depicting which of the predictions should be
            ignored (e.g. due to them resulting from padding a sequence
            with 0s)

        :type temp: float or tensor scalar
        :param temp: scalar representing the temperature that should be used
            when sampling from the output distribution

        :type reg: None or layer or theano scalar expression
        :param reg: additional regularization term that should be added to
            the cost

        :type scale: float or None or theano scalar
        :param scale: scaling factor with which the cost is multiplied

        :type additional_gradients: list of tuples of the form
            (param, gradient)
        :param additional_gradiens: A list of tuples. Each tuple has as its
            first element the parameter, and as second element a gradient
            expression that should be added to the gradient resulting from the
            cost. Not all parameters need to have an additional gradient.

        :type sum_over_time: bool or None
        :param sum_over_time: this flag overwrites the value given to this
            property in the constructor of the class

        :type use_noise: bool
        :param use_noise: flag. If true, noise is used when computing the
            output of the model

        :type no_noise_bias: bool
        :param no_noise_bias: flag, stating if weight noise should be added
            to the bias as well, or only to the weights
        """
        cost = self.get_cost(state_below,
                             target,
                             mask=mask,
                             reg=reg,
                             scale=scale,
                             sum_over_time=sum_over_time,
                             use_noise=use_noise,
                             additional_inputs=additional_inputs,
                             no_noise_bias=no_noise_bias)
        logger.debug("Get grads")
        grads = TT.grad(cost.mean(), self.params)
        logger.debug("Got grads")
        if additional_gradients:
            for p, gp in additional_gradients:
                if p in self.params:
                    grads[self.params.index(p)] += gp
        if self.additional_gradients:
            for new_grads, to_replace, properties in self.additional_gradients:
                gparams, params = new_grads
                prop_expr = [x[1] for x in properties]
                replace = [(x[0], TT.grad(cost, x[1])) for x in to_replace]
                rval = theano.clone(gparams + prop_expr,
                                    replace=replace)
                gparams = rval[:len(gparams)]
                prop_expr = rval[len(gparams):]
                self.properties += [(x[0], y)
                                    for x, y in zip(properties, prop_expr)]
                for gp, p in zip(gparams, params):
                    grads[self.params.index(p)] += gp

        self.cost = cost
        self.grads = grads
        return cost, grads

    def _get_samples(self, model, length=30, temp=1, *inps):
        """
        Sample a sequence from the model `model` whose output layer is given
        by `self`.

        :type model: groundhog model class
        :param model: model that has `self` as its output layer

        :type length: int
        :param length: length of the sequence to sample

        :type temp: float
        :param temp: temperature to use during sampling
        """

        raise NotImplemented


class LinearLayer(CostLayer):
    """
    Linear output layer.
    """

    def _init_params(self):
        """
        Initialize the parameters of the layer, either by using sparse initialization or small
        isotropic noise.
        """
        if self.rank_n_approx:
            W_em1 = self.init_fn(self.nin,
                                         self.rank_n_approx,
                                         self.sparsity,
                                         self.scale,
                                         self.rng)
            W_em2 = self.init_fn(self.rank_n_approx,
                                         self.nout,
                                         self.sparsity,
                                         self.scale,
                                         self.rng)
            self.W_em1 = theano.shared(W_em1,
                                       name='W1_%s'%self.name)
            self.W_em2 = theano.shared(W_em2,
                                       name='W2_%s'%self.name)
            self.b_em = theano.shared(
                numpy.zeros((self.nout,), dtype=theano.config.floatX),
                name='b_%s'%self.name)
            self.params += [self.W_em1, self.W_em2, self.b_em]
            self.myparams = []#[self.W_em1, self.W_em2, self.b_em]
            if self.weight_noise:
                self.nW_em1 = theano.shared(W_em1*0.,
                                            name='noise_W1_%s'%self.name)
                self.nW_em2 = theano.shared(W_em*0.,
                                            name='noise_W2_%s'%self.name)
                self.nb_em = theano.shared(b_em*0.,
                                           name='noise_b_%s'%self.name)
                self.noise_params = [self.nW_em1, self.nW_em2, self.nb_em]
                self.noise_params_shape_fn = [
                    constant_shape(x.get_value().shape)
                    for x in self.noise_params]

        else:
            W_em = self.init_fn(self.nin,
                                        self.nout,
                                        self.sparsity,
                                        self.scale,
                                        self.rng)
            self.W_em = theano.shared(W_em,
                                      name='W_%s'%self.name)
            self.b_em = theano.shared(
                numpy.zeros((self.nout,), dtype=theano.config.floatX),
                name='b_%s'%self.name)
            self.add_wghs = []
            self.n_add_wghs = []
            if self.additional_inputs:
                for pos, sz in enumerate(self.additional_inputs):
                    W_add = self.init_fn(sz,
                                        self.nout,
                                        self.sparsity,
                                        self.scale,
                                        self.rng)
                    self.add_wghs += [theano.shared(W_add,
                                      name='W_add%d_%s'%(pos, self.name))]
                    if self.weight_noise:
                        self.n_add_wghs += [theano.shared(W_add*0.,
                                                      name='noise_W_add%d_%s'%(pos,
                                                                               self.name))]

            self.params += [self.W_em, self.b_em] + self.add_wghs
            self.myparams = []#[self.W_em, self.b_em] + self.add_wghs
            if self.weight_noise:
                self.nW_em = theano.shared(W_em*0.,
                                           name='noise_W_%s'%self.name)
                self.nb_em = theano.shared(numpy.zeros((self.nout,),
                                                       dtype=theano.config.floatX),
                                           name='noise_b_%s'%self.name)
                self.noise_params = [self.nW_em, self.nb_em] + self.n_add_wghs
                self.noise_params_shape_fn = [
                    constant_shape(x.get_value().shape)
                    for x in self.noise_params]

    def _check_dtype(self, matrix, inp):
        if 'int' in inp.dtype and inp.ndim==2:
            return matrix[inp.flatten()]
        elif 'int' in inp.dtype:
            return matrix[inp]
        elif 'float' in inp.dtype and inp.ndim == 3:
            shape0 = inp.shape[0]
            shape1 = inp.shape[1]
            shape2 = inp.shape[2]
            return TT.dot(inp.reshape((shape0*shape1, shape2)), matrix)
        else:
            return TT.dot(inp, matrix)


    def fprop(self, state_below, temp = numpy.float32(1), use_noise=True,
             additional_inputs = None):
        """
        Constructs the computational graph of this layer.
        """

        if self.rank_n_approx:
            if use_noise and self.noise_params:
                emb_val = self._check_dtype(self.W_em1+self.nW_em1,
                                          state_below)
                emb_val = TT.dot(self.W_em2 + self.nW_em2, emb_val)
            else:
                emb_val = self._check_dtype(self.W_em1, state_below)
                emb_val = TT.dot(self.W_em2, emb_val)
        else:
            if use_noise and self.noise_params:
                emb_val = self._check_dtype(self.W_em + self.nW_em, state_below)
            else:
                emb_val = self._check_dtype(self.W_em, state_below)

        if additional_inputs:
            for st, wgs in zip(additional_inputs, self.add_wghs):
                emb_val += self._check_dtype(wgs, st)

        if use_noise and self.noise_params:
            emb_val = (emb_val + self.b_em+ self.nb_em)
        else:
            emb_val =  (emb_val + self.b_em)
        self.out = emb_val
        self.state_below = state_below
        self.model_output = emb_val
        return emb_val

    def get_cost(self, state_below, target=None, mask = None, temp=1,
                 reg = None, scale=None, sum_over_time=True, use_noise=True,
                additional_inputs=None):
        """
        This function computes the cost of this layer.

        :param state_below: theano variable representing the input to the
            softmax layer
        :param target: theano variable representing the target for this
            layer
        :return: mean cross entropy
        """
        class_probs = self.fprop(state_below, temp = temp,
                                 use_noise=use_noise,
                                additional_inputs=additional_inputs)
        pvals = class_probs
        assert target, 'Computing the cost requires a target'
        if target.ndim == 3:
            target = target.reshape((target.shape[0]*target.shape[1],
                                    target.shape[2]))
        assert 'float' in target.dtype
        cost = (class_probs - target)**2
        if mask:
            mask = mask.flatten()
            cost = cost * TT.cast(mask, theano.config.floatX)
        if sum_over_time is None:
            sum_over_time = self.sum_over_time
        if sum_over_time:
            if state_below.ndim ==3:
                sh0 = TT.cast(state_below.shape[0],
                             theano.config.floatX)
                sh1 = TT.cast(state_below.shape[1],
                             theano.config.floatX)
                self.cost = cost.sum()/sh1
            else:
                self.cost =cost.sum()
        else:
            self.cost = cost.mean()
        if scale:
            self.cost = self.cost*scale
        if reg:
            self.cost = self.cost + reg
        self.out = self.cost
        self.mask = mask
        self.cost_scale = scale
        return self.cost


    def get_grads(self, state_below, target, mask = None, reg = None,
                  scale=None, sum_over_time=True, use_noise=True,
                 additional_inputs=None):
        """
        This function implements both the forward and backwards pass of this
        layer. The reason we do this in a single function is because for the
        factorized softmax layer is hard to rely on grad and get an
        optimized graph. For uniformity I've implemented this method for
        this layer as well (though one doesn't need to use it)

        :param state_below: theano variable representing the input to the
            softmax layer
        :param target: theano variable representing the target for this
            layer
        :return: cost, dC_dstate_below, param_grads, new_properties
            dC_dstate_below is a computational graph representing the
            gradient of the cost wrt to state_below
            param_grads is a list containing the gradients wrt to the
            different parameters of the layer
            new_properties is a dictionary containing additional properties
            of the model; properties are theano expression that are
            evaluated and reported by the model
        """
        cost = self.get_cost(state_below,
                             target,
                             mask = mask,
                             reg = reg,
                             scale=scale,
                             sum_over_time=sum_over_time,
                             use_noise=use_noise,
                             additional_inputs=additional_inputs)
        grads = TT.grad(cost, self.params)
        if self.additional_gradients:
            for new_grads, to_replace, properties in self.additional_gradients:
                gparams, params = new_grads
                prop_expr = [x[1] for x in properties]
                replace = [(x[0], TT.grad(cost, x[1])) for x in to_replace]
                rval = theano.clone(gparams + prop_expr,
                                    replace=replace)
                gparams = rval[:len(gparams)]
                prop_expr = rval[len(gparams):]
                self.properties += [(x[0], y) for x,y in zip(properties,
                                                             prop_expr)]
                for gp, p in zip(gparams, params):
                    grads[self.params.index(p)] += gp

        self.cost = cost
        self.grads = grads
        def Gvs_fn(*args):
            w = (1 - self.model_output) * self.model_output * state_below.shape[1]
            Gvs = TT.Lop(self.model_output, self.params,
                         TT.Rop(self.model_output, self.params, args)/w)
            return Gvs
        self.Gvs = Gvs_fn
        return cost, grads


class SigmoidLayer(CostLayer):
    """
    Sigmoid output layer.
    """

    def _get_samples(self, model, length=30, temp=1, *inps):
        """
        See parent class.
        """

        if not hasattr(model, 'word_indxs_src'):
            model.word_indxs_src = model.word_indxs

        character_level = False
        if hasattr(model, 'character_level'):
            character_level = model.character_level
        if model.del_noise:
            model.del_noise()
        [values, probs] = model.sample_fn(length, temp, *inps)
        # Assumes values matrix
        #print 'Generated sample is:'
        #print
        if values.ndim > 1:
            for d in xrange(2):
                print '%d-th sentence' % d
                print 'Input: ',
                if character_level:
                    sen = []
                    for k in xrange(inps[0].shape[0]):
                        if model.word_indxs_src[inps[0][k][d]] == '<eol>':
                            break
                        sen.append(model.word_indxs_src[inps[0][k][d]])
                    print "".join(sen),
                else:
                    for k in xrange(inps[0].shape[0]):
                        print model.word_indxs_src[inps[0][k][d]],
                        if model.word_indxs_src[inps[0][k][d]] == '<eol>':
                            break
                print ''
                print 'Output: ',
                if character_level:
                    sen = []
                    for k in xrange(values.shape[0]):
                        if model.word_indxs[values[k][d]] == '<eol>':
                            break
                        sen.append(model.word_indxs[values[k][d]])
                    print "".join(sen),
                else:
                    for k in xrange(values.shape[0]):
                        print model.word_indxs[values[k][d]],
                        if model.word_indxs[values[k][d]] == '<eol>':
                            break
                print
                print
        else:
            print 'Input:  ',
            if character_level:
                sen = []
                for k in xrange(inps[0].shape[0]):
                    if model.word_indxs_src[inps[0][k]] == '<eol>':
                        break
                    sen.append(model.word_indxs_src[inps[0][k]])
                print "".join(sen),
            else:
                for k in xrange(inps[0].shape[0]):
                    print model.word_indxs_src[inps[0][k]],
                    if model.word_indxs_src[inps[0][k]] == '<eol>':
                        break
            print ''
            print 'Output: ',
            if character_level:
                sen = []
                for k in xrange(values.shape[0]):
                    if model.word_indxs[values[k]] == '<eol>':
                        break
                    sen.append(model.word_indxs[values[k]])
                print "".join(sen),
            else:
                for k in xrange(values.shape[0]):
                    print model.word_indxs[values[k]],
                    if model.word_indxs[values[k]] == '<eol>':
                        break
            print
            print

    def fprop(self,
              state_below,
              temp=numpy.float32(1),
              use_noise=True,
              additional_inputs=None,
              no_noise_bias=False):
        """
        Forward pass through the cost layer.

        :type state_below: tensor or layer
        :param state_below: The theano expression (or groundhog layer)
            representing the input of the cost layer

        :type temp: float or tensor scalar
        :param temp: scalar representing the temperature that should be used
            when sampling from the output distribution

        :type use_noise: bool
        :param use_noise: flag. If true, noise is used when computing the
            output of the model

        :type no_noise_bias: bool
        :param no_noise_bias: flag, stating if weight noise should be added
            to the bias as well, or only to the weights
        """

        if self.rank_n_approx:
            if use_noise and self.noise_params:
                emb_val = self.rank_n_activ(utils.dot(state_below,
                                                      self.W_em1+self.nW_em1))
                emb_val = TT.dot(self.W_em2 + self.nW_em2, emb_val)
            else:
                emb_val = self.rank_n_activ(utils.dot(state_below, self.W_em1))
                emb_val = TT.dot(self.W_em2, emb_val)
        else:
            if use_noise and self.noise_params:
                emb_val = utils.dot(state_below, self.W_em + self.nW_em)
            else:
                emb_val = utils.dot(state_below, self.W_em)

        if additional_inputs:
            if use_noise and self.noise_params:
                for inp, weight, noise_weight in zip(
                    additional_inputs, self.additional_weights,
                    self.noise_additional_weights):
                    emb_val += utils.dot(inp, (noise_weight + weight))
            else:
                for inp, weight in zip(additional_inputs, self.additional_weights):
                    emb_val += utils.dot(inp, weight)
        self.preactiv = emb_val
        if use_noise and self.noise_params and not no_noise_bias:
            emb_val = TT.nnet.sigmoid(temp *
                                      (emb_val + self.b_em + self.nb_em))
        else:
            emb_val = TT.nnet.sigmoid(temp * (emb_val + self.b_em))
        self.out = emb_val
        self.state_below = state_below
        self.model_output = emb_val
        return emb_val

    def compute_sample(self,
                       state_below,
                       temp=1,
                       additional_inputs=None,
                       use_noise=False):
        """
        See parent class.
        """
        class_probs = self.fprop(state_below,
                                 temp=temp,
                                 additional_inputs=additional_inputs,
                                 use_noise=use_noise)
        pvals = class_probs
        if pvals.ndim == 1:
            pvals = pvals.dimshuffle('x', 0)
        sample = self.trng.binomial(pvals.shape, p=pvals,
                                    dtype='int64')
        if class_probs.ndim == 1:
            sample = sample[0]
        self.sample = sample
        return sample

    def get_cost(self,
                 state_below,
                 target=None,
                 mask=None,
                 temp=1,
                 reg=None,
                 scale=None,
                 sum_over_time=None,
                 use_noise=True,
                 additional_inputs=None,
                 no_noise_bias=False):
        """
        See parent class
        """
        class_probs = self.fprop(state_below,
                                 temp=temp,
                                 use_noise=use_noise,
                                 additional_inputs=additional_inputs,
                                 no_noise_bias=no_noise_bias)
        pvals = class_probs
        assert target, 'Computing the cost requires a target'
        if target.ndim == 3:
            target = target.reshape((target.shape[0]*target.shape[1],
                                    target.shape[2]))
        assert 'float' in target.dtype
        # Do we need the safety net of 1e-12  ?
        cost = -TT.log(TT.maximum(1e-12, class_probs)) * target -\
            TT.log(TT.maximum(1e-12, 1 - class_probs)) * (1 - target)
        if cost.ndim > 1:
            cost = cost.sum(1)
        if mask:
            mask = mask.flatten()
            cost = cost * TT.cast(mask, theano.config.floatX)
        if sum_over_time is None:
            sum_over_time = self.sum_over_time
        if sum_over_time:
            if state_below.ndim == 3:
                sh0 = TT.cast(state_below.shape[0],
                              theano.config.floatX)
                sh1 = TT.cast(state_below.shape[1],
                              theano.config.floatX)
                self.cost = cost.sum()/sh1
            else:
                self.cost = cost.sum()
        else:
            self.cost = cost.mean()
        if scale:
            self.cost = self.cost*scale
        if reg:
            self.cost = self.cost + reg
        self.out = self.cost
        self.mask = mask
        self.cost_scale = scale
        return self.cost


class SoftmaxLayer(CostLayer):
    """
    Softmax output layer.
    """

    def _get_samples(self, model, length=30, temp=1, *inps):
        """
        See parent class
        """
        if not hasattr(model, 'word_indxs_src'):
            model.word_indxs_src = model.word_indxs

        character_level = False
        if hasattr(model, 'character_level'):
            character_level = model.character_level
        if model.del_noise:
            model.del_noise()
        [values, probs] = model.sample_fn(length, temp, *inps)
        #print 'Generated sample is:'
        #print
        if values.ndim > 1:
            for d in xrange(2):
                print '%d-th sentence' % d
                print 'Input: ',
                if character_level:
                    sen = []
                    for k in xrange(inps[0].shape[0]):
                        if model.word_indxs_src[inps[0][k][d]] == '<eol>':
                            break
                        sen.append(model.word_indxs_src[inps[0][k][d]])
                    print "".join(sen),
                else:
                    for k in xrange(inps[0].shape[0]):
                        print model.word_indxs_src[inps[0][k][d]],
                        if model.word_indxs_src[inps[0][k][d]] == '<eol>':
                            break
                print ''
                print 'Output: ',
                if character_level:
                    sen = []
                    for k in xrange(values.shape[0]):
                        if model.word_indxs[values[k][d]] == '<eol>':
                            break
                        sen.append(model.word_indxs[values[k][d]])
                    print "".join(sen),
                else:
                    for k in xrange(values.shape[0]):
                        print model.word_indxs[values[k][d]],
                        if model.word_indxs[values[k][d]] == '<eol>':
                            break
                print
                print
        else:
            print 'Input:  ',
            if character_level:
                sen = []
                for k in xrange(inps[0].shape[0]):
                    if model.word_indxs_src[inps[0][k]] == '<eol>':
                        break
                    sen.append(model.word_indxs_src[inps[0][k]])
                print "".join(sen),
            else:
                for k in xrange(inps[0].shape[0]):
                    print model.word_indxs_src[inps[0][k]],
                    if model.word_indxs_src[inps[0][k]] == '<eol>':
                        break
            print ''
            print 'Output: ',
            if character_level:
                sen = []
                for k in xrange(values.shape[0]):
                    if model.word_indxs[values[k]] == '<eol>':
                        break
                    sen.append(model.word_indxs[values[k]])
                print "".join(sen),
            else:
                for k in xrange(values.shape[0]):
                    print model.word_indxs[values[k]],
                    if model.word_indxs[values[k]] == '<eol>':
                        break
            print
            print

    def fprop(self,
              state_below,
              temp=numpy.float32(1),
              use_noise=True,
              additional_inputs=None,
              no_noise_bias=False,
              target=None,
              full_softmax=True):
        """
        Forward pass through the cost layer.

        :type state_below: tensor or layer
        :param state_below: The theano expression (or groundhog layer)
            representing the input of the cost layer

        :type temp: float or tensor scalar
        :param temp: scalar representing the temperature that should be used
            when sampling from the output distribution

        :type use_noise: bool
        :param use_noise: flag. If true, noise is used when computing the
            output of the model

        :type no_noise_bias: bool
        :param no_noise_bias: flag, stating if weight noise should be added
            to the bias as well, or only to the weights
        """
        if not full_softmax:
            assert target != None, 'target must be given'
        if self.rank_n_approx:
            if self.weight_noise and use_noise and self.noise_params:
                emb_val = self.rank_n_activ(utils.dot(state_below,
                                                      self.W_em1+self.nW_em1))
                nW_em = self.nW_em2
            else:
                emb_val = self.rank_n_activ(utils.dot(state_below, self.W_em1))
            W_em = self.W_em2
        else:
            W_em = self.W_em
            if self.weight_noise:
                nW_em = self.nW_em
            emb_val = state_below

        if full_softmax:
            if self.weight_noise and use_noise and self.noise_params:
                emb_val = TT.dot(emb_val, W_em + nW_em)
            else:
                emb_val = TT.dot(emb_val, W_em)

            if additional_inputs:
                if use_noise and self.noise_params:
                    for inp, weight, noise_weight in zip(
                        additional_inputs, self.additional_weights,
                        self.noise_additional_weights):
                        emb_val += utils.dot(inp, (noise_weight + weight))
                else:
                    for inp, weight in zip(additional_inputs, self.additional_weights):
                        emb_val += utils.dot(inp, weight)
            if self.weight_noise and use_noise and self.noise_params and \
               not no_noise_bias:
                emb_val = temp * (emb_val + self.b_em + self.nb_em)
            else:
                emb_val = temp * (emb_val + self.b_em)
        else:
            W_em = W_em[:, target]
            if self.weight_noise:
                nW_em = nW_em[:, target]
                W_em += nW_em
            if emb_val.ndim == 3:
                emb_val = emb_val.reshape([emb_val.shape[0]*emb_val.shape[1], emb_val.shape[2]])
            emb_val = (W_em.T * emb_val).sum(1) + self.b_em[target]
            if self.weight_noise and use_noise:
                emb_val += self.nb_em[target]
            emb_val = temp * emb_val

        self.preactiv = emb_val
        if full_softmax:
            emb_val = utils.softmax(emb_val)
        else:
            emb_val = TT.nnet.sigmoid(emb_val)
        self.out = emb_val
        self.state_below = state_below
        self.model_output = emb_val
        return emb_val

    def compute_sample(self,
                       state_below,
                       temp=1,
                       use_noise=False,
                       additional_inputs=None):

        class_probs = self.fprop(state_below,
                                 temp=temp,
                                 additional_inputs=additional_inputs,
                                 use_noise=use_noise)
        pvals = class_probs
        if pvals.ndim == 1:
            pvals = pvals.dimshuffle('x', 0)
        sample = self.trng.multinomial(pvals=pvals,
                                       dtype='int64').argmax(axis=-1)
        if class_probs.ndim == 1:
            sample = sample[0]
        self.sample = sample
        return sample

    def get_cost(self,
                 state_below,
                 target=None,
                 mask=None,
                 temp=1,
                 reg=None,
                 scale=None,
                 sum_over_time=False,
                 no_noise_bias=False,
                 additional_inputs=None,
                 use_noise=True):
        """
        See parent class
        """

        def _grab_probs(class_probs, target):
            shape0 = class_probs.shape[0]
            shape1 = class_probs.shape[1]
            target_ndim = target.ndim
            target_shape = target.shape
            if target.ndim > 1:
                target = target.flatten()
            assert target.ndim == 1, 'make sure target is a vector of ints'
            assert 'int' in target.dtype

            pos = TT.arange(shape0)*shape1
            new_targ = target + pos
            return class_probs.flatten()[new_targ]

        assert target, 'Computing the cost requires a target'
        target_shape = target.shape
        target_ndim = target.ndim
        target_shape = target.shape

        if self.use_nce:
            logger.debug("Using NCE")

            # positive samples: true targets
            class_probs = self.fprop(state_below,
                                     temp=temp,
                                     use_noise=use_noise,
                                     additional_inputs=additional_inputs,
                                     no_noise_bias=no_noise_bias,
                                     target=target.flatten(),
                                     full_softmax=False)
            # negative samples: a single uniform random sample per training sample
            nsamples = TT.cast(self.trng.uniform(class_probs.shape[0].reshape([1])) * self.n_out, 'int64')
            neg_probs = self.fprop(state_below,
                                     temp=temp,
                                     use_noise=use_noise,
                                     additional_inputs=additional_inputs,
                                     no_noise_bias=no_noise_bias,
                                     target=nsamples.flatten(),
                                     full_softmax=False)

            cost_target = class_probs
            cost_nsamples = 1. - neg_probs

            cost = -TT.log(cost_target)
            cost = cost - TT.cast(neg_probs.shape[0], 'float32') * TT.log(cost_nsamples)
        else:
            class_probs = self.fprop(state_below,
                                     temp=temp,
                                     use_noise=use_noise,
                                     additional_inputs=additional_inputs,
                                     no_noise_bias=no_noise_bias)
            cost = -TT.log(_grab_probs(class_probs, target))

        self.word_probs = TT.exp(-cost.reshape(target_shape))
        # Set all the probs after the end-of-line to one
        if mask:
            self.word_probs = self.word_probs * mask + 1 - mask
        if mask:
            cost = cost * TT.cast(mask.flatten(), theano.config.floatX)
        self.cost_per_sample = (cost.reshape(target_shape).sum(axis=0)
                if target_ndim > 1
                else cost)

        if sum_over_time is None:
            sum_over_time = self.sum_over_time
        if sum_over_time:
            if state_below.ndim == 3:
                cost = cost.reshape((state_below.shape[0],
                                     state_below.shape[1]))
                self.cost = cost.mean(1).sum()
            else:
                self.cost = cost.sum()
        else:
            self.cost = cost.mean()
        if scale:
            self.cost = self.cost*scale
        if reg:
            self.cost = self.cost + reg
        self.mask = mask
        self.cost_scale = scale
        return self.cost

class HierarchicalSoftmaxLayer(SoftmaxLayer):
    """
    Hierarchical Softmax output layer (2 layer)

    This is a preliminary implementation of 2-level hierarchical softmax layer (GPU only)
    """

    def __init__(self, rng,
                 n_in,
                 n_out,
                 scale,
                 sparsity,
                 weight_noise=False,
                 init_fn='sample_weights_classic',
                 bias_fn='init_bias',
                 bias_scale=0.,
                 sum_over_time=True,
                 grad_scale=1.,
                 name=None,
                 **kwargs):

        assert theano.config.device[:3] == 'gpu', 'Hierarchical softmax is not supported without GPU'

        from theano.sandbox.cuda.blocksparse import sparse_block_dot_SS
        self.sparse_block_dot_SS = sparse_block_dot_SS

        self.grad_scale = grad_scale
        super(CostLayer, self).__init__(n_in, n_out, rng, name)
        self.n_words_class = numpy.ceil(numpy.sqrt(self.n_out)).astype('int64') # oSize
        self.n_class = numpy.ceil(self.n_out/float(self.n_words_class)).astype('int64') # oBlocks
        logger.debug("n_words_class = %d, n_class = %d"%(self.n_words_class, self.n_class))
        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        if isinstance(bias_fn, str):
            self.bias_fn = eval(bias_fn)
        else:
            self.bias_fn = bias_fn
        self.bias_scale = bias_scale
        self.scale = scale

        self.sum_over_time = sum_over_time
        self.weight_noise = weight_noise
        self.sparsity = sparsity
        if self.sparsity < 0:
            self.sparsity = n_out
        if type(init_fn) is str:
            init_fn = eval(init_fn)
        self.init_fn = init_fn
        self._init_params()

    def _init_params(self):
        self.iBlocks = 1  # number of blocks in the input (from lower layer)

        W_em = self.init_fn(self.n_in,
                            self.n_class,
                            self.sparsity,
                            self.scale,
                            self.rng)
        self.W_em = theano.shared(W_em,
                                  name='W_%s' % self.name)
        self.b_em = theano.shared(
            self.bias_fn(self.n_class, self.bias_scale, self.rng),
            name='b_%s' % self.name)

        U_em = theano.shared(((self.rng.rand(self.iBlocks, self.n_class, 
            self.n_in, self.n_words_class)-0.5)/(self.n_words_class*self.n_in)
            ).astype(theano.config.floatX), name='U_%s'%self.name)
        self.U_em = U_em
        c_em = numpy.zeros((self.n_class, self.n_words_class), dtype='float32')
        n_words_last_class = self.n_out % self.n_words_class
        #c_em[-1, n_words_last_class:] = -numpy.inf
        self.c_em = theano.shared(c_em, name='c_%s' % self.name)

        self.params = [self.W_em, self.b_em, self.U_em, self.c_em]
        self.params_grad_scale = [self.grad_scale for x in self.params]

    def fprop(self,
              state_below,
              temp=numpy.float32(1),
              use_noise=True,
              additional_inputs=None,
              no_noise_bias=False,
              target=None,
              full_softmax=True,
              **kwargs):

        if not full_softmax:
            assert target != None, 'target must be given'

        W_em = self.W_em
        U_em = self.U_em
        b_em = self.b_em
        c_em = self.c_em

        emb_val = state_below
        bs = emb_val.shape[0]

        if full_softmax:
            # compute the probability of every word using scan

            # for all classes
            class_vecs = TT.arange(self.n_class)
            class_val = utils.softmax(TT.dot(emb_val, W_em) + b_em)

            def _compute_inclass(classid):
                # compute the word probabilities
                outputIdx = TT.alloc(classid, bs)[:, None]
                word_val = utils.softmax(TT.dot(emb_val, U_em[0, classid, :, :])+c_em[classid,:])
                word_val = word_val * class_val[:, classid][:,None]
                return word_val.T

            rval = theano.scan(_compute_inclass, class_vecs, None, name='compute_inclass')
            all_word_val = rval[0].reshape([rval[0].shape[0]*rval[0].shape[1], rval[0].shape[2]]).T
            all_word_val = all_word_val[:,:self.n_out]
            emb_val = all_word_val
        else:
            # compute only the probability of given targets
            if emb_val.ndim == 3:
                emb_val = emb_val.reshape([emb_val.shape[0]*emb_val.shape[1], emb_val.shape[2]])

            # extract class id's from target indices
            target = target.flatten()
            class_vec = target // self.n_words_class  # need to be int/int
            class_idx_vec = target % self.n_words_class
            outputIdx = class_vec[:, None]

            # compute the class probabilities
            class_val = utils.softmax(TT.dot(emb_val, W_em) + b_em)

            # compute the word probabilities
            word_val = utils.softmax(self.sparse_block_dot_SS(U_em, 
                emb_val[:, None, :], TT.zeros((bs, 1), dtype='int64'), c_em, outputIdx)[:, 0, :])

            class_val = class_val[TT.arange(bs), class_vec]
            word_val = word_val[TT.arange(bs), class_idx_vec]
            emb_val = class_val * word_val

        #self.preactiv = emb_val
        self.out = emb_val
        self.state_below = state_below
        self.model_output = emb_val
        return emb_val

    def get_cost(self,
                 state_below,
                 target=None,
                 mask=None,
                 temp=1,
                 reg=None,
                 scale=None,
                 sum_over_time=False,
                 no_noise_bias=False,
                 additional_inputs=None,
                 use_noise=True):
        """
        See parent class
        """

        assert target, 'Computing the cost requires a target'
        target_shape = target.shape
        target_ndim = target.ndim
        target_shape = target.shape

        if state_below.ndim == 3:
            shp = state_below.shape
            state_below = state_below.reshape([shp[0]*shp[1], shp[2]])

        class_probs = self.fprop(state_below,
                                 temp=temp,
                                 target=target,
                                 full_softmax=False,
                                 use_noise=use_noise,
                                 additional_inputs=additional_inputs,
                                 no_noise_bias=no_noise_bias)
        cost = -TT.log(class_probs)

        self.word_probs = TT.exp(-cost.reshape(target_shape))
        # Set all the probs after the end-of-line to one
        if mask:
            self.word_probs = self.word_probs * mask + 1 - mask
        if mask:
            cost = cost * TT.cast(mask.flatten(), theano.config.floatX)
        self.cost_per_sample = (cost.reshape(target_shape).sum(axis=0)
                if target_ndim > 1
                else cost)

        if sum_over_time is None:
            sum_over_time = self.sum_over_time
        if sum_over_time:
            if state_below.ndim == 3:
                cost = cost.reshape((state_below.shape[0],
                                     state_below.shape[1]))
                self.cost = cost.mean(1).sum()
            else:
                self.cost = cost.sum()
        else:
            self.cost = cost.mean()
        if scale:
            self.cost = self.cost*scale
        if reg:
            self.cost = self.cost + reg
        self.mask = mask
        self.cost_scale = scale
        return self.cost

