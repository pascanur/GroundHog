"""
Parent classes describing a layer, model or operator
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import numpy
import copy
import cPickle as pkl
import logging

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from groundhog.utils import utils
from groundhog.utils.utils import id_generator

logger = logging.getLogger(__name__)

class Container(object):
    """
    Root class. It contains some properties one would expect from any
    derived class.
    """

    def __init__(self):
        self.floatX = theano.config.floatX
        # Parameters of the model (shared variables)
        self.params                 = []
        # Factor scaling the gradient of the cost with respect to some
        # parameter
        self.params_grad_scale      = []
        # List of shared variables holding the noise used when applying
        # weight noise
        self.noise_params           = []
        # List of functions that compute the shape of a parameter
        self.noise_params_shape_fn  = []
        # Updates that the model require at each step
        self.updates                = []
        # Additional gradients that need to be added to the gradients of the
        # cost of a model with respect to the parameters
        self.additional_gradients   = []
        # Theano variables representing the inputs required to compute the
        # value of the container
        self.inputs                 = []
        # Schedules for updating shared variables involved in the
        # computation of the container
        self.schedules              = []
        # Additional properties that can be computed beside the output of
        # the container
        self.properties             = []

    def tensor_from_layer(self, arg):
        """
        Grab the theano tensor representing the computation of this
        layer/operator iff `arg` is a layer
        """
        if isinstance(arg, Container):
            return arg.out
        else:
            return arg

    def add_schedule(self, sched):
        """
        Add a new schedule to the list of schedules
        """
        self.schedules += [sched]

    def add_schedules(self, scheds):
        """
        Add a list of schedules to the current list of schedules
        """
        self.schedules += scheds

    def tensor_from_layer(self, arg, collect_params=True):
        """
        Grab the theano tensor representing the computation of this
        layer/operator iff `arg` is a layer.

        :type collect_params: bool
        :param collect_params: Flag. If true, also collect the parameters
            and inputs of the layer `arg` and make them parameters and inputs
            needed to compute the current layer/operator
        """
        if not collect_params:
            if isinstance(arg, Container):
                return arg.out
            else:
                return arg


        if isinstance(arg, Container):
            self.merge_params(arg)
            return arg.out
        elif isinstance(arg, theano.gof.Variable):
            inps = [x for x in theano.gof.graph.inputs([arg])
                    if not isinstance(x, (TT.Constant, theano.compile.SharedVariable))]
            self.add_inputs(inps)
            return arg
        else:
            return arg


    def add_inputs(self, inps):
        """
        Add to the current list of inputs the tensors in the `inps` list
        """
        if not isinstance(inps, (list, tuple)):
            inps = [inps]
        for inp in inps:
            if inp not in self.inputs:
                self.inputs = self.inputs + [inp]

    def merge_params(self, model):
        """
        Add to the current properties of the container (params, schedules,
        etc.) those of the layer/operator/model `model`.
        """
        new_params_grad_scale = [ps for ps, param in zip(model.params_grad_scale,
                                                         model.params)
                                 if param not in self.params]
        new_params  = [param for param in model.params if param not in self.params]

        assert len(new_params_grad_scale) == len(new_params)

        new_noise_params_shape_fn = [shape_fn
                for shape_fn,noise_param in zip(model.noise_params_shape_fn,
                                                model.noise_params)
                                    if noise_param not in self.noise_params]
        new_noise_params = [noise_param
                for noise_param in model.noise_params
                       if noise_param not in self.noise_params]
        new_inputs =[inp for inp in model.inputs if inp not in self.inputs]
        new_schedules =[schedule for schedule in model.schedules
                       if schedule not in self.schedules]
        new_updates = [update for update in model.updates
                       if update not in self.updates]
        new_additional_gradients = [additional_gradient
                for additional_gradient in model.additional_gradients
                       if additional_gradient not in self.additional_gradients]
        new_properties = [prop for prop in model.properties
                          if prop not in self.properties]

        self.params += new_params
        self.params_grad_scale += new_params_grad_scale
        assert len(self.params) == len(self.params_grad_scale)
        if hasattr(self, 'grads'):
            self.grads += [ 0 for param in new_params]
        self.noise_params += new_noise_params
        self.noise_params_shape_fn += new_noise_params_shape_fn
        self.inputs += new_inputs
        self.schedules += new_schedules
        self.updates += new_updates
        self.additional_gradients += new_additional_gradients
        self.properties += new_properties

    def save(self, filename):
        """
        Save the model to file `filename`
        """
        vals = dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez(filename, **vals)

    def load(self, filename):
        """
        Load the model.
        """
        vals = numpy.load(filename)
        for p in self.params:
            if p.name in vals:
                logger.debug('Loading {} of {}'.format(p.name, p.get_value(borrow=True).shape))
                if p.get_value().shape != vals[p.name].shape:
                    raise Exception("Shape mismatch: {} != {} for {}"
                            .format(p.get_value().shape, vals[p.name].shape, p.name))
                p.set_value(vals[p.name])
            else:
                # FIXME: do not stop loading even if there's a parameter value missing
                #raise Exception("No parameter {} given".format(p.name))
                logger.error( "No parameter {} given: default initialization used".format(p.name))
        unknown = set(vals.keys()) - {p.name for p in self.params}
        if len(unknown):
            logger.error("Unknown parameters {} given".format(unknown))

class Layer(Container):
    """
    Parent class for Layers.
    A layer is a segment of a computational pipeline. It is different from a
    model in the sense that it does not necessarly have a cost or gradients
    defined, neither does it respect the interface expected from the
    trainers.

    """
    def __init__(self, n_in=0, n_out=0, rng=None, name=None):
        super(Layer, self).__init__()
        if name:
            self.name = name
        else:
            self.name = 'unknown_'+ id_generator(4)
        self.rng = rng
        self.n_in = n_in
        self.n_out = n_out
        self.n_hid = n_out
        self.floatX = theano.config.floatX

    def reshape(self, shape):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        new_obj.out = new_obj.out.reshape(shape)
        return new_obj

    shape = property(lambda self: self.out.shape)

    def __str__(self):
        return self.name

    def __add__(self, other):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        other_var = new_obj.tensor_from_layer(other)
        new_obj.out = new_obj.out + other_var
        # Summing cost layers:
        if hasattr(new_obj, 'grads') and hasattr(other, 'grads'):
            for param, grad_param in zip(other.params, other.grads):
                pos = new_obj.params.index(param)
                new_obj.grads[pos] += grad_param
        elif hasattr(new_obj, 'grads') and \
                isinstance(other, theano.gof.Variable) and \
                other.ndim == 0:
            other_grads = TT.grad(other, new_obj.params,
                                  disconnected_inputs='ignore')
            new_obj.grads = [x + y for x,y in zip(new_obj.grads,
                                                  other_grads)]
        elif hasattr(new_obj, 'grads'):
            raise ValueError('I do not know how to compute the gradients'
                             ' of the added term' + str(other) + '. Call'
                             ' train on it if it is an output layer')
        return new_obj

    def __sub__(self, other):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        other_var = new_obj.tensor_from_layer(other)
        new_obj.out = new_obj.out - other_var
        if hasattr(new_obj, 'grads') and hasattr(other, 'grads'):
            for param, grad_param in zip(other.params, other.grads):
                pos = new_obj.params.index(param)
                new_obj.grads[pos] -= grad_param
        elif hasattr(new_obj, 'grads') and \
                isinstance(other, theano.gof.Variable) and \
                other.ndim == 0:
            other_grads = TT.grad(other, new_obj.params,
                                  disconnected_inputs='ignore')
            new_obj.grads = [x - y for x,y in zip(new_obj.grads,
                                                  other_grads)]
        elif hasattr(new_obj, 'grads'):
            raise ValueError('I do not know how to compute the gradients'
                             ' of the subtracted term' + str(other) + '. Call'
                             ' train on it if it is an output layer')
        return new_obj

    def __mul__(self, other):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        other_var = self.tensor_from_layer(other)

        if hasattr(new_obj, 'grads') and hasattr(other, 'grads'):
            new_obj.grads = [ x * other_var for x in new_obj.grads]
            for param, grad_param in zip(other.params, other.grads):
                pos = new_obj.params.index(param)
                new_obj.grads[pos] += new_obj.out * grad_param
        elif hasattr(new_obj, 'grads') and \
                isinstance(other, theano.gof.Variable) and \
                other.ndim == 0:

            new_obj.grads = [ x * other_var for x in new_obj.grads]
            other_grads = TT.grad(other, new_obj.params,
                                  disconnected_inputs='ignore')
            new_obj.grads = [x + new_obj.cost * y
                             for x,y in zip(new_obj.grads,
                                                  other_grads)]
        elif hasattr(new_obj, 'grads'):
            raise ValueError('I do not know how to compute the gradients'
                             ' of the subtracted term' + str(other) + '. Call'
                             ' train on it if it is an output layer')
        new_obj.out = new_obj.out * other_var
        return new_obj


    def __div__(self, other):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        other_var = new_obj.tensor_from_layer(other)
        if hasattr(new_obj, 'grads') and hasattr(other, 'grads'):
            new_obj.grads = [ x * other_var for x in new_obj.grads]
            for param, grad_param in zip(other.params, other.grads):
                pos = new_obj.params.index(param)
                new_obj.grads[pos] -= new_obj.out * grad_param
            new_obj.grads = [ x / (other_var**2) for x in new_obj.grads]
        elif hasattr(new_obj, 'grads') and \
                isinstance(other, theano.gof.Variable) and \
                other.ndim == 0:

            new_obj.grads = [ x * other_var for x in new_obj.grads]
            other_grads = TT.grad(other, new_obj.params,
                                  disconnected_inputs='ignore')
            new_obj.grads = [(x - new_obj.cost * y)/ (other_var**2)
                             for x,y in zip(new_obj.grads,
                                                  other_grads)]
        elif hasattr(new_obj, 'grads'):
            raise ValueError('I do not know how to compute the gradients'
                             ' of the subtracted term' + str(other) + '. Call'
                             ' train on it if it is an output layer')
        new_obj.out = new_obj.out / other_var
        return new_obj

    def __abs__(self, other):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        new_obj.out = abs(new_obj.out)
        if hasattr(new_obj, 'grads'):
            new_obj.grads = [TT.sgn(new_obj.out) * x for x in new_obj.grads]
        return new_obj

    def __pow__(self, power):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        power = self.tensor_from_layer(power)
        new_obj.out = new_obj.out**power
        if hasattr(new_obj, 'grads'):
            raise NotImplemented
        return new_obj


    def __lt__(self, other):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        other = self.tensor_from_layer(other)
        new_obj.out = new_obj.out.__lt__(other)
        if hasattr(new_obj, 'grads'):
            raise NotImplemented
        return new_obj

    def __le__(self, other):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        other = self.tensor_from_layer(other)
        new_obj.out = new_obj.out.__le__(other)
        if hasattr(new_obj, 'grads'):
            raise NotImplemented
        return new_obj

    def __gt__(self, other):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        other = self.tensor_from_layer(other)
        new_obj.out = new_obj.out.__gt__(other)
        if hasattr(new_obj, 'grads'):
            raise NotImplemented
        return new_obj

    def __ge__(self, other):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        other = self.tensor_from_layer(other)
        new_obj.out = new_obj.out.__ge__(other)
        if hasattr(new_obj, 'grads'):
            raise NotImplemented
        return new_obj

    def __getitem__(self, pos):
        assert hasattr(self, 'out'), 'all layers need a default output'
        new_obj = utils.copy(self)
        pos = self.tensor_from_layer(pos)
        new_obj.out = new_obj.out.__getitem__(pos)
        if hasattr(new_obj, 'grads'):
            raise NotImplemented
        return new_obj

    def _as_TensorVariable(self):
        print ('WARNING: you might loose track of parameters or inputs '\
               'because layer ' + self.name +' is being converted to a '\
               'theano variable')
        return self.out



    def validate(self, **kwargs):
        """
        Recompute the cost error (without the gradients)
        It only works for output layers !
        """
        if not hasattr(self, 'get_cost'):
            raise TypeError('Non-output layer does not support this method')
        new_obj = utils.copy(self)
        try:
            o_args, o_kwargs = new_obj.prev_args
        except:
            o_args, o_kwargs = ([], {})

        kwargs = dict([(k, new_obj.tensor_from_layer(v)) for k,v in kwargs.items()])
        for (k,v) in kwargs.items():
            o_kwargs[k] = v
        new_obj.prev_args = (o_args, o_kwargs)
        new_obj.get_cost(*o_args, **o_kwargs)
        return new_obj

    def train(self, **kwargs):
        """
        Compute the cost and gradients of the current layer with respect to
        its parameters.
        ! Only works for output layers
        """
        if not hasattr(self, 'get_grads'):
            raise TypeError('Non-output layer does not support this method')
        new_obj = utils.copy(self)
        try:
            o_args, o_kwargs = new_obj.prev_args
        except:
            o_args, o_kwargs = ([], {})
        kwargs = dict([(k, new_obj.tensor_from_layer(v)) for k,v in kwargs.items()])
        for (k,v) in kwargs.items():
            o_kwargs[k] = v
        new_obj.prev_args = (o_args, o_kwargs)
        new_obj.get_grads(*o_args, **o_kwargs)
        return new_obj

    def get_sample(self, **kwargs):
        """
        Get a sample from the curren model.
        ! Only works for output layers
        """
        if not hasattr(self, 'get_cost'):
            raise TypeError('Non-output layer does not support this method')
        new_obj = utils.copy(self)
        try:
            o_args, o_kwargs = new_obj.prev_args
        except:
            o_args, o_kwargs = ([], {})
        kwargs = dict([(k, new_obj.tensor_from_layer(v)) for k,v in kwargs.items()])
        for (k,v) in kwargs.items():
            o_kwargs[k] = v
        new_obj.prev_args = (o_args, o_kwargs)
        sample = new_obj.compute_sample(*o_args, **o_kwargs)
        return sample


    def __call__(self, *args, **kwargs):
        """
        Compose this layer with the inputs provided
        """
        if 'one_step' in kwargs and kwargs['one_step']:
            del kwargs['one_step']
            args = [self.tensor_from_layer(arg, False) for arg in args]
            kwargs = dict([(k, self.tensor_from_layer(v, False))
                           for k,v in kwargs.items()])
            if hasattr(self, 'step_fprop'):
                return self.step_fprop(*args, **kwargs)
            else:
                return self.fprop(*args, **kwargs)
        new_obj = utils.copy(self)

        args = [new_obj.tensor_from_layer(arg) for arg in args]
        kwargs = dict([(k, new_obj.tensor_from_layer(v)) for k,v in kwargs.items()])
        if 'do' in kwargs:
            kind = kwargs['do']
            del kwargs['do']
        else:
            kind = 'fprop'
        if 'one_step' in kwargs:
            del kwargs['one_step']
        new_obj.prev_args = (args, kwargs)
        if kind == 'fprop':
            new_obj.fprop(*args, **kwargs)
        elif kind == 'eval':
            new_obj.get_cost(*args, **kwargs)
        elif kind == 'train':
            new_obj.get_grads(*args, **kwargs)
        elif kind == 'run':
            return new_obj.run(*args, **kwargs)
        return new_obj


    def _init_params(self):
        raise NotImplementedError

    def fprop(self, state_below, state_before=None, state_after=None):
        raise NotImplementedError

class Model(Container):
    """
    Model class. It respects the interface expected by the trainer.
    """
    def __init__(self, output_layer,
                 sample_fn,
                 indx_word="/data/lisa/data/PennTreebankCorpus/dictionaries.npz",
                 indx_word_src=None,
                 rng =None):
        super(Model, self).__init__()
        if rng == None:
            rng = numpy.random.RandomState(123)
        assert hasattr(output_layer,'grads'), \
                'The model needs to have gradients defined'
        self.rng = rng
        self.trng = RandomStreams(rng.randint(1000)+1)
        self.sample_fn = sample_fn
        self.indx_word = indx_word
        self.indx_word_src = indx_word_src
        self.param_grads = output_layer.grads
        self.params = output_layer.params
        self.updates = output_layer.updates
        self.noise_params = output_layer.noise_params
        self.noise_params_shape_fn = output_layer.noise_params_shape_fn
        self.inputs = output_layer.inputs
        self.params_grad_scale = output_layer.params_grad_scale
        self.train_cost = output_layer.cost
        self.out = output_layer.out
        self.schedules = output_layer.schedules
        self.output_layer = output_layer
        self.properties = output_layer.properties
        self._get_samples = output_layer._get_samples

    def get_schedules(self):
        return self.schedules

    def validate(self, data_iterator):
        raise NotImplemented

    def clone(**new_inputs):
        new_obj = utils.copy(self)
        # Reorder inputs
        assert len(new_obj.inputs) == len(new_inputs.items())
        pairs=[(x, new_inputs[x.name]) for x in inputs]
        new_obj.inputs = new_inputs.values()
        new_obj.out = theano.clone(new_obj.out, replace=pairs)
        if hasattr(new_obj, 'cost'):
            new_obj.cost = theano.clone(new_obj.cost, replace=pairs)
        if hasattr(new_obj, 'grads'):
            new_obj.grads = theano.clone(new_obj.grads, replace=pairs)
        if hasattr(new_obj, 'sample'):
            new_obj.sample = theano.clone(new_obj.sample, replace=pairs)
        return new_obj

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

class Operator(Layer):
    def __init__(self,
#                 apply_operator=None,
                 n_in=0,
                 n_out = 0):
        super(Operator, self).__init__(n_in, n_out, rng=None)
        self.apply_operator = apply_operator

    def __call__(self, *args, **kwargs):
        rval = self.apply_operator(*args, **kwargs)
        if 'one_step' in kwargs and kwargs['one_step']:
            return rval
        self.params = rval.params
        self.noise_params = rval.noise_params
        self.noise_params_shape_fn = rval.noise_params_shape_fn
        self.params_grad_scale = rval.params_grad_scale

        self.inputs = rval.inputs
        self.schedules = rval.schedules
        return rval

    def on(self, *args, **kwargs):
        # Experimental
        if not hasattr(self, 'run_fn'):
            self.run_fn = theano.function(self.inputs, self.out)
        return self.run_fn(*args, **kwargs)
