"""
Utility functions


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import random
import string
import copy as pycopy

import theano
import theano.tensor as TT

def print_time(secs):
    if secs < 120.:
        return '%6.3f sec' % secs
    elif secs <= 60 * 60:
        return '%6.3f min' % (secs / 60.)
    else:
        return '%6.3f h  ' % (secs / 3600.)

def print_mem(context=None):
    if theano.sandbox.cuda.cuda_enabled:
        rvals = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
        # Avaliable memory in Mb
        available = float(rvals[0]) / 1024. / 1024.
        # Total memory in Mb
        total = float(rvals[1]) / 1024. / 1024.
        if context == None:
            print ('Used %.3f Mb Free  %.3f Mb, total %.3f Mb' %
                   (total - available, available, total))
        else:
            info = str(context)
            print (('GPU status : Used %.3f Mb Free %.3f Mb,'
                    'total %.3f Mb [context %s]') %
                    (total - available, available, total, info))

def const(value):
    return TT.constant(numpy.asarray(value, dtype=theano.config.floatX))

def as_floatX(variable):
    """
       This code is taken from pylearn2:
       Casts a given variable into dtype config.floatX
       numpy ndarrays will remain numpy ndarrays
       python floats will become 0-D ndarrays
       all other types will be treated as theano tensors
    """

    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)

def copy(x):
    new_x = pycopy.copy(x)
    new_x.params = [x for x in new_x.params]
    new_x.params_grad_scale      = [x for x in new_x.params_grad_scale    ]
    new_x.noise_params           = [x for x in new_x.noise_params         ]
    new_x.noise_params_shape_fn  = [x for x in new_x.noise_params_shape_fn]
    new_x.updates                = [x for x in new_x.updates              ]
    new_x.additional_gradients   = [x for x in new_x.additional_gradients ]
    new_x.inputs                 = [x for x in new_x.inputs               ]
    new_x.schedules              = [x for x in new_x.schedules            ]
    new_x.properties             = [x for x in new_x.properties           ]
    return new_x

def softmax(x):
    if x.ndim == 2:
        e = TT.exp(x)
        return e / TT.sum(e, axis=1).dimshuffle(0, 'x')
    else:
        e = TT.exp(x)
        return e/ TT.sum(e)

def sample_zeros(sizeX, sizeY, sparsity, scale, rng):
    return numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)

def sample_weights(sizeX, sizeY, sparsity, scale, rng):
    """
    Initialization that fixes the largest singular value.
    """
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.uniform(low=-scale, high=scale, size=(sparsity,))
        vals_norm = numpy.sqrt((new_vals**2).sum())
        new_vals = scale*new_vals/vals_norm
        values[dx, perm[:sparsity]] = new_vals
    _,v,_ = numpy.linalg.svd(values)
    values = scale * values/v[0]
    return values.astype(theano.config.floatX)

def sample_weights_classic(sizeX, sizeY, sparsity, scale, rng):
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)
    sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
    return values.astype(theano.config.floatX)

def sample_weights_orth(sizeX, sizeY, sparsity, scale, rng):
    sizeX = int(sizeX)
    sizeY = int(sizeY)

    assert sizeX == sizeY, 'for orthogonal init, sizeX == sizeY'

    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals

    u,s,v = numpy.linalg.svd(values)
    values = u * scale

    return values.astype(theano.config.floatX)

def init_bias(size, scale, rng):
    return numpy.ones((size,), dtype=theano.config.floatX)*scale

def id_generator(size=5, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for i in xrange(size))

def constant_shape(shape):
    return lambda *args, **kwargs : shape

def binVec2Int(binVec):
    add = lambda x,y: x+y
    return reduce(add,
                  [int(x) * 2 ** y
                   for x, y in zip(
                       list(binVec),range(len(binVec) - 1, -1,
                                                       -1))])

def Int2binVec(val, nbits=10):
    strVal = '{0:b}'.format(val)
    value = numpy.zeros((nbits,), dtype=theano.config.floatX)
    if theano.config.floatX == 'float32':
        value[:len(strVal)] = [numpy.float32(x) for x in strVal[::-1]]
    else:
        value[:len(strVal)] = [numpy.float64(x) for x in strVal[::-1]]
    return value

def dot(inp, matrix):
    """
    Decide the right type of dot product depending on the input
    arguments
    """
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

def dbg_hook(hook, x):
    if not isinstance(x, TT.TensorVariable):
        x.out = theano.printing.Print(global_fn=hook)(x.out)
        return x
    else:
        return theano.printing.Print(global_fn=hook)(x)
