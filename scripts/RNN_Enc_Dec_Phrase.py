'''
This is a test script for the RNN Encoder-Decoder

'''

from groundhog.datasets import TMIteratorPytables
from groundhog.trainer.SGD_adadelta import SGD
from groundhog.mainLoop import MainLoop
from groundhog.layers import \
        Layer,\
        MultiLayer, \
        RecurrentLayer, \
        SoftmaxLayer, \
        LastState, \
        DropOp, \
        UnaryOp, \
        Operator, \
        Shift, \
        GaussianNoise
from groundhog.models import LM_Model
from theano import scan

import numpy
import theano
import theano.tensor as TT
import theano.printing
import logging
import traceback
import sys
import pprint
import cPickle

theano.config.allow_gc = True

logger = logging.getLogger(__name__)

rect = 'lambda x:x*(x>0)'
htanh = 'lambda x:x*(x>-1)*(x<1)'

class ReplicateLayer(Layer):

    def __init__(self, n_times):
        self.n_times = n_times
        super(ReplicateLayer, self).__init__(0, 0, None)

    def fprop(self, matrix):
        # This is black magic based on broadcasting,
        # that's why variable names don't make any sense.
        a = TT.shape_padleft(matrix)
        b = TT.alloc(numpy.float32(1), self.n_times, 1, 1)
        self.out = a * b
        return self.out

def none_if_zero(x):
    if x == 0:
        return None
    return x

def dbg_sum(text, x):
    return x
    #if not isinstance(x, TT.TensorVariable):
    #    x.out = theano.printing.Print(text, attrs=['sum'])(x.out)
    #    return x
    #else:
    #    return theano.printing.Print(text, attrs=['sum'])(x)

def get_data(state, rng):

    def out_format (x, y, new_format=None):
        """A callback given to the iterator to transform data in suitable format

        :type x: list
        :param x: list of numpy.array's, each array is a batch of phrases
            in some of source languages

        :type y: list
        :param y: same as x but for target languages

        :param new_format: a wrapper to be applied on top of returned value

        :returns: a tuple (X, Xmask, Y, Ymask) where
            - X is a matrix, each column contains a source sequence
            - Xmask is 0-1 matrix, each column marks the sequence positions in X
            - Y and Ymask are matrices of the same format for target sequences
            OR new_format applied to the tuple

        Notes:
        * actually works only with x[0] and y[0]
        * len(x[0]) thus is just the minibatch size
        * len(x[0][idx]) is the size of sequence idx
        """

        # Similar length for all source sequences
        mx = numpy.minimum(state['seqlen'], max([len(xx) for xx in x[0]]))+1
        # Similar length for all target sequences
        my = numpy.minimum(state['seqlen'], max([len(xx) for xx in y[0]]))+1
        # Just batch size
        n = state['bs'] # FIXME: may become inefficient later with a large minibatch

        X = numpy.zeros((mx, n), dtype='int64')
        Y0 = numpy.zeros((my, n), dtype='int64')
        Y = numpy.zeros((my, n), dtype='int64')
        Xmask = numpy.zeros((mx, n), dtype='float32')
        Ymask = numpy.zeros((my, n), dtype='float32')

        # Fill X and Xmask
        for idx in xrange(len(x[0])):
            # Insert sequence idx in a column of matrix X
            if mx < len(x[0][idx]):
                # If sequence idx it too long,
                # we either choose random subsequence or just take a prefix
                if state['randstart']:
                    stx = numpy.random.randint(0, len(x[0][idx]) - mx)
                else:
                    stx = 0
                X[:mx, idx] = x[0][idx][stx:stx+mx]
            else:
                X[:len(x[0][idx]), idx] = x[0][idx][:mx]

            # Mark the end of phrase
            if len(x[0][idx]) < mx:
                X[len(x[0][idx]):, idx] = state['null_sym_source']

            # Initialize Xmask column with ones in all positions that
            # were just set in X
            Xmask[:len(x[0][idx]), idx] = 1.
            if len(x[0][idx]) < mx:
                Xmask[len(x[0][idx]), idx] = 1.

        # Fill Y and Ymask in the same way as X and Xmask in the previous loop
        for idx in xrange(len(y[0])):
            Y0[:len(y[0][idx]), idx] = y[0][idx][:my]
            if len(y[0][idx]) < my:
                Y0[len(y[0][idx]):, idx] = state['null_sym_target']
            Ymask[:len(y[0][idx]), idx] = 1.
            if len(y[0][idx]) < my:
                Ymask[len(y[0][idx]), idx] = 1.

        Y = Y0.copy()

        null_inputs = numpy.zeros(X.shape[1])

        # We say that an input pair is valid if both:
        # - either source sequence or target sequence is non-empty
        # - source sequence and target sequence have null_sym ending
        # Why did not we filter them earlier?
        for idx in xrange(X.shape[1]):
            if numpy.sum(Xmask[:,idx]) == 0 and numpy.sum(Ymask[:,idx]) == 0:
                null_inputs[idx] = 1
            if Xmask[-1,idx] and X[-1,idx] != state['null_sym_source']:
                null_inputs[idx] = 1
            if Ymask[-1,idx] and Y0[-1,idx] != state['null_sym_target']:
                null_inputs[idx] = 1

        valid_inputs = 1. - null_inputs

        # Leave only valid inputs
        X = X[:,valid_inputs.nonzero()[0]]
        Y = Y[:,valid_inputs.nonzero()[0]]
        Y0 = Y0[:,valid_inputs.nonzero()[0]]
        Xmask = Xmask[:,valid_inputs.nonzero()[0]]
        Ymask = Ymask[:,valid_inputs.nonzero()[0]]

        if len(valid_inputs.nonzero()[0]) <= 0:
            return None

        if n == 1:
            X = X[:,0]
            Y = Y[:,0]
            Y0 = Y0[:,0]
            Xmask = Xmask[:,0]
            Ymask = Ymask[:,0]
        if new_format:
            # Are Y and Y0 different?
            return new_format(X, Xmask, Y0, Y, Ymask)
        else:
            return X, Xmask, Y, Ymask

    new_format = lambda x,xm, y0, y, ym: {'x' : x, 'x_mask' :xm,
            'y': y0, 'y_mask' : ym}

    train_data = TMIteratorPytables(
        batch_size=int(state['bs']),
        target_lfiles=state['target'],
        source_lfiles=state['source'],
        output_format=lambda *args : out_format(*args,
                                                  new_format=new_format),
        can_fit=False,
        queue_size=10,
        cache_size=state['cache_size'],
        shuffle=state['shuffle'])

    valid_data = None
    test_data = None
    return train_data, valid_data, test_data

class Maxout(object):

    def __init__(self, maxout_part):
        self.maxout_part = maxout_part

    def __call__(self, x):
        shape = x.shape
        if x.ndim == 1:
            shape1 = TT.cast(shape[0] / self.maxout_part, 'int64')
            shape2 = TT.cast(self.maxout_part, 'int64')
            x = x.reshape([shape1, shape2])
            x = x.max(1)
        else:
            shape1 = TT.cast(shape[1] / self.maxout_part, 'int64')
            shape2 = TT.cast(self.maxout_part, 'int64')
            x = x.reshape([shape[0], shape1, shape2])
            x = x.max(2)
        return x

class EncoderDecoderBase(object):

    def _create_embedding_layers(self, prefix):
        self.approx_embedder = MultiLayer(
            self.rng,
            n_in=self.state['nins'],
            n_hids=[self.state['rank_n_approx']],
            activation=[self.state['rank_n_activ']],
            name='{}_approx_embdr'.format(prefix),
            **self.default_kwargs)

        # We have 3 embeddings for each word in each level,
        # the one used as input,
        # the one used to control resetting gate,
        # the one used to control update gate.
        self.input_embedders = [lambda x : 0] * self.num_levels
        self.reset_embedders = [lambda x : 0] * self.num_levels
        self.update_embedders = [lambda x : 0] * self.num_levels
        embedder_kwargs = dict(self.default_kwargs)
        embedder_kwargs.update(dict(
            n_in=self.state['rank_n_approx'],
            n_hids=[self.state['dim']],
            activation=['lambda x:x']))
        for level in range(self.num_levels):
            self.input_embedders[level] = MultiLayer(
                self.rng,
                name='{}_input_embdr_{}'.format(prefix, level),
                **embedder_kwargs)
            if self.state['rec_gating']:
                self.update_embedders[level] = MultiLayer(
                    self.rng,
                    learn_bias=False,
                    name='{}_update_embdr_{}'.format(prefix, level),
                    **embedder_kwargs)
            if self.state['rec_reseting']:
                self.reset_embedders[level] =  MultiLayer(
                    self.rng,
                    learn_bias=False,
                    name='{}_reset_embdr_{}'.format(prefix, level),
                    **embedder_kwargs)

    def _create_inter_level_layers(self, prefix):
        inter_level_kwargs = dict(self.default_kwargs)
        inter_level_kwargs.update(
                n_in=self.state['dim'],
                n_hids=self.state['dim'],
                activation=['lambda x:x'])

        self.inputers = [0] * self.num_levels
        self.reseters = [0] * self.num_levels
        self.updaters = [0] * self.num_levels
        for level in range(1, self.num_levels):
            self.inputters[level] = MultiLayer(self.rng,
                    name="{}_inputter_{}".format(prefix, level),
                    **inter_level_kwargs)
            if self.state['rec_reseting']:
                self.resetters[level] = MultiLayer(self.rng,
                    name="{}_reseter_{}".format(prefix, level),
                    **inter_level_kwargs)
            if self.state['rec_gating']:
                self.updaters[level] = MultiLayer(self.rng,
                    name="{}_updater_{}".format(prefix, level),
                    **inter_level_kwargs)

    def _create_transition_layers(self, prefix):
        self.transitions = []
        for level in range(self.num_levels):
            self.transitions.append(eval(self.state['rec_layer'])(
                    self.rng,
                    n_hids=self.state['dim'],
                    activation=self.state['activ'],
                    bias_scale=self.state['bias'],
                    scale=self.state['rec_weight_scale'],
                    init_fn=self.state['rec_weight_init_fn'],
                    weight_noise=self.state['weight_noise_rec'],
                    dropout=self.state['dropout_rec'],
                    gating=self.state['rec_gating'],
                    gater_activation=self.state['rec_gater'],
                    reseting=self.state['rec_reseting'],
                    reseter_activation=self.state['rec_reseter'],
                    name='{}_transition_{}'.format(prefix, level)))

class Encoder(EncoderDecoderBase):

    def __init__(self, state, rng):
        self.state = state
        self.rng = rng

        self.num_levels = self.state['encoder_stack']

    def create_layers(self):
        """ Create all elements of Encoder's computation graph"""

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'],
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self._create_embedding_layers('enc')
        self._create_transition_layers('enc')
        self._create_inter_level_layers('enc')
        self._create_representation_layers()

    def _create_representation_layers(self):
        # If we have a stack of RNN, then their last hidden states
        # are combined with a maxout layer.
        self.repr_contributors = [None] * self.num_levels
        for level in range(self.num_levels):
            self.repr_contributors[level] = MultiLayer(
                self.rng,
                n_in=self.state['dim'],
                n_hids=[self.state['dim'] * self.state['maxout_part']],
                activation=['lambda x: x'],
                name="enc_repr_contrib_{}".format(level),
                **self.default_kwargs)
        self.repr_calculator = UnaryOp(activation=eval(self.state['unary_activ']), name="enc_repr_calc")

    def build_encoder(self, x, x_mask, use_noise):
        """Create the computational graph of the RNN Encoder

        :param x: input variable, either vector of word indices or
        matrix of word indices, where each column is a sentence

        :param x_mask: when x is a matrix and input sequences are
        of variable length, this 1/0 matrix is used to specify
        the matrix positions where the input actually is

        :param use_noise: turns on addition of noise to weights
        """

        # Low rank embeddings of all the input words.
        # Shape in case of matrix input:
        #   (max_seq_len * batch_size, rank_n_approx),
        #   where max_seq_len is the maximum length of batch sequences.
        # Here and later n_words = max_seq_len * batch_size.
        # Shape in case of vector input:
        #   (seq_len, rank_n_approx)
        approx_embeddings = self.approx_embedder(x)
        dbg_sum("Approximate embeddings:", approx_embeddings)

        # Low rank embeddings are projected to contribute
        # to input, reset and update signals.
        # All the shapes: (n_words, dim)
        input_signals = []
        reset_signals = []
        update_signals = []
        for level in range(self.num_levels):
            input_signals.append(self.input_embedders[level](approx_embeddings))
            update_signals.append(self.update_embedders[level](approx_embeddings))
            reset_signals.append(self.reset_embedders[level](approx_embeddings))
            dbg_sum("Input embeddings:", input_signals[-1])
            dbg_sum("Update embeddings:", update_signals[-1])
            dbg_sum("Reset embeddings:", reset_signals[-1])

        # Hidden layers.
        # Shape in case of matrix input: (max_seq_len, batch_size, dim)
        # Shape in case of vector input: (seq_len, dim)
        hidden_layers = []
        for level in range(self.num_levels):
            # Each hidden layer (except the bottom one) receives
            # input, reset and update signals from below.
            # All the shapes: (n_words, dim)
            if level > 0:
                input_signals[level] += self.inputter[level](self.hidden_layers[-1])
                update_signals[level] += self.updater[level](self.hidden_layers[-1])
                reset_signals[level] += self.resetter[level](self.hidden_layers[-1])
            hidden_layers.append(self.transitions[level](
                    input_signals[level],
                    nsteps=x.shape[0],
                    batch_size=x.shape[1] if x.ndim == 2 else 1,
                    mask=x_mask,
                    gater_below=none_if_zero(update_signals[level]),
                    reseter_below=none_if_zero(reset_signals[level]),
                    use_noise=use_noise))

        # If we no stack of RNN but only a usual one,
        # then the last hidden state is used as a representation.
        # Return value shape in case of matrix input:
        #   (batch_size, dim)
        # Return value shape in case of vector input:
        #   (dim,)
        if self.num_levels == 1:
            return LastState()(hidden_layers[0])

        # If we have a stack of RNN, then their last hidden states
        # are combined with a maxout layer.
        # Return value however has the same shape.
        contributions = []
        for level in range(self.num_levels):
            contributions.append(self.repr_contributors(LastState()(hidden_layers[level])))
        c = self.repr_calculator(sum(contributions))
        dbg_sum("c:", c)
        return c

class Decoder(EncoderDecoderBase):

    def __init__(self, state, rng):
        self.state = state
        self.rng = rng

        self.num_levels = self.state['decoder_stack']

    def create_layers(self):
        """ Create all elements of Decoder's computation graph"""

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'],
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self._create_embedding_layers('dec')
        self._create_transition_layers('dec')
        self._create_inter_level_layers('dec')
        self._create_initialization_layers()
        self._create_decoding_layers()
        self._create_readout_layers()

    def _create_initialization_layers(self):
        self.initializers = [lambda x : None] * self.num_levels
        if self.state['bias_code']:
            for level in range(self.num_levels):
                self.initializers[level] = MultiLayer(
                    self.rng,
                    n_in=self.state['dim'],
                    n_hids=[self.state['dim']],
                    activation=[self.state['activ']],
                    bias_scale=[self.state['bias']],
                    name='dec_initializer_%d'%level,
                    **self.default_kwargs)

    def _create_decoding_layers(self):
        self.decode_inputters = [lambda x : 0] * self.num_levels
        self.decode_resetters = [lambda x : 0] * self.num_levels
        self.decode_updaters = [lambda x : 0] * self.num_levels
        decoding_kwargs = dict(self.default_kwargs)
        decoding_kwargs.update(dict(
                n_in=self.state['dim'],
                n_hids=self.state['dim'],
                activation=['lambda x:x']))
        for level in range(self.num_levels):
            self.decode_inputters[level] = MultiLayer(
                self.rng,
                name='dec_dec_inputter_{}'.format(level),
                learn_bias=False,
                **decoding_kwargs)
            if self.state['rec_gating']:
                self.decode_updaters[level] = MultiLayer(
                    self.rng,
                    name='dec_dec_updater_{}'.format(level),
                    learn_bias=False,
                    **decoding_kwargs)
            if self.state['rec_reseting']:
                self.decode_resetters[level] = MultiLayer(
                    self.rng,
                    name='dec_dec_reseter_{}'.format(level),
                    learn_bias=False,
                    **decoding_kwargs)

    def _create_readout_layers(self):
        self.repr_readout = MultiLayer(
                self.rng,
                n_in=self.state['dim'],
                n_hids=self.state['dim'],
                activation='lambda x: x',
                bias_scale=[self.state['bias_mlp']/3],
                learn_bias=False,
                name='dec_repr_readout',
                **self.default_kwargs)

        self.hidden_readouts = [None] * self.num_levels
        for level in range(self.num_levels):
            self.hidden_readouts[level] = MultiLayer(
                self.rng,
                n_in=self.state['dim'],
                n_hids=self.state['dim'],
                activation='lambda x: x',
                bias_scale=[self.state['bias_mlp']/3],
                name='dec_hid_readout_{}'.format(level),
                **self.default_kwargs)

        self.prev_word_readout = 0
        if self.state['bigram']:
            self.prev_word_readout = MultiLayer(
                self.rng,
                n_in=self.state['rank_n_approx'],
                n_hids=self.state['dim'],
                activation=['lambda x:x'],
                bias_scale=[self.state['bias_mlp']/3],
                learn_bias=False,
                name='dec_prev_readout_{}'.format(level),
                **self.default_kwargs)

        if self.state['deep_out']:
            act_layer = UnaryOp(activation=eval(self.state['unary_activ']))
            drop_layer = DropOp(rng=self.rng, dropout=self.state['dropout'])
            self.output_nonlinearities = [act_layer, drop_layer]
            self.output_layer = SoftmaxLayer(
                    self.rng,
                    self.state['dim'] / self.state['maxout_part'],
                    self.state['nouts'],
                    sparsity=-1,
                    rank_n_approx=self.state['rank_n_approx'],
                    name='dec_deep_softmax',
                    **self.default_kwargs)
        else:
            self.output_nonlinearities = []
            self.output_layer = SoftmaxLayer(
                    self.rng,
                    self.state['dim'],
                    self.state['nouts'],
                    sparsity=-1,
                    rank_n_approx=0,
                    name='dec_softmax',
                    sum_over_time=True,
                    **self.default_kwargs)

    def build_decoder(self, c, y, y_mask=None,
            for_sampling=False, given_init_states=None, T=1):
        """Create the computational graph of the RNN Decoder

        :param c: a representation produced by an encoder. Either (dim,)
        vector or (batch_size, dim) matrix

        :param y:
        if not for_sampling:
            target sequences, matrix of word indices,
            where each column is a sequence
        if for_sampling:
            a scalar corresponding to the previous word

        :param y_mask: if not for_sampling a 0/1 matrix determining lengths
            of the target sequences, must be None if for_sampling

        :param for_sampling: if True then builds the next word predictor,
            otherwise builds a log-likelihood evaluator

        :param given_init_states: if not None specifies
            the initial states of hidden layers

        :param T: sampling temperature
        """

        # Check parameter consistency
        if for_sampling:
            assert y_mask == None
            y = dbg_sum("Y:", y)

        # For log-likelihood evaluation the representation
        # be replicated for conveniency.
        # Shape if not for_sampling:
        #   (max_seq_len, batch_size, dim)
        # Shape if for_sampling:
        #   (dim,)
        if for_sampling:
            replicated_c = c
        else:
            replicated_c = ReplicateLayer(y.shape[0])(c)

        # Low rank embeddings of all the input words.
        # Shape if not for_sampling:
        #   (n_words, rank_n_approx),
        # Shape if for_sampling:
        #   (rank_n_approx, )
        approx_embeddings = dbg_sum("Approximate y embeddings:", self.approx_embedder(y))

        # Low rank embeddings are projected to contribute
        # to input, reset and update signals.
        # All the shapes if not for_sampling:
        #   (n_words, dim)
        # All the shape if for_sampling:
        #   (dim,)
        input_signals = []
        reset_signals = []
        update_signals = []
        for level in range(self.num_levels):
            # Contributions directly from input words.
            input_signals.append(self.input_embedders[level](approx_embeddings))
            update_signals.append(self.update_embedders[level](approx_embeddings))
            reset_signals.append(self.reset_embedders[level](approx_embeddings))

            # Contributions from the encoded source sentence.
            input_signals[level] += self.decode_inputters[level](replicated_c)
            update_signals[level] += self.decode_updaters[level](replicated_c)
            reset_signals[level] += self.decode_resetters[level](replicated_c)

            dbg_sum("Input signal:", input_signals[-1])
            dbg_sum("Update signal:", update_signals[-1])
            dbg_sum("Reset signal:", reset_signals[-1])

        # Hidden layers' initial states.
        # Shapes if not for_sampling:
        #   (batch_size, dim)
        # Shape if for_sampling:
        #   (,dim)
        init_states = given_init_states
        if not init_states:
            init_states = []
            for level in range(self.num_levels):
                init_states.append(self.initializers[level](c))

        # Hidden layers' states.
        # Shapes if not for_sampling:
        #  (seq_len, batch_size, dim)
        # Shapes if for_sampling:
        #  (,dim)
        hidden_layers = []
        for level in range(self.num_levels):
            if level > 0:
                input_signals[level] += self.inputter[level](self.hidden_layers[level - 1])
                update_signals[level] += self.updater[level](self.hidden_layers[level - 1])
                reset_signals[level] += self.resetter[level](self.hidden_layers[level - 1])
            hidden_layers.append(self.transitions[level](
                    input_signals[level],
                    mask=y_mask,
                    gater_below=none_if_zero(update_signals[level]),
                    reseter_below=none_if_zero(reset_signals[level]),
                    one_step=for_sampling,
                    use_noise=not for_sampling,
                    **(dict(state_before=init_states[level])
                        if for_sampling
                        else dict(init_state=init_states[level],
                            batch_size=self.state['bs'],
                            nsteps=y.shape[0]))
                        ))
            hidden_layers[-1] = dbg_sum("Hidden:", hidden_layers[-1])

        # In hidden_layers we do no have the initial state, but we need it.
        # Instead of it we have the last one, which we do not need.
        # So what we do is discard the last one and prepend the initial one.
        if not for_sampling:
            for level in range(self.num_levels):
                hidden_layers[level].out = TT.concatenate([
                        TT.shape_padleft(init_states[level].out),
                        hidden_layers[level].out])[:-1]

        # The output representation to be fed in softmax.
        # Shape if for_sampling:
        #   (n_words, dim_r)
        # Shape if not for_sampling:
        #   (,dim_r)
        # ... where dim_r depends on 'deep_out' option.
        readout = self.repr_readout(replicated_c)
        for level in range(self.num_levels):
            if for_sampling:
                read_from = init_states[level]
            else:
                read_from = hidden_layers[level]
            readout += self.hidden_readouts[level](read_from)
        if self.state['bigram']:
            if for_sampling:
                readout += self.prev_word_readout(approx_embeddings)
            else:
                if y.ndim == 1:
                    readout += Shift()(self.prev_word_readout(approx_embeddings).reshape(
                        (y.shape[0], 1, self.state['dim'])))
                else:
                    # This place needs explanation. When prev_word_readout is applied to
                    # approx_embeddings the resulting shape is
                    # (n_batches * sequence_length, repr_dimensionality). We first
                    # transform it into 3D tensor to shift forward in time. Then
                    # reshape it back.
                    readout += Shift()(self.prev_word_readout(approx_embeddings).reshape(
                        (y.shape[0], y.shape[1], self.state['dim']))).reshape(
                                readout.out.shape)
        for fun in self.output_nonlinearities:
            readout = fun(readout)

        if for_sampling:
            dbg_sum("Readout:", readout)
            sample = self.output_layer.get_sample(
                    state_below=readout,
                    temp=T)
            # Current SoftmaxLayer.get_cost is stupid,
            # that's why we have to reshape a lot.
            log_prob = self.output_layer.get_cost(
                    state_below=TT.shape_padleft(readout),
                    temp=T,
                    target=sample.reshape((1,)))
            return [sample] + [log_prob] + hidden_layers
        else:
            return self.output_layer.train(
                    state_below=readout,
                    target=y,
                    mask=y_mask,
                    reg=None)

    def sampling_step(self, *args):
        """Implements one step of sapling

        *args are necessary since the number of argument can vary"""

        args = iter(args)

        prev_word = next(args)
        assert prev_word.ndim == 0
        # skip the previous word log probability
        assert next(args).ndim == 0
        prev_hidden_states = [dbg_sum("PrevHidden:", next(args)) for k in range(self.num_levels)]
        assert prev_hidden_states[0].ndim == 1
        c = next(args)
        assert c.ndim == 1
        T = next(args)
        assert T.ndim == 0

        sample, log_prob, _ =  self.build_decoder(c, prev_word, for_sampling=True,
                given_init_states=prev_hidden_states, T=T)
        _1, _2, hidden_states = self.build_decoder(c, sample, for_sampling=True,
                given_init_states=prev_hidden_states, T=T)
        return sample, log_prob, hidden_states

    def build_sampler(self, n_steps, T, c):
        states = [TT.constant(0, dtype='int64'), TT.constant(0.0, dtype='float32')]\
                + [init(c).out for init in self.initializers]
        dbg_sum("Init:", states[2])
        params = [c, T]
        outputs, updates = scan(self.sampling_step,
                outputs_info=states,
                non_sequences=params,
                n_steps=n_steps,
                name="sampler_scan")
        return (outputs[0], outputs[1].sum()), updates

class RandomSamplePrinter(object):

    def __init__(self, state, model, train_iter, var_x, var_x_mask, var_y, var_y_mask):
        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

    def __call__(self):
        def cut_eol(words):
            for i, word in enumerate(words):
                if words[i] == '<eol>':
                    return words[:i + 1]

        sample_idx = 0
        while sample_idx < self.state['n_examples']:
            batch = self.train_iter.next()
            xs, ys = batch['x'], batch['y']
            for seq_idx in range(xs.shape[1]):
                if sample_idx == self.state['n_examples']:
                    break

                x, y = xs[:, seq_idx], ys[:, seq_idx]
                x_words = cut_eol(map(lambda w_idx : self.model.word_indxs_src[w_idx], x))
                y_words = cut_eol(map(lambda w_idx : self.model.word_indxs[w_idx], y))
                if len(x_words) == 0:
                    continue

                print "Input: {}".format(" ".join(x_words))
                print "Target: {}".format(" ".join(y_words))
                self.model.get_samples(self.state['seqlen'] + 1, self.state['n_samples'], x[:len(x_words)])
                sample_idx += 1

def do_experiment(state, channel):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug("Starting state: {}".format(pprint.pformat(state)))

    rng = numpy.random.RandomState(state['seed'])

    logger.debug("Load data")
    if state['loopIters'] > 0:
        train_data, _1, _2 = get_data(state, rng)
    else:
        # Skip loading if no training is planned
        train_data = None

    try:
        logger.debug("Create input variables")
        if state['bs'] == 1:
            x = TT.lvector('x')
            x_mask = TT.vector('x_mask')
            y = TT.lvector('y')
            y_mask = TT.vector('y_mask')
        else:
            x = TT.lmatrix('x')
            x_mask = TT.matrix('x_mask')
            y = TT.lmatrix('y')
            y_mask = TT.matrix('y_mask')

        logger.debug("Create encoder")
        encoder = Encoder(state, rng)
        encoder.create_layers()
        logger.debug("Build encoding computation graph")
        training_c = encoder.build_encoder(x, x_mask, use_noise=True)

        logger.debug("Create decoder")
        decoder = Decoder(state, rng)
        decoder.create_layers()
        logger.debug("Build log-likelihood computation graph")
        predictions = decoder.build_decoder(training_c, y, y_mask)

        logger.debug("Build sampling computation graph")
        sampling_x = TT.lvector("sampling_x")
        n_steps = TT.lscalar("n_steps")
        T = TT.scalar("T")
        sampling_c = encoder.build_encoder(sampling_x, x_mask=False, use_noise=False).out
        sampling_c = dbg_sum("Repr:", sampling_c)
        (sample, log_prob), updates = decoder.build_sampler(n_steps, T, sampling_c)
        logger.debug("Compile sampler")
        sample_fn = theano.function(
                inputs=[n_steps, T, sampling_x],
                outputs=[sample, log_prob],
                updates=updates,
                name="sample_fn")

        logger.debug("Create LM_Model and load dictionaries")
        lm_model = LM_Model(
            cost_layer=predictions,
            sample_fn=sample_fn,
            weight_noise_amount=state['weight_noise_amount'],
            indx_word=state['indx_word_target'],
            indx_word_src=state['indx_word'],
            rng=rng)
        lm_model.load_dict()
        logger.debug("Model params:\n{}".format(
            pprint.pformat([p.name for p in lm_model.params])))
        logger.debug("Create SGD")
        algo = SGD(lm_model, state, train_data, compil=state['loopIters'] > 0)
        logger.debug("Run training")
        main = MainLoop(train_data, None, None, lm_model, algo, state, channel,
                reset=state['reset'],
                hooks=[RandomSamplePrinter(state, lm_model, train_data,
                    x, x_mask, y, y_mask)])
        if state['reload']:
            main.load()
        if state['loopIters'] > 0:
            main.main()

        if state['sampler_test']:
            # This is a test script: we only sample
            indx_word = cPickle.load(open(state['word_indx'],'rb'))

            try:
                while True:
                    try:
                        seqin = raw_input('Input Sequence: ')
                        n_samples = int(raw_input('How many samples? '))
                        alpha = float(raw_input('Inverse Temperature? '))

                        seqin = seqin.lower()
                        seqin = seqin.split()

                        seqlen = len(seqin)
                        seq = numpy.zeros(seqlen+1, dtype='int64')
                        for idx,sx in enumerate(seqin):
                            try:
                                seq[idx] = indx_word[sx]
                            except:
                                seq[idx] = indx_word[state['oov']]
                        seq[-1] = state['null_sym_source']

                    except Exception:
                        print 'Something wrong with your input! Try again!'
                        continue

                    sentences = []
                    all_probs = []
                    for sidx in xrange(n_samples):
                        #import ipdb; ipdb.set_trace()
                        [values, probs] = lm_model.sample_fn(3 * seqlen, alpha, seq)
                        sen = []
                        for k in xrange(values.shape[0]):
                            if lm_model.word_indxs[values[k]] == '<eol>':
                                break
                            sen.append(lm_model.word_indxs[values[k]])
                        sentences.append(" ".join(sen))
                        all_probs.append(-probs)
                    sprobs = numpy.argsort(all_probs)
                    for pidx in sprobs:
                        print pidx,"(%f):"%(-all_probs[pidx]),sentences[pidx]
                    print

            except KeyboardInterrupt:
                print 'Interrupted'
                pass
    except:
        logger.debug("Exception in the main function:")
        traceback.print_exc()


def prototype_state():
    state = {}

    state['source'] = ["/data/lisatmp3/bahdanau/shuffled/phrase-table.en.h5"]
    state['target'] = ["/data/lisatmp3/bahdanau/shuffled/phrase-table.fr.h5"]
    state['indx_word'] = "/data/lisatmp3/chokyun/mt/ivocab_source.pkl"
    state['indx_word_target'] = "/data/lisatmp3/chokyun/mt/ivocab_target.pkl"
    state['word_indx'] = "/data/lisatmp3/chokyun/mt/vocab.en.pkl"
    state['oov'] = 'UNK'
    # TODO: delete this one
    state['randstart'] = False

    # These are end-of-sequence marks
    state['null_sym_source'] = 15000
    state['null_sym_target'] = 15000

    # These are vocabulary sizes for the source and target languages
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    # These are the number of input and output units
    state['nouts'] = state['n_sym_target']
    state['nins'] = state['n_sym_source']

    # This is for predicting the next target from the current one
    state['bigram'] = True

    # This for the hidden state initilization
    state['bias_code'] = True

    # This is for the input -> output shortcut
    state['avg_word'] = True

    state['eps'] = 1e-10

    # Dimensionality of hidden layers
    state['dim'] = 1000
    state['dim_mlp'] = state['dim']

    # Size of hidden layers' stack in encoder and decoder
    state['encoder_stack'] = 1
    state['decoder_stack'] = 1

    state['deep_out'] = True
    state['mult_out'] = False

    state['rank_n_approx'] = 100
    state['rank_n_activ'] = 'lambda x: x'

    # Hidden layer configuration
    state['rec_layer'] = 'RecurrentLayer'
    state['rec_gating'] = True
    state['rec_reseting'] = True
    state['rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'

    # Hidden-to-hidden activation function
    state['activ'] = 'lambda x: TT.tanh(x)'

    # This one is bias applied in the recurrent layer. It is likely
    # to be zero as MultiLayer already has bias.
    state['bias'] = 0.

    # This one is bias at the projection stage
    # TODO fully get what is it needed for
    state['bias_mlp'] = 0.

    # Specifiying the output layer
    state['maxout_part'] = 2.
    state['unary_activ'] = 'Maxout(2)'

    # Weight initialization parameters
    state['rec_weight_init_fn'] = 'sample_weights_orth'
    state['weight_init_fn'] = 'sample_weights_classic'
    state['rec_weight_scale'] = 1.
    state['weight_scale'] = 0.01

    # Dropout in output layer
    state['dropout'] = 1.
    # Dropout in recurrent layers
    state['dropout_rec'] = 1.

    # Random weight noise regularization settings
    state['weight_noise'] = False
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01

    # Threshold to cut the gradient
    state['cutoff'] = 1.
    # TODO: what does it do?
    state['cutoff_rescale_length'] = 0.

    # Adagrad setting
    state['adarho'] = 0.95
    state['adaeps'] = 1e-6

    # Learning rate stuff, not used in Adagrad
    state['patience'] = 1
    state['lr'] = 1.
    state['minlr'] = 0

    # Batch size
    state['bs']  = 64
    # TODO: not used???
    state['vbs'] = 64
    # Maximum sequence length
    state['seqlen'] = 30

    # Sampling hook settings
    state['n_samples'] = 3
    state['n_examples'] = 3

    # Starts a funny sampling regime
    state['sampler_test'] = True
    state['seed'] = 1234

    # Specifies whether old model should be reloaded first
    state['reload'] = True

    # Number of batches to process
    state['loopIters'] = 3000000
    # Maximum number of minutes to run
    state['timeStop'] = 24*60*7
    # Error level to stop at
    state['minerr'] = -1

    # Resetting data iterator during training
    state['reset'] = -1
    state['shuffle'] = False
    state['cache_size'] = 0

    # Frequency of training error reports (in number of batches)
    state['trainFreq'] = 1
    # Frequency of running hooks
    state['hookFreq'] = 10
    # Validation frequency
    state['validFreq'] = 500
    # Model saving frequency (in minutes)
    state['saveFreq'] = 1

    # Turns on profiling of training phase
    state['profile'] = 0

    # Raise exception if nan
    state['on_nan'] = 'raise'

    state['prefix'] = 'model_phrase_'

    # When set to 0 each new model dump will be saved in a new file
    state['overwrite'] = 1

    return state

def experiment(state, channel):
    proto = prototype_state()
    for k, v in proto.items():
        if not k in state:
            state[k] = v
    do_experiment(state, channel)

if __name__ == "__main__":
    state = prototype_state()
    options = eval("dict({})".format(", ".join(sys.argv[1:]))) if len(sys.argv) > 1 else dict()
    state.update(options)
    do_experiment(state, None)
