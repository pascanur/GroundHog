import numpy
import logging
import pprint
import operator
import itertools

import theano
import theano.tensor as TT

from groundhog.layers import\
        Layer,\
        MultiLayer,\
        SoftmaxLayer,\
        LSTMLayer, \
        RecurrentLayer,\
        RecursiveConvolutionalLayer,\
        UnaryOp,\
        Shift,\
        LastState,\
        DropOp
from groundhog.models import LM_Model
from groundhog.datasets import PytablesBitextIterator

logger = logging.getLogger(__name__)

def create_padded_batch(state, x, y, return_dict=False):
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
    #n = state['bs'] # FIXME: may become inefficient later with a large minibatch
    n = x[0].shape[0]

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

    if return_dict:
        # Are Y and Y0 different?
        return {'x' : X, 'x_mask' : Xmask,
            'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask

def get_batch_iterator(state, rng):

    class Iterator(PytablesBitextIterator):

        def __init__(self, *args, **kwargs):
            PytablesBitextIterator.__init__(self, *args, **kwargs)
            self.batch_iter = None
            self.peeked_batch = None

        def get_homogenous_batch_iter(self):
            while True:
                k_batches = state['sort_k_batches']
                batch_size = state['bs']
                data = [PytablesBitextIterator.next(self) for k in range(k_batches)]
                x = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(0), data))))
                y = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(1), data))))
                lens = numpy.asarray([map(len, x), map(len, y)])
                order = numpy.argsort(lens.max(axis=0))
                for k in range(k_batches):
                    indices = order[k * batch_size:(k + 1) * batch_size]
                    batch = create_padded_batch(state, [x[indices]], [y[indices]],
                            return_dict=True)
                    if batch:
                        yield batch

        def next(self, peek=False):
            if not self.batch_iter:
                self.batch_iter = self.get_homogenous_batch_iter()

            if self.peeked_batch:
                # Only allow to peek one batch
                assert not peek
                logger.debug("Use peeked batch")
                batch = self.peeked_batch
                self.peeked_batch = None
                return batch

            batch = next(self.batch_iter)
            if peek:
                self.peeked_batch = batch
            return batch

    train_data = Iterator(
        batch_size=int(state['bs']),
        target_file=state['target'][0],
        source_file=state['source'][0],
        can_fit=False,
        queue_size=1000,
        cache_size=state['cache_size'],
        shuffle=state['shuffle'],
        use_infinite_loop=state['use_infinite_loop'],
        max_len=state['seqlen'])
    return train_data

class ReplicateLayer(Layer):

    def __init__(self, n_times):
        self.n_times = n_times
        super(ReplicateLayer, self).__init__(0, 0, None)

    def fprop(self, x):
        # This is black magic based on broadcasting,
        # that's why variable names don't make any sense.
        a = TT.shape_padleft(x)
        padding = [1] * x.ndim
        b = TT.alloc(numpy.float32(1), self.n_times, *padding)
        self.out = a * b
        return self.out

def none_if_zero(x):
    if x == 0:
        return None
    return x

def dbg_sum(text, x):
    return x
    if not isinstance(x, TT.TensorVariable):
        x.out = theano.printing.Print(text, attrs=['sum'])(x.out)
        return x
    else:
        return theano.printing.Print(text, attrs=['sum'])(x)

def dbg_hook(hook, x):
    return x
    if not isinstance(x, TT.TensorVariable):
        x.out = theano.printing.Print(global_fn=hook)(x.out)
        return x
    else:
        return theano.printing.Print(global_fn=hook)(x)

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

def _prefix(state,p, s):
    if '%s_%s'%(p,s) in state:
        return state['%s_%s'%(p, s)]
    return state[s]

class EncoderDecoderBase(object):

    def _create_embedding_layers(self, prefix):
        logger.debug("_create_embedding_layers")
        self.approx_embedder = MultiLayer(
            self.rng,
            n_in=self.state['n_sym_source']
                if prefix == "enc"
                else self.state['n_sym_target'],
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
            n_hids=[self.state['dim'] * self.state['dim_mult']],
            activation=['lambda x:x']))
        for level in range(self.num_levels):
            self.input_embedders[level] = MultiLayer(
                self.rng,
                name='{}_input_embdr_{}'.format(prefix, level),
                **embedder_kwargs)
            if _prefix(self.state,prefix,'rec_gating'):
                self.update_embedders[level] = MultiLayer(
                    self.rng,
                    learn_bias=False,
                    name='{}_update_embdr_{}'.format(prefix, level),
                    **embedder_kwargs)
            if _prefix(self.state,prefix,'rec_reseting'):
                self.reset_embedders[level] =  MultiLayer(
                    self.rng,
                    learn_bias=False,
                    name='{}_reset_embdr_{}'.format(prefix, level),
                    **embedder_kwargs)

    def _create_inter_level_layers(self, prefix):
        logger.debug("_create_inter_level_layers")
        inter_level_kwargs = dict(self.default_kwargs)
        inter_level_kwargs.update(
                n_in=self.state['dim'],
                n_hids=self.state['dim'] * self.state['dim_mult'],
                activation=['lambda x:x'])

        self.inputers = [0] * self.num_levels
        self.reseters = [0] * self.num_levels
        self.updaters = [0] * self.num_levels
        for level in range(1, self.num_levels):
            self.inputers[level] = MultiLayer(self.rng,
                    name="{}_inputer_{}".format(prefix, level),
                    **inter_level_kwargs)
            if _prefix(self.state,prefix,'rec_reseting'):
                self.reseters[level] = MultiLayer(self.rng,
                    name="{}_reseter_{}".format(prefix, level),
                    **inter_level_kwargs)
            if _prefix(self.state,prefix,'rec_gating'):
                self.updaters[level] = MultiLayer(self.rng,
                    name="{}_updater_{}".format(prefix, level),
                    **inter_level_kwargs)

    def _create_transition_layers(self, prefix):
        logger.debug("_create_transition_layers")
        self.transitions = []
        for level in range(self.num_levels):
            self.transitions.append(eval(_prefix(self.state,prefix,'rec_layer'))(
                    self.rng,
                    n_hids=self.state['dim'],
                    activation=_prefix(self.state,prefix,'activ'),
                    bias_scale=self.state['bias'],
                    init_fn=(self.state['rec_weight_init_fn']
                        if not self.skip_init
                        else "sample_zeros"),
                    scale=_prefix(self.state,prefix,'rec_weight_scale'),
                    weight_noise=self.state['weight_noise_rec'],
                    dropout=self.state['dropout_rec'],
                    gating=_prefix(self.state,prefix,'rec_gating'),
                    gater_activation=_prefix(self.state,prefix,'rec_gater'),
                    reseting=_prefix(self.state,prefix,'rec_reseting'),
                    reseter_activation=_prefix(self.state,prefix,'rec_reseter'),
                    profile=self.state['profile'],
                    name='{}_transition_{}'.format(prefix, level)))

class Encoder(EncoderDecoderBase):

    def __init__(self, state, rng, skip_init=False):
        self.state = state
        self.rng = rng
        self.skip_init = skip_init

        self.num_levels = self.state['encoder_stack']
        
        if 'dim_mult' not in self.state:
            self.state['dim_mult'] = 1.

    def create_layers(self):
        """ Create all elements of Encoder's computation graph"""

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'] if not self.skip_init else "sample_zeros",
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self._create_embedding_layers('enc')
        self._create_transition_layers('enc')
        self._create_inter_level_layers('enc')
        self._create_representation_layers()

    def _create_representation_layers(self):
        logger.debug("_create_representation_layers")
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
        self.repr_calculator = UnaryOp(
                activation=eval(self.state['unary_activ']),
                name="enc_repr_calc")

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

        def inp_hook(op, x):
            if x.ndim == 2:
                values = x.sum(1).flatten()
            else:
                values = x.sum()
            logger.debug("Input signal: {}".format(values))
        input_signals[-1] = dbg_hook(inp_hook, input_signals[-1])

        # Hidden layers.
        # Shape in case of matrix input: (max_seq_len, batch_size, dim)
        # Shape in case of vector input: (seq_len, dim)
        hidden_layers = []
        for level in range(self.num_levels):
            # Each hidden layer (except the bottom one) receives
            # input, reset and update signals from below.
            # All the shapes: (n_words, dim)
            if level > 0:
                #if hidden_layers[-1].out.ndim == 3:
                #    hidden_layers[-1].out = hidden_layers[-1].out[:,:,:self.state['dim']]
                #else:
                #    hidden_layers[-1].out = hidden_layers[-1].out[:,:self.state['dim']]
                input_signals[level] += self.inputers[level](hidden_layers[-1])
                update_signals[level] += self.updaters[level](hidden_layers[-1])
                reset_signals[level] += self.reseters[level](hidden_layers[-1])
            hidden_layers.append(self.transitions[level](
                    input_signals[level],
                    nsteps=x.shape[0],
                    batch_size=x.shape[1] if x.ndim == 2 else 1,
                    mask=x_mask,
                    gater_below=none_if_zero(update_signals[level]),
                    reseter_below=none_if_zero(reset_signals[level]),
                    use_noise=use_noise))
        def hid_hook(op, x):
            if x.ndim == 3:
                values = x.sum(2).flatten()
            else:
                values = x.sum()
            logger.debug("Encoder hiddens: {}".format(values))
        hidden_layers[-1] = dbg_hook(hid_hook, hidden_layers[-1])

        # If we no stack of RNN but only a usual one,
        # then the last hidden state is used as a representation.
        # Return value shape in case of matrix input:
        #   (batch_size, dim)
        # Return value shape in case of vector input:
        #   (dim,)
        if self.num_levels == 1 or self.state['take_top']:
            c = LastState()(hidden_layers[-1])
            if c.out.ndim == 2:
                c.out = c.out[:,:self.state['dim']]
            else:
                c.out = c.out[:self.state['dim']]
            return c

        # If we have a stack of RNN, then their last hidden states
        # are combined with a maxout layer.
        # Return value however has the same shape.
        contributions = []
        for level in range(self.num_levels):
            contributions.append(self.repr_contributors[level](
                LastState()(hidden_layers[level])))
        # I do not know a good starting value for sum
        c = self.repr_calculator(sum(contributions[1:], contributions[0]))
        return c

class Decoder(EncoderDecoderBase):

    EVALUATION = 0
    SAMPLING = 1
    BEAM_SEARCH = 2

    def __init__(self, state, rng, skip_init=False):
        self.state = state
        self.rng = rng
        self.skip_init = skip_init

        self.num_levels = self.state['decoder_stack']
        
        if 'dim_mult' not in self.state:
            self.state['dim_mult'] = 1.

    def create_layers(self):
        """ Create all elements of Decoder's computation graph"""

        self.default_kwargs = dict(
            init_fn=self.state['weight_init_fn'] if not self.skip_init else "sample_zeros",
            weight_noise=self.state['weight_noise'],
            scale=self.state['weight_scale'])

        self._create_embedding_layers('dec')
        self._create_transition_layers('dec')
        self._create_inter_level_layers('dec')
        self._create_initialization_layers()
        self._create_decoding_layers()
        self._create_readout_layers()

    def _create_initialization_layers(self):
        logger.debug("_create_initialization_layers")
        self.initializers = [lambda x : None] * self.num_levels
        if self.state['bias_code']:
            for level in range(self.num_levels):
                self.initializers[level] = MultiLayer(
                    self.rng,
                    n_in=self.state['dim'],
                    n_hids=[self.state['dim']],
                    activation=[_prefix(self.state,'dec','activ')],
                    bias_scale=[self.state['bias']],
                    name='dec_initializer_%d'%level,
                    **self.default_kwargs)

    def _create_decoding_layers(self):
        logger.debug("_create_decoding_layers")
        self.decode_inputers = [lambda x : 0] * self.num_levels
        self.decode_reseters = [lambda x : 0] * self.num_levels
        self.decode_updaters = [lambda x : 0] * self.num_levels
        decoding_kwargs = dict(self.default_kwargs)
        decoding_kwargs.update(dict(
                n_in=self.state['dim'],
                n_hids=self.state['dim'] * self.state['dim_mult'],
                activation=['lambda x:x']))
        for level in range(self.num_levels):
            self.decode_inputers[level] = MultiLayer(
                self.rng,
                name='dec_dec_inputter_{}'.format(level),
                learn_bias=False,
                **decoding_kwargs)
            if _prefix(self.state,'dec','rec_gating'):
                self.decode_updaters[level] = MultiLayer(
                    self.rng,
                    name='dec_dec_updater_{}'.format(level),
                    learn_bias=False,
                    **decoding_kwargs)
            if _prefix(self.state,'dec','rec_reseting'):
                self.decode_reseters[level] = MultiLayer(
                    self.rng,
                    name='dec_dec_reseter_{}'.format(level),
                    learn_bias=False,
                    **decoding_kwargs)

    def _create_readout_layers(self):
        logger.debug("_create_readout_layers")
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
                    self.state['n_sym_target'],
                    sparsity=-1,
                    rank_n_approx=self.state['rank_n_approx'],
                    name='dec_deep_softmax',
                    **self.default_kwargs)
        else:
            self.output_nonlinearities = []
            self.output_layer = SoftmaxLayer(
                    self.rng,
                    self.state['dim'],
                    self.state['n_sym_target'],
                    sparsity=-1,
                    rank_n_approx=0,
                    name='dec_softmax',
                    sum_over_time=True,
                    **self.default_kwargs)

    def build_decoder(self, c, y, y_mask=None,
            mode=EVALUATION, given_init_states=None, T=1):
        """Create the computational graph of the RNN Decoder

        :param c: a representation produced by an encoder. Either (dim,)
        vector or (batch_size, dim) matrix

        :param y:
        if mode == evaluation:
            target sequences, matrix of word indices of shape (max_seq_len, batch_size),
            where each column is a sequence
        if mode != evaluation:
            a vector of previous words of shape (n_samples,)

        :param y_mask: if mode == evaluation a 0/1 matrix determining lengths
            of the target sequences, must be None otherwise

        :param mode: chooses on of three modes: evaluation, sampling and beam_search

        :param given_init_states: for sampling and beam_search. A list of hidden states
            matrices for each layer, each matrix is (n_samples, dim)

        :param T: sampling temperature
        """

        # Check parameter consistency
        if mode == Decoder.EVALUATION:
            assert not given_init_states
        else:
            assert not y_mask
            assert given_init_states
            if mode == Decoder.BEAM_SEARCH:
                assert T == 1

        # For log-likelihood evaluation the representation
        # be replicated for conveniency.
        # Shape if mode == evaluation
        #   (max_seq_len, batch_size, dim)
        # Shape if mode != evaluation
        #   (n_samples, dim)
        c = dbg_hook(lambda _, x : logger.debug("Representation: {}".format(x.sum())), c)
        replicated_c = ReplicateLayer(y.shape[0])(c)

        # Low rank embeddings of all the input words.
        # Shape if mode == evaluation
        #   (n_words, rank_n_approx),
        # Shape if mode != evaluation
        #   (n_samples, rank_n_approx)
        approx_embeddings = self.approx_embedder(y)

        # Low rank embeddings are projected to contribute
        # to input, reset and update signals.
        # All the shapes if mode == evaluation:
        #   (n_words, dim)
        # All the shape if mode != evaluation:
        #   (n_samples, dim)
        input_signals = []
        reset_signals = []
        update_signals = []
        for level in range(self.num_levels):
            # Contributions directly from input words.
            input_signals.append(self.input_embedders[level](approx_embeddings))
            update_signals.append(self.update_embedders[level](approx_embeddings))
            reset_signals.append(self.reset_embedders[level](approx_embeddings))

            # Contributions from the encoded source sentence.
            input_signals[level] += self.decode_inputers[level](replicated_c)
            update_signals[level] += self.decode_updaters[level](replicated_c)
            reset_signals[level] += self.decode_reseters[level](replicated_c)

        # Hidden layers' initial states.
        # Shapes if mode == evaluation:
        #   (batch_size, dim)
        # Shape if mode != evaluation:
        #   (n_samples, dim)
        init_states = given_init_states
        if not init_states:
            init_states = []
            for level in range(self.num_levels):
                init_states.append(self.initializers[level](c))

        # Hidden layers' states.
        # Shapes if mode == evaluation:
        #  (seq_len, batch_size, dim)
        # Shapes if mode != evaluation:
        #  (n_samples, dim)
        hidden_layers = []
        for level in range(self.num_levels):
            if level > 0:
                #if hidden_layers[level-1].out.ndim == 3:
                #    hidden_layers[level-1].out = hidden_layers[level-1].out[:,:,:self.state['dim']]
                #else:
                #    hidden_layers[level-1].out = hidden_layers[level-1].out[:,:self.state['dim']]
                input_signals[level] += self.inputers[level](hidden_layers[level - 1])
                update_signals[level] += self.updaters[level](hidden_layers[level - 1])
                reset_signals[level] += self.reseters[level](hidden_layers[level - 1])
            hidden_layers.append(self.transitions[level](
                    input_signals[level],
                    mask=y_mask,
                    gater_below=none_if_zero(update_signals[level]),
                    reseter_below=none_if_zero(reset_signals[level]),
                    one_step=mode != Decoder.EVALUATION,
                    use_noise=mode == Decoder.EVALUATION,
                    **(dict(state_before=init_states[level])
                        if mode != Decoder.EVALUATION
                        else dict(init_state=init_states[level],
                            batch_size=y.shape[1] if y.ndim == 2 else 1,
                            nsteps=y.shape[0]))
                        ))

            def hid_hook(op, x):
                if x.ndim == 3:
                    values = x.sum(2).flatten()
                else:
                    values = x.sum()
                logger.debug("Decoder hiddens: {}".format(values))
            hidden_layers[-1] = dbg_hook(hid_hook, hidden_layers[-1])

        # In hidden_layers we do no have the initial state, but we need it.
        # Instead of it we have the last one, which we do not need.
        # So what we do is discard the last one and prepend the initial one.
        if mode == Decoder.EVALUATION:
            for level in range(self.num_levels):
                if hidden_layers[level].out.shape[-1] != init_states[level].out.shape[-1]:
                    istate = TT.zeros_like(init_states[level].out)
                    istate = TT.concatenate([init_states[level].out, istate], axis=istate.shape.shape[0]-1)
                else:
                    istate = init_states[level].out

                hidden_layers[level].out = TT.concatenate([
                    TT.shape_padleft(istate),
                        hidden_layers[level].out])[:-1]

        # The output representation to be fed in softmax.
        # Shape if mode == evaluation
        #   (n_words, dim_r)
        # Shape if mode != evaluation
        #   (n_samples, dim_r)
        # ... where dim_r depends on 'deep_out' option.
        readout = self.repr_readout(replicated_c)
        for level in range(self.num_levels):
            if mode != Decoder.EVALUATION:
                read_from = init_states[level]
            else:
                read_from = hidden_layers[level]
            read_from_var = read_from if type(read_from) == theano.tensor.TensorVariable else read_from.out
            if read_from_var.ndim == 3:
                read_from_var = read_from_var[:,:,:self.state['dim']]
            else:
                read_from_var = read_from_var[:,:self.state['dim']]
            if type(read_from) != theano.tensor.TensorVariable:
                read_from.out = read_from_var
            else:
                read_from = read_from_var
            readout += self.hidden_readouts[level](read_from)
        if self.state['bigram']:
            if mode != Decoder.EVALUATION:
                check_first_word = (y > 0
                    if self.state['check_first_word']
                    else TT.ones((y.shape[0]), dtype="float32"))
                # padright is necessary as we want to multiply each row with a certain scalar
                readout += TT.shape_padright(check_first_word) * self.prev_word_readout(approx_embeddings).out
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

        def readout_hook(op, x):
            if x.ndim == 2:
                values = x.sum(1).flatten()
            else:
                values = x.sum()
            logger.debug("Readouts: {}".format(values))
        readout = dbg_hook(readout_hook, readout)

        if mode == Decoder.SAMPLING:
            sample = self.output_layer.get_sample(
                    state_below=readout,
                    temp=T)
            # Current SoftmaxLayer.get_cost is stupid,
            # that's why we have to reshape a lot.
            self.output_layer.get_cost(
                    state_below=readout.out,
                    temp=T,
                    target=sample)
            log_prob = self.output_layer.cost_per_sample
            return [sample] + [log_prob] + hidden_layers
        elif mode == Decoder.BEAM_SEARCH:
            return self.output_layer(
                    state_below=readout.out,
                    temp=T).out
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
        assert prev_word.ndim == 1
        # skip the previous word log probability
        assert next(args).ndim == 1
        prev_hidden_states = [dbg_sum("PrevHidden:", next(args)) for k in range(self.num_levels)]
        assert prev_hidden_states[0].ndim == 2
        c = next(args)
        assert c.ndim == 1
        T = next(args)
        assert T.ndim == 0

        sample, log_prob =  self.build_decoder(c, prev_word, mode=Decoder.SAMPLING,
                given_init_states=prev_hidden_states, T=T)[:2]
        hidden_states = self.build_decoder(c, sample, mode=Decoder.SAMPLING,
                given_init_states=prev_hidden_states, T=T)[2:]
        return [sample, log_prob] + hidden_states

    def build_initializers(self, c):
        return [init(c).out for init in self.initializers]

    def build_sampler(self, n_samples, n_steps, T, c):
        states = [TT.zeros(shape=(n_samples,), dtype='int64'),
                TT.zeros(shape=(n_samples,), dtype='float32')]\
                + [ReplicateLayer(n_samples)(init(c).out).out for init in self.initializers]
        dbg_sum("Init:", states[2])
        params = [c, T]
        outputs, updates = theano.scan(self.sampling_step,
                outputs_info=states,
                non_sequences=params,
                n_steps=n_steps,
                name="sampler_scan")
        return (outputs[0], outputs[1]), updates

    def build_next_probs_predictor(self, c, y, init_states):
        return self.build_decoder(c, y, mode=Decoder.BEAM_SEARCH,
                given_init_states=init_states)

    def build_next_states_computer(self, c, y, init_states):
        return self.build_decoder(c, y, mode=Decoder.SAMPLING,
                given_init_states=init_states)[2:]

class RNNEncoderDecoder(object):

    def __init__(self, state, rng, skip_init=False):
        self.state = state
        self.rng = rng
        self.skip_init = skip_init

    def build(self):
        logger.debug("Create input variables")
        self.x = TT.lmatrix('x')
        self.x_mask = TT.matrix('x_mask')
        self.y = TT.lmatrix('y')
        self.y_mask = TT.matrix('y_mask')
        self.inputs = [self.x, self.y, self.x_mask, self.y_mask]

        logger.debug("Create encoder")
        self.encoder = Encoder(self.state, self.rng, self.skip_init)
        self.encoder.create_layers()
        logger.debug("Build encoding computation graph")
        training_c = self.encoder.build_encoder(self.x, self.x_mask, use_noise=True)

        logger.debug("Create decoder")
        self.decoder = Decoder(self.state, self.rng, self.skip_init)
        self.decoder.create_layers()
        logger.debug("Build log-likelihood computation graph")
        self.predictions = self.decoder.build_decoder(training_c, self.y, self.y_mask)

        logger.debug("Build sampling computation graph")
        self.sampling_x = TT.lvector("sampling_x")
        self.n_samples = TT.lscalar("n_samples")
        self.n_steps = TT.lscalar("n_steps")
        self.T = TT.scalar("T")
        self.sampling_c = self.encoder.build_encoder(
                self.sampling_x, x_mask=None, use_noise=False).out
        (self.sample, self.sample_log_prob), self.sampling_updates =\
            self.decoder.build_sampler(self.n_samples, self.n_steps, self.T, self.sampling_c)

        logger.debug("Create auxiliary variables")
        self.c = TT.vector("c")
        self.current_states = [TT.matrix("cur_{}".format(i))
                for i in range(self.decoder.num_levels)]
        self.gen_y = TT.lvector("gen_y")

    def create_lm_model(self):
        if hasattr(self, 'lm_model'):
            return self.lm_model
        self.lm_model = LM_Model(
            cost_layer=self.predictions,
            sample_fn=self.create_sampler(),
            weight_noise_amount=self.state['weight_noise_amount'],
            indx_word=self.state['indx_word_target'],
            indx_word_src=self.state['indx_word'],
            rng=self.rng)
        self.lm_model.load_dict()
        logger.debug("Model params:\n{}".format(
            pprint.pformat([p.name for p in self.lm_model.params])))
        return self.lm_model

    def create_representation_computer(self):
        if not hasattr(self, "repr_fn"):
            self.repr_fn = theano.function(
                    inputs=[self.sampling_x],
                    outputs=[self.sampling_c],
                    name="repr_fn")
        return self.repr_fn

    def create_initializers(self):
        if not hasattr(self, "init_fn"):
            self.init_fn = theano.function(
                    inputs=[self.sampling_c],
                    outputs=self.decoder.build_initializers(self.sampling_c),
                    name="init_fn")
        return self.init_fn

    def create_sampler(self, many_samples=False):
        if hasattr(self, 'sample_fn'):
            return self.sample_fn
        logger.debug("Compile sampler")
        self.sample_fn = theano.function(
                inputs=[self.n_samples, self.n_steps, self.T, self.sampling_x],
                outputs=[self.sample, self.sample_log_prob],
                updates=self.sampling_updates,
                name="sample_fn")
        if not many_samples:
            def sampler(*args):
                return map(lambda x : x.squeeze(), self.sample_fn(1, *args))
            return sampler
        return self.sample_fn

    def create_scorer(self, batch=False):
        if not hasattr(self, 'score_fn'):
            logger.debug("Compile scorer")
            self.score_fn = theano.function(
                    inputs=self.inputs,
                    outputs=[-self.predictions.cost_per_sample],
                    name="score_fn")
        if batch:
            return self.score_fn
        def scorer(x, y):
            x_mask = numpy.ones(x.shape[0], dtype="float32")
            y_mask = numpy.ones(y.shape[0], dtype="float32")
            return self.score_fn(x[:, None], y[:, None],
                    x_mask[:, None], y_mask[:, None])
        return scorer

    def create_next_probs_computer(self):
        if not hasattr(self, 'next_probs_fn'):
            self.next_probs_fn = theano.function(
                    inputs=[self.c, self.gen_y] + self.current_states,
                    outputs=[self.decoder.build_next_probs_predictor(self.c, self.gen_y, self.current_states)],
                    name="next_probs_fn")
        return self.next_probs_fn

    def create_next_states_computer(self):
        if not hasattr(self, 'next_states_fn'):
            self.next_states_fn = theano.function(
                    inputs=[self.c, self.gen_y] + self.current_states,
                    outputs=self.decoder.build_next_states_computer(
                        self.c, self.gen_y, self.current_states),
                    name="next_states_fn")
        return self.next_states_fn


    def create_probs_computer(self):
        if not hasattr(self, 'probs_fn'):
            logger.debug("Compile probs computer")
            self.probs_fn = theano.function(
                    inputs=self.inputs,
                    outputs=[self.predictions.word_probs],
                    name="probs_fn")
        def probs_computer(x, y):
            x_mask = numpy.ones(x.shape[0], dtype="float32")
            y_mask = numpy.ones(y.shape[0], dtype="float32")
            return self.probs_fn(x[:, None], y[:, None],
                    x_mask[:, None], y_mask[:, None])
        return probs_computer

def parse_input(state, word2idx, line, raise_unk=False, idx2word=None):
    seqin = line.split()
    seqlen = len(seqin)
    seq = numpy.zeros(seqlen+1, dtype='int64')
    for idx,sx in enumerate(seqin):
        try:
            seq[idx] = word2idx[sx]
        except:
            if raise_unk:
                raise
            seq[idx] = word2idx[state['oov']]
    seq[-1] = state['null_sym_source']
    if idx2word:
        idx2word[state['null_sym_source']] = '<eos>'
        parsed_in = [idx2word[sx] for sx in seq]
        return seq, " ".join(parsed_in)

    return seq
