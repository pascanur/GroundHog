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
from theano.sandbox.scan import scan

import numpy
import theano
import theano.tensor as TT
import sys
import logging

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
        return a * b

def none_if_zero(x):
    if x == 0:
        return None
    return x

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

class EncoderDecoderBase(object):

    def _create_embedding_layers(self, prefix):
        self.approx_embedder = MultiLayer(
            self.rng,
            n_in=self.state['nins'],
            n_hids=[self.state['rank_n_approx']],
            activation=[self.state['rank_n_activ']],
            name='{}_approx_embedder'.format(prefix),
            **self.default_kwargs)

        # We have 3 embeddings for each word in each level,
        # the one used as input,
        # the one used to control resetting gate,
        # the one used to control update gate.
        self.input_embedders = [lambda x : 0] * self.num_levels
        self.reset_embedders = [lambda x : 0] * self.num_levels
        self.update_embedders = [lambda x : 0] * self.num_levels
        embedder_kwargs = self.default_kwargs
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
        inter_level_kwargs = self.default_kwargs
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
        self.transtitions = [None] * self.num_levels
        for level in range(self.num_levels):
            self.transitions[level] = eval(self.state['rec_layer'])(
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
                    name='{}_transition_{}'.format(prefix, level))

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

    def _create_represetation_layers(self):
        # If we have a stack of RNN, then their last hidden states
        # are combined with a maxout layer.
        self.repr_contributors = [None] * self.num_levels
        for level in range(self.num_levels):
            self.repr_contributors[level] = MultiLayer(
                self.rng,
                n_in=self.state['dim'],
                n_hids=[self.state['dim'] * self.state['maxout_part']],
                activation=['lambda x: x'],
                name="enc_repr_contrib_"%level,
                **self.default_kwargs)
        self.repr_calculator = UnaryOp(activation=eval(self.state['unary_activ']), name="enc_repr_calc")

    def build_encoder(self, x, x_mask, use_noise):
        """Create the computational graph of the RNN Encoder"""

        seq_length = x.out.shape[0] // self.state['bs']

        input_signals = []
        reset_signals = []
        update_signals = []
        for level in range(self.num_levels):
            input_signals.append(self.input_embedder[level](self.approx_embeddings))
            update_signals.append(self.update_embedder[level](self.approx_embeddings))
            reset_signals.append(self.reset_embedder[level](self.approx_embeddings))

        hidden_layers = []
        for level in range(self.num_levels):
            if level > 0:
                input_signals[level] += self.inputter[level](self.hidden_layers[level - 1])
                update_signals[level] += self.updater[level](self.hidden_layers[level - 1])
                reset_signals[level] += self.resetter[level](self.hidden_layers[level - 1])
            hidden_layers[level] = self.transitions[level](
                    input_signals[level],
                    n_steps=seq_length,
                    batch_size=self.state['bs'],
                    mask=x_mask,
                    gater_below=none_if_zero(update_signals[level]),
                    reseter_below=none_if_zero(reset_signals[level]),
                    use_noise=use_noise)

        # If we no stack of RNN but only a usual one,
        # then the last hidden state is used as a representation.
        if self.num_levels == 1:
            return LastState(hidden_layers[0])

        # If we have a stack of RNN, then their last hidden states
        # are combined with a maxout layer.
        contributions = []
        for level in range(self.num_levels):
            contributions.append(self.repr_contributors(LastState(hidden_layers[level])))
        return self.repr_calculator(sum(contributions))

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

    def _create_initilization_layers(self):
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
                n_hids=self.fstate['dim'],
                activation=['lambda x:x']))
        for level in range(self.num_levels):
            self.decode_inputters[level] = MultiLayer(
                self.rng,
                name='dec_dec_inputter_'%level,
                learn_bias=False,
                **decoding_kwargs)
            if self.state['rec_gating']:
                self.decode_updaters[level] = MultiLayer(
                    self.rng,
                    name='dec_dec_updater_'%level,
                    learn_bias=False,
                    **decoding_kwargs)
            if self.state['rec_reseting']:
                self.decode_reseters[level] = MultiLayer(
                    self.rng,
                    name='dec_dec_reseter_'%level,
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
                name='dev_prev_readout_{}'.format(level),
                **self.default_kwargs)

        if self.state['deep_out']:
            act_layer = UnaryOp(activation=eval(self.state['unary_activ']))
            drop_layer = DropOp(rng=self.rng, dropout=self.state['dropout'])
            self.output_nonlinearity = drop_layer(act_layer)
            self.output_layer = SoftmaxLayer(
                    self.rng,
                    self.state['dim'] / self.state['maxout_part'],
                    self.state['nouts'],
                    sparsity=-1,
                    rank_n_approx=0,
                    name='dec_deep_softmax',
                    **self.default_kwargs)
        else:
            self.output_nonlinearity = lambda x : x
            self.output_layer = SoftmaxLayer(
                    self.rng,
                    self.state['dim'],
                    self.state['nouts'],
                    sparsity=-1,
                    rank_n_approx=self.state['rank_n_approx'],
                    rank_n_activ=self.state['rank_n_activ'],
                    name='dec_softmax',
                    **self.default_kwargs)

    def build_predictor(self, c, y, y_mask, use_noise):
        """Create the computational graph of the RNN Decoder"""

        seq_length = y.out.shape[0] // self.state['bs']

        input_signals = []
        reset_signals = []
        update_signals = []
        for level in range(self.num_levels):
            # Contributions directly from input words.
            input_signals.append(self.input_embedder[level](self.approx_embeddings))
            update_signals.append(self.update_embedder[level](self.approx_embeddings))
            reset_signals.append(self.reset_embedder[level](self.approx_embeddings))

            # Contributions from the encoded source sentence.
            input_signals[level] += self.decode_inputters[level](c)
            update_signals[level] += self.update_inputters[level](c)
            reset_signals[level] += self.reset_inputters[level](c)

        init_states = []
        for level in range(self.num_levels):
            init_states.append(self.initializers[level](c))

        hidden_layers = []
        for level in range(self.num_levels):
            if level > 0:
                input_signals[level] += self.inputter[level](self.hidden_layers[level - 1])
                update_signals[level] += self.updater[level](self.hidden_layers[level - 1])
                reset_signals[level] += self.resetter[level](self.hidden_layers[level - 1])
            hidden_layers[level] = self.transitions[level](
                    input_signals[level],
                    init_state=init_states[level],
                    n_steps=seq_length,
                    batch_size=self.state['bs'],
                    mask=y_mask,
                    gater_below=none_if_zero(update_signals[level]),
                    reseter_below=none_if_zero(reset_signals[level]),
                    use_noise=use_noise)

        readout = self.repr_readout(c)
        for level in range(self.num_levels):
            readout += self.hidden_readouts(hidden_layers[level])
        if y.ndim == 1:
            readout += Shift(self.prev_word_readout(y).reshape(y.shape[0], 1, self.state['dim']))
        else:
            readout += Shift(self.prev_word_readout(y))

        return self.output_layer.train(
                state_below=readout,
                target=y,
                mask=y_mask,
                reg=None)

def do_experiment(state, channel):
    def maxout(x):
        shape = x.shape
        if x.ndim == 1:
            shape1 = TT.cast(shape[0] / state['maxout_part'], 'int64')
            shape2 = TT.cast(state['maxout_part'], 'int64')
            x = x.reshape([shape1, shape2])
            x = x.max(1)
        else:
            shape1 = TT.cast(shape[1] / state['maxout_part'], 'int64')
            shape2 = TT.cast(state['maxout_part'], 'int64')
            x = x.reshape([shape[0], shape1, shape2])
            x = x.max(2)
        return x

    rng = numpy.random.RandomState(state['seed'])

    logger.debug("Load data...")
    if state['loopIters'] > 0:
        train_data, _1, _2 = get_data(state, rng)
    else:
        # Skip loading if no training is planned
        train_data = None

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
    training_c = encoder.build_encoder(x, x_mask, use_noise=True)

    logger.debug("Create predictor")
    decoder = Decoder(state, rng)
    predictions = decoder.build_predictor(training_c, y, y_mask, use_noise=True)

    logger.debug("Create language model and optimization algorithm")
    lm_model = LM_Model(
        cost_layer=predictions,
        weight_noise_amount=state['weight_noise_amount'],
        indx_word=state['indx_word_target'],
        indx_word_src=state['indx_word'],
        rng=rng)
    algo = SGD(lm_model, state, train_data) if state['loopIters'] > 0 else 0

    logger.debug("Run training")
    main = MainLoop(train_data, None, None, lm_model, algo, state, channel,
            reset=state['reset'], hooks=None)
    if state['reload']:
        main.load()
    if state['loopIters'] > 0:
        main.main()

def prototype_state():
    state = {}

    state['source'] = ["/data/lisatmp3/chokyun/mt/phrase_table.en.h5"]
    state['target'] = ["/data/lisatmp3/chokyun/mt/phrase_table.fr.h5"]
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
    state['unary_activ'] = 'maxout'

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
    state['sample_reset'] = False
    state['sample_n'] = 1
    state['sample_max'] = 3

    # Starts a funny sampling regime
    state['sampler_test'] = True
    state['seed'] = 1234

    # Specifies whether old model should be reloaded first
    state['reload'] = True

    # Number of batches to process
    state['loopIters'] = 50000000
    # Maximum number of minutes to run
    state['timeStop'] = 24*60*7
    # Error level to stop at
    state['minerr'] = -1

    # Resetting data iterator during training
    state['reset'] = -1
    state['shuffle'] = True
    state['cache_size'] = 10

    # Frequency of training error reports (in number of batches)
    state['trainFreq'] = 1
    # Frequency of running hooks
    state['hookFreq'] = 100
    # Validation frequency
    state['validFreq'] = 500
    # Model saving frequency (in minutes)
    state['saveFreq'] = 5

    # Turns on profiling of training phase
    state['profile'] = 0

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
    do_experiment(prototype_state(), None)

