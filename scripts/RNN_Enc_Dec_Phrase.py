'''
This is a test script for the RNN Encoder-Decoder

'''

from groundhog.datasets import TMIteratorPytables
from groundhog.trainer.SGD_adadelta import SGD
from groundhog.mainLoop import MainLoop
from groundhog.layers import MultiLayer, \
        RecurrentLayer, \
        SoftmaxLayer, \
        LastState, \
        DropOp, \
        UnaryOp, \
        Operator, \
        Shift, \
        GaussianNoise
from groundhog.layers import last
from groundhog.models import LM_Model
from theano.sandbox.scan import scan

import numpy
import theano
import theano.tensor as TT
import sys
import logging

import cPickle as pkl

theano.config.allow_gc = True

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

rect = 'lambda x:x*(x>0)'
htanh = 'lambda x:x*(x>-1)*(x<1)'

def get_data(state):
    rng = numpy.random.RandomState(123)

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
        cache_size=10,
        shuffle=state['shuffle'])

    valid_data = None
    test_data = None
    return train_data, valid_data, test_data

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

    logger.info("Start loading")
    rng = numpy.random.RandomState(state['seed'])
    if state['loopIters'] > 0:
        train_data, valid_data, test_data = get_data(state)
    else:
        train_data = None
        valid_data = None
        test_data = None

    logger.info("Build layers")
    if state['bs'] == 1:
        x = TT.lvector('x')
        x_mask = TT.vector('x_mask')
        y = TT.lvector('y')
        y0 = y
        y_mask = TT.vector('y_mask')
    else:
        x = TT.lmatrix('x')
        x_mask = TT.matrix('x_mask')
        y = TT.lmatrix('y')
        y0 = y
        y_mask = TT.matrix('y_mask')

    bs = state['bs']

    # Dimensionality of word embedings.
    # The same as state['dim'] and in fact equals the number of hidden units.
    embdim = state['dim_mlp']

    logger.info("Source sentence")
    # Low-rank embeddings
    emb = MultiLayer(
        rng,
        n_in=state['nins'],
        n_hids=[state['rank_n_approx']],
        activation=[state['rank_n_activ']],
        init_fn=state['weight_init_fn'],
        weight_noise=state['weight_noise'],
        scale=state['weight_scale'],
        name='emb')

    emb_words = []
    if state['rec_gating']:
        gater_words = []
    if state['rec_reseting']:
        reseter_words = []
    # si always stands for the number in stack of RNNs (which is actually 1)
    for si in xrange(state['encoder_stack']):
        # In paper it is multiplication by W
        emb_words.append(MultiLayer(
            rng,
            n_in=state['rank_n_approx'],
            n_hids=[embdim],
            activation=['lambda x:x'],
            init_fn=state['weight_init_fn'],
            weight_noise=state['weight_noise'],
            scale=state['weight_scale'],
            name='emb_words_%d'%si))
        # In paper it is multiplication by W_z
        if state['rec_gating']:
            gater_words.append(MultiLayer(
                rng,
                n_in=state['rank_n_approx'],
                n_hids=[state['dim']],
                activation=['lambda x:x'],
                init_fn=state['weight_init_fn'],
                weight_noise=state['weight_noise'],
                scale=state['weight_scale'],
                learn_bias = False,
                name='gater_words_%d'%si))
        # In paper it is multiplication by W_r
        if state['rec_reseting']:
            reseter_words.append(MultiLayer(
                rng,
                n_in=state['rank_n_approx'],
                n_hids=[state['dim']],
                activation=['lambda x:x'],
                init_fn=state['weight_init_fn'],
                weight_noise=state['weight_noise'],
                scale=state['weight_scale'],
                learn_bias = False,
                name='reseter_words_%d'%si))

    add_rec_step = []
    rec_proj = []
    if state['rec_gating']:
        rec_proj_gater = []
    if state['rec_reseting']:
        rec_proj_reseter = []
    for si in xrange(state['encoder_stack']):
        if si > 0:
            rec_proj.append(MultiLayer(
                rng,
                n_in=state['dim'],
                n_hids=[embdim],
                activation=['lambda x:x'],
                init_fn=state['rec_weight_init_fn'],
                weight_noise=state['weight_noise'],
                scale=state['rec_weight_scale'],
                name='rec_proj_%d'%si))
            if state['rec_gating']:
                rec_proj_gater.append(MultiLayer(
                    rng,
                    n_in=state['dim'],
                    n_hids=[state['dim']],
                    activation=['lambda x:x'],
                    init_fn=state['weight_init_fn'],
                    weight_noise=state['weight_noise'],
                    scale=state['weight_scale'],
                    learn_bias=False,
                    name='rec_proj_gater_%d'%si))
            if state['rec_reseting']:
                rec_proj_reseter.append(MultiLayer(
                    rng,
                    n_in=state['dim'],
                    n_hids=[state['dim']],
                    activation=['lambda x:x'],
                    init_fn=state['weight_init_fn'],
                    weight_noise=state['weight_noise'],
                    scale=state['weight_scale'],
                    learn_bias=False,
                    name='rec_proj_reseter_%d'%si))

        # This should be U from paper
        add_rec_step.append(eval(state['rec_layer'])(
                rng,
                n_hids=state['dim'],
                activation = state['activ'],
                bias_scale = state['bias'],
                scale=state['rec_weight_scale'],
                init_fn=state['rec_weight_init_fn'],
                weight_noise=state['weight_noise_rec'],
                dropout=state['dropout_rec'],
                gating=state['rec_gating'],
                gater_activation=state['rec_gater'],
                reseting=state['rec_reseting'],
                reseter_activation=state['rec_reseter'],
                name='add_h_%d'%si))

    def _add_op(words_embeddings,
                words_mask=None,
                prev_val=None,
                si = 0,
                state_below = None,
                gater_below = None,
                reseter_below = None,
                one_step=False,
                bs=1,
                init_state=None,
                use_noise=True):
        seqlen = words_embeddings.out.shape[0]//bs
        rval = words_embeddings
        gater = None
        reseter = None
        if state['rec_gating']:
            gater = gater_below
        if state['rec_reseting']:
            reseter = reseter_below
        if si > 0:
            rval += rec_proj[si-1](state_below, one_step=one_step,
                    use_noise=use_noise)
            if state['rec_gating']:
                projg = rec_proj_gater[si-1](state_below, one_step=one_step,
                        use_noise = use_noise)
                if gater: gater += projg
                else: gater = projg
            if state['rec_reseting']:
                projg = rec_proj_reseter[si-1](state_below, one_step=one_step,
                        use_noise = use_noise)
                if reseter: reseter += projg
                else: reseter = projg

        if not one_step:
            rval= add_rec_step[si](
                rval,
                nsteps=seqlen,
                batch_size=bs,
                mask=words_mask,
                gater_below = gater,
                reseter_below = reseter,
                one_step=one_step,
                init_state=init_state,
                use_noise = use_noise)
        else:
            #Here we link the Encoder part
            rval= add_rec_step[si](
                rval,
                mask=words_mask,
                state_before=prev_val,
                gater_below = gater,
                reseter_below = reseter,
                one_step=one_step,
                init_state=init_state,
                use_noise = use_noise)
        return rval
    add_op = Operator(_add_op)

    logger.info("Target sequence")
    emb_t = MultiLayer(
        rng,
        n_in=state['nouts'],
        n_hids=[state['rank_n_approx']],
        activation=[state['rank_n_activ']],
        init_fn=state['weight_init_fn'],
        weight_noise=state['weight_noise'],
        scale=state['weight_scale'],
        name='emb_t')

    emb_words_t = []
    if state['rec_gating']:
        gater_words_t = []
    if state['rec_reseting']:
        reseter_words_t = []
    for si in xrange(state['decoder_stack']):
        emb_words_t.append(MultiLayer(
            rng,
            n_in=state['rank_n_approx'],
            n_hids=[embdim],
            activation=['lambda x:x'],
            init_fn=state['weight_init_fn'],
            weight_noise=state['weight_noise'],
            scale=state['weight_scale'],
            name='emb_words_t_%d'%si))
        if state['rec_gating']:
            gater_words_t.append(MultiLayer(
                rng,
                n_in=state['rank_n_approx'],
                n_hids=[state['dim']],
                activation=['lambda x:x'],
                init_fn=state['weight_init_fn'],
                weight_noise=state['weight_noise'],
                scale=state['weight_scale'],
                learn_bias=False,
                name='gater_words_t_%d'%si))
        if state['rec_reseting']:
            reseter_words_t.append(MultiLayer(
                rng,
                n_in=state['rank_n_approx'],
                n_hids=[state['dim']],
                activation=['lambda x:x'],
                init_fn=state['weight_init_fn'],
                weight_noise=state['weight_noise'],
                scale=state['weight_scale'],
                learn_bias=False,
                name='reseter_words_t_%d'%si))

    proj_everything_t = []
    if state['rec_gating']:
        gater_everything_t = []
    if state['rec_reseting']:
        reseter_everything_t = []
    for si in xrange(state['decoder_stack']):
        # This stands for the matrix C from the text
        proj_everything_t.append(MultiLayer(
            rng,
            n_in=state['dim'],
            n_hids=[embdim],
            activation=['lambda x:x'],
            init_fn=state['weight_init_fn'],
            weight_noise=state['weight_noise'],
            scale=state['weight_scale'],
            name='proj_everything_t_%d'%si,
            learn_bias = False))
        if state['rec_gating']:
            gater_everything_t.append(MultiLayer(
                rng,
                n_in=state['dim'],
                n_hids=[state['dim']],
                activation=['lambda x:x'],
                init_fn=state['weight_init_fn'],
                weight_noise=state['weight_noise'],
                scale=state['weight_scale'],
                name='gater_everything_t_%d'%si,
                learn_bias = False))
        if state['rec_reseting']:
            reseter_everything_t.append(MultiLayer(
                rng,
                n_in=state['dim'],
                n_hids=[state['dim']],
                activation=['lambda x:x'],
                init_fn=state['weight_init_fn'],
                weight_noise=state['weight_noise'],
                scale=state['weight_scale'],
                name='reseter_everything_t_%d'%si,
                learn_bias = False))

    add_rec_step_t = []
    rec_proj_t = []
    if state['rec_gating']:
        rec_proj_t_gater = []
    if state['rec_reseting']:
        rec_proj_t_reseter = []
    for si in xrange(state['decoder_stack']):
        if si > 0:
            rec_proj_t.append(MultiLayer(
                rng,
                n_in=state['dim'],
                n_hids=[embdim],
                activation=['lambda x:x'],
                init_fn=state['rec_weight_init_fn'],
                weight_noise=state['weight_noise'],
                scale=state['rec_weight_scale'],
                name='rec_proj_%d'%si))
            if state['rec_gating']:
                rec_proj_t_gater.append(MultiLayer(
                    rng,
                    n_in=state['dim'],
                    n_hids=[state['dim']],
                    activation=['lambda x:x'],
                    init_fn=state['weight_init_fn'],
                    weight_noise=state['weight_noise'],
                    scale=state['weight_scale'],
                    learn_bias=False,
                    name='rec_proj_t_gater_%d'%si))
            if state['rec_reseting']:
                rec_proj_t_reseter.append(MultiLayer(
                    rng,
                    n_in=state['dim'],
                    n_hids=[state['dim']],
                    activation=['lambda x:x'],
                    init_fn=state['weight_init_fn'],
                    weight_noise=state['weight_noise'],
                    scale=state['weight_scale'],
                    learn_bias=False,
                    name='rec_proj_t_reseter_%d'%si))

        # This one stands for gating, resetting and applying non-linearity in Decoder
        add_rec_step_t.append(eval(state['rec_layer'])(
                rng,
                n_hids=state['dim'],
                activation = state['activ'],
                bias_scale = state['bias'],
                scale=state['rec_weight_scale'],
                init_fn=state['rec_weight_init_fn'],
                weight_noise=state['weight_noise_rec'],
                dropout=state['dropout_rec'],
                gating=state['rec_gating'],
                gater_activation=state['rec_gater'],
                reseting=state['rec_reseting'],
                reseter_activation=state['rec_reseter'],
                name='add_h_t_%d'%si))

    if state['encoder_stack'] > 1:
        encoder_proj = []
        for si in xrange(state['encoder_stack']):
            encoder_proj.append(MultiLayer(
                rng,
                n_in=state['dim'],
                n_hids=[state['dim'] * state['maxout_part']],
                activation=['lambda x: x'],
                init_fn=state['weight_init_fn'],
                weight_noise=state['weight_noise'],
                scale=state['weight_scale'],
                name='encoder_proj_%d'%si,
                learn_bias = (si == 0)))

        encoder_act_layer = UnaryOp(activation=eval(state['unary_activ']),
                indim=indim, pieces=pieces, rng=rng)

    # Actually add target opp
    def _add_t_op(words_embeddings, everything = None, words_mask=None,
                prev_val=None,one_step=False, bs=1,
                init_state=None, use_noise=True,
                gater_below = None,
                reseter_below = None,
                si = 0, state_below = None):
        seqlen = words_embeddings.out.shape[0]//bs

        rval = words_embeddings
        gater = None
        if state['rec_gating']:
            gater = gater_below
        reseter = None
        if state['rec_reseting']:
            reseter = reseter_below
        if si > 0:
            if isinstance(state_below, list):
                state_below = state_below[-1]
            rval += rec_proj_t[si-1](state_below,
                    one_step=one_step, use_noise=use_noise)
            if state['rec_gating']:
                projg = rec_proj_t_gater[si-1](state_below, one_step=one_step,
                        use_noise = use_noise)
                if gater: gater += projg
                else: gater = projg
            if state['rec_reseting']:
                projg = rec_proj_t_reseter[si-1](state_below, one_step=one_step,
                        use_noise = use_noise)
                if reseter: reseter += projg
                else: reseter = projg
        if everything:
            rval = rval + proj_everything_t[si](everything)
            if state['rec_gating']:
                everyg = gater_everything_t[si](everything, one_step=one_step, use_noise=use_noise)
                if gater: gater += everyg
                else: gater = everyg
            if state['rec_reseting']:
                everyg = reseter_everything_t[si](everything, one_step=one_step, use_noise=use_noise)
                if reseter: reseter += everyg
                else: reseter = everyg

        if not one_step:
            rval = add_rec_step_t[si](
                rval,
                nsteps=seqlen,
                batch_size=bs,
                mask=words_mask,
                one_step=one_step,
                init_state=init_state,
                gater_below=gater,
                reseter_below=reseter,
                use_noise=use_noise)
        else:
            # Here we link the Decoder part
            rval = add_rec_step_t[si](
                rval,
                mask=words_mask,
                state_before=prev_val,
                one_step=one_step,
                gater_below=gater,
                reseter_below=reseter,
                use_noise=use_noise)
        return rval
    add_t_op = Operator(_add_t_op)

    outdim = state['dim_mlp']
    if not state['deep_out']:
        outdim = state['rank_n_approx']

    if state['bias_code']:
        bias_code = []
        for si in xrange(state['decoder_stack']):
            bias_code.append(MultiLayer(
                rng,
                n_in=state['dim'],
                n_hids=[state['dim']],
                activation = [state['activ']],
                bias_scale = [state['bias']],
                scale=state['weight_scale'],
                init_fn=state['weight_init_fn'],
                weight_noise=state['weight_noise'],
                name='bias_code_%d'%si))

    if state['avg_word']:
        word_code_nin = state['rank_n_approx']
        word_code = MultiLayer(
            rng,
            n_in=word_code_nin,
            n_hids=[outdim],
            activation = 'lambda x:x',
            bias_scale = [state['bias_mlp']/3],
            scale=state['weight_scale'],
            init_fn=state['weight_init_fn'],
            weight_noise=state['weight_noise'],
            learn_bias = False,
            name='word_code')

    proj_code = MultiLayer(
        rng,
        n_in=state['dim'],
        n_hids=[outdim],
        activation = 'lambda x: x',
        bias_scale = [state['bias_mlp']/3],
        scale=state['weight_scale'],
        init_fn=state['weight_init_fn'],
        weight_noise=state['weight_noise'],
        learn_bias = False,
        name='proj_code')

    proj_h = []
    for si in xrange(state['decoder_stack']):
        proj_h.append(MultiLayer(
            rng,
            n_in=state['dim'],
            n_hids=[outdim],
            activation = 'lambda x: x',
            bias_scale = [state['bias_mlp']/3],
            scale=state['weight_scale'],
            init_fn=state['weight_init_fn'],
            weight_noise=state['weight_noise'],
            name='proj_h_%d'%si))

    if state['bigram']:
        proj_word = MultiLayer(
            rng,
            n_in=state['rank_n_approx'],
            n_hids=[outdim],
            activation=['lambda x:x'],
            bias_scale = [state['bias_mlp']/3],
            init_fn=state['weight_init_fn'],
            weight_noise=state['weight_noise'],
            scale=state['weight_scale'],
            learn_bias = False,
            name='emb_words_lm')

    if state['deep_out']:
        indim = 0
        pieces = 0
        act_layer = UnaryOp(activation=eval(state['unary_activ']))
        drop_layer = DropOp(rng=rng, dropout=state['dropout'])

    if state['deep_out']:
        indim = state['dim_mlp'] / state['maxout_part']
        rank_n_approx = state['rank_n_approx']
        rank_n_activ = state['rank_n_activ']
    else:
        indim = state['rank_n_approx']
        rank_n_approx = 0
        rank_n_activ = None
    output_layer = SoftmaxLayer(
            rng,
            indim,
            state['nouts'],
            state['weight_scale'],
            -1,
            rank_n_approx = rank_n_approx,
            rank_n_activ = rank_n_activ,
            weight_noise=state['weight_noise'],
            init_fn=state['weight_init_fn'],
            name='out')

    def _pop_op(everything, accum, everything_max = None,
            everything_min = None, word = None, aword = None,
            one_step=False, use_noise=True):

        rval = proj_h[0](accum[0], one_step=one_step, use_noise=use_noise)
        for si in xrange(1,state['decoder_stack']):
            rval += proj_h[si](accum[si], one_step=one_step, use_noise=use_noise)
        if state['mult_out']:
            rval = rval * everything
        else:
            rval = rval + everything

        if aword and state['avg_word']:
            wcode = aword
            if one_step:
                if state['mult_out']:
                    rval = rval * wcode
                else:
                    rval = rval + wcode
            else:
                if not isinstance(wcode, TT.TensorVariable):
                    wcode = wcode.out
                shape = wcode.shape
                rshape = rval.shape
                rval = rval.reshape([rshape[0]/shape[0], shape[0], rshape[1]])
                wcode = wcode.dimshuffle('x', 0, 1)
                if state['mult_out']:
                    rval = rval * wcode
                else:
                    rval = rval + wcode
                rval = rval.reshape(rshape)
        if word and state['bigram']:
            if one_step:
                if state['mult_out']:
                    rval *= proj_word(emb_t(word, use_noise=use_noise),
                            one_step=one_step, use_noise=use_noise)
                else:
                    rval += proj_word(emb_t(word, use_noise=use_noise),
                            one_step=one_step, use_noise=use_noise)
            else:
                if isinstance(word, TT.TensorVariable):
                    shape = word.shape
                    ndim = word.ndim
                else:
                    shape = word.shape
                    ndim = word.out.ndim
                pword = proj_word(emb_t(word, use_noise=use_noise),
                        one_step=one_step, use_noise=use_noise)
                shape_pword = pword.shape
                if ndim == 1:
                    pword = Shift()(pword.reshape([shape[0], 1, outdim]))
                else:
                    pword = Shift()(pword.reshape([shape[0], shape[1], outdim]))
                if state['mult_out']:
                    rval *= pword.reshape(shape_pword)
                else:
                    rval += pword.reshape(shape_pword)
        if state['deep_out']:
            rval = drop_layer(act_layer(rval), use_noise=use_noise)
        return rval

    pop_op = Operator(_pop_op)

    logger.info("Construct the model")
    gater_below = None
    if state['rec_gating']:
        gater_below = gater_words[0](emb(x))
    reseter_below = None
    if state['rec_reseting']:
        reseter_below = reseter_words[0](emb(x))
    encoder_acts = [add_op(emb_words[0](emb(x)), x_mask,
        bs=x_mask.shape[1],
        si=0, gater_below=gater_below, reseter_below=reseter_below)]
    if state['encoder_stack'] > 1:
        everything = encoder_proj[0](last(encoder_acts[-1]))
    for si in xrange(1,state['encoder_stack']):
        gater_below = None
        if state['rec_gating']:
            gater_below = gater_words[si](emb(x))
        reseter_below = None
        if state['rec_reseting']:
            reseter_below = reseter_words[si](emb(x))
        encoder_acts.append(add_op(emb_words[si](emb(x)),
                x_mask, bs=x_mask.shape[1],
                si=si, state_below=encoder_acts[-1],
                gater_below=gater_below,
                reseter_below=reseter_below))
        if state['encoder_stack'] > 1:
            everything += encoder_proj[si](last(encoder_acts[-1]))

    if state['encoder_stack'] <= 1:
        encoder = encoder_acts[-1]
        everything = LastState(ntimes=True,n=y.shape[0])(encoder)
    else:
        everything = encoder_act_layer(everything)
        everything = everything.reshape([1, everything.shape[0], everything.shape[1]])
        everything = LastState(ntimes=True,n=y.shape[0])(everything)

    if state['bias_code']:
        init_state = [bc(everything[-1]) for bc in bias_code]
    else:
        init_state = [None for bc in bias_code]

    if state['avg_word']:
        shape = x.shape
        pword = emb(x).out.reshape([shape[0], shape[1], state['rank_n_approx']])
        pword = pword * x_mask.dimshuffle(0, 1, 'x')
        aword = pword.sum(0) / TT.maximum(1., x_mask.sum(0).dimshuffle(0, 'x'))
        aword = word_code(aword, use_noise=False)
    else:
        aword = None

    gater_below = None
    if state['rec_gating']:
        gater_below = gater_words_t[0](emb_t(y0))
    reseter_below = None
    if state['rec_reseting']:
        reseter_below = reseter_words_t[0](emb_t(y0))
    has_said = [add_t_op(emb_words_t[0](emb_t(y0)),
            everything,
            y_mask, bs=y_mask.shape[1],
            gater_below = gater_below,
            reseter_below = reseter_below,
            init_state=init_state[0],
            si=0)]
    for si in xrange(1,state['decoder_stack']):
        gater_below = None
        if state['rec_gating']:
            gater_below = gater_words_t[si](emb_t(y0))
        reseter_below = None
        if state['rec_reseting']:
            reseter_below = reseter_words_t[si](emb_t(y0))
        has_said.append(add_t_op(emb_words_t[si](emb_t(y0)),
                everything,
                y_mask, bs=y_mask.shape[1],
                state_below = has_said[-1],
                gater_below = gater_below,
                reseter_below = reseter_below,
                init_state=init_state[si],
                si=si))

    # has_said are hidden layer states

    if has_said[0].out.ndim < 3:
        for si in xrange(state['decoder_stack']):
            shape_hs = has_said[si].shape
            if y0.ndim == 1:
                shape = y0.shape
                has_said[si] = Shift()(has_said[si].reshape([shape[0], 1, state['dim_mlp']]))
            else:
                shape = y0.shape
                has_said[si] = Shift()(has_said[si].reshape([shape[0], shape[1], state['dim_mlp']]))
            has_said[si] = TT.set_subtensor(has_said[si][0, :, :], init_state[si])
            has_said[si] = has_said[si].reshape(shape_hs)
    else:
        for si in xrange(state['decoder_stack']):
            has_said[si] = Shift()(has_said[si])
            has_said[si] = TT.set_subtensor(has_said[si][0, :, :], init_state[si])

    model = pop_op(proj_code(everything), has_said, word=y0, aword = aword)

    nll = output_layer.train(state_below=model, target=y0,
                 mask=y_mask, reg=None) / TT.cast(y.shape[0]*y.shape[1], 'float32')

    valid_fn = None
    noise_fn = None

    x = TT.lvector(name='x')
    n_steps = TT.iscalar('nsteps')
    temp = TT.scalar('temp')
    gater_below = None
    if state['rec_gating']:
        gater_below = gater_words[0](emb(x))
    reseter_below = None
    if state['rec_reseting']:
        reseter_below = reseter_words[0](emb(x))
    encoder_acts = [add_op(emb_words[0](emb(x),use_noise=False),
            si=0,
            use_noise=False,
            gater_below=gater_below,
            reseter_below=reseter_below)]
    if state['encoder_stack'] > 1:
        everything = encoder_proj[0](last(encoder_acts[-1]), use_noise=False)
    for si in xrange(1,state['encoder_stack']):
        gater_below = None
        if state['rec_gating']:
            gater_below = gater_words[si](emb(x))
        reseter_below = None
        if state['rec_reseting']:
            reseter_below = reseter_words[si](emb(x))
        encoder_acts.append(add_op(emb_words[si](emb(x),use_noise=False),
                si=si,
                state_below=encoder_acts[-1], use_noise=False,
                gater_below = gater_below,
                reseter_below = reseter_below))
        if state['encoder_stack'] > 1:
            everything += encoder_proj[si](last(encoder_acts[-1]), use_noise=False)
    if state['encoder_stack'] <= 1:
        encoder = encoder_acts[-1]
        everything = last(encoder)
    else:
        everything = encoder_act_layer(everything)

    init_state = []
    for si in xrange(state['decoder_stack']):
        if state['bias_code']:
            init_state.append(TT.reshape(bias_code[si](everything,
                use_noise=False), [1, state['dim']]))
        else:
            init_state.append(TT.alloc(numpy.float32(0), 1, state['dim']))

    if state['avg_word']:
        aword = emb(x,use_noise=False).out.mean(0)
        aword = word_code(aword, use_noise=False)
    else:
        aword = None

    def sample_fn(*args):
        aidx = 0; word_tm1 = args[aidx]
        aidx += 1; prob_tm1 = args[aidx]
        has_said_tm1 = []
        for si in xrange(state['decoder_stack']):
            aidx += 1; has_said_tm1.append(args[aidx])
        aidx += 1; ctx = args[aidx]
        if state['avg_word']:
            aidx += 1; awrd = args[aidx]

        val = pop_op(proj_code(ctx), has_said_tm1, word=word_tm1,
                aword=awrd, one_step=True, use_noise=False)
        sample = output_layer.get_sample(state_below=val, temp=temp)
        logp = output_layer.get_cost(
                state_below=val.out.reshape([1, TT.cast(output_layer.n_in, 'int64')]),
                temp=temp, target=sample.reshape([1,1]), use_noise=False)
        gater_below = None
        if state['rec_gating']:
            gater_below = gater_words_t[0](emb_t(sample))
        reseter_below = None
        if state['rec_reseting']:
            reseter_below = reseter_words_t[0](emb_t(sample))
        has_said_t = [add_t_op(emb_words_t[0](emb_t(sample)),
                ctx,
                prev_val=has_said_tm1[0],
                gater_below=gater_below,
                reseter_below=reseter_below,
                one_step=True, use_noise=True,
                si=0)]
        for si in xrange(1, state['decoder_stack']):
            gater_below = None
            if state['rec_gating']:
                gater_below = gater_words_t[si](emb_t(sample))
            reseter_below = None
            if state['rec_reseting']:
                reseter_below = reseter_words_t[si](emb_t(sample))
            has_said_t.append(add_t_op(emb_words_t[si](emb_t(sample)),
                    ctx,
                    prev_val=has_said_tm1[si],
                    gater_below=gater_below,
                    reseter_below=reseter_below,
                    one_step=True, use_noise=True,
                    si=si, state_below=has_said_t[-1]))
        for si in xrange(state['decoder_stack']):
            if isinstance(has_said_t[si], list):
                has_said_t[si] = has_said_t[si][-1]
        rval = [sample, TT.cast(logp, 'float32')] + has_said_t
        return rval

    sampler_params = [everything]
    if state['avg_word']:
        sampler_params.append(aword)

    states = [TT.alloc(numpy.int64(0), n_steps)]
    states.append(TT.alloc(numpy.float32(0), n_steps))
    states += init_state

    outputs, updates = scan(sample_fn,
            states = states,
            params = sampler_params,
            n_steps= n_steps,
            name='sampler_scan'
            )
    samples = outputs[0]
    probs = outputs[1]

    sample_fn = theano.function(
        [n_steps, temp, x], [samples, probs.sum()],
        updates=updates,
        profile=False, name='sample_fn')

    model = LM_Model(
        cost_layer=nll,
        weight_noise_amount=state['weight_noise_amount'],
        valid_fn=valid_fn,
        sample_fn=sample_fn,
        clean_before_noise_fn = False,
        noise_fn=noise_fn,
        indx_word=state['indx_word_target'],
        indx_word_src=state['indx_word'],
        character_level=False,
        rng=rng)

    if state['loopIters'] > 0: algo = SGD(model, state, train_data)
    else: algo = None

    def hook_fn():
        if not hasattr(model, 'word_indxs'): model.load_dict()
        if not hasattr(model, 'word_indxs_src'):
            model.word_indxs_src = model.word_indxs
        old_offset = train_data.offset
        if state['sample_reset']: train_data.reset()
        ns = 0
        for sidx in xrange(state['sample_n']):
            while True:
                batch = train_data.next()
                if batch:
                    break
            x = batch['x']
            y = batch['y']
            #xbow = batch['x_bow']
            masks = batch['x_mask']
            if x.ndim > 1:
                for idx in xrange(x.shape[1]):
                    ns += 1
                    if ns > state['sample_max']:
                        break
                    print 'Input: ',
                    for k in xrange(x[:,idx].shape[0]):
                        print model.word_indxs_src[x[:,idx][k]],
                        if model.word_indxs_src[x[:,idx][k]] == '<eol>':
                            break
                    print ''
                    print 'Target: ',
                    for k in xrange(y[:,idx].shape[0]):
                        print model.word_indxs[y[:,idx][k]],
                        if model.word_indxs[y[:,idx][k]] == '<eol>':
                            break
                    print ''
                    senlen = len(x[:,idx])
                    if len(numpy.where(masks[:,idx]==0)[0]) > 0:
                        senlen = numpy.where(masks[:,idx]==0)[0][0]
                    if senlen < 1:
                        continue
                    xx = x[:senlen, idx]
                    #xx = xx.reshape([xx.shape[0], 1])
                    model.get_samples(state['seqlen']+1,  1, xx)
            else:
                ns += 1
                model.get_samples(state['seqlen']+1,  1, x)
            if ns > state['sample_max']:
                break
        train_data.offset = old_offset
        return

    main = MainLoop(train_data, valid_data, None, model, algo, state, channel,
            reset = state['reset'], hooks = hook_fn)
    if state['reload']: main.load()
    if state['loopIters'] > 0: main.main()

    if state['sampler_test']:
        # This is a test script: we only sample
        if not hasattr(model, 'word_indxs'): model.load_dict()
        if not hasattr(model, 'word_indxs_src'):
            model.word_indxs_src = model.word_indxs

        indx_word=pkl.load(open(state['word_indx'],'rb'))

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
                    [values, probs] = model.sample_fn(seqlen * 3, alpha, seq)
                    sen = []
                    for k in xrange(values.shape[0]):
                        if model.word_indxs[values[k]] == '<eol>':
                            break
                        sen.append(model.word_indxs[values[k]])
                    sentences.append(" ".join(sen))
                    all_probs.append(-probs)
                sprobs = numpy.argsort(all_probs)
                for pidx in sprobs:
                    print pidx,"(%f):"%(-all_probs[pidx]),sentences[pidx]
                print

        except KeyboardInterrupt:
            print 'Interrupted'
            pass

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
    state['shuffle'] = True

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

