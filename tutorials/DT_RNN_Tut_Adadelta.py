"""
Test of the classical LM model for language modelling
"""
from groundhog.datasets import LMIterator
from groundhog.trainer.SGD_adadelta import SGD
from groundhog.mainLoop import MainLoop
from groundhog.layers import MultiLayer, \
       RecurrentMultiLayer, \
       RecurrentMultiLayerInp, \
       RecurrentMultiLayerShortPath, \
       RecurrentMultiLayerShortPathInp, \
       RecurrentMultiLayerShortPathInpAll, \
       SoftmaxLayer, \
       LastState,\
       UnaryOp, \
       Operator, \
       Shift, \
       GaussianNoise, \
       SigmoidLayer
from groundhog.layers import maxpool, \
        maxpool_ntimes, \
        last, \
        last_ntimes,\
        tanh, \
        sigmoid, \
        rectifier,\
        hard_sigmoid, \
        hard_tanh
from groundhog.models import LM_Model
from theano import scan

import numpy
import theano
import theano.tensor as TT

linear = lambda x:x

theano.config.allow_gc = False

def get_text_data(state):
    def out_format (x, y, r):
        return {'x':x, 'y' :y, 'reset': r}
    def out_format_valid (x, y, r):
        return {'x':x, 'y' :y, 'reset': r}

    train_data = LMIterator(
            batch_size=state['bs'],
            path = state['path'],
            stop=-1,
            seq_len = state['seqlen'],
            mode="train",
            chunks=state['chunks'],
            shift = state['shift'],
            output_format = out_format,
            can_fit=True)

    valid_data = LMIterator(
            batch_size=state['bs'],
            path=state['path'],
            stop=-1,
            use_infinite_loop=False,
            allow_short_sequences = True,
            seq_len= state['seqlen'],
            mode="valid",
            reset =state['reset'],
            chunks=state['chunks'],
            shift = state['shift'],
            output_format = out_format_valid,
            can_fit=True)

    test_data = LMIterator(
            batch_size=state['bs'],
            path = state['path'],
            stop=-1,
            use_infinite_loop=False,
            allow_short_sequences=True,
            seq_len= state['seqlen'],
            mode="test",
            chunks=state['chunks'],
            shift = state['shift'],
            output_format = out_format_valid,
            can_fit=True)
    if 'wiki' in state['path']:
        test_data = None
    return train_data, valid_data, test_data

def jobman(state, channel):
    # load dataset
    rng = numpy.random.RandomState(state['seed'])

    # declare the dimensionalies of the input and output
    if state['chunks'] == 'words':
        state['n_in'] = 10000
        state['n_out'] = 10000
    else:
        state['n_in'] = 50
        state['n_out'] = 50
    train_data, valid_data, test_data = get_text_data(state)

    ## BEGIN Tutorial
    ### Define Theano Input Variables
    x = TT.lvector('x')
    y = TT.lvector('y')
    h0 = theano.shared(numpy.zeros((eval(state['nhids'])[-1],), dtype='float32'))

    ### Neural Implementation of the Operators: \oplus
    #### Word Embedding
    emb_words = MultiLayer(
        rng,
        n_in=state['n_in'],
        n_hids=eval(state['inp_nhids']),
        activation=eval(state['inp_activ']),
        init_fn='sample_weights_classic',
        weight_noise=state['weight_noise'],
        rank_n_approx = state['rank_n_approx'],
        scale=state['inp_scale'],
        sparsity=state['inp_sparse'],
        learn_bias = True,
        bias_scale=eval(state['inp_bias']),
        name='emb_words')

    #### Deep Transition Recurrent Layer
    rec = eval(state['rec_layer'])(
            rng,
            eval(state['nhids']),
            activation = eval(state['rec_activ']),
            #activation = 'TT.nnet.sigmoid',
            bias_scale = eval(state['rec_bias']),
            scale=eval(state['rec_scale']),
            sparsity=eval(state['rec_sparse']),
            init_fn=eval(state['rec_init']),
            weight_noise=state['weight_noise'],
            name='rec')

    #### Stiching them together
    ##### (1) Get the embedding of a word
    x_emb = emb_words(x, no_noise_bias=state['no_noise_bias'])
    ##### (2) Embedding + Hidden State via DT Recurrent Layer
    reset = TT.scalar('reset')
    rec_layer = rec(x_emb, n_steps=x.shape[0],
                    init_state=h0*reset,
                    no_noise_bias=state['no_noise_bias'],
                    truncate_gradient=state['truncate_gradient'],
                    batch_size=1)

    ### Neural Implementation of the Operators: \lhd
    #### Softmax Layer
    output_layer = SoftmaxLayer(
        rng,
        eval(state['nhids'])[-1],
        state['n_out'],
        scale=state['out_scale'],
        bias_scale=state['out_bias_scale'],
        init_fn="sample_weights_classic",
        weight_noise=state['weight_noise'],
        sparsity=state['out_sparse'],
        sum_over_time=True,
        name='out')

    ### Few Optional Things
    #### Direct shortcut from x to y
    if state['shortcut_inpout']:
        shortcut = MultiLayer(
            rng,
            n_in=state['n_in'],
            n_hids=eval(state['inpout_nhids']),
            activations=eval(state['inpout_activ']),
            init_fn='sample_weights_classic',
            weight_noise = state['weight_noise'],
            scale=eval(state['inpout_scale']),
            sparsity=eval(state['inpout_sparse']),
            learn_bias=eval(state['inpout_learn_bias']),
            bias_scale=eval(state['inpout_bias']),
            name='shortcut')

    #### Learning rate scheduling (1/(1+n/beta))
    state['clr'] = state['lr']
    def update_lr(obj, cost):
        stp = obj.step
        if isinstance(obj.state['lr_start'], int) and stp > obj.state['lr_start']:
            time = float(stp - obj.state['lr_start'])
            new_lr = obj.state['clr']/(1+time/obj.state['lr_beta'])
            obj.lr = new_lr
    if state['lr_adapt']:
        rec.add_schedule(update_lr)

    ### Neural Implementations of the Language Model
    #### Training
    if state['shortcut_inpout']:
        train_model = output_layer(rec_layer,
                                   no_noise_bias=state['no_noise_bias'],
                                   additional_inputs=[shortcut(x)]).train(target=y,
                scale=numpy.float32(1./state['seqlen']))
    else:
        train_model = output_layer(rec_layer,
                no_noise_bias=state['no_noise_bias']).train(target=y,
                        scale=numpy.float32(1./state['seqlen']))

    nw_h0 = rec_layer.out[rec_layer.out.shape[0]-1]
    if state['carry_h0']:
        train_model.updates += [(h0, nw_h0)]

    #### Validation
    h0val = theano.shared(numpy.zeros((eval(state['nhids'])[-1],), dtype='float32'))
    rec_layer = rec(emb_words(x, use_noise=False),
                    n_steps = x.shape[0],
                    batch_size=1,
                    init_state=h0val*reset,
                    use_noise=False)
    nw_h0 = rec_layer.out[rec_layer.out.shape[0]-1]
    if not state['shortcut_inpout']:
        valid_model = output_layer(rec_layer,
                use_noise=False).validate(target=y, sum_over_time=True)
    else:
        valid_model = output_layer(rec_layer,
                additional_inputs=[shortcut(x, use_noise=False)],
                use_noise=False).validate(target=y, sum_over_time=True)

    valid_updates = []
    if state['carry_h0']:
        valid_updates = [(h0val, nw_h0)]

    valid_fn = theano.function([x,y, reset], valid_model.cost,
          name='valid_fn', updates=valid_updates)

    #### Sampling
    ##### single-step sampling
    def sample_fn(word_tm1, h_tm1):
        x_emb = emb_words(word_tm1, use_noise = False, one_step=True)
        h0_ = rec(x_emb, state_before=h_tm1, one_step=True, use_noise=False)[-1]
        word = output_layer.get_sample(state_below=h0_, temp=1.)
        return word, h0_

    ##### scan for iterating the single-step sampling multiple times
    [samples, summaries], updates = scan(sample_fn,
                      states = [
                          TT.alloc(numpy.int64(0), state['sample_steps']),
                          TT.alloc(numpy.float32(0), 1, eval(state['nhids'])[-1])],
                      n_steps= state['sample_steps'],
                      name='sampler_scan')

    ##### define a Theano function
    sample_fn = theano.function([], [samples], 
        updates=updates, profile=False, name='sample_fn')
    
    ##### Load a dictionary
    dictionary = numpy.load(state['dictionary'])
    if state['chunks'] == 'chars':
        dictionary = dictionary['unique_chars']
    else:
        dictionary = dictionary['unique_words']
    def hook_fn():
        sample = sample_fn()[0]
        print 'Sample:',
        if state['chunks'] == 'chars':
            print "".join(dictionary[sample])
        else:
            for si in sample:
                print dictionary[si],
            print

    ### Build and Train a Model
    #### Define a model
    model = LM_Model(
        cost_layer = train_model,
        weight_noise_amount=state['weight_noise_amount'],
        valid_fn = valid_fn,
        clean_before_noise_fn = False,
        noise_fn = None,
        rng = rng)

    if state['reload']:
        model.load(state['prefix']+'model.npz')

    #### Define a trainer
    ##### Training algorithm (SGD)
    algo = SGD(model, state, train_data)
    ##### Main loop of the trainer
    main = MainLoop(train_data,
                    valid_data,
                    test_data,
                    model,
                    algo,
                    state,
                    channel,
                    train_cost = False,
                    hooks = hook_fn,
                    validate_postprocess =  eval(state['validate_postprocess']))
    main.main()
    ## END Tutorial


if __name__=='__main__':
    state = {}
    # complete path to data (cluster specific)
    state['seqlen'] = 100
    state['path']= "/data/lisa/data/PennTreebankCorpus/pentree_char_and_word.npz"
    state['dictionary']= "/data/lisa/data/PennTreebankCorpus/dictionaries.npz"
    state['chunks'] = 'chars'
    state['seed'] = 123

    # flag .. don't need to change it. It says what to do if you get cost to
    # be nan .. you could raise, though I would leave it to this
    state['on_nan'] = 'warn'

    # DATA

    # For wikipedia validation set is extremely large. Is very time
    # wasteful. This value is only used for validation set, and IMHO should
    # be something like seqlen * 10000 (i.e. the validation should be only
    # 10000 steps
    state['reset'] = -1
    # For music/ word level I think 50 is a good idea. For character this
    # should be at least 100 (I think there are problems with getting state
    # of the art otherwise). Note most people use 200 !

    # The job stops when learning rate declines to this value. It can be
    # useful, because sometimes is hopeless to wait for validation error to
    # get below minerr, or for the time to expire
    state['minlr'] = float(5e-7)

    # Layers
    # Input

    # Input weights are sampled from a gaussian with std=scale; this is the
    # standard way to initialize
    state['rank_n_approx'] = 0
    state['inp_nhids'] = '[400]'
    state['inp_activ'] = '[linear]'
    state['inp_bias'] = '[0.]'
    state['inp_sparse']= -1 # dense
    state['inp_scale'] = .1

    # This is for the output weights
    state['out_scale'] = .1
    state['out_bias_scale'] = -.5
    state['out_sparse'] = -1

    # HidLayer
    # Hidden units on for the internal layers of DT-RNN. Having a single
    # value results in a standard RNN
    state['nhids'] = '[200, 200]'
    # Activation of each layer
    state['rec_activ'] = '"TT.nnet.sigmoid"'
    state['rec_bias'] = '.0'
    state['rec_sparse'] ='20'
    state['rec_scale'] = '1.'
    # sample_weights - you rescale the weights such that the largest
    # singular value is scale
    # sample_weights_classic : just sample weights from a gaussian with std
    # equal to scale
    state['rec_init'] = "'sample_weights'"
    state['rec_layer'] = 'RecurrentMultiLayerShortPathInpAll'

    # SGD params
    state['bs'] = 1 # the size of the minibatch
    state['lr'] = 1. # initial learn_ing rate
    state['cutoff'] = 1. # threshold for gradient rescaling
    state['moment'] = 0.995 #-.1 # momentum

    # Do not optimize these
    state['weight_noise'] = True # white Gaussian noise in weights
    state['weight_noise_amount'] = 0.075 # standard deviation

    # maximal number of updates
    state['loopIters'] = int(1e8)
    # maximal number of minutes to wait until killing job
    state['timeStop'] = 48*60 # 48 hours

    # Construct linear connections from input to output. These are factored
    # (like the rank_n) to deal with the possible high dimensionality of the
    # input, but it is a linear projection that feeds into the softmax
    state['shortcut_inpout'] = False
    state['shortcut_rank'] = 200

    # Main Loop
    # Make this to be a decently large value. Otherwise you waste a lot of
    # memory keeping track of the training error (and other things) at each
    # step + the stdout becomes extremely large
    state['trainFreq'] = 100
    state['hookFreq'] = 5000
    state['validFreq'] = 1000

    state['saveFreq'] = 15 # save every 15 minutes
    state['prefix'] = 'model_' # prefix of the save files
    state['reload'] = False # reload
    state['overwrite'] = 1

    # Threhold should be 1.004 for PPL, for entropy (which is what
    # everything returns, it should be much smaller. Running value is 1.0002
    # We should not hyperoptimize this
    state['divide_lr'] = 2.
    state['cost_threshold'] = 1.0002
    state['patience'] = 1
    state['validate_postprocess'] = 'lambda x:10**(x/numpy.log(10))'

    state['truncate_gradient'] = 80 # truncated BPTT
    state['lr_adapt'] = 0 # 1/(1 + n/n0) scheduling
    state['lr_beta'] = 10*1900.
    state['lr_start'] = 'on_error'

    state['no_noise_bias'] = True # do not use weight noise for biases
    state['carry_h0'] = True # carry over h0 across updates

    state['sample_steps'] = 80

    # Do not change these
    state['minerr'] = -1
    state['shift'] = 1 # n-step forward prediction
    state['cutoff_rescale_length'] = False

    jobman(state, None)

