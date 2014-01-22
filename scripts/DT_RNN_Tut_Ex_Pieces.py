##############
outhid = outhid_dropout(outhid, use_noise=False)

##############
outhid_activ = UnaryOp(activation=eval(state['dout_activ']))

##############
emb_words_out = MultiLayer(
    rng,
    n_in=state['n_in'],
    n_hids=eval(state['dout_nhid']),
    activation=linear,
    init_fn='sample_weights_classic',
    weight_noise=state['weight_noise'],
    scale=state['dout_scale'],
    sparsity=state['dout_sparse'],
    rank_n_approx=state['dout_rank_n_approx'],
    learn_bias = False,
    bias_scale=eval(state['dout_bias']),
    name='emb_words_out')

##############
outhid = outhid_dropout(outhid)

##############
outhid_dropout = DropOp(dropout=state['dropout'], rng=rng)

#############
emb_state = MultiLayer(
    rng,
    n_in=eval(state['nhids'])[-1],
    n_hids=eval(state['dout_nhid']),
    activation=linear,
    init_fn=eval(state['dout_init']),
    weight_noise=state['weight_noise'],
    scale=state['dout_scale'],
    sparsity=state['dout_sparse'],
    learn_bias = True,
    bias_scale=eval(state['dout_bias']),
    name='emb_state')

##############
outhid = outhid_activ(emb_state(rec_layer) + emb_words_out(x))

