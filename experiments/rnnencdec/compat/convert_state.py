import argparse
import numpy
import cPickle

def separate_enc_dec_rec_layers(state):
    state['enc_rec_layer'] = state['rec_layer']
    state['enc_rec_gating'] = state['rec_gating']
    state['enc_rec_reseting'] = state['rec_reseting']
    state['enc_rec_gater'] = state['rec_gater']
    state['enc_rec_reseter'] = state['rec_reseter']

    state['dec_rec_layer'] = state['rec_layer']
    state['dec_rec_gating'] = state['rec_gating']
    state['dec_rec_reseting'] = state['rec_reseting']
    state['dec_rec_gater'] = state['rec_gater']
    state['dec_rec_reseter'] = state['rec_reseter']

    return state

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv-fn", help="Conversion function", default="")
    parser.add_argument("state", help="State to convert")
    parser.add_argument("changes",  nargs="?", help="Changes to state", default="")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.state, 'r') as src:
        state = cPickle.load(src)
    state.update(eval("dict({})".format(args.changes)))

    state = eval(args.conv_fn)(state)

    with open(args.state,'w') as tgt:
        cPickle.dump(state, tgt)


main()

