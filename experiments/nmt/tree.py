#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import copy

import networkx as nx

import numpy

import experiments.nmt
from experiments.nmt import RNNEncoderDecoder, parse_input

import theano
import theano.tensor as TT

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


logger = logging.getLogger(__name__)

class Timer(object):

    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--state-fn", help="Initialization function for state", default="prototype_state")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("changes",  nargs="?", help="Changes to state", default="")
    return parser.parse_args()

def main():
    args = parse_args()

    state = getattr(experiments.nmt, args.state_fn)()
    if hasattr(args, 'state') and args.state:
        with open(args.state) as src:
            state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    assert state['enc_rec_layer'] == "RecursiveConvolutionalLayer", "Only works with gated recursive convolutional encoder"

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)

    indx_word = cPickle.load(open(state['word_indx'],'rb'))
    idict_src = cPickle.load(open(state['indx_word'],'r'))

    x = TT.lvector()
    h = TT.tensor3()

    proj_x = theano.function([x], enc_dec.encoder.input_embedders[0](
        enc_dec.encoder.approx_embedder(x)).out, name='proj_x')
    new_h, gater = enc_dec.encoder.transitions[0].step_fprop(
        None, h, return_gates = True)
    step_up = theano.function([h], [new_h, gater], name='gater_step')

    while True:
        try:
            seqin = raw_input('Input Sequence: ')
            seq,parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
            print "Parsed Input:", parsed_in
        except Exception:
            print "Exception while parsing your input:"
            traceback.print_exc()
            continue

        # get the initial embedding
        new_h = proj_x(seq)
        new_h = new_h.reshape(new_h.shape[0], 1, new_h.shape[1])

        nodes = numpy.arange(len(seq)).tolist()
        node_idx = len(seq)-1
        rules = []
        nodes_level = copy.deepcopy(nodes)

        G = nx.DiGraph()

        input_nodes = []
        merge_nodes = []
        aggregate_nodes = []

        nidx = 0 
        vpos = 0
        nodes_pos = {}
        nodes_labels = {}
        # input nodes
        for nn in nodes[:-1]:
            nidx += 1
            G.add_node(nn, pos=(nidx, 0), ndcolor="blue", label="%d"%nn)
            nodes_pos[nn] = (nidx, vpos)
            nodes_labels[nn] = idict_src[seq[nidx-1]]
            input_nodes.append(nn)
        node_idx = len(seq) - 1

        vpos += 6
        for dd in xrange(len(seq)-1):
            new_h, gater = step_up(new_h)
            decisions = numpy.argmax(gater, -1)
            new_nodes_level = numpy.zeros(len(seq) - (dd+1))
            hpos = float(len(seq)+1) - 0.5 * (dd+1)
            last_node = True
            for nn in xrange(len(seq)-(dd+1)):
                hpos -= 1
                if not last_node:
                    # merge nodes
                    node_idx += 1
                    G.add_node(node_idx, ndcolor="red", label="m")
                    nodes_labels[node_idx] = ""
                    nodes_pos[node_idx] = (hpos, vpos)
                    G.add_edge(nodes_level[-(nn+1)], node_idx, weight=gater[-(nn+1),0,0])
                    G.add_edge(nodes_level[-(nn+2)], node_idx, weight=gater[-(nn+1),0,0])
                    merge_nodes.append(node_idx)

                    merge_node = node_idx
                    # linear aggregation nodes
                    node_idx += 1
                    G.add_node(node_idx, ndcolor="red", label="")
                    nodes_labels[node_idx] = "$+$"
                    nodes_pos[node_idx] = (hpos, vpos+6)
                    G.add_edge(merge_node, node_idx, weight=gater[-(nn+1),0,0])
                    G.add_edge(nodes_level[-(nn+2)], node_idx, weight=gater[-(nn+1),0,1])
                    G.add_edge(nodes_level[-(nn+1)], node_idx, weight=gater[-(nn+1),0,2])
                    aggregate_nodes.append(node_idx)

                    new_nodes_level[-(nn+1)] = node_idx
                last_node = False
            nodes_level = copy.deepcopy(new_nodes_level)
            vpos += 12

        # TODO: Show only strong edges.
        threshold = float(raw_input('Threshold: '))
        edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if d['weight'] > threshold]
        #edges = G.edges(data=True)

        use_weighting = raw_input('Color according to weight [Y/N]: ')
        if use_weighting == 'Y':
            cm = plt.get_cmap('binary') 
            cNorm  = colors.Normalize(vmin=0., vmax=1.)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            colorList = [scalarMap.to_rgba(d['weight']) for (u,v,d) in edges]
        else:
            colorList = 'k'

        nx.draw_networkx_nodes(G, pos=nodes_pos, nodelist=input_nodes, node_color='white', alpha=1., edge_color='white')
        nx.draw_networkx_nodes(G, pos=nodes_pos, nodelist=merge_nodes, node_color='blue', alpha=0.8, node_size=20)
        nx.draw_networkx_nodes(G, pos=nodes_pos, nodelist=aggregate_nodes, node_color='red', alpha=0.8, node_size=80)
        nx.draw_networkx_edges(G, pos=nodes_pos, edge_color=colorList, edgelist=edges)
        nx.draw_networkx_labels(G,pos=nodes_pos,labels=nodes_labels,font_family='sans-serif')
        plt.axis('off')
        figname = raw_input('Save to: ')
        if figname[-3:] == "pdf":
            plt.savefig(figname, type='pdf')
        else:
            plt.savefig(figname)
        plt.close()
        G.clear()



        




if __name__ == "__main__":
    main()
