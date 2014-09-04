#!/usr/bin/env python
import argparse
import cPickle
import gzip

import sys

import tables
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("source_input",
                    type=argparse.FileType('r'),
                    help="The source input HDF5 file")
parser.add_argument("target_input",
                    type=argparse.FileType('r'),
                    help="The target input HDF5 file")
parser.add_argument("source_output",
                    type=argparse.FileType('w'),
                    help="The source output HDF5 file")
parser.add_argument("target_output",
                    type=argparse.FileType('w'),
                    help="The target output HDF5 file")
args = parser.parse_args()

class Index(tables.IsDescription):
    pos = tables.UInt32Col()
    length = tables.UInt32Col()

infiles = [args.source_input, args.target_input]
outfiles = [args.source_output, args.target_output]
vlarrays_in = []
indices_in = []
vlarrays_out = []
indices_out = []

for i, f in enumerate(infiles):
    infiles[i] = tables.open_file(f.name, f.mode)
    vlarrays_in.append(infiles[i].get_node('/phrases'))
    indices_in.append(infiles[i].get_node('/indices'))

for i, f in enumerate(outfiles):
    outfiles[i] = tables.open_file(f.name, f.mode)
    vlarrays_out.append(outfiles[i].createEArray(outfiles[i].root, 'phrases',
            tables.Int32Atom(),shape=(0,)))
    indices_out.append(outfiles[i].createTable("/", 'indices', Index, "a table of indices and lengths"))

data_len = indices_in[0].shape[0]
print 'Data len:', data_len

idxs = numpy.arange(data_len)
numpy.random.shuffle(idxs)

print 'Shuffled'

count = 0
counts = numpy.zeros(2)

for ii in idxs:
    for fi in [0, 1]:
        pos = indices_in[fi][ii]['pos']
        length = indices_in[fi][ii]['length']
        vlarrays_out[fi].append(vlarrays_in[fi][pos:(pos+length)])
        ind = indices_out[fi].row
        ind['pos'] = counts[fi]
        ind['length'] = length
        ind.append()
        counts[fi] += length
    count +=1
    if count % 100000 == 0:
        print count,
        [i.flush() for i in indices_out]
        sys.stdout.flush()
    elif count % 10000 == 0:
        print '.',
        sys.stdout.flush()

for i in indices_out:
    i.flush()

for f in infiles:
    f.flush()
    f.close()
for f in outfiles:
    f.flush()
    f.close()

print 'processed', count, 'phrase pairs'
