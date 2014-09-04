#!/usr/bin/env python
import argparse
import cPickle as pkl
import gzip

import sys

import tables
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("input",
                    type=argparse.FileType('rb'),
                    help="Pickle file")
parser.add_argument("output",
                    type=argparse.FileType('w'),
                    help="Output HDF5 file")
args = parser.parse_args()

class Index(tables.IsDescription):
    pos = tables.UInt32Col()
    length = tables.UInt32Col()

f = args.output
f = tables.open_file(f.name, f.mode)
earrays = f.createEArray(f.root, 'phrases', 
    tables.Int32Atom(),shape=(0,))
indices = f.createTable("/", 'indices', 
    Index, "a table of indices and lengths")

sfile = open(args.input.name, args.input.mode)
sarray = pkl.load(sfile)
sfile.close()

count = 0
pos = 0
for x in sarray:
    earrays.append(numpy.array(x))
    ind = indices.row
    ind['pos'] = pos
    ind['length'] = len(x)
    ind.append()

    pos += len(x)
    count += 1

    if count % 100000 == 0:
        print count,
        sys.stdout.flush()
        indices.flush()
    elif count % 10000 == 0:
        print '.',
        sys.stdout.flush()

f.close()

print 'processed', count, 'phrases'
