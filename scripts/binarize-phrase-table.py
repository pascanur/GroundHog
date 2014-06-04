#!/usr/bin/env python

# Converts moses phrase table file to HDF5 files
# Written by Bart van Merrienboer (University of Montreal)

import argparse
import cPickle
import gzip

import sys

import tables
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("input",
                    type=argparse.FileType('r'),
                    help="The phrase table to be processed")
parser.add_argument("source_output",
                    type=argparse.FileType('w'),
                    help="The source output file")
parser.add_argument("target_output",
                    type=argparse.FileType('w'),
                    help="The target output file")
parser.add_argument("source_dictionary",
                    type=argparse.FileType('r'),
                    help="A pickled dictionary with words and IDs as keys and "
                         "values respectively")
parser.add_argument("target_dictionary",
                    type=argparse.FileType('r'),
                    help="A pickled dictionary with words and IDs as keys and "
                         "values respectively")
parser.add_argument("--labels",
                    type=int, default=15000,
                    help="Set the maximum word index")
args = parser.parse_args()

class Index(tables.IsDescription):
    pos = tables.UInt32Col()
    length = tables.UInt32Col()

files = [args.source_output, args.target_output]
vlarrays = []
indices = []
for i, f in enumerate(files):
    files[i] = tables.open_file(f.name, f.mode)
    vlarrays.append(files[i].createEArray(files[i].root, 'phrases',
            tables.Int32Atom(),shape=(0,)))
    indices.append(files[i].createTable("/", 'indices', Index, "a table of indices and lengths"))


sfile = gzip.open(args.input.name, args.input.mode)

source_table = cPickle.load(args.source_dictionary)
target_table = cPickle.load(args.target_dictionary)
tables = [source_table, target_table]

count = 0
counts = numpy.zeros(2).astype('int32')

freqs_sum = 0

for line in sfile:
    fields = line.strip().split('|||')
    for field_index in [0, 1]:
        words = fields[field_index].strip().split(' ')
        word_indices = [tables[field_index].get(word, 1) for word in words]
        if args.labels > 0:
            word_indices = [word_index if word_index < args.labels else 1
                            for word_index in word_indices]
        vlarrays[field_index].append(numpy.array(word_indices))
        pos = counts[field_index]
        length = len(word_indices)
        ind = indices[field_index].row
        ind['pos'] = pos
        ind['length'] = length
        ind.append()
        counts[field_index] += len(word_indices)

    count += 1
    if count % 100000 == 0:
        print count,
        [i.flush() for i in indices]
        sys.stdout.flush()
    elif count % 10000 == 0:
        print '.',
        sys.stdout.flush()

for f in indices:
    f.flush()

for f in files:
    f.close()

sfile.close()

print 'processed', count, 'phrase pairs'
