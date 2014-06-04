"""
Data iterator for text datasets that are used for translation model.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Caglar Gulcehre "
               "KyungHyun Cho ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy as np

import os, gc

import tables
import copy

import threading
import Queue

from multiprocessing import Process, Value, Array
import multiprocessing

import collections

class TMIterator(object):

    def __init__(self,
                 batch_size,
                 target_lfiles=None,
                 source_lfiles=None,
                 order = 0,
                 dtype="int64",
                 use_infinite_loop=True,
                 stop=-1,
                 output_format = None,
                 can_fit = False,
                 shuffle = False):

        assert type(source_lfiles) == list, "Target language file should be a list."

        if target_lfiles is not None:
            assert type(target_lfiles) == list, "Target language file should be a list."
            assert len(target_lfiles) == len(source_lfiles)

        self.batch_size = batch_size
        self.target_lfiles = target_lfiles
        self.source_lfiles = source_lfiles
        self.use_infinite_loop=use_infinite_loop
        self.target_langs = []
        self.source_langs = []
        self.order = order
        self.offset = 0
        self.data_len = 0
        self.stop = stop
        self.can_fit = can_fit
        self.dtype = dtype
        self.output_format = output_format
        self.shuffle = shuffle
        self.load_files()

    def load_files(self):
        mmap_mode = None
        if self.can_fit == False:
            mmap_mode = "r"
        if self.target_lfiles is not None:
            for target_lfile in self.target_lfiles:
                if target_lfile[-3:] == '.gz':
                    target_lang = np.loadtxt(target_lfile)
                else:
                    target_lang = np.load(target_lfile, mmap_mode=mmap_mode)
                self.target_langs.append(target_lang)

        for source_lfile in self.source_lfiles:
            if source_lfile[-3:] == '.gz':
                source_lang = np.loadtxt(source_lfile)
            else:
                source_lang = np.load(source_lfile, mmap_mode=mmap_mode)
            self.source_langs.append(source_lang)
        if isinstance(source_lang, list):
            self.data_len = len(source_lang)
        else:
            self.data_len = source_lang.shape[0]

        if self.shuffle and self.can_fit:
            shuffled_indx = np.arange(self.data_len)
            np.random.shuffle(shuffled_indx)
            if self.target_lfiles is not None:
                if isinstance(self.target_langs[0], list):
                    shuffled_target=[np.array([tt[si] for si in shuffled_indx]) for tt in self.target_langs]
                else:
                    shuffled_target = [tt[shuffled_indx] for tt in self.target_langs]
                self.target_langs = shuffled_target
            if isinstance(self.source_langs[0], list):
                shuffled_source=[np.array([tt[si] for si in shuffled_indx]) for tt in self.source_langs]
            else:
                shuffled_source = [tt[shuffled_indx] for tt in self.source_langs]
            self.source_langs = shuffled_source

    def __iter__(self):
        return self

    def reset(self):
        self.offset = 0

    def next(self):
        if self.stop != -1 and self.offset >= self.stop:
            self.offset = 0
            raise StopIteration
        else:
            while True:
                source_data = []
                target_data = []

                for source_lang in self.source_langs:
                    inc_offset = self.offset+self.batch_size
                    npos = 0
                    while not npos and inc_offset <= self.data_len:
                        npos = len([x for x in
                              source_lang[self.offset:inc_offset].tolist()
                              if len(x) > 0 ])
                        nzeros = self.batch_size - npos
                        inc_offset += nzeros

                    sents = np.asarray([np.cast[self.dtype](si) for si in
                                        source_lang[self.offset:inc_offset].tolist()
                                                    if len(si)>0])
                    if self.order:
                        sents = sents.T
                    source_data.append(sents)

                for target_lang in self.target_langs:
                    inc_offset = self.offset+self.batch_size
                    npos = 0
                    while not npos and inc_offset <= self.data_len:
                        npos = len([x for x in
                              target_lang[self.offset:inc_offset].tolist()
                              if len(x) > 0 ])
                        nzeros = self.batch_size - npos
                        inc_offset += nzeros

                    sents = np.asarray([np.cast[self.dtype](si) for si in target_lang[self.offset:inc_offset].tolist() if len(si) > 0])
                    if self.order:
                        sents = sents.T
                    target_data.append(sents)
                if inc_offset > self.data_len and self.use_infinite_loop:
                    print "Restarting the dataset iterator."
                    inc_offset = 0 #self.offset + self.batch_size
                elif inc_offset > self.data_len:
                    self.offset = 0
                    raise StopIteration
                if len(source_data[0]) < 1 or len(target_data[0]) < 1:
                    self.offset = inc_offset
                    inc_offset = self.offset+self.batch_size
                    continue
                break
            self.offset = inc_offset
        if not self.output_format:
            return source_data, target_data
        else:
            return self.output_format(source_data, target_data)

class TMIteratorPytablesGatherProcessing(Process):
    def __init__(self,
            datasetIter,
            exitFlag,
            queue):
        Process.__init__(self)
        self.datasetIter = datasetIter
        self.exitFlag = exitFlag
        self.queue = queue

    def run(self):
        self.target_langs = []
        self.source_langs = []

        if self.datasetIter.can_fit:
            driver = "H5FD_CORE"
        else:
            driver = None

        if self.datasetIter.target_lfiles is not None:
            for target_lfile in self.datasetIter.target_lfiles:
                target_lang = tables.open_file(target_lfile, 'r', driver=driver)
                self.target_langs.append([
                    target_lang.get_node(self.datasetIter.table_name), 
                    target_lang.get_node(self.datasetIter.index_name)])

        for source_lfile in self.datasetIter.source_lfiles:
            source_lang = tables.open_file(source_lfile, 'r', driver=driver)
            self.source_langs.append([
                source_lang.get_node(self.datasetIter.table_name),
                source_lang.get_node(self.datasetIter.index_name)])
            try:
                freqs = source_lang.get_node(self.datasetIter.freqs_name)
                self.source_langs[-1].append(freqs)
            except tables.NoSuchNodeError:
                pass
        self.data_len = self.source_langs[-1][1].shape[0]

        self.idxs = np.arange(self.data_len)

        if self.datasetIter.shuffle:
            np.random.shuffle(self.idxs)

        counter = 0
        while not self.exitFlag:
            last_batch = False
            while True:
                source_data = []
                target_data = []

                for source_lang in self.source_langs:
                    inc_offset = self.datasetIter.offset+self.datasetIter.batch_size
                    npos = 0
                    while not npos and inc_offset <= self.data_len:
                        sents = np.asarray([np.cast[self.datasetIter.dtype](si) for si in
                            [source_lang[0][source_lang[1][i]['pos']:(source_lang[1][i]['pos']+source_lang[1][i]['length'])]
                                for i in self.idxs[self.datasetIter.offset:inc_offset]]])
                        npos = len(sents)
                        nzeros = self.datasetIter.batch_size - npos
                        inc_offset += nzeros

                    if self.datasetIter.order:
                        sents = sents.T
                    source_data.append(sents)

                for target_lang in self.target_langs:
                    inc_offset = self.datasetIter.offset+self.datasetIter.batch_size
                    npos = 0
                    while not npos and inc_offset <= self.data_len:
                        sents = np.asarray([np.cast[self.datasetIter.dtype](si) for si in
                            [target_lang[0][target_lang[1][i]['pos']:(target_lang[1][i]['pos']+target_lang[1][i]['length'])]
                                for i in self.idxs[self.datasetIter.offset:inc_offset]] if
                            len(si)>0])
                        npos = len(sents)
                        nzeros = self.datasetIter.batch_size - npos
                        inc_offset += nzeros

                    if self.datasetIter.order:
                        sents = sents.T
                    target_data.append(sents)

                if inc_offset > self.data_len and self.datasetIter.use_infinite_loop:
                    print "Restarting the dataset iterator."
                    inc_offset = 0 
                    if self.datasetIter.shuffle:
                        np.random.shuffle(self.idxs)
                elif inc_offset > self.data_len:
                    last_batch = True

                if len(source_data[0]) < 1 or len(target_data[0]) < 1:
                    self.datasetIter.offset = inc_offset
                    inc_offset = self.datasetIter.offset+self.datasetIter.batch_size
                    continue
                break
            counter += 1
            self.datasetIter.offset = inc_offset

            if source_data[0] == None:
                continue
            
            self.queue.put([source_data, target_data])
            if last_batch:
                self.queue.put([None])
                return
            

class TMIteratorPytables(object):

    def __init__(self,
                 batch_size,
                 target_lfiles=None,
                 source_lfiles=None,
                 order = 0,
                 dtype="int64",
                 use_infinite_loop=True,
                 stop=-1,
                 output_format = None,
                 table_name = '/phrases', 
                 index_name = '/indices', 
                 freqs_name = '/counts', 
                 can_fit = False,
                 queue_size = 1000,
                 cache_size = 1000,
                 freqs_sum = 16791878,
                 shuffle = True):

        assert type(source_lfiles) == list, "Source language file should be a list."
        
        if target_lfiles is not None:
            assert type(target_lfiles) == list, "Target language file should be a list."
            assert len(target_lfiles) == len(source_lfiles)

        self.batch_size = batch_size
        self.target_lfiles = target_lfiles
        self.source_lfiles = source_lfiles
        self.use_infinite_loop=use_infinite_loop
        self.target_langs = []
        self.source_langs = []
        self.order = order
        self.offset = 0
        self.data_len = 0
        self.stop = stop
        self.can_fit = can_fit
        self.dtype = dtype
        self.output_format = output_format
        self.shuffle = shuffle
        self.table_name = table_name
        self.index_name = index_name
        self.freqs_name = freqs_name
        self.freqs_sum = freqs_sum

        self.exitFlag = False

        if cache_size > 0:
            self.cache = collections.deque(maxlen=cache_size)
        else:
            self.cache = None

        self.queue = multiprocessing.Queue(maxsize=queue_size)
        self.gather = TMIteratorPytablesGatherProcessing(self,
                self.exitFlag, self.queue)
        self.gather.start()

    def __del__(self):
        self.gather.exitFlag = True
        self.gather.join()

    def load_files(self):
        if self.target_lfiles is not None:
            for target_lfile in self.target_lfiles:
                target_lang = tables.open_file(target_lfile, 'r')
                self.target_langs.append([target_lang.get_node(self.table_name), 
                    target_lang.get_node(self.index_name)])

        for source_lfile in self.source_lfiles:
            source_lang = tables.open_file(source_lfile, 'r')
            self.source_langs.append([source_lang.get_node(self.table_name),
                source_lang.get_node(self.index_name)])
        self.data_len = self.source_langs[-1][1].shape[0]

        self.idxs = np.arange(self.data_len)
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def __iter__(self):
        return self

    def reset(self):
        self.offset = 0

    def next(self):
        while True:
            try:
                if self.cache != None:
                    batch = self.queue.get_nowait()
                    self.cache.append(batch)
                else:
                    batch = self.queue.get()
            except Queue.Empty:
                if self.cache != None:
                    try:
                        self.cache.rotate(-1)
                        batch = self.cache[0]
                    except IndexError:
                        batch = None
                else:
                    batch = None
            if batch:
                break

        if batch[0] == None:
            raise StopIteration

        if not self.output_format:
            return batch[0], batch[1]
        else:
            return self.output_format(batch[0], batch[1])

class NNJMContextIterator(object):

    def __init__(self,
                 batch_size,
                 order = 0,
                 path = None,
                 dtype = "int64",
                 use_infinite_loop = True,
                 stop = -1,
                 output_format = None,
                 can_fit = False):

        assert path is not None, "Path should not be empty!."

        self.source_ctxt = None
        self.target_ctxt = None
        self.targets = None

        self.batch_size = batch_size
        self.path = path
        self.use_infinite_loop = use_infinite_loop
        self.order = order
        self.offset = 0
        self.data_len = 0
        self.stop = stop
        self.can_fit = can_fit
        self.dtype = dtype
        self.output_format = output_format
        self.load_files()

    def load_files(self):
        mmap_mode = None
        if self.can_fit == False:
            mmap_mode = "r"

        data_file = np.load(self.path, mmap_mode=mmap_mode)

        self.source_ctxt = data_file["src_ctxt"]
        self.target_ctxt = data_file["tgt_ctxt"]
        self.targets = data_file["tgts"]
        self.targets = self.targets.reshape(self.targets.shape[0], 1)

        self.data_len = self.source_ctxt.shape[0]

    def __iter__(self):
        return self

    def reset(self):
        self.offset = 0

    def next(self):
        if self.stop != -1 and self.offset >= self.stop:
            self.offset = 0
            raise StopIteration
        else:
            while True:
                inc_offset = self.offset + self.batch_size
                if inc_offset > self.data_len and self.use_infinite_loop:
                    print "Restarting the dataset iterator."
                    inc_offset = 0 
                elif inc_offset > self.data_len:
                    self.offset = 0
                    raise StopIteration

                sents_s = np.asarray([np.cast[self.dtype](si) for si in
                                    self.source_ctxt[self.offset:inc_offset].tolist()
                                                if len(si)>0])

                if self.order:
                    sents_s = sents_s.T

                source_ctxt = sents_s
                sents_t = np.asarray([np.cast[self.dtype](si) for si in
                                    self.target_ctxt[self.offset:inc_offset].tolist()
                                                if len(si)>0])

                if self.order:
                    sents_t = sents_t.T

                target_ctxt = sents_t
                targets = np.asarray([np.cast[self.dtype](si) for si in
                                    self.targets[self.offset:inc_offset].tolist()
                                                if len(si)>0])

                if len(source_ctxt) < 1 or len(target_ctxt) < 1 or len(targets) < 1:
                    self.offset = inc_offset
                    inc_offset = self.offset + self.batch_size
                    continue
                break
            self.offset = inc_offset

        if not self.output_format:
            return source_ctxt, target_ctxt, targets
        else:
            return self.output_format(source_ctxt, target_ctxt, targets)


