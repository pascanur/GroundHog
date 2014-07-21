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
import logging

import threading
import Queue

import collections

logger = logging.getLogger(__name__)

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

class PytablesBitextFetcher(threading.Thread):
    def __init__(self, parent, start_offset):
        threading.Thread.__init__(self)
        self.parent = parent
        self.start_offset = start_offset

    def run(self):
        diter = self.parent

        driver = None
        if diter.can_fit:
            driver = "H5FD_CORE"

        target_table = tables.open_file(diter.target_file, 'r', driver=driver)
        target_data, target_index = (target_table.get_node(diter.table_name),
            target_table.get_node(diter.index_name))

        source_table = tables.open_file(diter.source_file, 'r', driver=driver)
        source_data, source_index = (source_table.get_node(diter.table_name),
            source_table.get_node(diter.index_name))

        assert source_index.shape[0] == target_index.shape[0]
        data_len = source_index.shape[0]

        offset = self.start_offset
        if offset == -1:
            offset = 0
            if diter.shuffle:
                offset = np.random.randint(data_len)
        logger.debug("{} entries".format(data_len))
        logger.debug("Starting from the entry {}".format(offset))

        while not diter.exit_flag:
            last_batch = False
            source_sents = []
            target_sents = []
            while len(source_sents) < diter.batch_size:
                if offset == data_len:
                    if diter.use_infinite_loop:
                        offset = 0
                    else:
                        last_batch = True
                        break

                slen, spos = source_index[offset]['length'], source_index[offset]['pos']
                tlen, tpos = target_index[offset]['length'], target_index[offset]['pos']
                offset += 1

                if slen > diter.max_len or tlen > diter.max_len:
                    continue
                source_sents.append(source_data[spos:spos + slen].astype(diter.dtype))
                target_sents.append(target_data[tpos:tpos + tlen].astype(diter.dtype))

            if len(source_sents):
                diter.queue.put([int(offset), source_sents, target_sents])
            if last_batch:
                diter.queue.put([None])
                return

class PytablesBitextIterator(object):

    def __init__(self,
                 batch_size,
                 target_file=None,
                 source_file=None,
                 dtype="int64",
                 table_name='/phrases',
                 index_name='/indices',
                 can_fit=False,
                 queue_size=1000,
                 cache_size=1000,
                 shuffle=True,
                 use_infinite_loop=True,
                 max_len=1000):

        args = locals()
        args.pop("self")
        self.__dict__.update(args)

        self.exit_flag = False

    def start(self, start_offset):
        self.queue = Queue.Queue(maxsize=self.queue_size)
        self.gather = PytablesBitextFetcher(self, start_offset)
        self.gather.daemon = True
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        batch = self.queue.get()
        if not batch:
            return None
        self.next_offset = batch[0]
        return batch[1], batch[2]

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


