"""
Data iterator for text datasets that are used for language modeling.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Caglar Gulcehre "
               "KyungHyun Cho ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import logging

logger = logging.getLogger(__name__)

class LMIterator(object):

    def __init__(self,
                 batch_size,
                 path=None,
                 use_infinite_loop=True,
                 stop=-1,
                 seq_len=10,
                 mode="train",
                 chunks="words",
                 shift = 1,
                 reset = None,
                 allow_short_sequences = False,
                 output_format = None,
                 can_fit = False):
        """
        :type batch_size: int
        :param batch_size: Make the data iterator return minibatches of this
            size. A minibatch has time as its first dimenion, the different
            examples as the second and the example dimensionality as the
            last. If set to 1, the iterator will return a matrix, where
            first dimension is time and second dimension is the data
            dimensionality

        :type path: string
        :param path: string; path where the dataset can be found

        :type use_infinite_loop: bool
        :param use_infinite_loop: If set to true, the iterator will not
            raise StopIteration, but rather loop for an infinite amound of
           steps

        :type seq_len: int
        :param seq_len: The length of each sequence in the minibatch

        :type mode: string
        :param mode: It can have one of the following values {'train',
            'valid', test'}, depicting which fold of the dataset it needs to
            loop over

        :type chunks: string
        :param chunks: It can be either {'word', 'char'}. It says what is
            the input of the dataset.

        :type shift: int
        :param shift: The target of at each time step is the `shift`-th
            word/charater in the future. For example `shift` is 1, means the
            next one, `shift` is 2 means predict the word at position t+2

        :type reset: int or None
        :param reset: If negative, the data iterator will stop after it
           looped over all examples in the dataset. If positive, the
           iterator will only loop over the first `reset` examples of the
           dataset.

        :type allow_short_sequences: bool
        :param allow_short_sequences: allow the iterator to return sequences
            that are shorter than the required `seq_len`

        :type output_format: None or function
        :param output_format: function wraps the returned samples into a
            different format. Specifically, the iterator will return
            `output_format`(x) instead of x

        :type can_fit: bool
        :param can_fit: Flag saying if the dataset fits in memory or not
        """

        assert type(path) is not None, "Target language file should be a list."


        self.reset = reset
        self.batch_size = batch_size
        self.offset = 0
        self.allow_short_sequences = allow_short_sequences
        self.output_format = output_format
        self.stop = stop
        self.can_fit = can_fit
        self.path = None
        self.chunks = chunks
        self.path = path
        self.mode = mode
        self.seq_len = seq_len
        self.batch_no = 0
        self.use_infinite_loop = use_infinite_loop
        self.data_len = 0
        self.load_files()
        self.shift = shift

    def load_files(self):
        mmap_mode = None
        if self.can_fit == False:
            mmap_mode = "r"

        penn_data = numpy.load(self.path, mmap_mode=mmap_mode)

        self.penn_nwords = penn_data["n_words"] if 'n_words' in penn_data else 0
        self.penn_nchars = penn_data["n_chars"] if 'n_chars' in penn_data else 0


        if self.chunks == "words":
            self.n_in = self.penn_nwords
            self.n_out = self.penn_nwords
            if self.mode == "train":
                self.data = penn_data['train_words']
                self.data_len = self.data.shape[0]
            elif self.mode == "valid":
                self.data = penn_data['valid_words']
                self.data_len = self.data.shape[0]
            elif self.mode == "test":
                self.data = penn_data['test_words']
                self.data_len = self.data.shape[0]
        elif self.chunks == "chars":
            self.n_in = self.penn_nchars
            self.n_out = self.penn_nchars
            if self.mode == "train":
                self.data = penn_data['train_chars']
                self.data_len = self.data.shape[0]
            elif self.mode == "valid":
                self.data = penn_data['valid_chars']
                self.data_len = self.data.shape[0]
            elif self.mode == "test":
                self.data = penn_data['test_chars']
                self.data_len = self.data.shape[0]
        if self.reset is not None and self.reset > 0:
            self.data = self.data[:self.reset]
            self.data_len = self.data.shape[0]
        print "data length is ", self.data_len
        del penn_data

    def get_length(self):
        return (self.data_len-self.shift) // (self.batch_size*self.seq_len)

    def start(self, start_offset):
        logger.debug("Not supported")
        self.next_offset = -1

    def __iter__(self):
        return self

    def next(self):
        if self.offset == 0:
            reset_h0 = 1
        else:
            reset_h0 = 0
        inc_offset = self.offset+self.batch_size*self.seq_len

        cond = self.offset > (self.data_len - self.shift)
        cond = cond or ((inc_offset >= self.data_len-self.shift) and
                        not self.allow_short_sequences)

        if cond and self.use_infinite_loop:
            print "Restarting the dataset iterator."
            self.offset = 0
            reset_h0 = 1
            inc_offset = self.offset+self.batch_size*self.seq_len
        elif cond:
            self.offset = 0
            raise StopIteration
        if inc_offset >= self.data.shape[0]-self.shift:
            inc_offset = self.data.shape[0]-self.shift
        data_part = self.data[self.offset:inc_offset]
        target = self.data[self.offset+self.shift:inc_offset+self.shift]

        self.offset = self.offset + self.batch_size*self.seq_len
        target = target.reshape(self.batch_size, -1)
        target = target.T
        data_part = data_part.reshape(self.batch_size,-1)
        data_part = data_part.T
        self.batch_no += 1
        if self.batch_size == 1:
            data_part = data_part[:,0]
            target = target[:,0]
        if not self.output_format:
            return data_part, target, numpy.float32(1-reset_h0)
        else:
            return self.output_format(data_part, target,
                                      numpy.float32(1-reset_h0))

