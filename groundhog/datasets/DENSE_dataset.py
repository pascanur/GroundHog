"""
Data iterator for dense datasets (eg MNIST/CIFAR-10).
"""
__docformat__ = 'restructedtext en'
__authors__ = "Razvan Pascanu "
__contact__ = "Razvan Pascanu <razp@google>"

import numpy
import logging

logger = logging.getLogger(__name__)

class DENSE_Iterator(object):

    def __init__(self,
                 batch_size,
                 path=None,
                 use_infinite_loop=True,
                 stop=-1,
                 mode="train",
                 reset = None,
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

        :type mode: string
        :param mode: It can have one of the following values {'train',
            'valid', test'}, depicting which fold of the dataset it needs to
            loop over

        :type reset: int or None
        :param reset: If negative, the data iterator will stop after it
           looped over all examples in the dataset. If positive, the
           iterator will only loop over the first `reset` examples of the
           dataset.

        :type output_format: None or function
        :param output_format: function wraps the returned samples into a
            different format. Specifically, the iterator will return
            `output_format`(x) instead of x

        :type can_fit: bool
        :param can_fit: Flag saying if the dataset fits in memory or not
        """

        assert type(path) is not None, "Target language file should be a list."


        self.reset             = reset
        self.batch_size        = batch_size
        self.offset            = 0
        self.output_format     = output_format
        self.stop              = stop
        self.can_fit           = can_fit
        self.path              = None
        self.path              = path
        self.mode              = mode
        self.batch_no          = 0
        self.use_infinite_loop = use_infinite_loop
        self.data_len          = 0
        self.load_files()

    def load_files(self):
        mmap_mode = None
        if self.can_fit == False:
            mmap_mode = "r"

        dense_data  = numpy.load(self.path, mmap_mode=mmap_mode)
        self.data_x = dense_data[self.mode + '_x']
        self.data_y = dense_data[self.mode + '_y']

        self.data_len = self.data_x.shape[0]
        self.n_in     = self.data_x.shape[1]
        self.n_out    = numpy.max(self.data_y)


        if self.reset is not None and self.reset > 0:
            self.data_x = self.data_x[:self.reset]
            self.data_y = self.data_y[:self.reset]
            self.data_len = self.data_len
        print "data length is ", self.data_len
        del dense_data

    def get_length(self):
        return self.data_len // self.batch_size

    def start(self, start_offset):
        logger.debug("Not supported")
        self.next_offset = -1

    def __iter__(self):
        return self

    def next(self):
        reset_h0 = 1
        inc_offset = self.offset+self.batch_size

        cond = self.offset > self.data_len
        cond = cond or (inc_offset >= self.data_len)

        if cond and self.use_infinite_loop:
            print "Restarting the dataset iterator."
            self.offset = 0
            reset_h0 = 1
            inc_offset = self.offset+self.batch_size
        elif cond:
            self.offset = 0
            raise StopIteration
        if inc_offset >= self.data_x.shape[0]:
            inc_offset = self.data_x.shape[0]
        data_part = self.data_x[self.offset:inc_offset]
        target    = self.data_y[self.offset:inc_offset]

        self.offset = self.offset + self.batch_size
        self.batch_no += 1
        if self.batch_size == 1:
            data_part = data_part[0]
            target = target[0]
        if not self.output_format:
            return data_part, target, numpy.float32(1-reset_h0)
        else:
            return self.output_format(data_part, target,
                                      numpy.float32(1-reset_h0))

