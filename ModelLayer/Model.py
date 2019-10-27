from abc import ABCMeta

import tensorflow as tf

from DataLayer.Data import Data


class Model(metaclass=ABCMeta):
    num_epoch: int  # number of train epochs.
    data: Data  # Source of data for training and testing.
    save_loss_batch_num: int  # Save the output of batch loss for every {save_loss_batch_num} batch.
    embedding_size: int
    learning_rate = 2e-4
    reg_loss_ratio = 5e-5  # The ratio of regularization loss.

    def __init__(self, epoch_num, data: Data, n_save_batch_loss=300,
                 embedding_size=64, learning_rate=2e-4, reg_loss_ratio=5e-5):
        # Init params.
        self.epoch_num = epoch_num
        self.data = data
        self.save_loss_batch_num = n_save_batch_loss
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.reg_loss_ratio = reg_loss_ratio

        self.initializer = tf.contrib.layers.xavier_initializer()
        self.t_node_dropout = tf.placeholder(tf.float32, shape=[None])

    def __get_model_name(self):
        def get_base_index(number):
            """
            Get the base and index of the number.
            For example, when the input is 1e-3, the output is 1 and -3
            """
            a = number
            index = 0
            while a < 1:
                a *= 10
                index += 1
            base = a
            return base, index

        class_name = self.__class__.__name__
        base_lr, index_lr = get_base_index(self.learning_rate)
        base_rr, index_rr = get_base_index(self.reg_loss_ratio)
        eb_size = self.embedding_size

        model_name = "%s_%s_eb%d_lr%de-%d_rr%de-%d" %\
                     (class_name, self.data.data_set_name,
                      eb_size, base_lr, index_lr, base_rr, index_rr)
        return model_name


