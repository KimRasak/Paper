from abc import ABCMeta, abstractmethod
from collections import namedtuple

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from DataLayer.Data import Data
from FileLayer.ckpt_file_layer import SaveManager
from FileLayer.log_file_layer import LogManager
from ModelLayertf2 import Metric


def _convert_sp_mat_to_sp_tensor(X):
    if X.getnnz() == 0:
        print("add one.", X.shape)
        return 0
        # X[0, 0] = 1
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()

    return tf.sparse.SparseTensor(indices, coo.data, dense_shape=X.shape)

class Loss:
    reg_loss: np.float
    mf_loss: np.float

    def __init__(self, reg_loss=0.0, mf_loss=0.0):
        self.reg_loss = reg_loss
        self.mf_loss = mf_loss

    def clear(self):
        self.reg_loss, self.mf_loss = 0.0, 0.0

    def add_loss(self, loss):
        self.reg_loss += loss.reg_loss
        self.mf_loss += loss.mf_loss

    def add_reg_loss(self, reg_loss):
        self.reg_loss += reg_loss

    def add_mf_loss(self, mf_loss):
        self.mf_loss += mf_loss

    def get_reg_loss(self):
        return self.reg_loss

    def get_mf_loss(self):
        return self.mf_loss

    def get_total_loss(self):
        return self.reg_loss + self.mf_loss

    def to_string(self):
        return "[reg_loss: %f, mf_loss: %f]" % (self.reg_loss, self.mf_loss)


class MDR(layers.Layer):
    def __init__(self, initializer, eb_size, track_num):
        super(MDR, self).__init__()
        self.B1 = tf.Variable(initializer([eb_size]), name="MDR_B1")
        self.B2 = tf.Variable(initializer([eb_size]), name="MDR_B2")
        self.track_biases = tf.Variable(initializer([track_num]), name="track_bias")

    def call(self, inputs, **kwargs):
        return self.__mdr_layer(inputs["user_ebs"], inputs["playlist_ebs"], inputs["track_ebs"],
                                inputs["track_entity_ids"])

    @staticmethod
    def __get_output(delta, B):
        B_delta = tf.multiply(B, delta)
        square = tf.square(B_delta)
        # print("square:", square, len(square.shape) - 1)
        return tf.reduce_sum(square, axis=len(square.shape) - 1)

    def __mdr_layer(self, user_ebs, playlist_ebs, track_ebs, track_entity_ids):
        delta_ut = user_ebs - track_ebs
        delta_pt = playlist_ebs - track_ebs

        o1 = MDR.__get_output(delta_ut, self.B1)
        o2 = MDR.__get_output(delta_pt, self.B2)
        # print("shape of o1/o2:", o1.shape, o2.shape)
        # print("-----")

        track_bias = tf.nn.embedding_lookup(self.track_biases, track_entity_ids)

        return o1 + o2 + track_bias

    def get_trainable_variables(self):
        return [self.B1, self.B2, self.track_biases]

    def get_reg_loss(self):
        return tf.nn.l2_loss(self.B1) + tf.nn.l2_loss(self.B2)


class BaseModel(metaclass=ABCMeta):
    epoch_num: int  # number of train epochs.
    embedding_size: int
    learning_rate = 2e-4
    reg_loss_ratio = 5e-5  # The ratio of regularization loss.

    def __init__(self, epoch_num, embedding_size=64, learning_rate=2e-4, reg_loss_ratio=5e-5):
        # Init params.
        self.epoch_num = epoch_num
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.reg_loss_ratio = reg_loss_ratio
        self.max_K = 20

    def init(self):
        # Init other variables.
        self.initializer = tf.initializers.VarianceScaling(scale=0.1)
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        # Init log manager and save manager.
        model_name = self.__get_model_name()
        self.log_manager = LogManager(model_name)

        # Build model of the layers.
        nets = self._build_model()
        self.save_manager = SaveManager(model_name, self.optimizer, nets)

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

    @abstractmethod
    def _build_model(self):
        pass

    def fit(self):
        cur_epoch_step = self.save_manager.get_train_step()
        for epoch in range(cur_epoch_step, self.epoch_num):
            epoch_loss, epoch_time = self._train_epoch(epoch)
            metrics, test_time = self._test(epoch)
            self.__output_epoch_message(epoch, epoch_time, epoch_loss)
            self._output_test_result(epoch, test_time, metrics)
            self.__save_model()

    @abstractmethod
    def _train_epoch(self, epoch):
        pass

    @abstractmethod
    def _test(self, epoch):
        pass

    def __output_epoch_message(self, epoch, epoch_time, epoch_loss: Loss):
        """
        Output the loss of the epoch and total time used in the epoch,
        """
        log = "Epoch %d used %f seconds. The epoch loss is: %s" % (epoch, epoch_time, epoch_loss.to_string())
        self.log_manager.print_and_write(log)

    @abstractmethod
    def _output_test_result(self, epoch, test_time, metrics: Metric):
        pass

    def __save_model(self):
        # if np.isnan(epoch_loss.get_total_loss()):
        #     print("Found nan loss. Don't save.")
        #     return

        self.save_manager.save_model()

