from abc import ABCMeta, abstractmethod
from collections import namedtuple

import tensorflow as tf
import numpy as np

from DataLayer.Data import Data
from FileLayer.ckpt_file_layer import SaveManager
from FileLayer.log_file_layer import LogManager
from ModelLayertf2 import Metric


class Loss:
    reg_loss: np.float
    mf_loss: np.float

    def __init__(self, reg_loss=0.0, mf_loss=0.0):
        self.clear()

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
        return "reg_loss: %f, mf_loss: %f" % (self.reg_loss, self.mf_loss)


class BaseModel(metaclass=ABCMeta):
    epoch_num: int  # number of train epochs.
    save_loss_batch_num: int  # Save the output of batch loss for every {save_loss_batch_num} batch.
    embedding_size: int
    learning_rate = 2e-4
    reg_loss_ratio = 5e-5  # The ratio of regularization loss.

    def __init__(self, epoch_num, save_loss_batch_num=300,
                 embedding_size=64, learning_rate=2e-4, reg_loss_ratio=5e-5):
        # Init params.
        self.epoch_num = epoch_num
        self.save_loss_batch_num = save_loss_batch_num
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.reg_loss_ratio = reg_loss_ratio

        # Build model of the layers.
        nets = self.__build_model()

        # Init log manager and save manager.
        model_name = self.__get_model_name()
        self.log_manager = LogManager(model_name)
        self.optimizer = tf.optimizers.Adam()
        self.save_manager = SaveManager(model_name, self.optimizer, nets)

        # Init other variables.
        self.initializer = tf.initializers.GlorotNormal()

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
    def __build_model(self):
        pass

    def fit(self):
        cur_epoch_step = self.save_manager.get_train_step()
        for epoch in range(cur_epoch_step, self.epoch_num):
            epoch_loss = self.__train_epoch(epoch)
            metrics: Metric = self.__test(epoch)
            self.__output_epoch_loss(epoch, epoch_loss)
            self.__write_test_result(metrics)
            self.__save_model()

    @abstractmethod
    def __train_epoch(self, epoch):
        pass

    @abstractmethod
    def __test(self, epoch):
        pass

    def __output_epoch_loss(self, epoch, epoch_loss: Loss):
        log = "Epoch %d complete. %s" % (epoch, epoch_loss.to_string())
        self.log_manager.print_and_write(log)

    def __write_test_result(self, metrics: Metric):
        self.log_manager.write(metrics.to_string())

    def __save_model(self):
        # if np.isnan(epoch_loss.get_total_loss()):
        #     print("Found nan loss. Don't save.")
        #     return

        self.save_manager.save_model()
