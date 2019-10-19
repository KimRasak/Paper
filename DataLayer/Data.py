import os
from time import time
from abc import ABCMeta, abstractmethod

import scipy.sparse as sp
import numpy as np


import FileLayer.data_file_layer as data_file_layer
from FileLayer import DatasetNum


class Data(metaclass=ABCMeta):
    # File names of data set.
    TRAIN_FILE_NAME = "train.txt"
    TEST_FILE_NAME = "test.txt"
    EVENTS_FILE_NAME = "event.txt"

    # File names of picked data set.
    PREFIX_PICK_FILE = "pick"
    PICK_TRAIN_FILE_NAME = PREFIX_PICK_FILE + "_" + TRAIN_FILE_NAME
    PICK_TEST_FILE_NAME = PREFIX_PICK_FILE + "_" + TEST_FILE_NAME
    PICK_EVENTS_FILE_NAME = PREFIX_PICK_FILE + "_" + EVENTS_FILE_NAME

    def __init__(self, data_set_name, use_picked_data=True,
                 batch_size=256, epoch_times=4, is_test_mode=False):
        t_all_start = time()
        self.use_picked_data = use_picked_data

        self.batch_size = batch_size

        # Define the file paths of the data set.
        train_file_path = data_file_layer.get_train_file_path(use_picked_data, data_set_name)
        test_file_path = data_file_layer.get_test_file_path(use_picked_data, data_set_name)
        count_file_path = data_file_layer.get_count_file_path(use_picked_data, data_set_name)

        # Define data set paths.
        if use_picked_data:
            print("{pick} == %r, you're using a sub-dataset" % use_picked_data)
        else:
            print("{pick} == %r, you're using a complete dataset" % use_picked_data)

        # Read number of entities in the data set.
        self.data_set_num = self.__read_count_file(count_file_path)
        self.sum = self.__get_data_sum(self.data_set_num)

        train_data = data_file_layer.read_playlist_data(train_file_path)
        self.__init_relation_data(train_data)

        test_data = data_file_layer.read_playlist_data(test_file_path)
        self.__init_test_data(test_data)

        # Print total time used.
        t_all_end = time()
        print("Reading data used %d seconds in all." % (t_all_end - t_all_start))


    @staticmethod
    def __read_count_file(count_file_path):
        return data_file_layer.read_count_file(count_file_path)

    @abstractmethod
    def __get_data_sum(self, data_set_num: DatasetNum):
        pass


    @abstractmethod
    def __init_relation_data(self, data: dict):
        """
        Initialize dicts for storing the relationship of entities.

        :return: null
        """
        pass

    @abstractmethod
    def __init_relation_dict(self, data: dict):
        """
        Init matrices of relations among
        :param data:
        :return:
        """
        pass

    @abstractmethod
    def __init_test_data(self, test_data: dict):
        pass



