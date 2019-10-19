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

        # Read train data for training.
        train_data = data_file_layer.read_playlist_data(train_file_path)
        self.__init_relation_data(train_data)

        # Generate test list data for testing.
        test_data = data_file_layer.read_playlist_data(test_file_path)
        self.test_list = self.__gen_test_list(test_data)

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

<<<<<<< HEAD
    @abstractmethod
    def __init_test_data(self, test_data: dict):
        pass
=======
    @staticmethod
    def __gen_test_list(test_data: dict):
        t_test_start = time()
        test_list = []
        for uid, user in test_data.items():
            for pid, tids in user.items():
                for tid in tids:
                    tuple = [uid, pid, tid]
                    test_list.append(tuple)
        t_test_end = time()
        print("Generate test list used %2f seconds." % (t_test_end - t_test_start))
        return test_list

    def read_test_file(self, test_filepath):
        # 设置test_set
        t_test_set = time()
        with open(test_filepath) as f:
            line = f.readline()
            while line:
                ids = [int(i) for i in line.split(' ') if i.isdigit()]
                uid, pid, tid = ids[0], ids[1], ids[2]
                test_tuple = [uid, pid, tid]
                self.test_set.append(test_tuple)
                line = f.readline()
        print("Used %d seconds. Have read test set." % (time() - t_test_set))
>>>>>>> 6fe9c21cdb8216f6b0609b6fd013af9c647843c4



