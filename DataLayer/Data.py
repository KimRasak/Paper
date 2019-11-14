import os
from time import time
from abc import ABCMeta, abstractmethod

import scipy.sparse as sp
import numpy as np


import FileLayer.data_file_layer as data_file_layer
from Common import DatasetNum


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

    # Entity names
    ENTITY_USER = "user"
    ENTITY_PLAYLIST = "playlist"
    ENTITY_TRACK = "track"

    def __init__(self, data_set_name, use_picked_data=True,
                 epoch_times=4, is_debug_mode=False):
        self.data_set_name = data_set_name
        self.use_picked_data = use_picked_data

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
        self.sum = self._get_data_sum(self.data_set_num)

        # Read train data for training.
        self.train_data = data_file_layer.read_playlist_data(train_file_path)
        self._init_relation_data(self.train_data)

        # Generate test list data for testing.
        self.test_data = data_file_layer.read_playlist_data(test_file_path)

    @staticmethod
    def __read_count_file(count_file_path):
        return data_file_layer.read_count_file(count_file_path)

    @abstractmethod
    def _get_data_sum(self, data_set_num: DatasetNum):
        """
        Given a DatasetNum Object, calculate and return the sum of all entities.
        """
        pass


    @abstractmethod
    def _init_relation_data(self, train_data: dict):
        """
        Initialize dicts for storing the relationship of entities.

        :return: null
        """
        pass

    @abstractmethod
    def _init_relation_dict(self):
        """
        Init matrices of relations among entities.
        :return: null
        """
        pass

    @staticmethod
    def __gen_test_list(test_data: dict):
        """Not used"""
        t_test_start = time()
        test_list = []
        # Iterate each u-p-t pair and add them to the test list.
        for uid, user in test_data.items():
            for pid, tids in user.items():
                for tid in tids:
                    tuple = [uid, pid, tid]
                    test_list.append(tuple)
        t_test_end = time()
        print("Generate test list used %2f seconds." % (t_test_end - t_test_start))
        return test_list

    @abstractmethod
    def sample_negative_test_track_ids(self, uid, pid):
        """
        Sample negative test tracks when testing, usually sample 100 negative tracks.
        """
        pass




