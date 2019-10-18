import os
from time import time
from abc import ABCMeta, abstractmethod

import scipy.sparse as sp


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

        train_file_path = data_file_layer.get_train_file_path(use_picked_data, data_set_name)
        test_file_path = data_file_layer.get_test_file_path(use_picked_data, data_set_name)
        count_file_path = data_file_layer.get_count_file_path(use_picked_data, data_set_name)

        # Define data set paths.
        if use_picked_data:
            print("{pick} == %r, Using picked playlist data. That is, you're using a sub-dataset" % use_picked_data)
        else:
            print("{pick} == %r, Using complete playlist data. That is, you're using a complete dataset" % use_picked_data)

        # 验证laplacian模式
        laplacian_modes = ["PT2", "PT4", "UT", "UPT", "None", "TestPT", "TestUPT",
                           "clusterPT2", "clusterPT4", "clusterUT", "clusterUPT"]

        # Read number of entities in the data set.
        self.data_set_num = self.__read_count_file(count_file_path)
        self.sum = self.__get_data_sum(self.data_set_num)

        self.__init_relation_dict()

        if "cluster" not in laplacian_mode:
            self.R_up = sp.dok_matrix((self.n_user, self.n_playlist), dtype=np.float64)
            self.R_ut = sp.dok_matrix((self.n_user, self.n_track), dtype=np.float64)
            self.R_pt = sp.dok_matrix((self.n_playlist, self.n_track), dtype=np.float64)
        self.pt = {}  # Storing playlist-track relationship of training set.
        self.up = {}  # Storing user-playlist relationship of training set.
        self.ut = {}  # Storing user-track relationship of training set.
        self.test_set = []  # Storing user-playlist-track test set.


        # Print time used for reading and pre-processing data.
        t_all_end = time()
        print("Reading data used %d seconds in all." % (t_all_end - t_all_start))


    @staticmethod
    def __read_count_file(count_file_path):
        return data_file_layer.read_count_file(count_file_path)

    @abstractmethod
    def __get_data_sum(self, data_set_num: DatasetNum):
        pass


    @abstractmethod
    def __init_relation_dict(self):
        """
        Initialize dicts for storing the relationship of entities.

        :return: nothing
        """
        pass


    def get_dataset_name(self):
        return self.data_base_path.split("/")[-1]