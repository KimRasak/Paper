from abc import abstractmethod

import numpy as np
from DataLayer.Data import Data
import FileLayer.data_file_layer as data_file_layer


class NormalData(Data):
    def __init__(self, data_set_name, use_picked_data=True,
                 epoch_times=4, is_debug_mode=False, batch_size=256, use_laplacian=False):
        super().__init__(data_set_name, use_picked_data,
                 epoch_times, is_debug_mode)
        self.batch_size = batch_size
        if use_laplacian:
            self.laplacian_matrices = self._get_laplacian_matrices(self.train_data, self.data_set_num)
            pass

    def _init_relation_data(self, train_data):
        """
        Initialize dicts for storing the relationship of entities.

        :return: null
        """
        self._init_relation_matrix()
        self._init_relation_dict()

    @abstractmethod
    def _init_relation_matrix(self):
        """
        Init matrices of relationships among entities.
        :return: null
        """
        pass

    @abstractmethod
    def _get_batch_num(self):
        pass

    @abstractmethod
    def get_batches(self):
        pass

    @abstractmethod
    def _get_laplacian_matrices(self, train_data, data_set_num):
        pass


