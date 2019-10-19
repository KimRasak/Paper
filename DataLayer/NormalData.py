from abc import abstractmethod

from DataLayer.Data import Data
import FileLayer.data_file_layer as data_file_layer


class NormalData(Data):
    def __init__(self, base_data_path):
        super().__init__(base_data_path)

    def __init_relation_data(self, train_data):
        """
        Initialize dicts for storing the relationship of entities.

        :return: null
        """
        self.__init_relation_matrix(train_data)
        self.__init_relation_dict(train_data)

    @abstractmethod
    def __init_relation_matrix(self, data):
        """
        Init matrices of relationships among entities.
        :return: null
        """
        pass

