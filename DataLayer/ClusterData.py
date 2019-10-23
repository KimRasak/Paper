from abc import ABC, abstractmethod

from DataLayer.Data import Data


class ClusterStrategy(ABC):
    @abstractmethod
    def cluster(self, data):
        pass


class UTClusterStrategy(ClusterStrategy):
    def cluster(self, data):
        pass


class PTClusterStrategy(ClusterStrategy):
    def cluster(self, data):
        pass


class UPTClusterStrategy(ClusterStrategy):
    def cluster(self, data):
        pass


class ClusterData(Data, ABC):
    def __init__(self, data_set_name, use_picked_data=True,
                 batch_size=256, epoch_times=4, is_test_mode=False,
                 num_cluster=50):
        super().__init__(data_set_name, use_picked_data=True, batch_size=256, epoch_times=4, is_test_mode=False)
        self.num_cluster = num_cluster

    def __init_relation_data(self, train_data: dict):
        """
        This function only needs to init dicts storing the relationships.
        :param train_data:
        :return:
        """
        self.__init_relation_dict(train_data)
