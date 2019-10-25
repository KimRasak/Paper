from abc import ABC, abstractmethod

import numpy as np
from Common import DatasetNum
from DataLayer.Cluster.ClusterStrategyI import ClusterStrategyI
from DataLayer.Data import Data


class ClusterData(Data, ABC):
    def __init__(self, data_set_name, cluster_strategy, use_picked_data=True,
                 batch_size=256, epoch_times=4, is_test_mode=False,
                 num_cluster=50):
        super().__init__(data_set_name, use_picked_data=use_picked_data, batch_size=batch_size, epoch_times=epoch_times, is_test_mode=is_test_mode)
        self.num_cluster = num_cluster

        # Get cluster of the nodes.
        data_set_num = self.data_set_num
        train_data = self.train_data
        self.cluster_strategy: ClusterStrategyI = cluster_strategy
        self.parts: np.ndarray = self.cluster_strategy.get_cluster(data_set_num, train_data, num_cluster)

        # Extract training tuples from the cluster parts.
        self.cluster_sizes, self.global_id_cluster_id_map = self.__gen_global_id_cluster_id_map(self.parts, data_set_num)
        self.cluster_pos_train_tuples = self.__gen_train_tuples(train_data, data_set_num, self.parts, self.global_id_cluster_id_map)

        # Extract track ids of each cluster.
        self.cluster_track_ids = self.__gen_cluster_track_ids(self.parts, data_set_num, self.global_id_cluster_id_map)

        # Extract test tuples from the cluster parts.
        self.__gen_test_tuples(self.test_list, data_set_num, self.global_id_cluster_id_map)

    def __init_relation_data(self, train_data: dict):
        """
        This function only needs to init dicts storing the relationships.
        The ids in the dicts are local to its entity.
        """
        self.__init_relation_dict(train_data)

    @abstractmethod
    def __gen_global_id_cluster_id_map(self, parts, data_set_num: DatasetNum):
        pass

    @abstractmethod
    def __gen_train_tuples(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
        pass

    @abstractmethod
    def __map_entity_id_to_global_id(self, entity_id, data_set_num, entity_name):
        pass

    @abstractmethod
    def __gen_cluster_track_ids(self, parts, data_set_num, global_id_cluster_id_map):
        pass

    @abstractmethod
    def __gen_test_tuples(self, test_list, data_set_num: DatasetNum, global_id_cluster_id_map):
        pass
