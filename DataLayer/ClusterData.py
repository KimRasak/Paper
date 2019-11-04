from abc import ABC, abstractmethod

import numpy as np
from Common import DatasetNum
from DataLayer.Cluster.ClusterStrategyI import ClusterStrategyI
from DataLayer.Data import Data


class ClusterData(Data, ABC):
    def __init__(self, data_set_name, cluster_strategy, use_picked_data=True,
                 batch_size=256, epoch_times=4, is_debug_mode=False,
                 cluster_num=50, ut_alpha=1):
        super().__init__(data_set_name, use_picked_data=use_picked_data, batch_size=batch_size, epoch_times=epoch_times,
                         is_debug_mode=is_debug_mode)
        self.cluster_num = cluster_num

        # Get cluster of the nodes.
        data_set_num = self.data_set_num
        train_data = self.train_data
        self.cluster_strategy: ClusterStrategyI = cluster_strategy
        self.parts: np.ndarray = self.cluster_strategy.get_cluster(data_set_num, train_data, cluster_num)

        self.cluster_sizes = self.__get_cluster_sizes(self.parts, data_set_num)
        self.cluster_bounds = self.__get_cluster_bounds(self.cluster_sizes)

        # Extract training tuples from the cluster parts.
        self.global_id_cluster_id_map = self.__gen_global_id_cluster_id_map(self.parts)
        self.cluster_pos_train_tuples = self.__gen_train_tuples(train_data, data_set_num, self.parts,
                                                                self.global_id_cluster_id_map)

        # Extract track ids of each cluster. This is for negative sampling.
        self.cluster_track_ids = self.__gen_cluster_track_ids(self.parts, data_set_num, self.global_id_cluster_id_map)

        # Extract test tuples from the cluster parts. This is for testing.
        self.test_pos_tuples = self.__gen_test_pos_tuples(self.test_data,
                                                          data_set_num, self.global_id_cluster_id_map, self.parts)

        self.cluster_connections = self.__get_cluster_connections(train_data, data_set_num,
                                                                  self.parts, self.global_id_cluster_id_map)

        # Generate laplacian matrices.
        self.clusters_laplacian_matrices: dict = self.__gen_laplacian_matrices(self.cluster_pos_train_tuples,
                                                                               self.cluster_sizes,
                                                                               self.cluster_connections,
                                                                               ut_alpha=ut_alpha)

    def __init_relation_data(self, train_data: dict):
        """
        This function only needs to init dicts storing the relationships.
        The ids in the dicts are local to its entity.
        """
        self.__init_relation_dict(train_data)

    @abstractmethod
    def get_entity_names(self):
        pass

    @abstractmethod
    def __get_cluster_sizes(self, parts, data_set_num: DatasetNum):
        """
        Get sizes of each entity in the cluster and the cluster's size.
        :return A list of dict. Each dict stores the size of each entity and the total size of the cluster.
        """
        pass

    @abstractmethod
    def __get_cluster_bounds(self, cluster_sizes):
        pass

    @abstractmethod
    def __gen_global_id_cluster_id_map(self, parts):
        """
        Generate an numpy array that stores the mappings from global id to cluster id.
        """
        pass

    @abstractmethod
    def __gen_train_tuples(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
        """
        Generate the train tuples of each cluster.
        :return: A list of dicts. Each dict stores the training tuples of a cluster.
        """
        pass

    @abstractmethod
    def __map_entity_id_to_global_id(self, entity_id, data_set_num, entity_name):
        """
        Map an entity id to a global id.
        :return: The related global id.
        """
        pass

    @abstractmethod
    def __gen_cluster_track_ids(self, parts, data_set_num, global_id_cluster_id_map):
        """
        Generate a list to store the track ids(global id and cluster id) of each cluster.
        :return:
        """
        pass

    @abstractmethod
    def __gen_test_pos_tuples(self, test_data, data_set_num: DatasetNum, global_id_cluster_id_map, parts):
        """
        Generate postive test tuples according to the test data.
        Note that negative sampling of each test tuple will be done later.
        :return: A dict, each key is a component of a test tuple, and each value is a numpy array.
        """
        pass

    @abstractmethod
    def __get_cluster_connections(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
        pass

    @abstractmethod
    def __gen_laplacian_matrices(self, cluster_pos_train_tuples, cluster_sizes, cluster_connections, ut_alpha):
        pass

    @abstractmethod
    def sample_negative_test_track_ids(self, uid, pid):
        pass
