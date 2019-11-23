from abc import ABC, abstractmethod
from time import time

import numpy as np
from Common import DatasetNum
from DataLayer.Cluster.ClusterStrategyI import ClusterStrategyI
from DataLayer.Data import Data


class ClusterData(Data, ABC):
    def __init__(self, data_set_name, cluster_strategy, use_picked_data=True,
                 batch_size=256, epoch_times=4, is_debug_mode=False,
                 cluster_num=50, ut_alpha=1):
        super().__init__(data_set_name, use_picked_data=use_picked_data, epoch_times=epoch_times,
                         is_debug_mode=is_debug_mode)
        all_start_t = time()
        self.cluster_num = cluster_num

        # Get cluster of the nodes.
        data_set_num = self.data_set_num
        train_data = self.train_data
        self.cluster_strategy: ClusterStrategyI = cluster_strategy
        self.parts: np.ndarray = self.cluster_strategy.get_cluster(data_set_num, train_data, cluster_num)

        self.cluster_sizes = self._get_cluster_sizes(self.parts, data_set_num)
        self.cluster_bounds = self._get_cluster_bounds(self.cluster_sizes)

        # Extract training tuples from the cluster parts.
        self.global_id_cluster_id_map = self._gen_global_id_cluster_id_map(self.parts)
        self.cluster_id_entity_id_map = self._gen_cluster_id_entity_id_map(self.parts, self.cluster_sizes, self.global_id_cluster_id_map)

        self.cluster_pos_train_tuples = self._gen_train_tuples(train_data, data_set_num, self.parts,
                                                               self.global_id_cluster_id_map)

        # Extract track ids of each cluster. This is for negative sampling.
        self.cluster_track_ids = self._gen_cluster_track_ids(self.parts, data_set_num, self.global_id_cluster_id_map)

        # Extract test tuples from the cluster parts. This is for testing.
        self.test_pos_tuples = self._gen_test_pos_tuples(self.test_data,
                                                         data_set_num, self.global_id_cluster_id_map, self.parts)

        self.inter_cluster_connections = self._get_inter_cluster_connections(train_data, data_set_num,
                                                                             self.parts, self.global_id_cluster_id_map)

        self.all_cluster_connections = self._get_all_cluster_connections(train_data, data_set_num, self.parts, self.global_id_cluster_id_map)

        # Generate laplacian matrices.
        gen_laplacian_start_t = time()
        self.single_cluster_laplacian_matrices: dict = self._gen_single_cluster_laplacian_matrices(self.cluster_pos_train_tuples,
                                                                                                   self.cluster_sizes,
                                                                                                   self.inter_cluster_connections,
                                                                                                   ut_alpha=ut_alpha)
        gen_laplacian_end_t = time()
        print("Generating laplacian matrices used %f seconds." % (gen_laplacian_end_t - gen_laplacian_start_t))

        # Print total time used.
        all_end_t = time()
        print("Reading data used %d seconds in all." % (all_end_t - all_start_t))

    def _init_relation_data(self, train_data: dict):
        """
        This function only needs to init dicts storing the relationships.
        The ids in the dicts are local to its entity.
        """
        self._init_relation_dict()

    @abstractmethod
    def get_entity_names(self):
        pass

    @abstractmethod
    def _get_cluster_sizes(self, parts, data_set_num: DatasetNum):
        """
        Get sizes of each entity in the cluster and the cluster's size.
        :return A list of dict. Each dict stores the size of each entity and the total size of the cluster.
        """
        pass

    @abstractmethod
    def _get_cluster_bounds(self, cluster_sizes):
        pass

    @abstractmethod
    def _gen_global_id_cluster_id_map(self, parts):
        """
        Generate the map of global id -> cluster id.
        That is, generate an numpy array that stores the mappings from global id to cluster id.
        """
        pass

    @abstractmethod
    def _gen_cluster_id_entity_id_map(self, parts, cluster_sizes, global_id_cluster_id_map):
        """
        Generate the map of cluster id -> entity id.
        """
        pass

    @abstractmethod
    def _gen_train_tuples(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
        """
        Read the train_data, generate the train tuples of each cluster.
        :return: A list of dicts. Each dict stores the training tuples of a cluster.
        """
        pass

    @abstractmethod
    def _map_entity_id_to_global_id(self, entity_id, data_set_num, entity_name):
        """
        Map an entity id to a global id.
        :return: The related global id.
        """
        pass

    @abstractmethod
    def _map_global_id_to_entity_id(self, global_id, data_set_num: DatasetNum):
        """
        Map a global id to an entity id.
        :return: The related entity id.
        """
        pass

    @abstractmethod
    def _gen_cluster_track_ids(self, parts, data_set_num, global_id_cluster_id_map):
        """
        Generate a list to store the track ids(global id and cluster id) of each cluster.
        :return:
        """
        pass

    @abstractmethod
    def _gen_test_pos_tuples(self, test_data, data_set_num: DatasetNum, global_id_cluster_id_map, parts):
        """
        Generate postive test tuples according to the test data.
        Note that negative sampling of each test tuple will be done later.
        :return: A dict, each key is a component of a test tuple, and each value is a numpy array.
        """
        pass

    @abstractmethod
    def _get_inter_cluster_connections(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
        """
        Generate the data storing the inter cluster connections.
        """
        pass

    @abstractmethod
    def _get_all_cluster_connections(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
        """
        Generate the data storing inter and intra cluster connections.
        """
        pass

    @abstractmethod
    def _gen_single_cluster_laplacian_matrices(self, cluster_pos_train_tuples, cluster_sizes, cluster_connections, ut_alpha):
        pass

