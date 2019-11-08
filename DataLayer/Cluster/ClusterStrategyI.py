import os
from abc import ABC, abstractmethod

import numpy as np

from Common import DatasetNum
import FileLayer.cluster_file_layer as cluster_file_layer
from DataLayer.Cluster.DoClusterStrategy import DoClusterStrategyI, DoPTClusterStrategy, DoUPTClusterStrategy


class ClusterStrategyI(ABC):
    __do_cluster_strategy: DoClusterStrategyI

    def __init__(self, data_set_name):
        self.__cluster_dir_path = cluster_file_layer.get_cluster_dir_path(data_set_name)

    def get_cluster(self, data_set_num: DatasetNum, data, num_cluster):
        """
        Return the cluster of the ids.
        If the cluster file exists, read the cluster from the file,
        else generate the cluster of the ids and write to the cluster file.
        :param data_set_num: The DatasetNum.
        :param data: The playlist data, usually train data.
        :param num_cluster: Number of cluster of the ids.
        :return:
        """
        cluster_file_path = self.__get_cluster_file_path(num_cluster)

        if os.path.exists(cluster_file_path):
            return self.__read_cluster_file(cluster_file_path)
        parts = self._cluster(data_set_num, data, num_cluster)
        self.__write_cluster_file(cluster_file_path, parts)
        return parts

    def __write_cluster_file(self, cluster_file_path, parts):
        cluster_file_layer.write_cluster_file(cluster_file_path, parts)

    def __read_cluster_file(self, cluster_file_path):
        return np.array(cluster_file_layer.read_cluster_file(cluster_file_path))

    @abstractmethod
    def _cluster(self, data_set_num: DatasetNum, data, num_cluster):
        """
        Generate clusters of the ids and return the parts.
        :return numpy.ndarray
        """
        pass

    @abstractmethod
    def _get_cluster_type_name(self):
        pass

    def __get_cluster_file_path(self, cluster_num):
        cluster_dir_path = self.__cluster_dir_path
        cluster_file_name = self._get_cluster_file_name(cluster_num)

        return os.path.join(cluster_dir_path, cluster_file_name)

    @abstractmethod
    def _get_cluster_file_name(self, num_cluster):
        pass


class PTClusterStrategy(ClusterStrategyI):
    __do_cluster_strategy = DoPTClusterStrategy()

    def _cluster(self, data_set_num, data, num_cluster):
        return self.__do_cluster_strategy.do_cluster(data_set_num, data, num_cluster)

    def _get_cluster_type_name(self):
        return "clusterPT"

    def _get_cluster_file_name(self, num_cluster):
        return "%s_%d" % (self._get_cluster_type_name(), num_cluster)


class UPTClusterStrategyI(ClusterStrategyI, ABC):
    class PickClusterStrategyI(ABC):
        @abstractmethod
        def pick_cluster(self, user: dict, pt_parts: np.ndarray, dataset_num: DatasetNum):
            """
            Pick a cluster number for a user id that belongs to no cluster.
            :return:
            """
            pass

        @abstractmethod
        def get_name(self):
            pass

    class MostNumPickClusterStrategy(PickClusterStrategyI):
        def pick_cluster(self, user: dict, pt_parts, dataset_num: DatasetNum):
            """
            The playlists of the user belong to different clusters,
            so we choose and return the cluster number of the cluster
             having the most tracks that the user has collected.
            """
            cluster_track_num = dict()
            for pid, tids in user.items():
                pid_offset = 0
                cluster_number = pt_parts[pid + pid_offset]

                if cluster_number not in cluster_track_num:
                    cluster_track_num[cluster_number] = 0
                else:
                    cluster_track_num[cluster_number] += len(tids)

            cluster_numbers = list(cluster_track_num.keys())
            cluster_numbers = sorted(cluster_numbers, key=lambda n: cluster_track_num[n], reverse=True)
            return cluster_numbers[0]

        def get_name(self):
            return "MostNumPick"

    class FirstChoicePickClusterStrategy(PickClusterStrategyI):
        def pick_cluster(self, user: dict, pt_parts, dataset_num: DatasetNum):
            """
            Return the cluster number of the first playlist.
            :param user: Dict, where the keys are playlist ids
             and the values are the ids of the playlists' tracks.
            :param pt_parts: Cluster parts of the playlists and tracks.
            """
            for entity_pid in user.keys():
                global_pid = dataset_num.user + entity_pid
                assert dataset_num.user <= global_pid < dataset_num.user + dataset_num.playlist
                return pt_parts[global_pid]

        def get_name(self):
            return "FirstChoicePick"


class UPTFromPTClusterStrategy(UPTClusterStrategyI):
    __pick_cluster_strategy: UPTClusterStrategyI.PickClusterStrategyI

    def __init__(self, data_set_name, pick_cluster_strategy):
        super().__init__(data_set_name)
        self.__do_cluster_strategy = DoPTClusterStrategy()
        self.__pick_cluster_strategy = pick_cluster_strategy

    def _cluster(self, data_set_num, data, num_cluster):
        data_set_sum = data_set_num.user + data_set_num.playlist + data_set_num.track
        parts = np.full((data_set_sum, ), -1)

        # Generate the clusters from playlist and track ids.
        pt_parts = self.__do_cluster_strategy.do_cluster(data_set_num, data, num_cluster)

        zero_num = 0
        for global_id, cluster_no in enumerate(pt_parts):
            if cluster_no == 0:
                zero_num += 1

        # Pick a cluster number for each user id.
        for entity_uid, user in data.items():
            global_uid = entity_uid
            assert 0 <= global_uid < data_set_num.user
            picked_cluster_no = self.__pick_cluster_strategy.pick_cluster(user, pt_parts, data_set_num)
            parts[global_uid] = picked_cluster_no

        # copy the cluster numbers in pt_parts to parts
        for playlist_or_track_id, cluster_no in enumerate(pt_parts):
            global_id = data_set_num.user + playlist_or_track_id
            assert data_set_num.user <= global_id< data_set_sum
            parts[global_id] = pt_parts[playlist_or_track_id]

        zero_num = 0
        for global_id, cluster_no in enumerate(parts):
            if cluster_no == 0:
                zero_num += 1
            assert cluster_no != -1, "Wrong cluster number!"

        # Put user ids into parts
        return parts

    def _get_cluster_type_name(self):
        return "clusterUPTFromPT"

    def _get_cluster_strategy_name(self):
        return self.__pick_cluster_strategy.get_name()

    def _get_cluster_file_name(self, num_cluster):
        return "%s_%s_%d" % (self._get_cluster_type_name(), self._get_cluster_strategy_name(), num_cluster)


class UPTFromUPTClusterStrategy(UPTClusterStrategyI):
    def __init__(self, data_set_name):
        super().__init__(data_set_name)
        self.__do_cluster_strategy = DoUPTClusterStrategy()

    def _cluster(self, data_set_num, data, num_cluster):
        parts = self.__do_cluster_strategy.do_cluster(data_set_num, data, num_cluster)
        return parts

    def _get_cluster_type_name(self):
        return "clusterUPTFromUPT"

    def _get_cluster_file_name(self, num_cluster):
        return "%s_%d" % (self._get_cluster_type_name(), num_cluster)
