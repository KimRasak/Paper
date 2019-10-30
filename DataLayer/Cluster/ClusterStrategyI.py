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
        parts = self.__cluster(data_set_num, data, num_cluster)
        self.__write_cluster_file(cluster_file_path, parts)
        return parts

    def __write_cluster_file(self, cluster_file_path, parts):
        cluster_file_layer.write_cluster_file(cluster_file_path, parts)

    def __read_cluster_file(self, cluster_file_path):
        return np.array(cluster_file_layer.read_cluster_file(cluster_file_path))

    @abstractmethod
    def __cluster(self, data_set_num: DatasetNum, data, num_cluster):
        """
        Generate clusters of the ids and return the parts.
        :return numpy.ndarray
        """
        pass

    @abstractmethod
    def __get_cluster_type_name(self):
        pass

    def __get_cluster_file_path(self, num_cluster):
        cluster_dir_path = self.__cluster_dir_path
        cluster_file_name = self.__get_cluster_file_name(num_cluster)

        return os.path.join(cluster_dir_path, cluster_file_name)

    @abstractmethod
    def __get_cluster_file_name(self, num_cluster):
        pass


class PTClusterStrategy(ClusterStrategyI):
    __do_cluster_strategy = DoPTClusterStrategy()

    def __cluster(self, data_set_num, data, num_cluster):
        return self.__do_cluster_strategy.do_cluster(data_set_num, data, num_cluster)

    def __get_cluster_type_name(self):
        return "clusterPT"

    def __get_cluster_file_name(self, num_cluster):
        return "%s_%d" % (self.__get_cluster_type_name(), num_cluster)


class UPTClusterStrategyI(ClusterStrategyI, ABC):
    class PickClusterStrategyI(ABC):
        @abstractmethod
        def pick_cluster(self, user: dict, pt_parts: np.ndarray):
            """
            Pick a cluster number for a user id that belongs to no cluster.
            :return:
            """
            pass

        @abstractmethod
        def get_name(self):
            pass

    class MostNumPickClusterStrategy(PickClusterStrategyI):
        def pick_cluster(self, user: dict, pt_parts):
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
            return "MostNum"

    class FirstChoicePickClusterStrategy(PickClusterStrategyI):
        def pick_cluster(self, user: dict, pt_parts):
            """
            Return the cluster number of the first playlist.
            :param user: Dict, where the keys are pids
             and the values are the tids.
            :param pt_parts: Cluster parts of the playlists and tracks.
            """
            for pid in user.keys():
                pid_offset = 0
                return pt_parts[pid + pid_offset]

        def get_name(self):
            return "First"


class UPTFromPTClusterStrategy(UPTClusterStrategyI):
    __pick_cluster_strategy: UPTClusterStrategyI.PickClusterStrategyI

    def __init__(self, data_set_name, pick_cluster_strategy):
        super().__init__(data_set_name)
        self.__do_cluster_strategy = DoPTClusterStrategy()
        self.__pick_cluster_strategy = pick_cluster_strategy

    def __cluster(self, data_set_num, data, num_cluster):
        data_set_sum = data_set_num.user + data_set_num.playlist + data_set_num.track
        parts = np.array((data_set_sum, ), dtype=np.int)

        # Generate the clusters from playlist and track ids.
        pt_parts = self.__do_cluster_strategy.do_cluster(data_set_num, data, num_cluster)

        for uid, user in data.items():
            picked_cluster = self.__pick_cluster_strategy.pick_cluster(user, pt_parts)
            parts[uid] = picked_cluster

        for playlist_or_track_id in pt_parts:
            offset = data_set_num.user
            parts[playlist_or_track_id + offset] = pt_parts[playlist_or_track_id]

        # Put user ids into parts
        return parts

    def __get_cluster_type_name(self):
        return "clusterUPT"

    def __get_cluster_strategy_name(self):
        return self.__pick_cluster_strategy.get_name()

    def __get_cluster_file_name(self, num_cluster):
        return "%s_%s_%d" % (self.__get_cluster_type_name(), self.__get_cluster_strategy_name(), num_cluster)


class UPTFromUPTClusterStrategy(UPTClusterStrategyI):
    def __init__(self, data_set_name):
        super().__init__(data_set_name)
        self.__do_cluster_strategy = DoUPTClusterStrategy()

    def __cluster(self, data_set_num, data, num_cluster):
        parts = self.__do_cluster_strategy.do_cluster(data_set_num, data, num_cluster)
        return parts

    def __get_cluster_type_name(self):
        return "clusterUPT"

    def __get_cluster_file_name(self, num_cluster):
        return "%s_%d" % (self.__get_cluster_type_name(), num_cluster)
