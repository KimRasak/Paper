from abc import ABC, abstractmethod
from time import time

import numpy as np
import metis
import networkx as nx

from Common import DatasetNum


class DoClusterStrategyI(ABC):
    @abstractmethod
    def _gen_global_id_pairs(self, data, data_set_num: DatasetNum):
        pass

    @abstractmethod
    def _get_sum(self, data_set_num: DatasetNum):
        pass

    def do_cluster(self, data_set_num, data, num_cluster):
        print("Generating the clusters...")
        # The order of nodes stored in graph G: U, P, T
        # for example: uid1, uid2, ..., pid1, pid2, ..., tid1, tid2, ...

        # Generate graph.
        G = nx.Graph()

        # Add nodes to graph G.
        entity_sum = self._get_sum(data_set_num)
        G.add_nodes_from([i for i in range(entity_sum)])

        # Add edges for graph G.
        global_id_pairs = self._gen_global_id_pairs(data, data_set_num)
        for from_id, to_id in global_id_pairs:
            assert 0 <= from_id < entity_sum and 0 <= to_id < entity_sum
            G.add_edge(from_id, to_id)

        # Cluster the graph.
        part_graph_start_t = time()
        (edgecuts, parts) = metis.part_graph(G, num_cluster)
        part_graph_end_t = time()

        print("Generating %d clusters used %d seconds. There are %d nodes in the clusters." %
              (num_cluster, part_graph_end_t - part_graph_start_t, len(parts)))

        # Make asserts.
        assert len(parts) == entity_sum
        for part in parts:
            assert 0 <= part < num_cluster

        return np.array(parts)


class DoUTClusterStrategy(DoClusterStrategyI):
    def _get_sum(self, data_set_num: DatasetNum):
        return data_set_num.user + data_set_num.track

    def _gen_global_id_pairs(self, data, data_set_num: DatasetNum):
        pass


class DoPTClusterStrategy(DoClusterStrategyI):
    def _get_sum(self, data_set_num: DatasetNum):
        return data_set_num.playlist + data_set_num.track

    def _gen_global_id_pairs(self, data, data_set_num: DatasetNum):
        pairs = []

        # Define offsets of entities.
        track_offset = data_set_num.playlist

        for uid, user in data.items():
            for pid, tids in user.items():
                for tid in tids:
                    global_pid, global_tid = pid, tid + track_offset

                    pt_pair = (global_pid, global_tid)

                    pairs.append(pt_pair)
        return pairs


class DoUPTClusterStrategy(DoClusterStrategyI):
    def _get_sum(self, data_set_num: DatasetNum):
        return data_set_num.user + data_set_num.playlist + data_set_num.track

    def _gen_global_id_pairs(self, data, data_set_num: DatasetNum):
        pairs = []

        # Define offsets of entities.
        playlist_offset = data_set_num.playlist
        track_offset = data_set_num.user + data_set_num.track

        for uid, user in data.items():
            for pid, tids in user.items():
                for tid in tids:
                    global_uid, global_pid, global_tid = uid, pid + playlist_offset, tid + track_offset

                    up_pair = (global_uid, global_pid)
                    pt_pair = (global_pid, global_tid)
                    ut_pair = (global_uid, global_tid)

                    pairs.append(up_pair)
                    pairs.append(pt_pair)
                    pairs.append(ut_pair)
        return pairs

