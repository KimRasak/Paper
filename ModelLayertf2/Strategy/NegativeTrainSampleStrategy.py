from abc import ABCMeta, abstractmethod
import numpy as np


class NegativeTrainSampleStrategy(metaclass=ABCMeta):
    def __init__(self, cluster_track_ids):
        self.cluster_track_ids = cluster_track_ids

    @abstractmethod
    def sample_negative_tids(self, cluster_no, track_num):
        pass


class OtherClusterStrategyTrain(NegativeTrainSampleStrategy):
    def sample_negative_tids(self, pos_cluster_no, picked_num):
        # Sample negative cluster
        cluster_num = len(self.cluster_track_ids)
        neg_cluster_no = np.random.randint(0, cluster_num)
        while neg_cluster_no == pos_cluster_no:
            neg_cluster_no = np.random.randint(0, cluster_num)
        picked_neg_cluster = self.cluster_track_ids[neg_cluster_no]
        neg_tid_num = picked_neg_cluster["num"]

        # Pick indices of tids.
        # picked_num = self.cluster_track_ids[pos_cluster_no]["num"]
        picked_indices = []
        while len(picked_indices) < picked_num:
            picked_index = np.random.randint(0, neg_tid_num)
            picked_indices.append(picked_index)
        picked_indices = np.array(list(picked_indices), dtype=int)

        # Generate picked track ids.
        picked_negative_tids = {
            "entity_id": picked_neg_cluster["entity_id"][picked_indices],
            "cluster_id": picked_neg_cluster["cluster_id"][picked_indices]
        }

        # Sample negative tids.
        return neg_cluster_no, picked_negative_tids
