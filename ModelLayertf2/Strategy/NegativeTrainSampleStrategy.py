from abc import ABCMeta, abstractmethod
import numpy as np


class NegativeTrainSampleStrategy(metaclass=ABCMeta):
    def __init__(self, pos_train_tuples, cluster_track_ids):
        self.cluster_pos_train_tuples: list = pos_train_tuples
        self.cluster_track_ids = cluster_track_ids

    @abstractmethod
    def sample_negative_tids(self, pos_cluster_no):
        pass


class OtherClusterStrategy(NegativeTrainSampleStrategy):
    def sample_negative_tids(self, pos_cluster_no):
        # Sample negative cluster
        cluster_num = len(self.cluster_track_ids)
        neg_cluster_no = np.random.randint(0, cluster_num)
        while neg_cluster_no == pos_cluster_no:
            neg_cluster_no = np.random.randint(0, cluster_num)
        picked_neg_cluster = self.cluster_track_ids[neg_cluster_no]
        neg_tid_num = picked_neg_cluster["num"]

        # Pick indices of tids.
        pos_train_tuples: dict = self.cluster_pos_train_tuples[pos_cluster_no]
        picked_num = pos_train_tuples["length"]
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


class SameClusterStrategy(NegativeTrainSampleStrategy):
    """
    Pick negative training samples from the same cluster of the positive training tuples.
    """
    def __init__(self, pos_train_tuples, cluster_track_ids, pt):
        super().__init__(pos_train_tuples, cluster_track_ids)
        self.pt: dict = pt

    def pick_negative_tid(self, pos_cluster_no, playlist_entity_id):
        pos_track_ids = self.cluster_track_ids[pos_cluster_no]
        playlist_tids = self.pt[playlist_entity_id]

        # Pick a negative tid.
        tid_num, entity_ids, cluster_ids = pos_track_ids["num"], pos_track_ids["entity_id"], pos_track_ids["cluster_id"]

        picked_index = np.random.randint(0, tid_num)
        picked_tid = entity_ids[picked_index]
        while picked_tid in playlist_tids:
            # We only pick a tid in the cluster, so if the tid isn't in the playlist, it must be a negative sample.
            # Else if it's in the playlist, it must be a positive sample.
            # Those who are in the playlist but not in the cluster, won't be picked,
            # because we only pick in the same cluster.
            # 我们会在同簇之内抽取tid, 如果它不在歌单内，一定是负样本
            # 如果它在歌单内，又因为一定在同簇，就一定是正样本
            # 那些同歌单内不同簇的歌曲，在此不会被抽到，因为我们只在同簇之内抽取tid.
            picked_index = np.random.randint(0, tid_num)
            picked_tid = entity_ids[picked_index]

        picked_entity_tid = picked_tid
        picked_cluster_tid = cluster_ids[picked_index]
        return picked_entity_tid, picked_cluster_tid

    def sample_negative_tids(self, pos_cluster_no):
        # Sample negative samples in the same cluster.
        neg_tids = {
            "entity_id": np.array([], dtype=int),
            "cluster_id": np.array([], dtype=int)
        }

        pos_train_tuples: dict = self.cluster_pos_train_tuples[pos_cluster_no]
        for i in range(pos_train_tuples["length"]):
            picked_entity_tid, picked_cluster_tid = \
                self.pick_negative_tid(pos_cluster_no, pos_train_tuples["playlist_entity_id"][i])
            neg_tids["entity_id"] = np.append(neg_tids["entity_id"], picked_entity_tid)
            neg_tids["cluster_id"] = np.append(neg_tids["cluster_id"], picked_cluster_tid)

        return pos_cluster_no, neg_tids