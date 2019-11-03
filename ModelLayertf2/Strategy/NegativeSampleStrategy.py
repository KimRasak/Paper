from abc import ABCMeta


class NegativeSampleStrategy(metaclass=ABCMeta):
    def __init__(self, cluster_track_ids):
        self.cluster_tids = cluster_track_ids

    def sample_negative_tids(self):
        pass