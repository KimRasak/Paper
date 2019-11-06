from ModelLayertf2.ClusterModel import ClusterModel
import tensorflow as tf


class ClusterUPTModel(ClusterModel):
    def _train_epoch(self, epoch):
        for cluster_no in range(self.cluster_num):
            self._train_cluster(cluster_no)

        pass

