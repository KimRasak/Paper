from time import time

from ModelLayertf2.BaseModel import Loss
from ModelLayertf2.ClusterModel import ClusterModel
import tensorflow as tf


class ClusterUPTModel(ClusterModel):
    def _train_epoch(self, epoch):
        epoch_loss = Loss()
        epoch_start_time = time()
        for cluster_no in range(self.cluster_num):
            cluster_loss: Loss = self._train_cluster(cluster_no)
            epoch_loss.add_loss(cluster_loss)
        epoch_end_time = time()
        return epoch_loss, epoch_end_time - epoch_start_time

