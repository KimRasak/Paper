from DataLayer.Data import Data
from ModelLayer.Model import Model


class ClusterModel(Model):
    cluster_dropout_flag: bool  # If use dropout tensors for nodes in the cluster.
    node_dropout_ratio: float  # The ratio of dropout. It is only useful if node_dropout_flag

    def __init__(self, epoch_num, data: Data, n_save_batch_loss=300, embedding_size=64,
                 learning_rate=2e-4, reg_loss_ratio=5e-5,
                 cluster_dropout_flag=True, node_dropout_ratio=0.1):
        super().__init__(epoch_num, data, n_save_batch_loss, embedding_size, learning_rate, reg_loss_ratio)

        self.cluster_dropout_flag = cluster_dropout_flag
        self.node_dropout_ratio = node_dropout_ratio