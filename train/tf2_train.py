from DataLayer.Cluster.ClusterStrategyI import UPTClusterStrategyI, UPTFromPTClusterStrategy
from DataLayer.ClusterData import ClusterData
from DataLayer.ClusterUPTData import ClusterUPTData
from FileLayer import DatasetName
import os

from ModelLayertf2.G6_concat_MDR import G6_concat_MDR

if __name__ == '__main__':
    epoch_num = 300
    batch_size = 256
    cluster_num = 100

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    data_set_name = DatasetName.THIRTY_MUSIC
    pick_cluster_strategy = UPTClusterStrategyI.FirstChoicePickClusterStrategy()
    cluster_strategy = UPTFromPTClusterStrategy(data_set_name, pick_cluster_strategy)

    data = ClusterUPTData(data_set_name, cluster_strategy, use_picked_data=False,
                 batch_size=256, epoch_times=4, is_debug_mode=False,
                 cluster_num=100, ut_alpha=1)
    model = G6_concat_MDR(data, epoch_num,
                          save_loss_batch_num=300, embedding_size=64, learning_rate=2e-4, reg_loss_ratio=1e-4,
                          cluster_dropout_flag=True, node_dropout_ratio=0.1,
                          gnn_layer_num=3)
    model.fit()
