import tensorflow as tf
from DataLayer.Cluster.ClusterStrategyI import UPTFromUPTClusterStrategy
from DataLayer.ClusterDatas.ClusterUPTData import ClusterUPTData
from DataLayer.NormalDatas.NormalUPTData import NormalUPTData
from FileLayer import DatasetName
import os

from ModelLayertf2.ClusterModels.G6_concat_MDR import G6_concat_MDR

# 在使用UPTFromPTClusterStrategy的情况下，出现很多cluster没有训练集可用
from ModelLayertf2.NormalModels.MDRModel import MDRModel

if __name__ == '__main__':
    epoch_num = 100
    batch_size = 256

    data_set_name = DatasetName.THIRTY_MUSIC
    data = NormalUPTData(data_set_name, use_picked_data=True,
                 epoch_times=4, is_debug_mode=False, batch_size=256)
    model = MDRModel(data, epoch_num, embedding_size=8, learning_rate=2e-4, reg_loss_ratio=5e-5)
    model.init()
    model.fit()
