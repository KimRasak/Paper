import tensorflow as tf
from DataLayer.Cluster.ClusterStrategyI import UPTFromUPTClusterStrategy
from DataLayer.ClusterDatas.ClusterUPTData import ClusterUPTData
from FileLayer import DatasetName
import os

from ModelLayertf2.ClusterModels.G6_concat_MDR import G6_concat_MDR

# 在使用UPTFromPTClusterStrategy的情况下，出现很多cluster没有训练集可用
if __name__ == '__main__':
    epoch_num = 300
    batch_size = 256
    cluster_num = 100
    print("-----")
    print("is built with cuda: ", tf.test.is_built_with_cuda())
    # print("is gpu available: ", tf.test.is_gpu_available())
    # tf.debugging.set_log_device_placement(True)
    print("-----")

    print("-----")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # gpus = tf.config.experimental.get_visible_devices('GPU')
    print("All available gpus:")
    for gpu in gpus:
        print("=====")
        print("Name:", gpu.name, "  Type:", gpu.device_type)
        tf.config.experimental.set_memory_growth(gpu, True)

    # print("Only use gpus [{}, {})".format(gpu_start, gpu_end))
    # tf.config.experimental.set_visible_devices(gpus[2:3], device_type='GPU')

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # logical_devices = tf.config.experimental.list_logical_devices('GPU')
    visible_devices = tf.config.experimental.get_visible_devices()
    for gpu in visible_devices:
        print("******")
        print("Visible device Name:", gpu.name, "  Type:", gpu.device_type)

    # Logical device was not created for first GPU
    # assert len(logical_devices) == len(physical_devices) - 2

    data_set_name = DatasetName.THIRTY_MUSIC
    # pick_cluster_strategy = UPTClusterStrategyI.FirstChoicePickClusterStrategy()
    # cluster_strategy = UPTFromPTClusterStrategy(data_set_name, pick_cluster_strategy)
    cluster_strategy = UPTFromUPTClusterStrategy(data_set_name)

    data = ClusterUPTData(data_set_name, cluster_strategy, use_picked_data=False,
                 epoch_times=4, is_debug_mode=False,
                 cluster_num=100, ut_alpha=1)
    model = G6_concat_MDR(data, epoch_num, embedding_size=8, learning_rate=2e-4, reg_loss_ratio=1e-4,
                          cluster_dropout_flag=True, cluster_dropout_ratio=0.1,
                          gnn_layer_num=3)
    # with tf.device(gpus[2].name):
      # with tf.device('/device:gpu:2'):
      # model.init()
    model.init()
    model.fit()
