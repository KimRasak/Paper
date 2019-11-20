import os
os.environ['METIS_DLL'] = '/home/jinzili/.local/lib/libmetis.so'

import tensorflow as tf
from DataLayer.NormalDatas.NormalUPTData import NormalUPTData
from FileLayer import DatasetName

from ModelLayertf2.ClusterModels.G6_concat_MDR import G6_concat_MDR

# 在使用UPTFromPTClusterStrategy的情况下，出现很多cluster没有训练集可用
from ModelLayertf2.NormalModels.MDRModel import MDRModel

if __name__ == '__main__':
    epoch_num = 100
    batch_size = 256

    print("All available gpus:")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print("=====")
        print("Name:", gpu.name, "  Type:", gpu.device_type)
        tf.config.experimental.set_memory_growth(gpu, True)

    data_set_name = DatasetName.THIRTY_MUSIC
    data = NormalUPTData(data_set_name, use_picked_data=False,
                 epoch_times=4, is_debug_mode=False, batch_size=256)
    model = MDRModel(data, epoch_num, embedding_size=64, learning_rate=2e-4, reg_loss_ratio=1e-4)
    model.init()
    model.fit()
