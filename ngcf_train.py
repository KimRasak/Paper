import os

from DataLayer.NormalDatas.RatingData import RatingData
from ModelLayertf2.NormalModels.NGCF import NGCF

os.environ['METIS_DLL'] = '/home/jinzili/.local/lib/libmetis.so'

import tensorflow as tf
from FileLayer import DatasetName
if __name__ == '__main__':
    epoch_num = 1000
    batch_size = 256

    print("All available gpus:")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # logical_devices = tf.config.experimental.list_logical_devices('GPU')
    visible_devices = tf.config.experimental.get_visible_devices()
    for gpu in visible_devices:
        print("******")
        print("Visible device Name:", gpu.name, "  Type:", gpu.device_type)

    data_set_name = DatasetName.RATING
    data = RatingData(data_set_name, use_picked_data=False, is_debug_mode=False, batch_size=256, use_laplacian=True)

    model: NGCF = NGCF(data, epoch_num, embedding_size=8, learning_rate=1e-3, reg_loss_ratio=1e-3, dropout_flag=False)
    model.init()
    model.fit()
