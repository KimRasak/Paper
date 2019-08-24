from Model.BPR_PT import BPR_PT
from Model.utility.data_helper import Data
import os
if __name__ == '__main__':
    path = "./data/aotm"
    num_epoch = 300
    batch_size = 1024

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data = Data(path, pick=False, batch_size=batch_size, laplacian_mode="PT", reductive_ut=True)
    model = BPR_PT(num_epoch, data, embedding_size=64, learning_rate=1e-3, reg_rate=1e-3)
    model.fit()
