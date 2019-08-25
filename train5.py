from Model.MDR_G6_att import MDR_G6_att
from Model.utility.data_helper import Data
import os

if __name__ == '__main__':
    path = "./data/aotm"
    num_epoch = 300
    batch_size = 512

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data = Data(path, pick=False, batch_size=batch_size, laplacian_mode="UPT", reductive_ut=True)
    model = MDR_G6_att(num_epoch, data, embedding_size=64, learning_rate=1e-3, reg_rate=1e-3)
    print("Start training...")
    model.fit()
