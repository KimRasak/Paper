from Model.BPR_PT import BPR_PT
from Model.MDR import MDR
from Model.MDR_G6_att import MDR_G6
from Model.MENGCF_PT_bi_PT import MENGCF_PT_bi_PT
from Model.ME_NGCF import ME_NGCF
from Model.NGCF_PT import NGCF_PT
from Model.NGCF_PT_PT import NGCF_PT_PT
from Model.utility.data_helper import Data
import tensorflow as tf

if __name__ == '__main__':
    path = "./data/30music"
    num_epoch = 300
    batch_size = 1024

    # data = Data(path, batch_size=batch_size, laplacian_mode="UPT", reductive_ut=True)
    # model = MDR_G6(num_epoch, data)
    # model.fit()
    data = Data(path, batch_size=batch_size, laplacian_mode="PT", reductive_ut=True)
    model = NGCF_PT_PT(num_epoch, data)
    model.fit()
