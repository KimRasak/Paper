from Model.MDR import MDR
from Model.MDR_G6_att import MDR_G6_att
from Model.NGCF_PT_PT import NGCF_PT_PT
from Model.NGCF_PT_bi_PT_att import NGCF_PT_bi_PT_att
from Model.utility.data_helper import Data

if __name__ == '__main__':
    path = "./data/aotm"
    num_epoch = 300
    batch_size = 512

    data = Data(path, pick=False, batch_size=batch_size, laplacian_mode="PT", reductive_ut=True)
    model = NGCF_PT_PT(num_epoch, data)
    model.fit()
