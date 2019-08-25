from Model.MDR_G6_att import MDR_G6_att
from Model.utility.data_helper import Data

if __name__ == '__main__':
    path = "./data/30music"
    num_epoch = 300
    batch_size = 512

    data = Data(path, batch_size=batch_size, laplacian_mode="UPT", reductive_ut=True)
    model = MDR_G6_att(num_epoch, data)
