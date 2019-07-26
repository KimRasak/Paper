import scipy.sparse

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread

from MF_BPR.bpr import BPR


def read_pair_data(file_path):
    """
    Read data from path.
    :param file_path: The file path of the data.
    :return: 2-dimension numpy.ndarray.
    The format is:
    [[uid1, tid1],
    [uid2, tid2],
    ...]
    """
    with open(file_path) as f:
        title = f.readline()

        data = []
        line = f.readline()
        while line:
            pair = [int(s) for s in line.split()]  # Line format is "uid tid"
            data.append(pair)
            line = f.readline()
    return np.array(data)


def read_interaction_data(file_path):
    """
    Read interaction data from path
    :param file_path: The file path of the interaction data.
    :return: 2-dimension scipy.sparse.spmatrix
    Containing the user-track interactions.
    """
    data = mmread(file_path).tocsr()
    return data


if __name__ == '__main__':
    print("Reading train data.")
    train_data_path = "../preprocess/30music_train.txt"
    train_data = read_pair_data(train_data_path)

    test_data_path = "../preprocess/30music_test.txt"
    test_data = read_pair_data(test_data_path)

    interaction_data_path = "../preprocess/30music_interaction.mtx"
    interaction_data = read_interaction_data(interaction_data_path)

    print("There are %d interactions." % interaction_data.getnnz())

    bpr = BPR(train_data, test_data, interaction_data, use_model=False)
    bpr.fit()
    # lr:0.0001 embedding-size:64 batch:256
    # Epoch:50 total_loss:4081 hr@10:0.665074 Epoch:60 total_loss:3496 hr@10:0.68
    #
