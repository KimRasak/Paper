import scipy.sparse

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread

from MF_BPR.bpr import BPR

def read_train_data(file_path):
    """
    Read train data from path.
    :param file_path: The file path of the train data.
    :return: 2-dimension numpy.ndarray.
    The format is:
    [[uid1, tid1, neg1],
    [uid2, tid2, neg2],
    ...]
    """
    with open(file_path) as f:
        title = f.readline()

        train_data = []
        line = f.readline()
        while line:
            traid = [int(s) for s in line.split()]  # Line format is "uid tid negative_sample"
            train_data.append(traid)
            line = f.readline()
    return np.array(train_data)


def read_test_data(file_path):
    """
    Read test data from path.
    :param file_path: The file path of the test data.
    :return: 2-dimension numpy.ndarray.
    The format is:
    [[uid1, tid1],
    [uid2, tid2],
    ...]
    """

    with open(file_path) as f:
        title = f.readline()

        test_data = []
        line = f.readline()
        while line:
            traid = [int(s) for s in line.split()]  # line format is "uid tid negative_sample"
            test_data.append(traid)
            line = f.readline()
    return np.array(test_data)


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
    train_data = read_train_data(train_data_path)

    test_data_path = "../preprocess/30music_test.txt"
    test_data = read_test_data(test_data_path)

    interaction_data_path = "../preprocess/30music_interaction.mtx"
    interaction_data = read_interaction_data(interaction_data_path)

    print("There are %d interactions." % interaction_data.getnnz())

    bpr = BPR(train_data, test_data, interaction_data)
    bpr.fit()
