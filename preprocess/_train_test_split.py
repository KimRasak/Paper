# coding=utf-8
import scipy as spy
import scipy.sparse as sp
import numpy as np
from scipy.io import mmwrite, mmread
from surprise.model_selection import LeaveOneOut
from preprocess._gen_30music_dataset import filter_and_compact, convert_to_sparse_matrix, transfer_row_column
import random
from preprocess._util import random_pick_subset, filter_dataset, get_unique_nonzero_indices


percent_subset_data_path = "30music_interaction_subset_min10x10_percent.mtx"

def train_test_split(X: sp.csr_matrix):
    row_indices = get_unique_nonzero_indices(X)

    train_data = []
    test_data = []

    for row_index in row_indices:
        col_indices = X.getrow(row_index).indices

        test_index = np.random.choice(col_indices, 1)[0]
        train_data.extend([(row_index, col_index) for col_index in col_indices if col_index != test_index])
        test_data.append((row_index, test_index))
    return train_data, test_data


def save_pair_data(data: list, file_path):
    with open(file_path, 'w') as f:
        f.write("uid, tid\n")
        for uid, tid in data:
            f.write("%d %d\n" % (uid, tid))
    return

if __name__ == '__main__':
    print("Reading .mtx file...")
    mtx_data: sp.csr_matrix = mmread(percent_subset_data_path).tocsr()
    train_data, test_data = train_test_split(mtx_data)

    train_data_path = "30music_train.txt"
    save_pair_data(train_data, train_data_path)
    test_data_path = "30music_test.txt"
    save_pair_data(test_data, test_data_path)