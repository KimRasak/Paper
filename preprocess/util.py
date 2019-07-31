import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite, mmread
import random

def get_unique_nonzero_indices(X: sp.spmatrix):
    nonzero_row_indices, _ = X.nonzero()
    return np.unique(nonzero_row_indices)

def remove_zero_rows(X: sp.spmatrix):
    """
    X is a scipy sparse matrix. We want to remove all zero rows from it.
    :param X: scipy sparse matrix.
    :return: Scipy sparse matrix without zero rows.
    """
    row_indices = get_unique_nonzero_indices(X)
    return X[row_indices]


def random_pick_rows(X: sp.csr_matrix, percent):
    """
    Pick rows to form a new scipy sparse matrix.
    :param X: Scipy sparse matrix.
    :param percent: The percent of picked rows.
    :return: Scipy sparse matrix formed by picked rows.
    """

    # 1.Get all row indices.
    row_indices = get_unique_nonzero_indices(X)

    # 2. Randomly pick indices.
    num_rows = len(row_indices)
    num_picked_rows = int(num_rows * percent)
    picked_row_indices = np.random.choice(row_indices, num_picked_rows)

    # 3. Return the new matrix
    return X[picked_row_indices]


def random_pick_subset(X, row_percent, col_percent):
    """
    Randomly Pick some percent of rows and columns from matrix {X}.
    :param X: Scipy sparse matrix.
    :param row_percent:
    :param col_percent:
    :return: The sub dataset.
    """
    X = random_pick_rows(X, row_percent)

    T_X: sp.csr_matrix = X.transpose()
    T_X = random_pick_rows(T_X, col_percent)

    X = T_X.transpose()
    X = remove_zero_rows(X)
    return X


def filter_rows(X: sp.csr_matrix, min_row_values=10):
    row_indices = get_unique_nonzero_indices(X)

    # Set elements of filtered rows to 0.
    for row_index in row_indices:
        col_indices = X.getrow(row_index).indices
        if len(col_indices) < min_row_values:
            for col_index in col_indices:
                X[row_index, col_index] = 0

    # Remove zero rows.
    X = remove_zero_rows(X)
    return X


def filter_dataset(X: sp.csr_matrix, min_ui_count=10, min_iu_count=10):

    X = filter_rows(X, min_ui_count)
    T_X = X.transpose()
    T_X = filter_rows(T_X, min_iu_count)
    X = T_X.transpose()
    return X

if __name__ == '__main__':
    row = np.array([0, 0, 0, 1, 2, 2, 2, 5])
    col = np.array([0, 1, 3, 2, 0, 1, 3, 5])
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    mtx: sp.csr_matrix = sp.csr_matrix((data, (row, col)), shape=(6, 8))

    print(random_pick_rows(mtx, 0.5))
    print("---")
    print(filter_dataset(mtx, min_ui_count=2, min_iu_count=1))
