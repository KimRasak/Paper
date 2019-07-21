# coding=utf-8
import scipy as spy
import scipy.sparse as sp
import numpy as np
import preprocess.examine_30music_events as ex
from scipy.io import mmwrite

if __name__ == '__main__':
    data = ex.read_file_events_tuple(ex.file_path_events)
    row = []
    col = []
    value = []
    for uid, tid in data:
        row.append(uid - 1)  # user的第一个编号为1，track的第一个编号为0，因此uid要统一起始为0。
        col.append(tid)
        value.append(1)
    print("Read events data completed.")

    m = max(row) + 1
    n = max(col) + 1

    print("value", value[:10])
    print("row", row[:10])
    print("col", col[:10])
    print("m", m)
    print("n", n)

    sparse_matrix = sp.csr_matrix((value, (row, col)), shape=(m, n))
    mtx_file_path = "./sparse_matrix.mtx"
    print("Writing the sparse matrix to file[%s]" % mtx_file_path)
    mmwrite(mtx_file_path, sparse_matrix)