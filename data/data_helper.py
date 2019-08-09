import os
import numpy as np
import scipy.sparse as sp

"""
数据读取来源
首先设置路径，尝试读取.mat文件。
若失败，则尝试读取train/test文件，并以此生成.mat文件。
若再失败，则抛出异常。

先试试直接读取train/test文件，得到矩阵的速度。
"""

class Data():
    def __init__(self, path, pick=True):
        self.path = path

        if pick is True:
            print("{pick} == %r, Using picked playlist data. That is, you're using a sub-dataset" % pick)
            train_filepath = path + "/pick_train.txt"
            test_filepath = path + "/pick_test.txt"
        else:
            print("{pick} == %r, Using complete playlist data. That is, you're using a complete dataset" % pick)
            train_filepath = path + "/train.txt"
            test_filepath = path + "/test.txt"

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_items, self.test_set = {}, {}

        if not os.path.exists(train_filepath) or not os.path.exists(test_filepath):
            raise Exception("train/test file not found.")




    def _read_playlist_file(self):
        pass

    def _read_event_file(self):
        pass

    def next_batch(self):
        pass





if __name__ == '__main__':
    pass