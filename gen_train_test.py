import scipy as sp
import numpy as np

from FileLayer import data_file_layer, DatasetName, WHOLE_PLAYLIST_PATH
from FileLayer.data_file_layer import get_train_file_path, get_test_file_path

"""
读取交互数据文件、歌单数据文件，并切分训练/测试集。
1. 读取文件
2. 压缩id
3. 切分数据集并保存

"""


def read_events(file_path):
    data = dict()
    with open(file_path) as f:
        head_title = f.readline()

        line = f.readline()
        while line:
            ids = [int(s) for s in line.split() if s.isdigit()]
            uid = ids[0]
            tid = ids[1]

            if uid not in data:
                data[uid] = {tid}
            else:
                data[uid].add(tid)

            line = f.readline()
    return data


def split_train_test(playlist_data: dict, train_file_path, test_file_path, proportion=0.8):
    train_data = dict()
    test_data = dict()

    for uid, user in playlist_data.items():
        if uid not in train_data:
            train_data[uid] = {}
        if uid not in test_data:
            test_data[uid] = {}

        for pid, tids in user.items():
            # num_test = max(int(len(tids) * (1 - proportion)), 1)
            num_test = 1
            num_train = len(tids) - num_test
            train_tids = np.random.choice(tids, num_train, replace=False)
            test_tids = [tid for tid in tids if tid not in train_tids]

            train_data[uid][pid] = train_tids
            test_data[uid][pid] = test_tids

    data_file_layer.write_playlist_data(train_data, train_file_path)
    data_file_layer.write_playlist_data(test_data, test_file_path)


def main():
    # Read whole playlist data set.
    data_set_name = DatasetName.AOTM
    whole_playlist_path = WHOLE_PLAYLIST_PATH[data_set_name]
    whole_playlist_data = data_file_layer.read_playlist_data(whole_playlist_path)

    # Split whole data set into train/test data set.
    train_file_path = get_train_file_path(False, data_set_name)
    test_file_path = get_test_file_path(False, data_set_name)
    split_train_test(whole_playlist_data, train_file_path, test_file_path)


if __name__ == '__main__':
    main()

