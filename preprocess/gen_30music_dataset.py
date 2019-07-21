# coding=utf-8
import scipy as spy
import scipy.sparse as sp
import numpy as np
from scipy.io import mmwrite
from surprise.model_selection import LeaveOneOut
import random

import preprocess.examine_30music_events as ex

def filter_and_compact(data: dict, min_ui_count = 10):
    """
    Filter noise data out.
    Compact the user ids to make them continuous.
    :param data:
    key: A unique user id
    value: list. Containing all track ids that the user has interact with.
    e.g. { uid1: [tid1, tid2, ...], uid2: [...] }
    :param min_ui_count: Let uis be the user interactions.
    If count(uis) < min_ui_count, the user interactions are
    too few, and the user should be filtered.
    :return: Filtered data.
    Number of users.
    """

    # Filter noise data.
    print("user num before filter: ", len(data))
    for uid in list(data.keys()):
        tids = data[uid]

        if len(tids) < min_ui_count:
            data.pop(uid)
    print("user num after filter: ", len(data))

    # Compact the user ids.
    compact_data = dict()
    for uid, (old_uid, tids) in enumerate(data.items()):
        compact_data[uid] = tids

    return compact_data, len(compact_data)

def split_train_test_dataset(data, num_user, num_track):
    """
    Split the dataset into train and test dataset.
    Sample 1 interaction from each user from the dataset, to construct the test dataset.
    The remaining dataset is the train dataset.
    :param data: The dataset.
    :param num_user: Number of users.
    :param num_track: Number of tracks.
    :return: train_data, test_data
    """
    test_data = dict()
    for uid, tids in data.items():
        random_pick = random.sample(tids, 1)[0]
        tids.remove(random_pick)
        test_data[uid] = random_pick
    return data, test_data

def generate_negative_samples(data, num_track, num_negative_sample = 10):
    tripled_data = dict()
    for uid, tids in data.items():
        for tid in tids:
            negative_samples = []
            for i in range(num_negative_sample):
                # Generate random unvisited track.
                max_tid = num_track - 1
                random_pick = random.randint(0, max_tid)
                while random_pick in tids and random_pick in negative_samples:
                    random_pick = random.randint(0, max_tid)
                negative_samples.append(random_pick)
            tripled_data[(uid, tid)] = negative_samples
    return tripled_data




def convert_to_sparse_matrix(data: dict, num_user, num_track):
    """
    Convert the data to type of scipy sparse matrix.
    :param data:
    key: A unique user id
    value: list. Containing all track ids that the user has interact with.
    :param num_user: Number of user.
    :param num_track: Number of track.
    e.g. { uid1: [tid1, tid2, ...], uid2: [...] }
    :return: scipy.sparse.matrix.
    """
    # Convert to the form of row, col value,
    # which are used to create the scipy sparse matrix.

    row = []
    col = []
    value = []
    for uid, tids in data.items():
        for tid in tids:
            row.append(uid)
            col.append(tid)
            value.append(1)
    print("Convert to row, col and value completed.")

    # Print some data.
    print("value of the first 10 records", value[:10])
    print("row of the first 10 records", row[:10])
    print("col of the first 10 records", col[:10])
    print("num of user", num_user)
    print("num of track", num_track)

    # Convert to the form of scipy sparse matrix.
    sparse_matrix = sp.csr_matrix((value, (row, col)), shape=(num_user, num_track))
    return sparse_matrix


def save_train_dataset(file_path, train_data):
    with open(file_path, 'w') as f:
        f.write("uid tid negative_sample\n")
        for (uid, tid), negative_samples in train_data.items():
            for neg in negative_samples:
                f.write("%d %d %d\n" % (uid, tid, neg))

def save_test_dataset(file_path, test_data):
    with open(file_path, 'w') as f:
        f.write("uid tid\n")
        for uid, tid in test_data.items():
            f.write("%d %d\n" % (uid, tid))


if __name__ == '__main__':
    # Read data.
    data, num_user_before_filter, num_track = ex.read_file_events(ex.file_path_events)

    # Filter users having too few interactions.
    data, num_user = filter_and_compact(data, min_ui_count=20)

    # Split train/test dataset.
    train_data, test_data = split_train_test_dataset(data, num_user, num_track)

    # Generate negative samples for each pair of user-track interaction.
    tripled_train_data = generate_negative_samples(train_data, num_track)

    # Save datasets.
    train_data_path = "30music_train.txt"
    save_train_dataset(train_data_path, tripled_train_data)

    test_data_path = "30music_test.txt"
    save_test_dataset(test_data_path, test_data)

    interaction_data_path = "30music_interaction.mtx"
    sparse_matrix = convert_to_sparse_matrix(data, num_user, num_track)
    mmwrite(interaction_data_path, sparse_matrix)

    # Save train/test dataset.

    # Convert to type of scipy sparse matrix.
    # sparse_matrix = convert_to_sparse_matrix(data_filtered)


    # data_tuple_set = ex.read_file_events_tuple(ex.file_path_events)
    #
    # sparse_matrix = convert_to_sparse_matrix(data_tuple_set)
    #
    # matrix_file_path = "./sparse_matrix.mtx"
    # print("Writing the sparse matrix to file[%s]" % matrix_file_path)
    # mmwrite(matrix_file_path, sparse_matrix)

# 以pair为单位切分训练/测试数据集

# 为训练集的每个pair采样10个负样本，从而为每个二元组生成10个三元组

# 保存测试集文件

# 保存训练集文件