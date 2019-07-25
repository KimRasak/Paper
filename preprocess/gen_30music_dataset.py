# coding=utf-8
import scipy as spy
import scipy.sparse as sp
import numpy as np
from scipy.io import mmwrite
from surprise.model_selection import LeaveOneOut
import random

import preprocess.examine_30music_events as ex

def column_first_to_row_first(data_column_first: dict):
    """
    Change the data from column first to row first.
    e.g. The column first data is like:
    { tid1: [uid1, uid2, ...], tid2: [...] }
    The row first data is like:
    { uid1: [tid1, tid2, ...], uid2: [...] }
    :param data_column_first: The column first data.
    :return: The row first data.
    """
    data_row_first = dict()
    for tid, uids in data_column_first.items():
        for uid in uids:
            if uid not in data_row_first:
                data_row_first[uid] = []

            tids = data_row_first[uid]
            if tid not in tids:
                data_row_first[uid].append(tid)

    return data_row_first


def filter_and_compact(data_column_first: dict, min_iu_count = 10, min_ui_count = 10):
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
    # Remove track ids with too few interactions.
    print("track num before filter: ", len(data_column_first))
    # The keys may not be accessed sequentially.
    # That is, the keys may be accessed by 1, 6, 2, 10, ...
    # instead of 0, 1, 2, 3, ...
    # But that doesn't matter, you can say the track ids are re-indexed.
    for tid in list(data_column_first.keys()):
        uids = data_column_first[tid]

        if len(uids) < min_iu_count:
            data_column_first.pop(tid)
    print("track num after filter: ", len(data_column_first))

    # Compact the track ids
    compact_column_data = dict()
    for tid, (old_tid, uids) in enumerate(data_column_first.items()):
        compact_column_data[tid] = uids

    # Change to row first expression.
    num_track = len(compact_column_data)
    compact_column_data = column_first_to_row_first(compact_column_data)

    # Remove user ids with too few interactions.
    # The keys may not be accessed sequentially.
    # But that doesn't matter.
    print("user num before filter: ", len(compact_column_data))
    for uid in list(compact_column_data.keys()):
        tids = compact_column_data[uid]

        if len(tids) < min_ui_count:
            compact_column_data.pop(uid)
    print("user num after filter: ", len(compact_column_data))

    # Compact the user ids.
    compact_data = dict()
    for uid, (old_uid, tids) in enumerate(compact_column_data.items()):
        compact_data[uid] = tids

    return compact_data, len(compact_data), num_track


def split_train_test_dataset(data, num_user, num_track):
    """
    Split the dataset into train and test dataset.
    For uach user, sample 1 interaction from its interactions to construct the test dataset.
    The remaining dataset is the train dataset.
    :param data: The dataset.
    :param num_user: Number of users.
    :param num_track: Number of tracks.
    :return: train_data, test_data
    """
    print("Start spliting train/test dataset.")
    test_data = dict()
    for uid, tids in data.items():
        random_pick = random.sample(tids, 1)[0]
        tids.remove(random_pick)
        test_data[uid] = random_pick
    return data, test_data


def generate_negative_samples(data, num_track, num_negative_sample = 10):
    """
    Not used for now.
    :param data:
    :param num_track:
    :param num_negative_sample:
    :return:
    """
    print("Start generating negative samples.")
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
    data_column_first, num_user_before_filter, num_track_before_filter = ex.read_file_events_column_first(ex.file_path_events)

    # Filter users having too few interactions.
    compact_data, num_user, num_track = filter_and_compact(data_column_first, min_ui_count=300, min_iu_count=10)
    # min_ui_count~user_num 100~17000 350~9000

    # Split train/test dataset.
    train_data, test_data = split_train_test_dataset(compact_data, num_user, num_track)

    # Generate negative samples for each pair of user-track interaction.
    tripled_train_data = generate_negative_samples(train_data, num_track)

    # Save datasets.
    train_data_path = "30music_train.txt"
    save_train_dataset(train_data_path, tripled_train_data)

    test_data_path = "30music_test.txt"
    save_test_dataset(test_data_path, test_data)

    interaction_data_path = "30music_interaction.mtx"
    sparse_matrix = convert_to_sparse_matrix(compact_data, num_user, num_track)
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
