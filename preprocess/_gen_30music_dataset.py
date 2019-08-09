# coding=utf-8
import random

import numpy as np
import scipy.sparse as sp

import preprocess._examine_30music_events as ex


def transfer_row_column(data: dict):
    """
    The input data represents a sparse matrix,
    and this function transfers the matrix.
    e.g. If the data is {0: [0, 1, 2], 1: [0], 2: [1, 3]}
    The output will be {0: [0, 1], 1: [0, 2], 2: [0], 3: [2]}
    :param data: Dict, representing a sparse matrix.
    :return: If the input is column first, it's converted to row first.
    If the input is row first, it's converted to column first.
    """
    data_converted = dict()
    for row_index, col_indexes in data.items():
        for col_index in col_indexes:
            if col_index not in data_converted:
                data_converted[col_index] = []

            row_indexes = data_converted[col_index]
            if row_index not in row_indexes:
                data_converted[col_index].append(row_index)

    return data_converted


def filter_and_compact(data_uid: dict, min_iu_count=10, min_ui_count=10):
    """
    1. Filter the data(user or track) with too few interactions out.
    2. Compact the user ids and track ids to make them continuous starting from 0.
    We
    :param data_uid:
    key: A unique user id
    value: List. Containing ids of tracks that the user has interacted with
    e.g. { uid1: [tid1, tid2, ...], uid2: [...] }
    The user/track ids may not be continuous, that is,
    the first id may be 1, and the second id may be 3, but there's no id 2.
    :param original_num_user: Original num of users.
    :param original_num_track: Original num of tracks.
    :param min_ui_count: Let uis be the user's interactions.
    If count(uis) < min_ui_count, the user's interactions are
    too few, and the user should be filtered.
    :param min_iu_count: Let ius be the track's interactions.
    If count(ius) < min_iu_count, the track's interactions are
    too few, and the track should be filtered.
    :return: 1. Dict. Filtered data with continuous ids starting from 0.
    2. Number of users.
    3. Number of tracks.
    """
    filter_count = 1
    old_num_user = len(data_uid)
    old_num_track = max([len(row) for row in data_uid.values()])

    data_filter, new_num_user, new_num_track, should_continue_filter = \
        do_filter_and_compact(data_uid, old_num_user, old_num_track, min_ui_count, min_iu_count)

    # Loop until every user/track meets the need(having at least min_ui_count/min_iu_count interactions.)
    while should_continue_filter:
        filter_count += 1
        old_num_user = new_num_user
        old_num_track = new_num_track

        data_filter, new_num_user, new_num_track, should_continue_filter = \
            do_filter_and_compact(data_filter, old_num_user, old_num_track, min_ui_count, min_iu_count)

    print("Loop Filter %d times." % filter_count)
    return data_filter, new_num_user, new_num_track


def filter_and_compact_rows(data: dict, min_row_values):
    """
    1. Filter the rows with too few values.
    2. Compact the row indexes to make them continuous.
    In this scene, each row represents a user/track,
    and each value in the value list represent a track/user that has interaction with it.
    :param data: Dict. Representing the sparse matrix.
    :param min_row_values: Minimum number of values that each row should have.
    Rows with less than {min_row_values} values will be filtered.
    :return: Filtered data with continuous row indexes.
    """
    # Remove track ids with too few interactions.
    for row_index in list(data.keys()):
        values = data[row_index]

        if len(values) < min_row_values:
            data.pop(row_index)

    # Compact the row indexes to make them continuous starting from 0.
    # The order of row indexes may be shuffled, due to random access to original row indexes,
    # but that doesn't matter, because the interaction relationships don't change.
    compact_row_data = dict()
    for row_index, (old_row_index, uids) in enumerate(data.items()):
        compact_row_data[row_index] = uids

    return compact_row_data


def do_filter_and_compact(data_uid: dict, old_num_user, old_num_track, min_ui_count = 10, min_iu_count = 10):
    """
    1. Filter the data(user or track) with too few interactions out.
    2. Compact the user ids and track ids to make them continuous starting from 0.
    :param data_uid:
    key: A unique user id
    value: List. Containing ids of tracks that the user has interacted with.
    e.g. {uid1: [tid1, tid2, ...], uid2: [...] }
    :param min_ui_count: Let uis be the user's interactions.
    If count(uis) < min_ui_count, the user's interactions are
    too few, and the user should be filtered.
    :param min_iu_count: Let ius be the track's interactions.
    If count(ius) < min_iu_count, the track's interactions are
    too few, and the track should be filtered.
    :return: 1. Filtered data with continuous ids starting from 0.
    2. Number of users.
    3. Number of tracks.
    4. If this dataset should be filtered again.
    """
    # Filter out the users with too few interactions.
    data_filter_uid = filter_and_compact_rows(data_uid, min_ui_count)
    new_num_user = len(data_filter_uid)

    # Transform the sparse matrix. Now the row indexes is track ids.
    data_tid = transfer_row_column(data_filter_uid)

    # Filter out the tracks with too few interactions.
    data_filter_tid = filter_and_compact_rows(data_tid, min_iu_count)
    new_num_track = len(data_filter_tid)

    # The data has filtered user ids and track ids.
    data_filter = transfer_row_column(data_filter_tid)

    # If no user/track is filtered out, there is no need to filter the data again.
    should_continue_filter = old_num_user != new_num_user or old_num_track != new_num_track
    return data_filter, new_num_user, new_num_track, should_continue_filter


def split_train_test_dataset(data):
    """
    Split the dataset into train and test dataset.
    For uach user, sample 1 interaction from its interactions to construct the test dataset.
    The remaining dataset is the train dataset.
    :param data: The dataset.
    :return: train_data, test_data
    """
    print("Start spliting train/test dataset.")
    test_data = dict()
    for uid, tids in data.items():
        random_pick = random.sample(tids, 1)[0]
        tids.remove(random_pick)  # Remove the test item from train data.
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
    # print("value of the first 10 records", value[:10])
    # print("row of the first 10 records", row[:10])
    # print("col of the first 10 records", col[:10])
    print("num of user", num_user)
    print("num of track", num_track)

    # Convert to the form of scipy sparse matrix.
    sparse_matrix = sp.csr_matrix((value, (row, col)), shape=(num_user, num_track))
    return sparse_matrix


def save_train_dataset(file_path, train_data):
    with open(file_path, 'w') as f:
        f.write("uid tid\n")
        for uid, tids in train_data.items():
            for tid in tids:
                f.write("%d %d\n" % (uid, tid))

def save_test_dataset(file_path, test_data):
    with open(file_path, 'w') as f:
        f.write("uid tid\n")
        for uid, tid in test_data.items():
            f.write("%d %d\n" % (uid, tid))

def check_dataset_distribution(data, save_file_path="dataset_30music_distribution"):
    ui_distribution = {}
    below_10 = 0
    on_10_below_100 = 0
    on_100_below_500 = 0
    on_500_below_1000 = 0
    on_1000 = 0

    for uid, tids in data.items():
        num_ui = len(tids)
        index = num_ui // 10

        if index not in ui_distribution:
            ui_distribution[index] = 0

        if num_ui <= 10:
            below_10 += 1
        elif num_ui > 10 and num_ui <= 100:
            on_10_below_100 += 1
        elif num_ui > 100 and num_ui <= 500:
            on_100_below_500 += 1
        elif num_ui > 500 and num_ui <= 1000:
            on_500_below_1000 += 1
        elif num_ui > 1000:
            on_1000 += 1

        ui_distribution[index] += 1

    print("below on: \n%d %d %d %d %d" % (below_10, on_10_below_100, on_100_below_500,
                                       on_500_below_1000, on_1000))
    T_data = transfer_row_column(data)

    iu_distribution = {}
    for tid, uids in T_data.items():
        num_iu = len(uids)
        index = num_iu // 10

        if index not in iu_distribution:
            iu_distribution[index] = 0

        iu_distribution[index] += 1

    ui_indexes = np.sort(list(ui_distribution.keys()))
    iu_indexes = np.sort(list(iu_distribution.keys()))

    with open(save_file_path, 'w') as f:
        f.write("ui_distribution\n")

        for step in ui_indexes:
            num = ui_distribution[step]
            f.write("step:%d num:%d\n" % (step, num))
        f.write("iu_distribution\n")

        for step in iu_indexes:
            num = iu_distribution[step]
            f.write("%d %d\n" % (step, num))

    return

if __name__ == '__main__':
    # Read data.
    data, max_uid, max_tid = ex.read_file_events(ex.file_path_events)
    # Filter users/tracks with too few interactions.
    # while True:
    #     min_ui_count = input("input min_ui_num")
    #     min_iu_count = input("input min_iu_num")
    #     filtered_data, num_user, num_track = filter_and_compact(data, min_ui_count=200, min_iu_count=10)

    # (100, 10)~(28342, 246118) (200, 10) ~ (16111, 202001)
    check_dataset_distribution(data)

    # filtered_data, num_user, num_track = filter_and_compact(data, min_ui_count=10, min_iu_count=10)
    #
    #
    # # Split train/test dataset.
    # train_data, test_data = split_train_test_dataset(filtered_data)
    #
    # # Save datasets.
    # train_data_path = "30music_train.txt"
    # save_train_dataset(train_data_path, train_data)
    #
    # test_data_path = "30music_test.txt"
    # save_test_dataset(test_data_path, test_data)
    #
    # interaction_data_path = "30music_interaction.mtx"
    # sparse_matrix = convert_to_sparse_matrix(filtered_data, num_user, num_track)
    # mmwrite(interaction_data_path, sparse_matrix)
