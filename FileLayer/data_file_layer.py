import numpy as np
from FileLayer import TRAIN_FILE_PATH, PICK_TRAIN_FILE_PATH, TEST_FILE_PATH, PICK_TEST_FILE_PATH, \
    COUNT_FILE_NAME, PICK_COUNT_FILE_PATH, WHOLE_COUNT_FILE_PATH, DatasetName
from Common import DatasetNum


"""
Provide read/write API of playlist files and count files.
Playlist files include "whole playlist file"/"train file"/"test file"
"""


def get_train_file_path(use_picked_data, data_set_name):
    if not use_picked_data:
        return TRAIN_FILE_PATH[data_set_name]
    else:
        return PICK_TRAIN_FILE_PATH[data_set_name]


def get_test_file_path(use_picked_data, data_set_name):
    if not use_picked_data:
        return TEST_FILE_PATH[data_set_name]
    else:
        return PICK_TEST_FILE_PATH[data_set_name]


def get_count_file_path(use_picked_data, data_set_name):
    if not use_picked_data:
        return WHOLE_COUNT_FILE_PATH[data_set_name]
    else:
        return PICK_COUNT_FILE_PATH[data_set_name]


# Define read functions for dataset "rating"
def read_rating_data(path):
    data = dict()
    with open(path) as f:
        head_line = f.readline()
        line = f.readline()
        while line:
            items = line.split(" ")
            user_id = int(items[0])
            # Prepare user dict.
            if user_id not in data:
                data[user_id] = dict()

            assert items[-1] == "\n"
            for pair in items[1:-1]:
                # Extract id and score from str.
                assert pair[0] == "(" and pair[-1] == ")"
                splits = pair[1:-1].split(",")
                item_id, score = int(splits[0]), int(splits[1])

                # Fill in data.
                data[user_id][item_id] = score

            line = f.readline()
    return data


def read_rating_count(path):
    with open(path) as f:
        f.readline()
        user_num = int(f.readline())
        f.readline()
        item_num = int(f.readline())

    return DatasetNum(1, user_num, item_num, 1)


# read/write functions for general data.
def write_playlist_data(playlist_data: dict, playlist_data_path):
    """
    Write playlist data to file.
    :param playlist_data: The playlist data.
    :param playlist_data_path: The file storing the playlist data.
    :return:
    """
    with open(playlist_data_path, 'w') as f:
        f.write("user_id playlist_id track_ids\n")
        for uid, user in playlist_data.items():
            for pid, tids in user.items():
                f.write("%d %d " % (uid, pid))
                for tid in tids:
                    f.write("%d " % tid)
                f.write("\n")


def read_playlist_data(playlist_data_path):
    """
    Read playlist data from file. All ids in the file must be continuous starting from 0.
    """
    # Special read function for dataset rating.
    if playlist_data_path == TRAIN_FILE_PATH[DatasetName.RATING] or playlist_data_path == TEST_FILE_PATH[DatasetName.RATING]:
        return read_rating_data(playlist_data_path)

    data = dict()
    with open(playlist_data_path) as f:
        head_title = f.readline()  # title: "user_id playlist_id track_ids"

        line = f.readline()
        while line:
            ids = [int(s) for s in line.split() if s.isdigit()]
            uid, pid, tids = ids[0], ids[1], ids[2:]

            if uid not in data:
                data[uid] = {pid: np.array(tids)}
            elif pid not in data[uid]:
                data[uid][pid] = np.array(tids)

            line = f.readline()
    return data


def write_count_file(dataset_num: DatasetNum, count_file_path):
    # Write the number of user/playlist/track into the file.
    num_user = dataset_num.user
    num_playlist = dataset_num.playlist
    num_track = dataset_num.track
    num_interaction = dataset_num.interaction

    with open(count_file_path, 'w') as f:
        f.write("number of user\n")
        f.write("{}\n".format(num_user))
        f.write("number of playlist\n")
        f.write("{}\n".format(num_playlist))
        f.write("number of track\n")
        f.write("{}\n".format(num_track))
        f.write("number of interaction\n")
        f.write("{}\n".format(num_interaction))


def read_count_file(count_file_path):
    # Special read function for dataset rating.
    if count_file_path == WHOLE_COUNT_FILE_PATH[DatasetName.RATING]:
        return read_rating_count(count_file_path)

    # Read the number of entities of the file.
    with open(count_file_path) as f:
        f.readline()
        user_num = int(f.readline())
        f.readline()
        playlist_num = int(f.readline())
        f.readline()
        track_num = int(f.readline())
        f.readline()
        interaction_num = int(f.readline())

    return DatasetNum(user_num, playlist_num, track_num, interaction_num)
