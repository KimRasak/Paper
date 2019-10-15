from FileLayer import DatasetNum, TRAIN_FILE_PATH, PICK_TRAIN_FILE_PATH, TEST_FILE_PATH, PICK_TEST_FILE_PATH, \
    COUNT_FILE_NAME, PICK_COUNT_FILE_PATH


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
        return COUNT_FILE_NAME[data_set_name]
    else:
        return PICK_COUNT_FILE_PATH[data_set_name]

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
    data = dict()
    with open(playlist_data_path) as f:
        head_title = f.readline()  # title: "user_id playlist_id track_ids"

        line = f.readline()
        while line:
            ids = [int(s) for s in line.split() if s.isdigit()]
            uid, pid, tids = ids[0], ids[1], ids[2:]

            if uid not in data:
                data[uid] = {pid: tids}
            elif pid not in data[uid]:
                data[uid][pid] = tids

            line = f.readline()
    return data


def write_count_file(dataset_num, count_file_path):
    # Write the number of user/playlist/track into the file.
    num_user = dataset_num.user
    num_playlist = dataset_num.playlist
    num_track = dataset_num.track

    with open(count_file_path, 'w') as f:
        f.write("number of user\n")
        f.write(num_user + "\n")
        f.write("number of playlist\n")
        f.write(num_playlist + "\n")
        f.write("number of track\n")
        f.write(num_track + "\n")


def read_count_file(count_file_path):
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
