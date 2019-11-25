import os
import numpy as np

import FileLayer.raw_file_layer as raw_file_layer
import FileLayer.data_file_layer as data_file_layer
from FileLayer import DatasetName, RAW_PLAYLIST_PATH, PICK_COUNT_FILE_PATH, WHOLE_PLAYLIST_PATH, PICK_PLAYLIST_PATH, \
    WHOLE_COUNT_FILE_PATH
from Common import DatasetNum

"""
提供各种对数据集的操作函数, 如下:
1. 读取user-playlist-song数据
2. 过滤用户和歌单
3. 以user-playlist-song数据为基础，
4. 保存数据集到playlist.txt中, 保存计数到count.txt中
5. 生成子数据集。保存到pick_events.txt和pick_playlist.txt中
在30music中，提供的读取数据的方式:
1. 从events.idomaar读取用户id和歌曲id
2. 从playlist.idomaar读取用户id、歌单id和包含的歌曲id
"""


# Define read functions for data sets.
ReadFileFunction = {
    DatasetName.THIRTY_MUSIC: raw_file_layer.read_raw_30music_playlists,
    DatasetName.AOTM: raw_file_layer.read_raw_aotm_playlists
}


def filter_playlist_data(data, max_n_playlist=1000, min_n_track=5, max_n_track=1000):
    # Filter the ids of users and ids of playlists which have too few/many interactions.
    # This function may cause the ids to be non-continuous,
    # for example, the ids of users can be 1, 2, 3, 4, and when this function runs, 3 is deleted,
    # so the remaining ids of users are 1, 2, 4, which is non-continuous(3 is gone).

    uids = list(data.keys())
    for uid in uids:
        user = data[uid]
        pids = list(user.keys())

        if len(pids) > max_n_playlist:  # The user has too many playlists.
            del data[uid]
            continue

        for pid in pids:
            playlist_tids = user[pid]
            n_track = len(playlist_tids)
            if n_track < min_n_track or n_track > max_n_track:  # Playlist too small/large.
                del data[uid][pid]
                if len(data[uid].keys()) == 0:  # Now the user has no playlist.
                    del data[uid]


def get_unique_ids(data: dict):
    # Read and return the unique ids of users/playlists/tracks
    unique_uids = set(list(data.keys()))
    unique_pids = set()
    unique_tids = set()
    interactions_num = 0

    for uid, user in data.items():
        for pid, tids in user.items():
            unique_pids.add(pid)
            for tid in tids:
                interactions_num += 1
                unique_tids.add(tid)
    return unique_uids, unique_pids, unique_tids, interactions_num


def compact_data_ids(playlist_data: dict, event_data: dict = None, uids=None, pids=None, tids=None):
    # Compact the ids of data.
    # The ids may be non-continuous,
    # so this function rearrange the ids to make them continuous, starting from 0.
    if uids is None or pids is None or tids is None:
        uids, pids, tids, _ = get_unique_ids(playlist_data)
    uid_dict = {uid: new_uid for new_uid, uid in enumerate(uids)}
    pid_dict = {pid: new_pid for new_pid, pid in enumerate(pids)}
    tid_dict = {tid: new_tid for new_tid, tid in enumerate(tids)}

    new_playlist_data = dict()
    for uid, user in playlist_data.items():
        new_uid = uid_dict[uid]

        if new_uid not in new_playlist_data:
            new_playlist_data[new_uid] = dict()

        for pid, tids in user.items():
            new_pid = pid_dict[pid]

            if new_pid not in new_playlist_data[new_uid]:
                new_playlist_data[new_uid][new_pid] = set()

            for tid in tids:
                new_tid = tid_dict[tid]
                new_playlist_data[new_uid][new_pid].add(new_tid)

    if event_data is None:
        return new_playlist_data

    new_events_data = dict()
    for uid, tids in event_data.items():
        new_uid = uid_dict[uid]

        if new_uid not in new_events_data:
            new_events_data[new_uid] = set()

        for tid in tids:
            new_tid = tid_dict[tid]
            new_events_data[new_uid].add(new_tid)
    return new_playlist_data, new_events_data


def generate_subset(playlist_data: dict, unique_pids=None, proportion=0.2):
    # Pick the sub-data-set from the whole data-set.
    # Some proportion of the playlists are picked.
    if unique_pids is None:
        _, unique_pids, _, _ = get_unique_ids(playlist_data)

    # Generate subset pids.
    num_pick_pids = int(len(unique_pids) * proportion)
    pick_pids = np.random.choice(list(unique_pids), num_pick_pids, replace=False)

    # Filter playlist data playlists.
    pick_playlist_data = dict()
    for uid, user in playlist_data.items():
        new_user = {pid: user[pid] for pid in user.keys() if pid in pick_pids}
        if len(new_user) > 0:
            pick_playlist_data[uid] = new_user

    uids, unique_pids, tids, num_playlist_interactions = get_unique_ids(pick_playlist_data)

    print("generate_subset: The [sub]-dataset has %d user ids, %d playlist ids, %d track ids and %d interactions" % (
        len(uids), len(unique_pids), len(tids), num_playlist_interactions))

    return pick_playlist_data


def gen_whole_dataset(dataset_name):
    # Read the raw data to generate whole data set,
    # and write the data set into files.

    # 1. Get function for reading file and read the playlist data.
    read_file_function = ReadFileFunction[dataset_name]
    filepath = raw_file_layer.get_raw_file_path(DatasetName.THIRTY_MUSIC)
    playlist_data = read_file_function(filepath)

    # 2. Filter out some playlists that don't meet the need.
    filter_playlist_data(playlist_data)

    # 3. Extrack the user/playlist/track ids from the filtered playlist data,
    # and count the number of ids.
    uids, pids, tids, interaction_num = get_unique_ids(playlist_data)
    dataset_num = DatasetNum(len(uids), len(pids), len(tids), interaction_num)
    print("The dataset has %d user ids, %d playlist ids, %d track ids and %d interactions" % (
        dataset_num.user, dataset_num.playlist, dataset_num.track, dataset_num.interaction))

    # 4. Save the number of each entity.
    count_filepath = WHOLE_COUNT_FILE_PATH[dataset_name]
    data_file_layer.write_count_file(dataset_num, count_filepath)

    # 5. Compact whole data set ids. Save data-set.
    compact_playlist_data = compact_data_ids(playlist_data)
    playlist_filepath = WHOLE_PLAYLIST_PATH[dataset_name]
    data_file_layer.write_playlist_data(compact_playlist_data, playlist_data_path=playlist_filepath)

    return compact_playlist_data


def gen_sub_dataset(dataset_name, playlist_data, proportion=0.6):
    # Generate sub-data-set according to the proportion of whole data set,
    # and write the sub-data-set into files.

    # 1. Generate sub-data-setset from the whole data set,
    # and compact the ids of the sub-data-set.
    pick_playlist_data = generate_subset(playlist_data, proportion=proportion)
    pick_playlist_data = compact_data_ids(pick_playlist_data)

    # 2. Extrack the user/playlist/track ids from the playlist data,
    # and save the number of entities of sub-data-set.
    uids, pids, tids, pick_interactions_num = get_unique_ids(pick_playlist_data)
    pick_dataset_num = DatasetNum(len(uids), len(pids), len(tids), pick_interactions_num)
    print("The dataset has %d user ids, %d playlist ids, %d track ids and %d interactions" % (
        pick_dataset_num.user, pick_dataset_num.playlist, pick_dataset_num.track, pick_dataset_num.interaction))

    count_path = PICK_COUNT_FILE_PATH[dataset_name]
    data_file_layer.write_count_file(pick_dataset_num, count_path)

    # 3. Save the sub-data-set.
    playlist_path = PICK_PLAYLIST_PATH[dataset_name]
    data_file_layer.write_playlist_data(pick_playlist_data, playlist_data_path=playlist_path)


def gen_dataset(dataset_name):
    # Read raw dataset files, generate data set and sub-data-set,
    # and write dataset files.
    whole_playlist_data = gen_whole_dataset(dataset_name)
    gen_sub_dataset(dataset_name, whole_playlist_data, proportion=0.6)


if __name__ == '__main__':
    gen_dataset(DatasetName.AOTM)
