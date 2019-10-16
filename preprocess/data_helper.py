import os
import re
import json
import time
from collections import namedtuple

import numpy as np

"""
提供各种对数据集的操作函数, 如下:
1. 读取user-song数据(如果有)和user-playlist-song数据
2. 过滤用户和歌单
3. 以user-playlist-song数据为基础，过滤user-song的数据(如果有)。user-song的uid和tid都必须局限在user-playlist-song关系出现过的数据中。
4. 保存数据集到events.txt和playlist.txt中
5. 生成子数据集。保存到pick_events.txt和pick_playlist.txt中
在30music中，读取数据的方式:
1. 从events.idomaar读取用户id和歌曲id
2. 从playlist.idomaar读取用户id、歌单id和包含的歌曲id
"""

DatasetNum = namedtuple("DatasetNum", ["user", "playlist", "track", "interaction"])


# Define names of data sets.
class DatasetName:
    THIRTY_MUSIC = "30music"
    AOTM = "aotm"


# Define paths of raw data.
RAW_DATA_BASE_PATH = "../raw-data"
RAW_THIRTY_MUSIC_PATH = os.path.join(RAW_DATA_BASE_PATH, DatasetName.THIRTY_MUSIC)
RAW_AOTM_PATH = os.path.join(RAW_DATA_BASE_PATH, DatasetName.AOTM)

RAW_PLAYLIST_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(RAW_THIRTY_MUSIC_PATH, "entities/playlist.idomaar"),
    DatasetName.AOTM: os.path.join(RAW_AOTM_PATH, "aotm2011_playlists.json")
}

# Define paths of data sets.
# A data set usually contains a playlist file containing the playlist data
# and a count file containing the number of ids.
DATA_BASE_PATH = "../data"
THIRTY_MUSIC_PATH = os.path.join(DATA_BASE_PATH, DatasetName.THIRTY_MUSIC)
AOTM_PATH = os.path.join(DATA_BASE_PATH, DatasetName.AOTM)

PLAYLIST_FILE_NAME = "playlist.txt"
COUNT_FILE_NAME = "count.txt"

PICK_PLAYLIST_FILE_NAME = "pick_playlist.txt"
PICK_COUNT_FILE_NAME = "pick_count.txt"

WHOLE_PLAYLIST_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, PLAYLIST_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, PLAYLIST_FILE_NAME)
}

WHOLE_COUNT_FILE_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, COUNT_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, COUNT_FILE_NAME)
}

PICK_PLAYLIST_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, PICK_PLAYLIST_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, PICK_PLAYLIST_FILE_NAME)
}

PICK_COUNT_FILE_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, PICK_COUNT_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, PICK_COUNT_FILE_NAME)
}


def read_30music_events(filepath="../raw-data/30music/relations/events.idomaar"):
    # Read '30music' raw events data and return structured data.
    # Usually the events data is not used, so this function is not used.
    """
    Read the user-item interactions data.
    """
    data = dict()  # The interactions data. (key=user id, value=track ids that the user has listened.)

    read_count = 0
    max_uid = 0
    max_tid = 0

    # Print Message every n records.
    print_n = 100000
    print("Print Message every %d records." % print_n)

    # Record read start time.
    time_st = int(time.time())
    print("Start Reading events.idomaar")
    with open(filepath) as f:
        for line in f:
            pattern = re.compile('{"subjects".+}]}')
            match_result = re.findall(pattern, line)
            play_event = match_result[0]

            obj = json.loads(play_event)
            uid = obj["subjects"][0]["id"] - 1  # User id. Note that user id starts with 1. We Make them start with 0.
            tid = obj["objects"][0]["id"]  # Track id. Note that track id starts with 0.

            if uid not in data:
                data[uid] = {tid}
            else:
                data[uid].add(tid)
            # count user id/track id num.
            max_uid = max(max_uid, uid)
            max_tid = max(max_tid, tid)

            # Print Message.
            read_count += 1
            if read_count % print_n == 0:
                print("Having read %d records." % read_count)

    time_ed = int(time.time())
    print("Read event dataset comlete. Cost %d seconds. Read %d records. Max uid: %d, max tid: %d"
          % (time_ed - time_st, read_count, max_uid, max_tid))
    return data, max_uid, max_tid  # Note that the first track id is 0.


def read_30music_playlists(filepath=RAW_PLAYLIST_PATH[DatasetName.THIRTY_MUSIC]):
    # Read '30music' raw playlist data and return structured data.

    # 数据的完整性、正确性已经检测过, 因此没有添加assert语句。
    # The completion and correctness of data has been inspected, so many "assert" functions are ignored.
    data = dict()
    read_count = 0

    # Print Message every n records.
    print_n = 10000
    print("Print Message every %d records." % print_n)

    # Record read start time.
    time_st = int(time.time())
    print("Start Reading events.idomaar")
    with open(filepath) as f:
        for line in f:
            playlist_id_pattern = re.compile(r'{"ID":(\d+),"Title":')  # Containing Playlist info
            user_track_pattern = re.compile('{"subjects":.+]}')  # Containing user and tracks.

            pid = int(re.search(playlist_id_pattern, line).group(1))
            user_track = json.loads(re.findall(user_track_pattern, line)[-1])

            # Extract user id.
            assert len(user_track['subjects']) == 1
            assert user_track['subjects'][0]['type'] == 'user'
            uid = user_track['subjects'][0]['id']

            # Extract playlist's tids.
            if (len(user_track['objects']) == 0) or (
                    not isinstance(user_track['objects'][0], dict)):  # Ignore empty playlists.
                continue
            tids = set([track['id'] for track in user_track['objects']])

            # Put data.
            if uid not in data:
                data[uid] = {pid: tids}
            else:
                data[uid][pid] = tids

            # Print Message.
            read_count += 1
            if read_count % print_n == 0:
                print("Having read %d records." % read_count)

    time_ed = int(time.time())
    print("Read event dataset comlete. Cost %d seconds. Read %d playlists." % (time_ed - time_st, read_count))
    return data # Note that the first track id is 0.


def read_aotm_playlists(filepath=RAW_PLAYLIST_PATH[DatasetName.AOTM]):
    # Read 'aotm' raw playlist data and return structured data.
    with open('../raw-data/aotm/aotm2011_playlists.json', 'r') as file_desc:
        raw_playlists = json.loads(file_desc.read())
        data = {}

        st_users = set()
        st_playlists = set()
        st_tracks = set()

        for playlist in raw_playlists:
            pid = playlist['mix_id']
            username = playlist['user']['name']
            tracks = {(t[0][0], t[0][1]) for t in playlist['playlist']}

            if username not in data:
                data[username] = {}

            user = data[username]
            assert pid not in user
            user[pid] = tracks

            st_users.add(username)
            st_playlists.add(pid)
            for t in tracks:
                st_tracks.add(t)
    return data


# Define read functions for data sets.
ReadFileFunction = {
    DatasetName.THIRTY_MUSIC: read_30music_playlists,
    DatasetName.AOTM: read_aotm_playlists
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
    unique_uids = set(data.keys())
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


def save_dataset_num(dataset_num, n_filepath):
    # Write the number of user/playlist/track into the file.
    num_user = dataset_num.user
    num_playlist = dataset_num.playlist
    num_track = dataset_num.track

    with open(n_filepath, 'w') as f:
        f.write("number of user\n")
        f.write(num_user + "\n")
        f.write("number of playlist\n")
        f.write(num_playlist + "\n")
        f.write("number of track\n")
        f.write(num_track + "\n")


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


def save_data_playlist_and_events(playlist_data: dict, p_filepath, e_filepath,
                                  event_data: dict=None):
    # Write playlist data to file.
    with open(p_filepath, 'w') as f:
        f.write("user_id playlist_id track_ids\n")
        for uid, user in playlist_data.items():
            for pid, tids in user.items():
                f.write("%d %d " % (uid, pid))
                for tid in tids:
                    f.write("%d " % tid)
                f.write("\n")

    if event_data is None:
        # No events data to write. just return.
        return

    # Write events data to file.
    with open(e_filepath, 'w') as f:
        f.write("user_id track_ids\n")
        for uid, tids in event_data.items():
            f.write("%d " % uid)
            for tid in tids:
                f.write("%d " % tid)
            f.write("\n")


def main_30music():
    playlist_data = read_30music_playlists()

    # 1.1 Filter out some playlists that don't meet the need.
    filter_playlist_data(playlist_data)

    # 1.2 Extrack the user/playlist/track ids from the playlist data.
    uids, pids, tids, num_playlist_interactions = get_unique_ids(playlist_data)
    num_user = len(uids)
    num_playlist = len(pids)
    num_track = len(tids)
    dataset_num = DatasetNum(num_user, num_playlist, num_track)
    print("The dataset has %d user ids, %d playlist ids, %d track ids and %d interactions" % (
        num_user, num_playlist, num_track, num_playlist_interactions))

    # 1.3 Save the number of entities of dataset.
    n_filepath = "../data/30music/count.txt"
    save_dataset_num(dataset_num, n_filepath)

    # 1.4 Compact whole dataset ids. Save data-set.
    playlist_data = compact_data_ids(playlist_data)
    p_filepath = "../data/30music/playlist.txt"
    e_filepath = "../data/30music/events.txt"
    save_data_playlist_and_events(playlist_data, p_filepath=p_filepath, e_filepath=e_filepath)

    # 2.1 Generate subset. Compact subset ids. Save sub-dataset.
    pick_playlist_data = generate_subset(playlist_data, proportion=0.6)
    pick_playlist_data = compact_data_ids(pick_playlist_data)
    pick_p_filepath = "../data/30music/pick_playlist.txt"
    pick_e_filepath = "../data/30music/pick_events.txt"

    # 2.2 Extrack the user/playlist/track ids from the playlist data.
    pick_uids, pick_pids, pick_tids, num_pick_playlist_interactions = get_unique_ids(playlist_data)
    num_user = len(uids)
    num_playlist = len(pids)
    num_track = len(tids)
    dataset_num = DatasetNum(num_user, num_playlist, num_track)
    print("The dataset has %d user ids, %d playlist ids, %d track ids and %d interactions" % (
        num_user, num_playlist, num_track, num_playlist_interactions))
    pick_dataset_num = DatasetNum()

    # 2.3 Save the number of entities of sub-data-set.
    n_filepath = "../data/30music/pick_count.txt"
    save_dataset_num(dataset_num, n_filepath)

    # 2.4 Save the sub-data-set.
    save_data_playlist_and_events(pick_playlist_data, p_filepath=pick_p_filepath, e_filepath=pick_e_filepath)
    # Read event dataset comlete. Cost 2 seconds. Read 48422 playlists. Max uid: 45169, max pid: 11766362, max tid: 5023056

    # Before filter.
    # The dataset has 15102 user ids, 48422 playlist ids, 466244 track ids and 1602290 interactions
    # The events implicit feedbacks have 14938 user ids, 361142 track ids and 3160026 interactions

    # Ater filter.
    # The dataset has 13417 user ids, 39524 playlist ids, 461383 track ids and 1579282 interactions
    # The events implicit feedbacks have 13268 user ids, 343788 track ids and 2793905 interactions

    # 60%
    # The dataset has 13417 user ids, 39524 playlist ids, 461383 track ids and 1579282 interactions
    # generate_subset: The [sub]-dataset has 10674 user ids, 23714 playlist ids, 345510 track ids and 949721 interactions


def main_aotm():
    playlist_data = read_aotm_playlists()

    filter_playlist_data(playlist_data)
    uids, pids, tids, num_playlist_interactions = get_unique_ids(playlist_data)

    print("The dataset has %d user ids, %d playlist ids, %d track ids and %d interactions" % (
        len(uids), len(pids), len(tids), num_playlist_interactions))

    # Compact whole dataset ids. Save dataset.
    playlist_data = compact_data_ids(playlist_data)
    p_filepath = "../data/aotm/playlist.txt"
    e_filepath = "../data/aotm/events.txt"
    save_data_playlist_and_events(playlist_data, p_filepath=p_filepath, e_filepath=e_filepath)

    # Generate subset. Compact subset ids. Save sub-dataset.
    pick_playlist_data = generate_subset(playlist_data, proportion=0.4)
    pick_playlist_data = compact_data_ids(pick_playlist_data)
    pick_p_filepath = "../data/aotm/pick_playlist.txt"
    pick_e_filepath = "../data/aotm/pick_events.txt"
    save_data_playlist_and_events(pick_playlist_data, p_filepath=pick_p_filepath, e_filepath=pick_e_filepath)
    # aotm filtered data.
    # The dataset has 15863 user ids, 100013 playlist ids, 970678 track ids and 1981525 interactions
    # 20%
    # generate_subset: The [sub]-dataset has 6974 user ids, 20002 playlist ids, 270124 track ids and 396295 interactions
    # 40%
    # generate_subset: The [sub]-dataset has 10170 user ids, 40005 playlist ids, 473852 track ids and 792283 interactions


def gen_whole_dataset(dataset_name):
    # 1. Get function for reading file and read the playlist data.
    read_file_function = ReadFileFunction[dataset_name]
    filepath = RAW_PLAYLIST_PATH[DatasetName.THIRTY_MUSIC]
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
    save_dataset_num(dataset_num, count_filepath)

    # 5. Compact whole data set ids. Save data-set.
    compact_playlist_data = compact_data_ids(playlist_data)
    playlist_filepath = WHOLE_PLAYLIST_PATH[dataset_name]
    save_data_playlist_and_events(compact_playlist_data, p_filepath=playlist_filepath)

    pass


def gen_sub_dataset(dataset_name):
    pass


def gen_dataset(dataset_name):
    # Read raw dataset files and write dataset files.
    gen_whole_dataset(dataset_name)
    gen_sub_dataset(dataset_name)


if __name__ == '__main__':
    gen_dataset(DatasetName.THIRTY_MUSIC)
