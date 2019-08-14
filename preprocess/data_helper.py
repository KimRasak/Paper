import re
import json
import time
import numpy as np

"""
提供各种对数据集的操作函数
1. 读取user-song数据和user-playlist-song数据
2. 过滤用户和歌单
3. 以user-playlist-song数据为基础，过滤user-song的数据。user-song的uid和tid都必须局限在user-playlist-song出现过的数据中。
4. 保存数据集到events.txt和playlist.txt中
5. 生成子数据集。保存到pick_events.txt和pick_playlist.txt中
30music:
1. 从events.idomaar读取用户id和歌曲id
2. 从playlist.idomaar读取用户id、歌单id和包含的歌曲id
"""


def read_30music_events(filepath="../raw-data/30music/relations/events.idomaar"):
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


def read_30music_playlists(filepath="../raw-data/30music/entities/playlist.idomaar"):
    # 数据的完整性、正确性已经检测过, 因此忽视了很多assert语句。
    # The completion and correctness of data has been inspected, so many "assert" functions are ignored.
    data = dict()

    read_count = 0
    max_uid = 0
    max_pid = 0
    max_tid = 0

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
            if (len(user_track['objects']) == 0) or (not isinstance(user_track['objects'][0], dict)):  # Ignore empty playlists.
                continue
            tids = set([track['id'] for track in user_track['objects']])

            # Put data.
            if uid not in data:
                data[uid] = {pid: tids}
            else:
                data[uid][pid] = tids

            # Get max id.
            max_uid = max(max_uid, uid)
            max_pid = max(max_pid, pid)
            max_tid = max(max_tid, max(tids))

            # Print Message.
            read_count += 1
            if read_count % print_n == 0:
                print("Having read %d records." % read_count)

    time_ed = int(time.time())
    print("Read event dataset comlete. Cost %d seconds. Read %d playlists. Max uid: %d, max pid: %d, max tid: %d"
          % (time_ed - time_st, read_count, max_uid, max_pid, max_tid))
    return data, max_uid, max_pid, max_tid  # Note that the first track id is 0.


def read_30music(filepath):
    pass


def filter_playlist_data(data, max_n_playlist=1000, min_n_track=5, max_n_track=1000):
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


def get_playlist_ids(data: dict):
    unique_uids = data.keys()
    unique_pids = set()
    unique_tids = set()
    num_interactions = 0

    for uid, user in data.items():
        for pid, tids in user.items():
            unique_pids.add(pid)
            for tid in tids:
                num_interactions += 1
                unique_tids.add(tid)
    return unique_uids, unique_pids, unique_tids, num_interactions


def get_events_ids(data: dict):
    unique_uids = data.keys()
    unique_tids = set()
    num_interactions = 0

    for uid, tids in data.items():
        for tid in tids:
            num_interactions += 1
            unique_tids.add(tid)

    return unique_uids, unique_tids, num_interactions


def filter_events_data(events_data: dict, valid_uids, valid_tids):
    uids = list(events_data.keys())
    for uid in uids:
        if uid not in valid_uids:
            del events_data[uid]
            continue

        tids = events_data[uid]
        events_data[uid] = {tid for tid in tids if tid in valid_tids}
        if len(events_data[uid]) == 0:
            del events_data[uid]


def compact_data_ids(playlist_data: dict, event_data: dict, uids=None, pids=None, tids=None):
    if uids is None or pids is None or tids is None:
        uids, pids, tids, _ = get_playlist_ids(playlist_data)
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

    new_events_data = dict()
    for uid, tids in event_data.items():
        new_uid = uid_dict[uid]

        if new_uid not in new_events_data:
            new_events_data[new_uid] = set()

        for tid in tids:
            new_tid = tid_dict[tid]
            new_events_data[new_uid].add(new_tid)
    return new_playlist_data, new_events_data


def generate_subset(playlist_data: dict, event_data: dict, pids=None, proportion=0.2):
    if pids is None:
        _, pids, _, _ = get_playlist_ids(playlist_data)

    # Generate subset pids.
    num_pick_pids = int(len(pids) * proportion)
    pick_pids = np.random.choice(list(pids), num_pick_pids, replace=False)

    # Filter playlist data playlists.
    pick_playlist_data = dict()
    for uid, user in playlist_data.items():
        new_user = {pid: user[pid] for pid in user.keys() if pid in pick_pids}
        if len(new_user) > 0:
            pick_playlist_data[uid] = new_user

    uids, pids, tids, num_playlist_interactions = get_playlist_ids(pick_playlist_data)
    filter_events_data(event_data, uids, tids)
    event_uids, event_tids, num_event_interactions = get_events_ids(event_data)

    print("generate_subset: The [sub]-dataset has %d user ids, %d playlist ids, %d track ids and %d interactions" % (len(uids), len(pids), len(tids), num_playlist_interactions))
    print("generate_subset: The [subset] events implicit feedbacks have %d user ids, %d track ids and %d interactions" % (len(event_uids), len(event_tids), num_event_interactions))

    return pick_playlist_data, event_data


def save_data(playlist_data: dict, event_data: dict, p_filepath="../data/30music/playlist.txt", e_filepath="../data/30music/events.txt"):
    with open(p_filepath, 'w') as f:
        f.write("user_id playlist_id track_ids\n")
        for uid, user in playlist_data.items():
            for pid, tids in user.items():
                f.write("%d %d " % (uid, pid))
                for tid in tids:
                    f.write("%d " % tid)
                f.write("\n")

    with open(e_filepath, 'w') as f:
        f.write("user_id track_ids\n")
        for uid, tids in event_data.items():
            f.write("%d " % uid)
            for tid in tids:
                f.write("%d " % tid)
            f.write("\n")


if __name__ == '__main__':
    event_data, _, _ = read_30music_events()
    playlist_data, _, _, _ = read_30music_playlists()

    filter_playlist_data(playlist_data)
    uids, pids, tids, num_playlist_interactions = get_playlist_ids(playlist_data)
    filter_events_data(event_data, uids, tids)
    event_uids, event_tids, num_event_interactions = get_events_ids(event_data)


    print("The dataset has %d user ids, %d playlist ids, %d track ids and %d interactions" % (len(uids), len(pids), len(tids), num_playlist_interactions))
    print("The events implicit feedbacks have %d user ids, %d track ids and %d interactions" % (len(event_uids), len(event_tids), num_event_interactions))

    # Compact whole dataset ids. Save dataset.
    playlist_data, event_data = compact_data_ids(playlist_data, event_data)
    save_data(playlist_data, event_data)

    # Generate subset. Compact subset ids. Save sub-dataset.
    pick_playlist_data, pick_event_data = generate_subset(playlist_data, event_data)
    pick_playlist_data, pick_event_data = compact_data_ids(pick_playlist_data, pick_event_data)
    pick_p_filepath = "../data/30music/pick_playlist.txt"
    pick_e_filepath = "../data/30music/pick_events.txt"
    save_data(pick_playlist_data, pick_event_data, p_filepath=pick_p_filepath, e_filepath=pick_e_filepath)
    # Read event dataset comlete. Cost 2 seconds. Read 48422 playlists. Max uid: 45169, max pid: 11766362, max tid: 5023056

    # Before filter.
    # The dataset has 15102 user ids, 48422 playlist ids, 466244 track ids and 1602290 interactions
    # The events implicit feedbacks have 14938 user ids, 361142 track ids and 3160026 interactions

    # Ater filter.
    # The dataset has 13417 user ids, 39524 playlist ids, 461383 track ids and 1579282 interactions
    # The events implicit feedbacks have 13268 user ids, 343788 track ids and 2793905 interactions
    # The [sub]-dataset has 5305 user ids, 7904 playlist ids, 170691 track ids and 319299 interactions
    # The [subset] events implicit feedbacks have 5207 user ids, 108607 track ids and 719713 interactions




