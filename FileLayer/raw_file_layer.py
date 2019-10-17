import json
import re
import time


def read_raw_30music_events(filepath="../raw-data/30music/relations/events.idomaar"):
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


def read_raw_30music_playlists(filepath):
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


def read_raw_aotm_playlists(filepath):
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

