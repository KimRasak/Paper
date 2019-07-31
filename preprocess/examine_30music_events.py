# coding=utf-8
import re
import os
import json
import numpy as np
# 检查数据的格式:
# 1. events中每一行都能读到user id和track id？ 【是的】
# 2. 每一个user id都是raw-data/30music/entities/persons.idomaar中的一个id？ 【不是】
# 3. 每一个user id都是raw-data/30music/entities/users.idomaar中的一个id？ 【user id 45168到45175并不包含在users.idomaar中，其它都包含】
# 4. raw-data/30music/entities/users.idomaar的id是连续的，从1到45167？ 【是的】

file_path_persons = "../raw-data/30music/entities/persons.idomaar"
file_path_users = "../raw-data/30music/entities/users.idomaar"
file_path_tracks = "../raw-data/30music/entities/tracks.idomaar"
file_path_events = "../raw-data/30music/relations/events.idomaar"


# 从persons.idomaar读取所有person id
def read_file_persons(file_path_persons):
    pids = set()
    with open(file_path_persons) as f:
        for line in f:
            pattern_person = 'person\s\d+\s'
            pid_with_prefix = re.findall(pattern_person, line)[0]

            pattern_number = '\d+'
            pid = re.findall(pattern_number, pid_with_prefix)[0]

            if not str.isnumeric(pid):
                raise Exception("Error! extracted string isn'preprocess number")
            pids.add(int(pid))
    return pids


# 从users.idomaar读取所有user id
def read_file_users(file_path_users):
    uids = set()
    with open(file_path_users) as f:
        for line in f:
            pattern_user = 'user\s\d+\s'
            uid_with_prefix = re.findall(pattern_user, line)[0]

            pattern_number = '\d+'
            uid = re.findall(pattern_number, uid_with_prefix)[0]

            if not str.isnumeric(uid):
                raise Exception("Error! extracted string isn'preprocess number")
            uids.add(int(uid))
    return uids


# 从tracks.idomaar读取所有track id
def read_file_tracks(file_path_tracks):
    tids = set()
    with open(file_path_tracks) as f:
        for line in f:
            pattern_track = 'track\s\d+\s'
            tid_with_prefix = re.findall(pattern_track, line)[0]

            pattern_number = '\d+'
            tid = re.findall(pattern_number, tid_with_prefix)[0]

            if not str.isnumeric(tid):
                raise Exception("Error! extracted string isn'preprocess number")
            tids.add(int(tid))
    return tids


def read_file_events(file_path_events):
    """
    Read the user-item interactions data.
    The first id of user is 1, but the first id of track is 0.
    :param file_path_events: The path of events.idomaar
    :return: The data, max user id, max track id.
    """
    # key: A unique user id
    # value: list. Containing all track ids that the user has interact with.
    # e.g. { uid1: [tid1, tid2, ...], uid2: [...] }
    data = dict()

    read_count = 0
    print_every_n_records = 100000
    print("Print Message every %d records." % print_every_n_records)

    max_uid = 0
    max_tid = 0

    with open(file_path_events) as f:
        for line in f:
            pattern = re.compile('{"subjects".+}]}')
            match_result = re.findall(pattern, line)
            play_event = match_result[0]

            obj = json.loads(play_event)
            uid = obj["subjects"][0]["id"] - 1 # User id. Note that user id starts with 1. We Make them start with 0.
            tid = obj["objects"][0]["id"]  # Track id. Note that track id starts with 0.

            # Record the interaction.
            if uid not in data:
                data[uid] = []
            uis = data[uid] # All track ids that the user has interacted with
            if tid not in uis:
                data[uid].append(tid)

            # Print Message.
            read_count += 1
            if read_count % print_every_n_records == 0:
                print("Having read %d records." % read_count)

            # count user id/track id num.
            max_uid = max(max_uid, uid)
            max_tid = max(max_tid, tid)

            # if read_count == 30: debug
            #     return data, max_uid, max_tid

        print("There are %d records in all." % read_count)
        print("There are %d users and %d tracks." % (max_uid, max_tid))
        print("Read dataset comlete.")
    return data, max_uid, max_tid  # Note that the first track id is 0.


def read_file_events_column_first(file_path_events):
    """
        Read the user-item interactions data.
        The first id of user is 1, but the first id of track is 0.
        :param file_path_events: The path of events.idomaar
        :return: The data, num of user, and num of track.
        """
    # key: A unique track id
    # value: list. Containing all user ids that have interacted with the track.
    # e.g. { tid1: [uid1, uid2, ...], tid2: [...] }
    data = dict()

    read_count = 0
    print_every_n_records = 100000
    print("Print Message every %d records." % print_every_n_records)

    max_uid = 0
    max_tid = 0

    with open(file_path_events) as f:
        for line in f:
            pattern = re.compile('{"subjects".+}]}')
            match_result = re.findall(pattern, line)
            play_event = match_result[0]

            obj = json.loads(play_event)
            uid = obj["subjects"][0]["id"] - 1  # User id. Note that user id starts with 1. We Make them start with 0.
            tid = obj["objects"][0]["id"]  # Track id. Note that track id starts with 0.

            # Record the interaction.
            if tid not in data:
                data[tid] = []
            ius = data[tid]  # All track ids that the user has interacted with
            if uid not in ius:
                data[tid].append(uid)

            # Print Message.
            read_count += 1
            if read_count % print_every_n_records == 0:
                print("Having read %d records." % read_count)

            # For debugging, interrupt directly.
            # if read_count == 1000:
            #     return data, max_uid + 1, max_tid + 1

            # count user id/track id num.
            max_uid = max(max_uid, uid)
            max_tid = max(max_tid, tid)

        print("There are %d records in all." % read_count)
        print("There are %d users and %d tracks." % (max_uid, max_tid))
        print("Read dataset comlete.")
    return data, max_uid + 1, max_tid + 1  # Note that the first track id is 0.


# 查看每行里的user id是否都包含在ids里
# ids提取自persons.idomaar或users.idomaar文件
def check_file_events(file_path_events, uids, tids):
    unexpected_uids = []
    unexpected_tids = []

    with open(file_path_events) as f:
        line_num = 0
        for line in f:
            pattern = re.compile('{"subjects".+}]}')
            match_result = re.findall(pattern, line)
            play_event = match_result[0]

            obj = json.loads(play_event)
            uid = obj["subjects"][0]["id"]  # user id
            tid = obj["objects"][0]["id"]  # track id

            if (uid not in uids) and (uid not in unexpected_uids):
                print("Error! current extracted 【user id】 %d isn'preprocess in stored 【user ids】.\n"
                                "line: %s" % (uid, line))
                unexpected_uids.append(uid)

            if (tid not in tids) and (tid not in unexpected_tids):
                print("Error! current extracted 【track id】 %d isn'preprocess in stored 【track ids】.\n"
                                "line: %s" % (tid, line))
                unexpected_tids.append(tid)

            line_num += 1
        print("There are %d lines in file events" % line_num)
        print("unexpected uids: ", unexpected_uids)  # [45168, 45169, ..., 45175] events中只有几个多出的user id
        print("unexpected tids: ", unexpected_tids)  # [3893304, ...] events中有比较多多出的track id
    return unexpected_uids, unexpected_tids


if __name__ == '__main__':
    print(os.curdir)

    uids = read_file_users(file_path_users)
    tids = read_file_tracks(file_path_tracks)

    # length of ids == the last element of ids
    print("uids is continuous(%d compared to %d): %r" % (len(uids), max(uids), len(uids) == max(uids)))
    # length of ids == the last element of ids
    print("tids is continuous(%d compared to %d): %r" % (len(tids), max(tids), len(tids) == max(tids)))

    # 查看events文件中相比users文件和tracks文件多出的id
    unexpected_uids, unexpected_tids = check_file_events(file_path_events, uids, tids)

    # 【raw data】 (45175 users * 5023108 tracks)
    # 【论文数据】 (12336 users * 276142 tracks)









