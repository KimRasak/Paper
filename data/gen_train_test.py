import scipy as sp
import numpy as np

"""
读取交互数据文件、歌单数据文件，并切分训练/测试集。
1. 读取文件
2. 压缩id
3. 切分数据集并保存

"""


def read_playlist(filepath):
    """
    Read playlist data from file. All ids in the file must be continuous starting from 0.
    :param filepath: Playlist file path.
    :return:
    """
    data = dict()
    max_uid = 0
    max_pid = 0
    max_tid = 0
    with open(filepath) as f:
        head_title = f.readline()

        line = f.readline()
        while line:
            ids = [int(s) for s in line.split() if s.isdigit()]
            uid, pid, tids = ids[0], ids[1], ids[2:]

            if uid not in data:
                data[uid] = {pid: tids}
            elif pid not in data[uid]:
                data[uid][pid] = tids

            max_uid = max(max_uid, uid)
            max_pid = max(max_pid, pid)
            max_tid = max(max_tid, max(tids))

            line = f.readline()
    return data, max_uid + 1, max_pid + 1, max_tid + 1


def read_events(filepath):
    data = dict()
    with open(filepath) as f:
        head_title = f.readline()

        line = f.readline()
        while line:
            ids = [int(s) for s in line.split() if s.isdigit()]
            uid = ids[0]
            tid = ids[1]

            if uid not in data:
                data[uid] = {tid}
            else:
                data[uid].add(tid)

            line = f.readline()
    return data


def split_train_test(playlist_data: dict, train_filepath, test_filepath, num_user, num_playlist, num_track, proportion=0.8):
    train_data = dict()
    test_data = dict()

    f_train = open(train_filepath, 'w')
    f_test = open(test_filepath, 'w')

    f_train.write("whole dataset: %d users, %d playlists, %d tracks\n" % (num_user, num_playlist, num_track))  # Write title
    for uid, user in playlist_data.items():
        if uid not in train_data:
            train_data[uid] = {}
        if uid not in test_data:
            test_data[uid] = {}

        for pid, tids in user.items():
            # num_test = max(int(len(tids) * (1 - proportion)), 1)
            num_test = 1
            num_train = len(tids) - num_test
            train_tids = np.random.choice(tids, num_train, replace=False)
            test_tids = [tid for tid in tids if tid not in train_tids]


            train_data[uid][pid] = train_tids
            test_data[uid][pid] = test_tids

            # Write user's playlist to train_file and test_file.
            f_train.write("%d %d " % (uid, pid))
            f_test.write("%d %d " % (uid, pid))
            for tid in train_tids:
                f_train.write("%d " % tid)
            for tid in test_tids:
                f_test.write("%d " % tid)
            f_train.write("\n")
            f_test.write("\n")

    f_train.close()
    f_test.close()



if __name__ == '__main__':
    p_filepath = '30music/playlist.txt'
    train_filepath = '30music/train.txt'
    test_filepath = '30music/test.txt'
    p_data, num_user, num_playlist, num_track = read_playlist(p_filepath)
    split_train_test(p_data, train_filepath, test_filepath, num_user, num_playlist, num_track)


