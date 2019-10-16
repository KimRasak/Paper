import numpy as np
from preprocess.data_helper import read_30music_playlists, filter_playlist_data, get_unique_ids

"""
本文将读取30music数据集，并对其完整性进行检查。
1. 要验证两个文件的user id和track id是一致的
   1. playlist.idomaar文件中的每个user id都在events.idomaar中有存在，反之亦然
   2. 两个文件中的track id互相在另一个文件中存在
2. playlist.idomaar文件中
   - 每个用户拥有的歌单数
   - 一个歌单可能有重复的歌曲
   - 查看歌单含有的歌曲的数量分布情况
3. events.idomaar文件中
   - 查看每个用户通过的歌曲分布情况

* 我们不检查用户听歌曲的交互记录，也就是说，听歌记录过少的用户不会被过滤。
* 当引入用户"喜好"记录时，情况可能变得更加复杂。


原文的数据集预处理:
For data preprocessing, we removed duplicate songs in playlists.
Then we adopted a widely used k-core preprocessing step [12, 47]
(with k-core = 5), filtering out playlists with less than 5 songs. We
also removed users with an extremely large number of playlists,
and extremely large playlists (i.e., containing thousands of songs).
Since the datasets did not have song order information for playlists
(i.e., which song was added to a playlist first, then next, and so on),
we randomly shuffled the song order of each playlist and used it in
the sequential recommendation baseline models to compare with
our models. The two datasets are implicit feedback datasets.

1. 对于歌单
   - 删除歌单中的重复歌曲
   - 删除数量少于5首歌的歌单
   - 删除数量庞大的歌单(几千首歌)
2. 对于用户
   - 删除有太多歌单的用户

"""

def count_playlist_data(data):
    num_users = len(data.keys())
    num_playlists = sum([len(p.keys()) for p in data.values()])

    tids = set()
    num_interactions = 0
    for uid, user in data.items():
        for pid, user_tids in user.items():
            for tid in user_tids:
                tids.add(tid)
                num_interactions += 1
    num_tracks = len(tids)
    return num_users, num_playlists, num_tracks, num_interactions


def check_playlist_data(data, output_filepath='../data/30music/playlist_distribution.txt'):
    print("-----Checking playlist data-----")
    num_users, num_playlists, num_tracks, num_interactions = count_playlist_data(data)

    # Check number of every user's playlist.
    distribution_user_num_playlist = {}
    for uid, user in data.items():
        n_playlist = len(user.keys())
        num_range = n_playlist
        if num_range not in distribution_user_num_playlist:
            distribution_user_num_playlist[num_range] = 1
        else:
            distribution_user_num_playlist[num_range] += 1

    # Check number of every playlist's tracks.
    distribution_playlist_num_track = {}
    for uid, user in data.items():
        for pid, user_tids in user.items():
            n_tracks = len(user_tids)
            num_range = n_tracks
            if num_range not in distribution_playlist_num_track:
                distribution_playlist_num_track[num_range] = 1
            else:
                distribution_playlist_num_track[num_range] += 1

    num_below_5 = 0
    for i in range(1, 5):
        if i in distribution_playlist_num_track:
            num_below_5 += distribution_playlist_num_track[i]
    if num_below_5 > 0:
        print("Number of playlists below 5:", num_below_5)

    print("There are %d users, %d playlists and %d tracks. There are %d user-playlist-song interactions." % (num_users, num_playlists, num_tracks, num_interactions))

    with open(output_filepath, 'w') as f:
        f.write("distribution_user_num_playlist\n")
        for num_range, num in sorted(distribution_user_num_playlist.items()):
            f.write("range: %d, num: %d\n" % (num_range, num))

        f.write("distribution_playlist_num_track\n")
        for num_range, num in sorted(distribution_playlist_num_track.items()):
            f.write("range: %d, num: %d\n" % (num_range, num))


if __name__ == '__main__':
    playlist_data, _, _, _ = read_30music_playlists()
    uids, pids, tids, num_playlist_interactions = get_unique_ids(playlist_data)
    print("There are %d users, %d playlists and %d tracks. There are %d user-playlist-song interactions." % (len(uids), len(pids), len(tids), num_playlist_interactions))

    filter_playlist_data(playlist_data)
    print("Filter data...")
    check_playlist_data(playlist_data)
    # 30music whole dataset
    # There are 15102 users, 48422 playlists and 466244 tracks. There are 1602290 user-playlist-song interactions.
    # Filter data...
    # -----Checking playlist data-----
    # There are 13417 users, 39524 playlists and 461383 tracks. There are 1579282 user-playlist-song interactions.