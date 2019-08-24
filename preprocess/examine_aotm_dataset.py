
import json
import gzip
from preprocess.data_helper import read_aotm_playlists, filter_playlist_data, get_playlist_ids
#
# with open('../raw-data/aotm/aotm2011_playlists.json', 'r') as file_desc:
#     raw_playlists = json.loads(file_desc.read())
#     up = {}
#     pt = {}
#
#     st_users = set()
#     st_playlists = set()
#     st_tracks = set()
#     num_interactions = 0
#
#     for playlist in raw_playlists:
#         pid = playlist['mix_id']
#         username = playlist['user']['name']
#         tracks = [(t[0][0], t[0][1]) for t in playlist['playlist']]
#
#         if username not in up:
#             up[username] = {}
#
#         user = up[username]
#         assert pid not in user
#         user[pid] = tracks
#
#         st_users.add(username)
#         st_playlists.add(pid)
#         for t in tracks:
#             st_tracks.add(t)
#             num_interactions += 1
#
#         # print(len(playlists))
#         # print(pid)
#         # print(username)
#         # print(tracks)
#         # print(playlists[0])
#     print(len(st_users), len(st_playlists), len(st_tracks), num_interactions)
#     # 16204 users, 101343 playlists, 972081 tracks, 1990022 interactions

if __name__ == '__main__':
    playlist_data = read_aotm_playlists()
    uids, pids, tids, num_playlist_interactions = get_playlist_ids(playlist_data)
    print("There are %d users, %d playlists and %d tracks. There are %d user-playlist-song interactions." % (len(uids), len(pids), len(tids), num_playlist_interactions))
    filter_playlist_data(playlist_data)
    uids, pids, tids, num_playlist_interactions = get_playlist_ids(playlist_data)
    print("There are %d users, %d playlists and %d tracks. There are %d user-playlist-song interactions." % (len(uids), len(pids), len(tids), num_playlist_interactions))
    # aotm whole dataset
    # There are 16204 users, 101343 playlists and 972081 tracks. There are 1984355 user-playlist-song interactions.
    # filter data.
    # There are 15863 users, 100013 playlists and 970678 tracks. There are 1981525 user-playlist-song interactions.