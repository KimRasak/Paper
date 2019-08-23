
import json
import gzip

with open('../raw-data/aotm/aotm2011_playlists.json', 'r') as file_desc:
    playlists = json.loads(file_desc.read())
    up = {}
    pt = {}

    for playlist in playlists:
        pid = playlist['mix_id']
        username = playlist['user']['name']
        tracks = [(t[0][0], t[0][1]) for t in playlist['playlist']]

        if username not in up:
            up[username] = {}

        user = up[username]
        assert pid not in user
        user[pid] = []

        # print(len(playlists))
        # print(pid)
        # print(username)
        # print(tracks)
        # print(playlists[0])