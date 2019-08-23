
import json
import gzip

with open('../raw-data/aotm/aotm2011_playlists.json', 'r') as file_desc:
    raw_playlists = json.loads(file_desc.read())
    up = {}
    pt = {}

    st_users = set()
    st_playlists = set()
    st_tracks = set()

    for playlist in raw_playlists:
        pid = playlist['mix_id']
        username = playlist['user']['name']
        tracks = [(t[0][0], t[0][1]) for t in playlist['playlist']]

        if username not in up:
            up[username] = {}

        user = up[username]
        assert pid not in user
        user[pid] = tracks

        st_users.add(username)
        st_playlists.add(pid)
        for t in tracks:
            st_tracks.add(t)

        # print(len(playlists))
        # print(pid)
        # print(username)
        # print(tracks)
        # print(playlists[0])
    print(len(st_users), len(st_playlists), len(st_tracks))