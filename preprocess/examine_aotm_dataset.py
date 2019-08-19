
import json
import gzip

with open('../raw-data/aotm/aotm2011_playlists.json', 'r') as file_desc:
    playlists = json.loads(file_desc.read())
    print(playlists[0])