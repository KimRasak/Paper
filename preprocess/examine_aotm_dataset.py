
import cjson
import gzip

with gzip.open('aotm2011_playlists.json.gz', 'r') as file_desc:
    playlists = cjson.decode(file_desc.read())