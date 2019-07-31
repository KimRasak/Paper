# coding=utf-8
import scipy as spy
import scipy.sparse as sp
import numpy as np
from scipy.io import mmwrite, mmread
import re
import json

# 查看所有playlist,确认以下: 1. 每个playlist只有一个所有者 2. 所有者拥有的playlist数量阶梯分布, 写入文件
# 3. 每个type都是track
# 去除歌单中的重复歌曲, 过滤少于5首歌的playlist, 太大的playlist, 以及拥有太多playlist的用户


def deal_with_illegal_quote(line):
    new_line = ""
    for i, c in enumerate(line):
        if c == '"':
            if line[i - 1] == '{' or line[i - 1] == ',' or line[i - 1] == ':' or line[i + 1] == ':' or line[
                i + 1] == ',':
                new_line += c
            else:
                new_line += '\\' + c
        elif c == "\\":
            new_line += "\\" + c
        else:
            new_line += c
    return new_line


def match_row(line, playlist_dict):
    # Example 1.
    # Take the first line as an example:
    # playlist	0	1216545588	{"ID":2973549,"Title":"my_favorites","numtracks":27,"duration":6522}	{"subjects":[{"type":"user","id":41504}],"objects":[{"type":"track","id":3006631}, ...]}

    # Example 2.
    # It is below, note that there are 2 '[]' in 'objects'
    #  playlist	10	1172149901	{"ID":136481,"Title":"-kamyk-'s playlist","numtracks":89,"duration":30451}	{"subjects":[{"type":"user","id":43704}],"objects":[[]]}

    try:
        playlist_info_pattern = re.compile('{"ID":.+?}')  # Containing Playlist info
        user_track_pattern = re.compile('{"subjects".+]}')  # Containing user and tracks.

        playlist_info = json.loads(deal_with_illegal_quote(re.findall(playlist_info_pattern, line)[0]))
        user_track = json.loads(re.findall(user_track_pattern, line)[0])

        # print(playlist_info)
        # print(user_track)

        pid = playlist_info['ID']
        title = playlist_info['Title']
        duration = playlist_info['duration']


        # Check the 'subjects'
        assert len(user_track['subjects']) == 1
        assert user_track['subjects'][0]['type'] == 'user'

        uid = user_track['subjects'][0]['id']

        # Check the 'objects'
        tids = set()
        for track in user_track['objects']:
            if type(track) != type(dict()):  # Found Example 2.
                break

            assert track['type'] == 'track'
            tids.add(track['id'])

        assert pid not in playlist_dict
        playlist_dict[pid] = {
            'pid': pid,
            'title': title,
            'duration': duration,
            'tids': tids
        }

        # print(pid, title, numtracks, duration)
        # print(uid)
    except Exception as e:
        print("----")
        print(pid, "|%s|" % title, duration)
        print("user_track['objects']:", user_track['objects'])
        print("len(user_track['objects'])", len(user_track['objects']))
        print("----")
        raise






def read_file(path="../raw-data/30music/entities/playlist.idomaar"):
    with open(path) as f:
        i = 0
        line = f.readline()
        playlist_dict = {}
        while line:
            i += 1
            if i == 1:
                print(line)
            try:
                match_row(line, playlist_dict)
            except Exception as e:
                print("Error! The line number is: [%d], and the line is:\n" % i, line)
                raise

            line = f.readline()

if __name__ == '__main__':
    read_file()