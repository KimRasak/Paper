import scipy as sp
import numpy as np
import networkx as nx
import metis
import matplotlib.pyplot as plt
from time import time

"""
1. 检查孤立歌曲数目
2. 测试划分簇时间
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

            for tid in tids:
                if tid not in data:
                    data[tid] = {pid}
                else:
                    data[tid].add(pid)

            max_uid = max(max_uid, uid)
            max_pid = max(max_pid, pid)
            max_tid = max(max_tid, max(tids))

            line = f.readline()
    return data, max_uid + 1, max_pid + 1, max_tid + 1


def read_graph(filepath):
    """
        Read playlist data from file. All ids in the file must be continuous starting from 0.
        :param filepath: Playlist file path.
        :return:
        """
    # 读取数据到dict中
    data = dict()

    set_pids = set()
    set_tids = set()
    with open(filepath) as f:
        head_title = f.readline()

        line = f.readline()
        while line:
            ids = [int(s) for s in line.split() if s.isdigit()]
            uid, pid, tids = ids[0], ids[1], ids[2:]

            set_pids.add(pid)
            if uid not in data:
                data[uid] = {pid: tids}
            elif pid not in data[uid]:
                data[uid][pid] = tids

            for tid in tids:
                set_tids.add(tid)
            line = f.readline()

    t1 = time()
    num_user = len(list(data.keys()))
    num_playlist = len(set_pids)
    num_track = len(set_tids)
    num_total = num_user + num_playlist + num_track

    # 生成Graph
    G = nx.Graph()
    # Add nodes.
    G.add_nodes_from([i for i in range(num_total)])
    # Add edges.
    for uid, user in data.items():
        assert uid < num_user
        real_uid = uid
        for pid, tids in user.items():
            assert pid < num_playlist
            real_pid = pid + num_user
            G.add_edge(real_uid, real_pid)
            for tid in tids:
                assert tid < num_track
                real_tid = tid + num_user + num_playlist
                G.add_edge(real_pid, real_tid)
    print("生产Graph用了%d秒" % (time() - t1))
    return G


def check_isolated_num():
    # 结论: 很多歌曲只在一个歌单内, 划分train/test后有一些歌曲会不属于任何歌单, 相当于从图中孤立了
    # 30music为例, 6.7k的歌曲不属于任何歌单.
    # 但这个情况应该没事, 因为尽管孤立歌曲不进行信息交互, 也不影响计算.
    filepath = '30music/playlist.txt'
    data, _, num_playlist, num_track = read_playlist(filepath)
    print(len(list(data.keys())))
    print("Num of playlists %d." % num_playlist)
    num_belonged_track = len(list(data.keys()))
    num_isolated_track = num_track - num_belonged_track

    print("Total num of track: %d. Num of isolated tracks: %d" % (num_track, num_isolated_track))
    # 30music 歌曲有46w首(461383首), train/test划分后, train数据集有6707首歌不在任何歌单内

    num_isolated_track = 0
    for tid, pids in data.items():
        if len(pids) == 1:
            num_isolated_track += 1
    print("Num of track: %d. Num of isolated tracks: %d" % (num_track, num_isolated_track))
    # 30music 歌曲有46w首, 其中25w首只在一个歌单内


def cluster_and_show(G, num_cluster):
    # 进行分割
    t1 = time()
    (edgecuts, parts) = metis.part_graph(G, num_cluster)
    print("分割图为%d个簇, 用了%d秒" % (num_cluster, time() - t1))
    for part in parts:
        assert 0 <= part < num_cluster

    print("There are %d clustered parts" % len(parts))
    if len(parts) <= 50:
        print(parts)

    # 输出分簇结果到文件
    filename = "temp_cluster.txt"
    with open(filename, "w") as f:
        for id, part in enumerate(parts):
            f.write("%d %d\n" % (id, part))

    # 再将分簇结果读入
    t2 = time()
    with open(filename) as f:
        line = f.readline()
        while line:
            ids = [int(s) for s in line.split() if s.isdigit()]
            assert len(ids) == 2
            line = f.readline()
    print("Read cluster file used %f seconds." % (time() - t2))

    # 显示分割结果
    colors = ['red', 'blue', 'green']
    if num_cluster > len(colors):
        print("颜色太多, 不予显示")
        return
    node_colors = []
    for i, p in enumerate(parts):
        node_colors.append(colors[p])
        G.node[i]['color'] = colors[p]
    nx.draw_networkx(G, node_color=node_colors)
    plt.show()


def demo_show_graph():
    G = nx.Graph()
    nodes = [i for i in range(10)]
    G.add_nodes_from(nodes)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    print("graph: ", G.node)
    cluster_and_show(G, 2)


def try_cluster_dataset():
    # ppt
    filepath = '30music/playlist.txt'

    # 读取数据Graph
    G = read_graph(filepath)

    # 分组输出
    cluster_and_show(G, 100)


if __name__ == '__main__':
    try_cluster_dataset()