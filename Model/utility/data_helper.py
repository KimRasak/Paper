import os
from time import time

import metis as metis
import numpy as np
import scipy.sparse as sp
import networkx as nx

"""
数据读取来源
首先设置路径，尝试读取.mat文件。
若失败，则尝试读取train/test文件，并以此生成.mat文件。
若再失败，则抛出异常。

先试试直接读取train/test文件，得到矩阵的速度。
"""


def get_laplacian(A: sp.spmatrix, A_alpha=None, log=True):
    t = time()

    def get_D(adj: sp.spmatrix,
              power=-0.5):  # Get degree diagonal matrix, where a node's degree is the number of edges starting from this node.
        rowsum = np.sum(adj, axis=1).flatten().A[0]

        # Get inverse( x^(-1) ) of every element, but zero remains zero.
        with np.errstate(divide='ignore'):
            d_inv = np.float_power(rowsum, power)
        d_inv[np.isinf(d_inv)] = 0
        d_mat_inv = sp.diags(d_inv)
        return d_mat_inv

    def get_symmetric_normalized_laplacian(adj: sp.spmatrix, A_alpha: sp.spmatrix):
        D = get_D(adj)
        if A_alpha != None:
            return D.dot(A_alpha).dot(D).tocoo()
        else:
            return D.dot(adj).dot(D).tocoo()

    L: sp.spmatrix = get_symmetric_normalized_laplacian(A, A_alpha)
    if log:
        print('Used %f seconds. Get symmetric normalized laplacian matrix.' % (time() - t))
    return L.tocsr()


def set_maxtrix_value(A, R, m_offset, n_offset, alpha=1):
    cx = R.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        assert v == 1
        A[i + m_offset, j + n_offset] = alpha
        A[j + n_offset, i + m_offset] = alpha


def get_A_3(R_up: sp.spmatrix, R_ut: sp.spmatrix, R_pt: sp.spmatrix, alpha):
    # Get matrix "A" among user-playlist-track relationship.
    """
    A = [ 0      R_up   R_ut
          R_up_T  0     R_pt
          R_ut_T R_pt_T 0   ]
        (m+n+l) * (m+n+l)
    Where m, n and l is the number of users, playlists and tracks.
    R_up is the interaction matrix between users and playlists.
    Matrix R_up_T is the transpose of matrix R_up. So as the others.
    :return: matrix A
    """
    t = time()
    m = R_up.shape[0]  # Number of users.
    n = R_pt.shape[0]  # Number of playlists.
    l = R_ut.shape[1]  # Number of tracks.
    assert R_ut.shape[1] == R_pt.shape[1]

    A = sp.lil_matrix((m+n+l, m+n+l), dtype=np.float64)
    set_maxtrix_value(A, R_up, 0, n)
    set_maxtrix_value(A, R_ut, 0, m+n)
    set_maxtrix_value(A, R_pt, m, m+n)

    A_alpha = None
    if alpha != 1:
        A_alpha = sp.lil_matrix((m+n+l, m+n+l), dtype=np.float64)
        set_maxtrix_value(A_alpha, R_up, 0, n)
        set_maxtrix_value(A_alpha, R_ut, 0, m+n, alpha=alpha)
        set_maxtrix_value(A_alpha, R_pt, m, m+n)
    print('Used %d seconds. Already create adjacency matrix(A_3). shape of A: %r' % (time() - t, A.shape))
    return A, A_alpha


def get_A_2(R: sp.spmatrix):  # Get matrix "A" between user-item relationship.
    t = time()
    m = R.shape[0]
    n = R.shape[1]

    A = sp.lil_matrix((m + n, m + n), dtype=np.float64)
    set_maxtrix_value(A, R, 0, m)
    print('Used %d seconds. Already create adjacency matrix(A_2). shape of A: %r' % (time() - t, A.shape))
    return A


def sample_pos_track_for_playlist(playlist: int, pt: dict):
    pos_tracks = pt[playlist]
    return np.random.choice(pos_tracks, 1)[0]


class Data:
    def __init__(self, path, pick=True, laplacian_mode="PT2", batch_size=256, reductive_ut=True, alpha=1, epoch_times=4,
                 num_cluster=50):
        t0 = time()
        self.path = path
        self.dataset_name = path.split("/")[-1]
        self.batch_size = batch_size
        self.alpha = alpha
        self.num_cluster = num_cluster


        if pick is True:
            print("{pick} == %r, Using picked playlist data. That is, you're using a sub-dataset" % pick)
            train_filepath = path + "/pick_train.txt"
            test_filepath = path + "/pick_test.txt"
            event_filepath = path + "/pick_events.txt"
            cluster_map_filepath = path + "/pick_cluster_%s_%d.txt" % (laplacian_mode, num_cluster)
        else:
            print("{pick} == %r, Using complete playlist data. That is, you're using a complete dataset" % pick)
            train_filepath = path + "/train.txt"
            test_filepath = path + "/test.txt"
            event_filepath = path + "/events.txt"
            cluster_map_filepath = path + "/cluster_%s_%d.txt" % (laplacian_mode, num_cluster)

        # 验证laplacian模式
        laplacian_modes = ["PT2", "PT4", "UT", "UPT", "None", "TestPT", "TestUPT",
                           "clusterPT2", "clusterPT4", "clusterUT", "clusterUPT"]
        if laplacian_mode not in laplacian_modes:
            raise Exception("Wrong laplacian mode. Expected one of %r, got %r" % (laplacian_modes, laplacian_mode))
        self.laplacian_mode = laplacian_mode

        # Read the number of entities and print statistics.
        self.read_entity_num(train_filepath)

        # Define matrices and dicts for later use.
        if "cluster" not in laplacian_mode:
            self.R_up = sp.dok_matrix((self.n_user, self.n_playlist), dtype=np.float64)
            self.R_ut = sp.dok_matrix((self.n_user, self.n_track), dtype=np.float64)
            self.R_pt = sp.dok_matrix((self.n_playlist, self.n_track), dtype=np.float64)
        self.pt = {}  # Storing playlist-track relationship of training set.
        self.up = {}  # Storing user-playlist relationship of training set.
        self.ut = {}  # Storing user-track relationship of training set.
        self.test_set = []  # Storing user-playlist-track test set.

        if laplacian_mode != "Test":
            # Initialize R_ut
            if reductive_ut:
                print("Using Reductive R_ut, not reading events data.")
            else:
                self.read_events_file(event_filepath)

            # Initialize matrix R_up, matrix R_pt, dict up and dict pt.
            self.read_train_file(train_filepath, reductive_ut=reductive_ut)

            # Initialize self.test_set
            self.read_test_file(test_filepath)

        self.n_batch = int(np.ceil(self.n_train / self.batch_size)) * epoch_times
        print("There are %d training instances. Each epoch has %d batches with batch size of %d." % (self.n_train, self.n_batch, self.batch_size))

        # set u/p/t offset.
        self.set_offset()

        # 读取cluster映射
        if "cluster" in laplacian_mode:
            print("laplacian_mode=%r, loading clustered laplacian matrix." % laplacian_mode)
            self.read_cluster_laplacian_matrix(num_cluster, cluster_map_filepath)
            self.map_ids()
        elif laplacian_mode != "None":
            print("laplacian_mode=%r, loading laplacian matrix." % laplacian_mode)
            self.gen_normal_laplacian_matrix()
        elif laplacian_mode == "None":
            print("laplacian_mode=%r. Don't load laplacian matrix." % laplacian_mode)

        print("Read data used %d seconds in all." % (time() - t0))

    def set_offset(self):
        if "cluster" not in self.laplacian_mode:
            self.u_offset = self.p_offset = self.t_offset = 0
        else:
            if "UPT" in self.laplacian_mode:
                self.u_offset = 0
                self.p_offset = self.n_user
                self.t_offset = self.n_user + self.n_playlist
            elif "PT" in self.laplacian_mode:
                self.u_offset = 0
                self.p_offset = 0
                self.t_offset = self.n_playlist
            elif "UT" in self.laplacian_mode:
                self.u_offset = 0
                self.p_offset = 0
                self.t_offset = self.n_user

    def map_ids(self):
        # 将原本的id序列映射到cluster的id序列下
        laplacian_mode = self.laplacian_mode

        # map up
        temp_up = dict()
        for uid, pids in self.up.items():
            # "UPT" and "PT"
            if "UPT" in laplacian_mode:
                new_pids = pids + self.n_user
            else:
                new_pids = pids

            temp_up[self.cluster_id_map[uid]] = self.cluster_id_map[new_pids]
        self.up = temp_up

        # map pt
        temp_pt = dict()
        for pid, tids in self.pt.items():
            # "UPT" and "PT"
            if "UPT" in laplacian_mode:
                mapped_pid = pid + self.n_user
            else:
                mapped_pid = pid

            if "UPT" in laplacian_mode:
                new_tids = tids + self.n_user + self.n_playlist
            elif "PT" in laplacian_mode:
                new_tids = tids + self.n_playlist
            else:  # "UT" mode
                new_tids = tids + self.n_user
            temp_pt[self.cluster_id_map[mapped_pid]] = self.cluster_id_map[new_tids]
        self.pt = temp_pt

        # map ut
        temp_ut = dict()
        for uid, tids in self.ut.items():
            # "UPT" and "UT"
            if "UPT" in laplacian_mode:
                new_tids = tids + self.n_user + self.n_playlist
            else:  # "UT" mode
                new_tids = tids + self.n_user
            temp_ut[self.cluster_id_map[uid]] = self.cluster_id_map[new_tids]
        self.ut = temp_ut

        # map test_set
        for test_i, test_tuple in enumerate(self.test_set):
            test_tuple[0] = self.cluster_id_map[test_tuple[0] + self.u_offset]
            test_tuple[1] = self.cluster_id_map[test_tuple[1] + self.p_offset]
            test_tuple[2] = self.cluster_id_map[test_tuple[2] + self.t_offset]
            # if "cluster" in self.laplacian_mode:
            #     tid_biases = self.cluster_id_map_reverse[test_tuple[2]] - self.t_offset
            #     test_tuple.append(tid_biases)

    def gen_normal_laplacian_matrix(self):
        laplacian_mode = self.laplacian_mode
        assert "cluster" not in laplacian_mode and laplacian_mode != "None"

        self.L = dict()
        self.LI = dict()
        if laplacian_mode in ["PT2", "PT4", "TestPT"]:
            # Prepare matrix L and LI.
            if laplacian_mode == "TestPT":
                L = LI = sp.lil_matrix((self.n_sum, self.n_sum), dtype=np.float64)  # (n * l)
            else:
                A = get_A_2(self.R_pt)  # (n * l)
                L: sp.spmatrix = get_laplacian(A)  # Normalized laplacian matrix of A. (n+l * n+l)
                LI: sp.spmatrix = L + sp.eye(L.shape[0])  # A + I. where I is the identity matrix.

            # Prepare the needed matrices.
            if laplacian_mode in ["PT2", "TestPT"]:
                self.L["complete"] = L
                self.LI["complete"] = LI
            if laplacian_mode in ["PT4", "TestPT"]:
                self.L["p"] = L[:self.n_playlist, :]
                self.L["t"] = L[self.n_playlist:, :]
                self.LI["p"] = LI[:self.n_playlist, :]
                self.LI["t"] = LI[self.n_playlist:, :]
        elif laplacian_mode == "UT":
            A: sp.spmatrix = get_A_2(self.R_ut)  # (m * n)
            L: sp.spmatrix = get_laplacian(A)  # Normalized laplacian matrix of A. (m+n * m+n)
            LI: sp.spmatrix = L + sp.eye(L.shape[0])  # A + I. where I is the identity matrix.

            self.L["u"] = L[:self.n_user, :]
            self.L["t"] = L[self.n_user:, :]
            self.LI["u"] = LI[:self.n_user, :]
            self.LI["t"] = LI[self.n_user:, :]
        elif laplacian_mode in ["TestUPT", "UPT"]:
            if laplacian_mode == "UPT":
                A, A_alpha = get_A_3(self.R_up, self.R_ut, self.R_pt, self.alpha)  # (m+n+l * m+n+l)
                L: sp.spmatrix = get_laplacian(A, A_alpha)  # Normalized laplacian matrix of A. (m+n+l * m+n+l)
                LI: sp.spmatrix = L + sp.eye(L.shape[0])  # A + I. where I is the identity matrix.
            else:  # Mode is "TestUPT".
                L = LI = sp.lil_matrix((self.n_sum, self.n_sum), dtype=np.float64)

            self.L["u"] = L[:self.n_user, :]
            self.L["p"] = L[self.n_user:self.n_user + self.n_playlist, :]
            self.L["t"] = L[self.n_user + self.n_playlist:, :]
            self.LI["u"] = LI[:self.n_user, :]
            self.LI["p"] = LI[self.n_user:self.n_user + self.n_playlist, :]
            self.LI["t"] = LI[self.n_user + self.n_playlist:, :]

        if laplacian_mode.startswith("PT"):
            # 验证pt与R_pt是一致的
            for pid, tids in self.pt.items():
                for tid in tids:
                    assert self.R_pt[pid, tid] == 1

    def gen_cluster_parts_from_list(self, num_cluster):
        D = [[] for i in range(self.n_sum)]
        # Add edges for graph
        if "UPT" in self.laplacian_mode:
            for uid, pids in self.up.items():
                for pid in pids:
                    real_uid = uid
                    real_pid = pid + self.n_user
                    D[real_uid].append(real_pid)
            for pid, tids in self.pt.items():
                for tid in tids:
                    real_pid = pid + self.n_user
                    real_tid = tid + self.n_user + self.n_playlist
                    D[real_pid].append(real_tid)
            for uid, tids in self.ut.items():
                for tid in tids:
                    real_uid = uid
                    real_tid = tid + self.n_user + self.n_playlist
                    D[real_uid].append(real_tid)
        elif "PT" in self.laplacian_mode:
            for pid, tids in self.pt.items():
                for tid in tids:
                    real_pid = pid
                    real_tid = tid + self.n_playlist
                    D[real_pid].append(real_tid)
        elif "UT" in self.laplacian_mode:
            for uid, pids in self.up.items():
                for pid in pids:
                    real_uid = uid
                    real_pid = pid + self.n_user
                    D[real_uid].append(real_pid)
        else:
            raise Exception("Wrong laplacian mode %r" % self.laplacian_mode)

        # 进行分割
        t1 = time()
        (edgecuts, parts) = metis.part_graph(adjacency=D, nparts=num_cluster)
        print("分割图为%d个簇, 用了%d秒" % (num_cluster, time() - t1))
        print("There are %d clustered nodes." % len(parts))
        assert len(parts) == self.n_sum
        for part in parts:
            assert 0 <= part < num_cluster

        return parts

    def gen_cluster_parts(self, num_cluster):
        assert "cluster" in self.laplacian_mode
        print("生成cluster映射中...")
        # G的节点存储顺序: U, P, T
        # 比如: uid, uid... pid, pid... tid, tid...
        G = nx.Graph()
        G.add_nodes_from([i for i in range(self.n_sum)])

        # Add edges for graph
        if "UPT" in self.laplacian_mode:
            for uid, pids in self.up.items():
                for pid in pids:
                    real_uid = uid
                    real_pid = pid + self.n_user
                    G.add_edge(real_uid, real_pid)
            for pid, tids in self.pt.items():
                for tid in tids:
                    real_pid = pid + self.n_user
                    real_tid = tid + self.n_user + self.n_playlist
                    G.add_edge(real_pid, real_tid)
            for uid, tids in self.ut.items():
                for tid in tids:
                    real_uid = uid
                    real_tid = tid + self.n_user + self.n_playlist
                    G.add_edge(real_uid, real_tid)
        elif "PT" in self.laplacian_mode:
            for pid, tids in self.pt.items():
                for tid in tids:
                    real_pid = pid
                    real_tid = tid + self.n_playlist
                    G.add_edge(real_pid, real_tid)
        elif "UT" in self.laplacian_mode:
            for uid, pids in self.up.items():
                for pid in pids:
                    real_uid = uid
                    real_pid = pid + self.n_user
                    G.add_edge(real_uid, real_pid)
        else:
            raise Exception("Wrong laplacian mode %r" % self.laplacian_mode)

        # 进行分割
        t1 = time()
        (edgecuts, parts) = metis.part_graph(G, num_cluster)
        print("分割图为%d个簇, 用了%d秒" % (num_cluster, time() - t1))
        print("There are %d clustered nodes." % len(parts))
        assert len(parts) == self.n_sum
        for part in parts:
            assert 0 <= part < num_cluster

        return parts

    def gen_cluster_map(self, parts, num_cluster):
        laplacian_mode = self.laplacian_mode
        assert "cluster" in laplacian_mode

        # 对每个id, 放置到对应cluster的u, p或t类别中.
        # temp_clusters存储的id都是经过cluster偏移计算的.
        temp_clusters = dict()
        for id, cluster_no in enumerate(parts):
            if cluster_no not in temp_clusters:
                temp_clusters[cluster_no] = dict()

            # Decide the key of entity name to put in dict.
            if "UPT" in laplacian_mode:
                if id < self.n_user:
                    entity_name = "u"
                elif id < self.n_user + self.n_playlist:
                    entity_name = "p"
                elif id < self.n_sum:
                    entity_name = "t"
                else:
                    raise Exception("Unexpected id: %d" % id)
            elif "PT" in laplacian_mode:
                if "PT2" in laplacian_mode:
                    entity_name = "complete"
                elif "PT4" in laplacian_mode:
                    if id < self.n_playlist:
                        entity_name = "p"
                    elif id < self.n_sum:
                        entity_name = "t"
                    else:
                        raise Exception("Unexpected id: %d" % id)
                else:
                    raise Exception("Unexpected laplacian_mode: %r" % laplacian_mode)
            elif "UT" in laplacian_mode:
                if id < self.n_user:
                    entity_name = "u"
                elif self.n_user <= id < self.n_user + self.n_track:
                    entity_name = "t"
                else:
                    raise Exception("Unexpected id: %d" % id)
            else:
                raise Exception("Unexpected laplacian_mode: %r" % laplacian_mode)

            assert entity_name != None
            if entity_name not in temp_clusters[cluster_no]:
                temp_clusters[cluster_no][entity_name] = [id]
            else:
                temp_clusters[cluster_no][entity_name].append(id)
        assert len(list(temp_clusters.keys())) == num_cluster  # 保证总共的num_cluster个簇都分到了节点

        # 设置cluster_id_map
        # 和clusters中每个cluster的ids, offset和size.
        # 还有每个cluster每种entity的ids, offset和size.
        # 存储的id都是经过cluster偏移计算的.
        clusters = dict()
        cluster_id_map = np.array([-1 for i in range(self.n_sum)])
        cluster_id_map_reverse = np.array([-1 for i in range(self.n_sum)])
        map_id = 0
        cluster_offset = 0
        entity_num_per_cluster = -1
        for cluster_no, temp_cluster in temp_clusters.items():
            # 计算存储cluster的id和每类实体的大小
            assert cluster_no not in clusters
            cluster = clusters[cluster_no] = dict()

            cluster["ids"] = []
            entity_offset = 0
            cur_entity_num = len(list(temp_cluster.keys()))
            if entity_num_per_cluster != -1:
                assert entity_num_per_cluster == cur_entity_num, ("%d %d" % (entity_num_per_cluster, cur_entity_num))
            else:
                entity_num_per_cluster = cur_entity_num

            for entity_name, entity_ids in temp_cluster.items():
                assert entity_name not in cluster
                assert len(entity_ids) != 0

                cluster[entity_name] = dict()
                cluster[entity_name]["ids"] = np.array(entity_ids)
                cluster[entity_name]["offset"] = entity_offset
                cluster[entity_name]["size"] = len(entity_ids)
                cluster["ids"].extend(entity_ids)
                entity_offset += len(entity_ids)

            cluster["ids"] = np.array(cluster["ids"])
            # 计算存储cluster的size
            cluster_size = 0
            for entity_name, entity_ids in temp_cluster.items():
                cluster_size += len(entity_ids)
                for id in entity_ids:
                    assert cluster_id_map[id] == -1 and cluster_id_map_reverse[map_id] == -1
                    cluster_id_map[id] = map_id
                    cluster_id_map_reverse[map_id] = id
                    map_id += 1
            cluster["size"] = cluster_size

            # 存储cluster偏移
            cluster["offset"] = cluster_offset
            cluster_offset += cluster_size
        assert -1 not in cluster_id_map
        return cluster_id_map, cluster_id_map_reverse, clusters

    def get_cluster_A(self, cluster, size):
        # 太慢
        laplacian_mode = self.laplacian_mode
        A = sp.dok_matrix((size, size), dtype=np.float64)
        if "UPT" in laplacian_mode:
            for uid in cluster["u"]["ids"]:
                x = np.where(cluster["ids"] == uid)[0][0]
                old_uid = uid
                up_pids = self.up[old_uid] + self.n_user
                ut_tids = self.ut[old_uid] + self.n_user + self.n_playlist
                pids_in_cluster = np.intersect1d(cluster["p"]["ids"], up_pids)
                tids_in_cluster = np.intersect1d(cluster["t"]["ids"], ut_tids)
                for pid in pids_in_cluster:
                    y = np.where(cluster["ids"] == pid)[0][0]
                    A[x, y] = 1
                    A[y, x] = 1
                for tid in tids_in_cluster:
                    y = np.where(cluster["ids"] == tid)[0][0]
                    A[x, y] = 1
                    A[y, x] = 1

            for pid in cluster["p"]["ids"]:
                x = np.where(cluster["ids"] == pid)[0][0]
                old_pid = pid - self.n_user
                pt_tids = self.pt[old_pid] + self.n_user + self.n_playlist
                tids_in_cluster = np.intersect1d(cluster["t"]["ids"], pt_tids)
                for tid in tids_in_cluster:
                    y = np.where(cluster["ids"] == tid)[0][0]
                    A[x, y] = 1
                    A[y, x] = 1

        elif "PT" in laplacian_mode:
            A = sp.dok_matrix((size, size), dtype=np.float64)
            for old_pid in cluster["p"]["ids"]:
                old_pid = old_pid
                x = cluster["ids"].index(old_pid)
                for tid in self.pt[old_pid]:
                    new_tid = tid + self.n_playlist
                    if new_tid not in cluster["ids"]:
                        continue
                    y = cluster["ids"].index(new_tid)
                    A[x, y] = 1
                    A[y, x] = 1
        elif "UT" in laplacian_mode:
            for uid in cluster["u"]["ids"]:
                old_uid = uid
                for tid in self.ut[old_uid]:
                    new_tid = tid + self.n_user
                    if new_tid not in cluster["ids"]:
                        continue
                    x = cluster["ids"].index(uid)
                    y = cluster["ids"].index(new_tid)
                    A[x, y] = 1
                    A[y, x] = 1
        return A

    def set_L_and_LI_for_cluster(self, cluster, L, LI):
        # 0.004608409020173726 laplacian矩阵比较稀疏
        n_user = cluster["u"]["size"]
        n_playlist = cluster["p"]["size"]
        n_track = cluster["t"]["size"]
        laplacian_mode = self.laplacian_mode
        if "UPT" in laplacian_mode:
            cluster["L"] = {
                "u": L[:n_user, :],
                "p": L[n_user:n_user + n_playlist, :],
                "t": L[n_user + n_playlist:, :]
            }
            cluster["LI"] = {
                "u": LI[:n_user, :],
                "p": LI[n_user:n_user + n_playlist, :],
                "t": LI[n_user + n_playlist:, :]
            }
        elif "PT" in laplacian_mode:
            cluster["L"] = {
                "p": L[:n_playlist, :],
                "t": L[n_playlist:, :]
            }
            cluster["LI"] = {
                "p": LI[:n_playlist, :],
                "t": LI[n_playlist:, :]
            }
        elif "UT" in laplacian_mode:
            cluster["L"] = {
                "u": L[:n_user, :],
                "t": L[n_user:, :]
            }
            cluster["LI"] = {
                "u": LI[:n_user, :],
                "t": LI[n_user:, :]
            }
        # for entity_name, l_matrix in cluster["L"].items():
        #     n_nonzero = l_matrix.getnnz()
        #     n_whole = l_matrix.get_shape()[0] * l_matrix.get_shape()[1]
        #     print(n_nonzero / n_whole)

    def do_gen_cluster_laplacian_matrix(self, clusters):
        laplacian_mode = self.laplacian_mode
        assert "cluster" in laplacian_mode
        t_func = time()
        for cluster_no, cluster in clusters.items():
            offset = cluster["offset"]
            size = cluster["size"]

            # get matrix A.
            t_cluster = time()
            A = self.get_cluster_A(cluster, size)
            t1 = time()
            L: sp.spmatrix = get_laplacian(A, log=False)  # Normalized laplacian matrix of cluster A.
            t2 = time()
            LI: sp.spmatrix = L + sp.eye(L.shape[0])  # A + I. where I is the identity matrix.
            t3 = time()
            self.set_L_and_LI_for_cluster(cluster, L, LI)
            t4 = time()
            # print("%f, %f, %f, %f" % (t1 - t_cluster, t2 - t1, t3 - t2, t4 - t3))
            # print("Laplacian matrix for cluster %d used %f seconds" % (cluster_no, time() - t_cluster))
        print("Generate cluster laplacian matrices used %f seconds" % (time() - t_func))

    def read_cluster_laplacian_matrix(self, num_cluster, cluster_map_filepath):
        # 生成id map和每个part的长度
        # 每一part都有R, L, LI. 视情况会有up, pt.

        # 分簇
        parts = self.read_cluster_parts(num_cluster, cluster_map_filepath)
        # parts = self.gen_cluster_parts(num_cluster)

        # 生成id映射
        self.cluster_id_map, self.cluster_id_map_reverse, self.clusters = self.gen_cluster_map(parts, num_cluster)

        # 为每个cluster生成L, LI. 视情况会有up, pt.
        self.do_gen_cluster_laplacian_matrix(self.clusters)

    def do_read_cluster_parts(self, filepath):
        parts = []
        with open(filepath) as f:
            line = f.readline()
            while line:
                ids = [int(s) for s in line.split() if s.isdigit()]
                assert len(ids) == 2

                id = ids[0]
                map_id = ids[1]
                parts.append(map_id)

                line = f.readline()
        return parts

    def read_cluster_parts(self, num_cluster, filepath):
        """
        读取簇映射数据文件, 若不存在则生成之.
        :param filepath: Cluster map filepath.
        :return: 无
        """
        if not os.path.exists(filepath):
            parts = self.gen_cluster_parts(num_cluster)
            assert len(parts) == self.n_sum
            # Write cluster map file.
            with open(filepath, "w") as f:
                for id, part in enumerate(parts):
                    f.write("%d %d\n" % (id, part))
            print("已生成cluster映射文件...")
            return parts
        t1 = time()
        print("存在已有的cluster映射文件, 开始读取...")
        parts = self.do_read_cluster_parts(filepath)
        assert len(parts) == self.n_sum
        print("读取cluster映射文件%s完成, 用了%f秒" % (filepath, time() - t1))
        return parts

    def read_entity_num(self, train_filepath):
        # Read and print statistics.
        # Set n_user, n_playlist, n_track
        with open(train_filepath) as f:
            head_title = f.readline()
            ids = [int(i) for i in head_title.split(' ') if i.isdigit()]
            self.n_user, self.n_playlist, self.n_track = ids[0], ids[1], ids[2]

        # Set n_sum
        if self.laplacian_mode == "None" or "UPT" in self.laplacian_mode:
            self.n_sum = self.n_user + self.n_playlist + self.n_track
        elif "PT" in self.laplacian_mode:
            self.n_sum = self.n_playlist + self.n_track
        elif "UT" in self.laplacian_mode:
            self.n_sum = self.n_user + self.n_track
        else:
            raise Exception("Wrong laplacian mode: %r" % self.laplacian_mode)
        self.print_statistics()

    def read_events_file(self, event_filepath):
        t_event = time()
        with open(event_filepath) as f:
            head_title = f.readline()

            line = f.readline()
            while line:
                ids = [int(i) for i in line.split(' ') if i.isdigit()]
                uid, tids = ids[0], ids[1:]
                for tid in tids:
                    self.R_ut[uid, tid] = 1
                line = f.readline()
        print("Used %d seconds. Have read R_ut." % (time() - t_event))

    def read_train_file(self, train_filepath, reductive_ut=True):
        # 设置R_up, R_pt, R_ui, up, pt
        t_upt = time()
        self.n_train = 0
        with open(train_filepath) as f:
            head_title = f.readline()

            line = f.readline()
            while line:
                ids = [int(i) for i in line.split(' ') if i.isdigit()]
                assert len(ids) >= 3
                uid, pid, tids = ids[0], ids[1], ids[2:]

                # 非分簇模式时, 全局R矩阵可用
                if "cluster" not in self.laplacian_mode:
                    # Add element to R_up
                    self.R_up[uid, pid] = 1
                    # Add element to R_pt
                    for tid in tids:
                        self.R_pt[pid, tid] = 1
                        if reductive_ut:
                            self.R_ut[uid, tid] = 1

                # Add element to up
                if uid not in self.up:
                    self.up[uid] = [pid]
                else:
                    self.up[uid].append(pid)
                # Add element to pt
                assert pid not in self.pt
                self.pt[pid] = tids

                # Add element to ut
                if uid not in self.ut:
                    self.ut[uid] = tids
                else:
                    self.ut[uid].extend(tids)

                # Add number of train.
                self.n_train += len(tids)

                line = f.readline()
        for uid in self.up.keys():
            self.up[uid] = np.array(self.up[uid])
        for pid in self.pt.keys():
            self.pt[pid] = np.array(self.pt[pid])
        for uid in self.ut.keys():
            self.ut[uid] = np.array(self.ut[uid])
        print("Used %d seconds. Have read matrix R_up, matrix R_pt, dict up and dict pt." % (time() - t_upt))

    def read_test_file(self, test_filepath):
        # 设置test_set
        t_test_set = time()
        with open(test_filepath) as f:
            line = f.readline()
            while line:
                ids = [int(i) for i in line.split(' ') if i.isdigit()]
                uid, pid, tid = ids[0], ids[1], ids[2]
                test_tuple = [uid, pid, tid]
                self.test_set.append(test_tuple)
                line = f.readline()
        print("Used %d seconds. Have read test set." % (time() - t_test_set))

    def sample_neg_track_for_playlist(self, playlist: int, pt: dict, t_offset, n_track):
        pos_tracks = pt[playlist]
        neg_track = np.random.randint(t_offset, t_offset + n_track)
        if "cluster" in self.laplacian_mode:
            neg_track = self.cluster_id_map[neg_track]
        while neg_track in pos_tracks:
            neg_track = np.random.randint(t_offset, t_offset + n_track)
            if "cluster" in self.laplacian_mode:
                neg_track = self.cluster_id_map[neg_track]
        return neg_track

    def next_batch_ut(self):
        pass

    def next_batch_pt(self) -> dict:
        batch = {
            "playlists": [np.random.randint(0, self.n_playlist) for _ in range(self.batch_size)],
            "pos_tracks": [],
            "neg_tracks": []
        }

        for playlist in batch["playlists"]:
            pos_track = sample_pos_track_for_playlist(playlist, self.pt)
            neg_track = self.sample_neg_track_for_playlist(playlist, self.pt, self.t_offset, self.n_track)

            batch["pos_tracks"].append(pos_track)
            batch["neg_tracks"].append(neg_track)

        return batch

    def next_batch_upt(self):
        batch = {
            "users":  np.array([np.random.randint(0, self.n_user) for _ in range(self.batch_size)]),
            "playlists": [],
            "pos_tracks": [],
            "neg_tracks": []
        }

        if "cluster" in self.laplacian_mode:
            batch["users"] = self.cluster_id_map[batch["users"]]
            batch["pos_track_biases"] = []
            batch["neg_track_biases"] = []

        for user in batch["users"]:
            playlist = np.random.choice(self.up[user], 1)[0]
            pos_track = sample_pos_track_for_playlist(playlist, self.pt)
            neg_track = self.sample_neg_track_for_playlist(playlist, self.pt, self.t_offset, self.n_track)

            batch["playlists"].append(playlist)
            batch["pos_tracks"].append(pos_track)
            batch["neg_tracks"].append(neg_track)

            if "cluster" in self.laplacian_mode:
                batch["pos_track_biases"].append(self.cluster_id_map_reverse[pos_track] - self.t_offset)
                batch["neg_track_biases"].append(self.cluster_id_map_reverse[neg_track] - self.t_offset)

        return batch

    def sample_negative_item(self, observed_tids):
        neg_tid = np.random.randint(self.t_offset, self.t_offset + self.n_track)
        if "cluster" in self.laplacian_mode:
            neg_tid = self.cluster_id_map[neg_tid]
        while neg_tid in observed_tids:
            neg_tid = np.random.randint(self.t_offset, self.t_offset + self.n_track)
            if "cluster" in self.laplacian_mode:
                neg_tid = self.cluster_id_map[neg_tid]
        return neg_tid

    def sample_hundred_negative_item(self, pid):
        observed_tids = self.pt[pid]
        neg_tids = []

        for _ in range(100):
            neg_tid = self.sample_negative_item(observed_tids)
            assert neg_tid not in observed_tids
            while neg_tid in neg_tids:
                neg_tid = self.sample_negative_item(observed_tids)
                assert neg_tid not in observed_tids

            neg_tids.append(neg_tid)

        return neg_tids

    def print_statistics(self):
        print('n_users=%d, n_playlists=%d, n_tracks=%d' % (self.n_user, self.n_playlist, self.n_track))
        # num interactions/sparsity