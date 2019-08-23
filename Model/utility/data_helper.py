import os
from time import time

import numpy as np
import scipy.sparse as sp

"""
数据读取来源
首先设置路径，尝试读取.mat文件。
若失败，则尝试读取train/test文件，并以此生成.mat文件。
若再失败，则抛出异常。

先试试直接读取train/test文件，得到矩阵的速度。
"""


def get_laplacian(A: sp.spmatrix):
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

    def get_symmetric_normalized_laplacian(adj: sp.spmatrix):
        D = get_D(adj)
        return D.dot(adj).dot(D).tocoo()

    L: sp.spmatrix = get_symmetric_normalized_laplacian(A)
    print('Used %d seconds. Get symmetric normalized laplacian matrix.' % (time() - t))
    return L.tocsr()


def get_A_3(R_up: sp.spmatrix, R_ut: sp.spmatrix, R_pt: sp.spmatrix):  # Get matrix "A" among user-playlist-track relationship.
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

    A = sp.lil_matrix((m+n+l, m+n+l), dtype=np.float32)
    A[:m, m:m+n] = R_up  # (m * n)
    A[:m, m+n:] = R_ut  # (m * l)
    A[m:m+n, m+n:] = R_pt  # (n * l)

    A[m:m+n, :m] = R_up.T  # (n * m)
    A[m+n:, :m] = R_ut.T  # (l * m)
    A[m+n:, m:m+n] = R_pt.T  # (l * n)

    print('Used %d seconds. Already create adjacency matrix(A_3). shape of A: %r' % (time() - t, A.shape))
    return A


def get_A_2(R: sp.spmatrix):  # Get matrix "A" between user-item relationship.
    t = time()
    m = R.shape[0]
    n = R.shape[1]

    A = sp.lil_matrix((m + n, m + n), dtype=np.float32)

    A[:m, m:] = R
    A[m:, :m] = R.T
    print('Used %d seconds. Already create adjacency matrix(A_2). shape of A: %r' % (time() - t, A.shape))
    return A

def sample_pos_track_for_playlist(playlist: int, pt: dict):
    pos_tracks = pt[playlist]
    return np.random.choice(pos_tracks, 1)[0]


def sample_neg_track_for_playlist(playlist: int, pt: dict, n_track):
    pos_tracks = pt[playlist]
    neg_track = np.random.randint(0, n_track)
    while neg_track in pos_tracks:
        neg_track = np.random.randint(0, n_track)
    return neg_track

class Data():
    def __init__(self, path, pick=True, laplacian_mode="PT", batch_size=256, reductive_ut=True, alpha=0.5):
        t0 = time()
        self.path = path
        self.batch_size = batch_size
        self.alpha = alpha

        if pick is True:
            print("{pick} == %r, Using picked playlist data. That is, you're using a sub-dataset" % pick)
            train_filepath = path + "/pick_train.txt"
            test_filepath = path + "/pick_test.txt"
            event_filepath = path + "/pick_events.txt"
        else:
            print("{pick} == %r, Using complete playlist data. That is, you're using a complete dataset" % pick)
            train_filepath = path + "/train.txt"
            test_filepath = path + "/test.txt"
            event_filepath = path + "/events.txt"

        # Read and print statistics.
        with open(train_filepath) as f:
            head_title = f.readline()
            ids = [int(i) for i in head_title.split(' ') if i.isdigit()]
            self.n_user, self.n_playlist, self.n_track = ids[0], ids[1], ids[2]
        self.print_statistics()

        self.R_up = sp.dok_matrix((self.n_user, self.n_playlist), dtype=np.float32)
        self.R_ut = sp.dok_matrix((self.n_user, self.n_track), dtype=np.float32)
        self.R_pt = sp.dok_matrix((self.n_playlist, self.n_track), dtype=np.float32)
        self.pt = {}  # Storing playlist-track relationship of training set.
        self.up = {}  # Storing user-playlist relationship of training set.
        self.test_set = {}  # Storing user-playlist-track test set.

        if laplacian_mode != "Test":
            # Initialize R_ut
            if reductive_ut:
                print("Using Reductive R_ut, not reading events data.")
            else:
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

            # Initialize matrix R_up, matrix R_pt, dict up and dict pt.
            t_upt = time()
            with open(train_filepath) as f:
                head_title = f.readline()

                line = f.readline()
                while line:
                    ids = [int(i) for i in line.split(' ') if i.isdigit()]
                    uid, pid, tids = ids[0], ids[1], ids[2:]
                    # Add element to R_up
                    self.R_up[uid, pid] = 1
                    # Add element to R_pt
                    for tid in tids:
                        self.R_pt[pid, tid] = 1
                        if reductive_ut:
                            self.R_ut[uid, tid] = self.alpha
                    # Add element to up
                    if uid not in self.up:
                        self.up[uid] = [pid]
                    else:
                        self.up[uid].append(pid)
                    # Add element to pt
                    self.pt[pid] = tids

                    line = f.readline()
            print("Used %d seconds. Have read matrix R_up, matrix R_pt, dict up and dict pt." % (time() - t_upt))

            # Initialize test_set
            t_test_set = time()
            with open(test_filepath) as f:
                line = f.readline()
                while line:
                    ids = [int(i) for i in line.split(' ') if i.isdigit()]
                    uid, pid, tids = ids[0], ids[1], ids[2:]
                    if uid not in self.test_set:
                        self.test_set[uid] = dict()
                    self.test_set[uid][pid] = tids
                    line = f.readline()
            print("Used %d seconds. Have read test set." % (time() - t_test_set))

        self.n_train = self.R_pt.getnnz()
        self.n_batch = int(np.ceil(self.n_train / self.batch_size)) * 4

        laplacian_modes = ["PT", "UT", "UPT", "None", "Test"]
        if laplacian_mode not in laplacian_modes:
            raise Exception("Wrong laplacian mode. Expected one of %r, got %r" % (laplacian_modes, laplacian_mode))
        print("laplacian_mode=%r, loading laplacian matrix." % laplacian_mode)
        self.laplacian_mode = laplacian_mode
        if laplacian_mode == "PT":
            self.A: sp.spmatrix = get_A_2(self.R_pt)  # (n * l)

            self.L: sp.spmatrix = get_laplacian(self.A)  # Normalized laplacian matrix of A. (n+l * n+l)
            self.L_p = self.L[:self.n_playlist, :]
            self.L_t = self.L[self.n_playlist:, :]

            self.LI: sp.spmatrix = self.L + sp.eye(self.L.shape[0])  # A + I. where I is the identity matrix.
            self.LI_p = self.LI[:self.n_playlist, :]
            self.LI_t = self.LI[self.n_playlist:, :]
        elif laplacian_mode == "UT":
            self.A: sp.spmatrix = get_A_2(self.R_ut)  # (m * n)

            self.L: sp.spmatrix = get_laplacian(self.A)  # Normalized laplacian matrix of A. (m+n * m+n)
            self.L_u = self.L[:self.n_user, :]
            self.L_t = self.L[self.n_user:, :]

            self.LI: sp.spmatrix = self.L + sp.eye(self.L.shape[0])  # A + I. where I is the identity matrix.
            self.LI_u = self.LI[:self.n_user, :]
            self.LI_t = self.LI[self.n_user:, :]
        elif laplacian_mode == "UPT":
            self.A: sp.spmatrix = get_A_3(self.R_up, self.R_ut, self.R_pt)  # (m+n+l * m+n+l)

            self.L: sp.spmatrix = get_laplacian(self.A)  # Normalized laplacian matrix of A. (m+n+l * m+n+l)
            self.L_u = self.L[:self.n_user, :]
            self.L_p = self.L[self.n_user:self.n_user+self.n_playlist, :]
            self.L_t = self.L[self.n_user+self.n_playlist:, :]

            self.LI: sp.spmatrix = self.L + sp.eye(self.L.shape[0])  # A + I. where I is the identity matrix.
            self.LI_u = self.LI[:self.n_user, :]
            self.LI_p = self.LI[self.n_user:self.n_user+self.n_playlist, :]
            self.LI_t = self.LI[self.n_user+self.n_playlist:, :]
        elif laplacian_mode == "Test":
            self.A: sp.spmatrix = sp.lil_matrix((self.n_user + self.n_playlist + self.n_track, self.n_user + self.n_playlist + self.n_track), dtype=np.float32)
            self.L: sp.spmatrix = sp.lil_matrix((self.n_user + self.n_playlist + self.n_track, self.n_user + self.n_playlist + self.n_track), dtype=np.float32)
            self.L_u = self.L[:self.n_user, :]
            self.L_p = self.L[self.n_user:self.n_user + self.n_playlist, :]
            self.L_t = self.L[self.n_user + self.n_playlist:, :]

            self.LI: sp.spmatrix = self.L + sp.eye(self.L.shape[0])  # A + I. where I is the identity matrix.
            self.LI_u = self.LI[:self.n_user, :]
            self.LI_p = self.LI[self.n_user:self.n_user + self.n_playlist, :]
            self.LI_t = self.LI[self.n_user + self.n_playlist:, :]
            # self.A: sp.spmatrix = sp.lil_matrix((self.n_playlist + self.n_track, self.n_user + self.n_track), dtype=np.float32)  # (n * l)
            #
            # self.L: sp.spmatrix = sp.lil_matrix((self.n_playlist + self.n_track, self.n_playlist + self.n_track), dtype=np.float32)  # Normalized laplacian matrix of A. (n+l * n+l)
            # self.L_p = self.L[:self.n_playlist, :]
            # self.L_t = self.L[self.n_playlist:, :]
            #
            # self.LI: sp.spmatrix = self.L + sp.eye(self.L.shape[0])  # A + I. where I is the identity matrix.
            # self.LI_p = self.LI[:self.n_playlist, :]
            # self.LI_t = self.LI[self.n_playlist:, :]
        print("Read data used %d seconds in all." % (time() - t0))
        # self.laplacian_ut, laplacian_pt = self.get_laplacian(self.R_ut.tolil()), self.get_laplacian(self.R_pt.tolil())

    def next_batch(self):
        if self.batch_size > self.n_user:
            raise Exception("Batch size too large(batch size > number of users).")

        users = [np.random.randint(0, self.n_user) for _ in range(self.batch_size)]
        playlists = []
        pos_tracks = []
        neg_tracks = []

        # Randomly choose a playlist for the user, and the positive track、negative track.
        for user in users:
            playlist = np.random.choice(self.up[user], 1)[0]
            pos_track = sample_pos_track_for_playlist(playlist, self.pt)
            neg_track = sample_neg_track_for_playlist(playlist, self.pt, self.n_track)

            playlists.append(playlist)
            pos_tracks.append(pos_track)
            neg_tracks.append(neg_track)

        return users, playlists, pos_tracks, neg_tracks

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
            neg_track = sample_neg_track_for_playlist(playlist, self.pt, self.n_track)

            batch["pos_tracks"].append(pos_track)
            batch["neg_tracks"].append(neg_track)

        return batch

    def next_batch_upt(self):
        batch = {
            "users":  [np.random.randint(0, self.n_user) for _ in range(self.batch_size)],
            "playlists": [],
            "pos_tracks": [],
            "neg_tracks": []
        }

        for user in batch["users"]:
            playlist = np.random.choice(self.up[user], 1)[0]
            pos_track = sample_pos_track_for_playlist(playlist, self.pt)
            neg_track = sample_neg_track_for_playlist(playlist, self.pt, self.n_track)

            batch["playlists"].append(playlist)
            batch["pos_tracks"].append(pos_track)
            batch["neg_tracks"].append(neg_track)

        return batch

    def sample_negative_item(self, observed_tids):
        neg_tid = np.random.randint(0, self.n_track)
        while neg_tid in observed_tids:
            neg_tid = np.random.randint(0, self.n_track)
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