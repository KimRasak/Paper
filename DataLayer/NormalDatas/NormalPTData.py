from Common import DatasetNum
from DataLayer.NormalDatas.NormalData import NormalData

import numpy as np
import scipy.sparse as sp


class NormalPTData(NormalData):
    def _get_laplacian_matrices(self, train_data, data_set_num: DatasetNum):
        # The neighbourhood matrix will be [P T] vertically.
        playlist_num = data_set_num.playlist
        track_num = data_set_num.track
        total_size = playlist_num + track_num

        # Init laplacian matrix.
        L_matrix = sp.dok_matrix((total_size, total_size), dtype=np.float64)
        track_offset = playlist_num
        for pid, tids in self.pt.items():
            for tid in tids:
                x = pid
                y = tid + track_offset
                L_matrix[x, y] = 1
                L_matrix[y, x] = 1
        LI_matrix = L_matrix + sp.eye(L_matrix.shape[0])
        return {
            "L": L_matrix,
            "LI": LI_matrix
        }

    def _get_data_sum(self, data_set_num: DatasetNum):
        return data_set_num.playlist + data_set_num.track

    def _init_relation_dict(self):
        # init up dict and pt dict.
        self.pt = dict()

        for uid, user in self.train_data.items():
            for pid, tids in user.items():
                # Add element to pt
                assert pid not in self.pt
                self.pt[pid] = np.array(tids)

        # Init test_pt dict.
        self.test_pt = dict()
        for uid, user in self.test_data.items():
            for pid, tids in user.items():
                self.test_pt[pid] = tids

    def _init_relation_matrix(self):
        self.R_pt = sp.dok_matrix((self.data_set_num.playlist, self.data_set_num.track), dtype=np.float64)

        for uid, user in self.train_data.items():
            for pid, tids in user.items():
                for tid in tids:
                    self.R_pt[pid, tid] = 1

    def _get_batch_num(self):
        train_num = self.R_pt.getnnz()
        return int(np.ceil(train_num / self.batch_size))

    def get_batches(self):
        def __sample_neg_track_for_playlist(pid: int):
            train_pids = self.pt[pid]
            test_pids = self.test_pt[pid]

            neg_tid = np.random.randint(0, self.data_set_num.track)
            while (neg_tid in train_pids) or (neg_tid in test_pids):
                # Random pick again.
                neg_tid = np.random.randint(0, self.data_set_num.track)
            return neg_tid

        for batch_no in range(self._get_batch_num()):
            batch = {
                "size": self.batch_size,
                "pids": [np.random.randint(0, self.data_set_num.playlist) for _ in range(self.batch_size)],
                "pos_tids": [],
                "neg_tids": []
            }  # Only "pos_tids" and "neg_tids" need to be initialized.

            for pid in batch["playlists"]:
                pos_tid = np.random.choice(self.pt[pid], 1)[0]
                neg_tid = __sample_neg_track_for_playlist(pid)

                batch["pos_tids"].append(pos_tid)
                batch["neg_tids"].append(neg_tid)

            for k, v in batch.items():
                if isinstance(v, list):
                    batch[k] = np.array(batch[k])

            yield batch

    def sample_negative_test_track_ids(self, uid, pid):
        # Sample 100 track id, and ensure each tid doesn't appear in train data and test data.
        track_num = self.data_set_num.track
        negative_test_tids = {
            "num": 0,
            "id": np.array([], dtype=int),
        }
        while negative_test_tids["num"] < 100:
            picked_tid = np.random.randint(0, track_num)
            if (not picked_tid in self.train_data[uid][pid]) \
                    and (not picked_tid in self.test_data[uid][pid]):
                negative_test_tids["num"] += 1
                negative_test_tids["id"] = np.append(negative_test_tids["id"], picked_tid)

        return negative_test_tids