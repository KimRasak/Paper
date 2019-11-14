from DataLayer.NormalDatas.NormalData import NormalData
from Common import DatasetNum

import scipy.sparse as sp
import numpy as np


class NormalUPTData(NormalData):
    def _init_relation_dict(self):
        self.up = dict()
        self.pt = dict()
        self.ut = dict()

        for uid, user in self.train_data.items():
            for pid, tids in user.items():
                # Add element to up
                if uid not in self.up:
                    self.up[uid] = np.array([pid])
                else:
                    self.up[uid] = np.append(self.up[uid], pid)

                # Add element to pt
                assert pid not in self.pt
                self.pt[pid] = np.array(tids)

                # Add element to ut
                if uid not in self.ut:
                    self.ut[uid] = np.array(tids)
                else:
                    self.ut[uid] = np.concatenate((self.ut[uid], tids))

    def _init_relation_matrix(self):
        self.R_up = sp.dok_matrix((self.data_set_num.user, self.data_set_num.playlist), dtype=np.float64)
        self.R_ut = sp.dok_matrix((self.data_set_num.user, self.data_set_num.track), dtype=np.float64)
        self.R_pt = sp.dok_matrix((self.data_set_num.playlist, self.data_set_num.track), dtype=np.float64)

        for uid, user in self.train_data.items():
            for pid, tids in user.items():
                self.R_up[uid, pid] = 1
                for tid in tids:
                    self.R_pt[pid, tid] = 1
                    self.R_ut[uid, tid] = 1

    def _get_batch_num(self):
        train_num = self.R_pt.getnnz()
        return int(np.ceil(train_num / self.batch_size))

    def _get_data_sum(self, data_set_num: DatasetNum):
        return data_set_num.user + data_set_num.playlist + data_set_num.track

    def get_batches(self):
        def __sample_neg_track_for_playlist(uid: int, pid: int):
            train_pids = self.train_data[uid][pid]
            test_pids = self.test_data[uid][pid]

            neg_tid = np.random.randint(0, self.data_set_num.track)
            while (neg_tid in train_pids) or (neg_tid in test_pids):
                # Random pick again.
                neg_tid = np.random.randint(0, self.data_set_num.track)
            return neg_tid

        for batch_no in range(self._get_batch_num()):
            batch = {
                "size": self.batch_size,
                "uids": [np.random.randint(0, self.data_set_num.user) for _ in range(self.batch_size)],
                "pids": np.array([np.random.randint(0, self.data_set_num.playlist) for _ in range(self.batch_size)], dtype=int),
                "pos_tids": np.array([], dtype=int),
                "neg_tids": np.array([], dtype=int)
            }  # Only "pos_tids" and "neg_tids" need to be initialized.

            for uid in batch["uids"]:
                pid = np.random.choice(self.up[uid], 1)[0]
                pos_tid = np.random.choice(self.pt[pid], 1)[0]
                neg_tid = __sample_neg_track_for_playlist(uid, pid)

                batch["pos_tids"] = np.append(batch["pos_tids"], pos_tid)
                batch["neg_tids"] = np.append(batch["neg_tids"], neg_tid)

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
            if (picked_tid not in self.train_data[uid][pid]) \
                    and (picked_tid not in self.test_data[uid][pid]):
                negative_test_tids["num"] += 1
                negative_test_tids["id"] = np.append(negative_test_tids["id"], picked_tid)

        return negative_test_tids