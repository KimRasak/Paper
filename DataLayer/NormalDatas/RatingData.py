
import numpy as np
import scipy.sparse as sp
from Common import DatasetNum
from DataLayer.NormalDatas.NormalData import NormalData


class RatingData(NormalData):
    def _init_relation_matrix(self):
        user_num = self.data_set_num.playlist
        item_num = self.data_set_num.track

        self.R = sp.dok_matrix((user_num, item_num), dtype=np.float64)

        for user_id, user in self.train_data.items():
            for item_id, score in user.items():
                self.R[user_id, item_id] = 1

    def _get_batch_num(self):
        train_num = self.R.getnnz()
        return int(np.ceil(train_num / self.batch_size))

    def get_batches(self):
        for batch_no in range(self._get_batch_num()):
            user_num = self.data_set_num.playlist  # Simple for use.

            batch = {
                "size": self.batch_size,
                "user_ids": [np.random.randint(0, user_num) for _ in range(self.batch_size)],
                "item_ids": [],
                "scores": []
            }  # Only "pos_tids" and "neg_tids" need to be initialized.

            for user_id in batch["user_ids"]:
                rand_index = np.random.randint(0, len(self.ui[user_id]))
                item_id, score = self.ui[user_id][rand_index]
                batch["item_ids"].append(item_id)
                batch["scores"].append(score)

            for k, v in batch.items():
                if isinstance(v, list):
                    batch[k] = np.array(batch[k])

            yield batch

    def _get_laplacian_matrices(self, train_data, data_set_num):
        # The neighbourhood matrix will be [P T] vertically.
        user_num = data_set_num.playlist
        item_num = data_set_num.track
        total_size = user_num + item_num

        # Init laplacian matrix.
        L_matrix = sp.dok_matrix((total_size, total_size), dtype=np.float64)
        item_offset = user_num
        for user_id, pairs in self.ui.items():
            for item_id, score in pairs:
                x = user_id
                y = item_id + item_offset
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
        # init ui dict.
        self.ui = dict()

        for user_id, user in self.train_data.items():
            # Add element to ui
            assert user_id not in self.ui
            self.ui[user_id] = [(item_id, score) for item_id, score in user.items()]

    def sample_negative_test_track_ids(self, uid, pid):
        # Not used.
        # We don't need to sample negative samples in dataset "rating".
        pass

