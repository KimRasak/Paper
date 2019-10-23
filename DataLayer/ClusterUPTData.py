from DataLayer.ClusterData import ClusterData
from DataLayer.NormalData import NormalData
from Common import DatasetNum


class ClusterUPTData(ClusterData):
    def __init__(self, base_data_path, use_reductive_ut_data=True, alpha_ut=1):
        super().__init__(base_data_path)

    def __init_relation_dict(self, train_data):
        self.up = dict()
        self.pt = dict()
        self.ut = dict()

        for uid, user in train_data.items():
            for pid, tids in user.items():
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

    def __get_data_sum(self, data_set_num: DatasetNum):
        return data_set_num.user + data_set_num.playlist + data_set_num.track