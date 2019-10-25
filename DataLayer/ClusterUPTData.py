import numpy as np

from DataLayer.ClusterData import ClusterData
from DataLayer.NormalData import NormalData
from Common import DatasetNum


class ClusterUPTData(ClusterData):
    def __init__(self, data_set_name, cluster_strategy):
        super().__init__(data_set_name, cluster_strategy)

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

    def __gen_global_id_cluster_id_map(self, parts, data_set_num: DatasetNum):
        # Init clusters
        cluster_num = np.max(parts)
        cluster_sizes = np.zeros((cluster_num, ), dtype=int)
        global_id_cluster_id_map = np.zeros((len(parts), ), dtype=int)

        for global_id, cluster_number in enumerate(parts):
            assert global_id_cluster_id_map[global_id] is 0
            global_id_cluster_id_map[global_id] = cluster_sizes[cluster_number]
            cluster_sizes[cluster_number] += 1

        return cluster_sizes, global_id_cluster_id_map

    def __map_entity_id_to_global_id(self, entity_id, data_set_num, entity_name):
        if entity_name not in [self.ENTITY_USER, self.ENTITY_PLAYLIST, self.ENTITY_TRACK]:
            raise Exception("Wrong entity name!")

        if entity_name is self.ENTITY_USER:
            return entity_id
        elif entity_name is self.ENTITY_PLAYLIST:
            return data_set_num.user + entity_id
        elif entity_id is self.ENTITY_TRACK:
            return data_set_num.user + data_set_num.playlist + entity_id

    def __gen_train_tuples(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
        cluster_num = np.max(parts)
        cluster_pos_train_tuples = [dict() for i in range(cluster_num)]

        for pos_train_tuple in cluster_pos_train_tuples:
            pos_train_tuple.update({
                "user_X": np.array([]),
                "playlist_X": np.array([]),
                "pos_track_X": np.array([]),
                "pos_track_bias_X": np.array([])
                # "neg_track_X": [],
                # "neg_track_bias_X": []
            })

        def ids_are_in_same_cluster(global_uid, global_pid, global_tid, parts):
            # Check if the connection exists.
            user_cluster_number, playlist_cluster_number, track_cluster_number = \
                parts[global_uid], parts[global_pid], parts[global_tid]

            return user_cluster_number == playlist_cluster_number == track_cluster_number, user_cluster_number

        for entity_uid, user in train_data.items():
            for entity_pid, entity_tids in user.items():
                for entity_tid in entity_tids:
                    # for each u-p-t tuple, add to train tuples if their connection exists.

                    # Get global ids:
                    global_uid, global_pid, global_tid = \
                        self.__map_entity_id_to_global_id(entity_uid, data_set_num, self.ENTITY_USER), \
                        self.__map_entity_id_to_global_id(entity_pid, data_set_num, self.ENTITY_PLAYLIST), \
                        self.__map_entity_id_to_global_id(entity_tid, data_set_num, self.ENTITY_TRACK)

                    are_same_cluster, cluster_number = ids_are_in_same_cluster(global_uid, global_pid, global_tid, parts)
                    if not are_same_cluster:  # The connection exists.
                        continue
                    pos_train_tuple = cluster_pos_train_tuples[cluster_number]

                    cluster_uid, cluster_pid, cluster_tid = \
                        global_id_cluster_id_map[global_uid], global_id_cluster_id_map[global_pid], global_id_cluster_id_map[global_tid]

                    np.append(pos_train_tuple["user_X"], cluster_uid)
                    np.append(pos_train_tuple["playlist_X"], cluster_pid)
                    np.append(pos_train_tuple["pos_track_X"], cluster_tid)
                    np.append(pos_train_tuple["pos_track_bias_X"], entity_tid)
        return cluster_pos_train_tuples

    def __gen_cluster_track_ids(self, parts, data_set_num: DatasetNum, global_id_cluster_id_map):
        cluster_num = np.max(parts)
        cluster_track_ids = [{"global_id": np.array([]), "cluster_id": np.array([])} for _ in range(cluster_num)]

        tid_offset = data_set_num.user + data_set_num.playlist
        for entity_tid in range(data_set_num.track):
            global_tid = entity_tid + tid_offset
            cluster_tid = global_id_cluster_id_map[global_tid]
            cluster_number = parts[global_tid]
            np.append(cluster_track_ids[cluster_number]["global_id"], global_tid)
            np.append(cluster_track_ids[cluster_number]["cluster_id"], cluster_tid)

        return cluster_track_ids

    def __gen_test_tuples(self, test_list, data_set_num: DatasetNum, global_id_cluster_id_map):
        # TODO
        test_tuples = {
            "user_X": np.array([]),
            "playlist_X": np.array([]),
            "track_X": np.array([]),
            "track_bias_x": np.array([])
        }

        for entity_uid, entity_pid, entity_tid in test_list:
            # Get global ids:
            global_uid, global_pid, global_tid = \
                self.__map_entity_id_to_global_id(entity_uid, data_set_num, self.ENTITY_USER), \
                self.__map_entity_id_to_global_id(entity_pid, data_set_num, self.ENTITY_PLAYLIST), \
                self.__map_entity_id_to_global_id(entity_tid, data_set_num, self.ENTITY_TRACK)

            cluster_uid, cluster_pid, cluster_tid = \
                global_id_cluster_id_map[global_uid], global_id_cluster_id_map[global_pid], global_id_cluster_id_map[
                    global_tid]

            np.append(pos_train_tuple["user_X"], cluster_uid)
            np.append(pos_train_tuple["playlist_X"], cluster_pid)
            np.append(pos_train_tuple["pos_track_X"], cluster_tid)