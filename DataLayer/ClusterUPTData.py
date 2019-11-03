import scipy.sparse as sp
import numpy as np


from DataLayer.ClusterData import ClusterData
from DataLayer.NormalData import NormalData
from Common import DatasetNum


class ClusterUPTData(ClusterData):
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

    def get_entity_names(self):
        return ["u", "p", "t"]

    def __get_cluster_sizes(self, parts, data_set_num: DatasetNum):
        cluster_sizes = [{"u": 0, "p": 0, "t": 0, "total": 0} for _ in range(self.cluster_num)]

        for global_id in range(self.sum):
            cluster_number = parts[global_id]
            cluster_size = cluster_sizes[cluster_number]
            cluster_size["total"] += 1
            if global_id < data_set_num.user:  # User id.
                cluster_size["u"] += 1
            elif data_set_num.user <= global_id < data_set_num.user + data_set_num.playlist:  # Playlist id.
                cluster_size["p"] += 1
            elif data_set_num.user + data_set_num.playlist <= global_id < self.sum:  # Track id.
                cluster_size["t"] += 1

        return cluster_sizes

    def __get_cluster_bounds(self, cluster_sizes):
        return [{
            "u": (0, size["u"]),  # Bound start, bound end.
            "p": (size["u"], size["u"] + size["p"]),
            "t": (size["u"] + size["p"], size["u"] + size["p"] + size["t"])
        } for size in self.cluster_sizes]

    def __gen_global_id_cluster_id_map(self, parts):
        """
        Ids in parts are in the order of user, playlist, track,
        so the cluster id will also be in this order.
        :param parts:
        :return:
        """
        # Init clusters
        cluster_total_sizes = np.zeros((self.cluster_num, ), dtype=int)
        global_id_cluster_id_map = np.zeros((len(parts), ), dtype=int)

        for global_id, cluster_number in enumerate(parts):
            assert global_id_cluster_id_map[global_id] is 0
            global_id_cluster_id_map[global_id] = cluster_total_sizes[cluster_number]
            cluster_total_sizes[cluster_number] += 1

        return global_id_cluster_id_map

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
        cluster_pos_train_tuples = [dict() for i in range(self.cluster_num)]

        for pos_train_tuple in cluster_pos_train_tuples:
            pos_train_tuple.update({
                "length": 0,
                # Entity ids of u-p-t tuples.
                "user_entity_id": np.array([]),
                "playlist_entity_id": np.array([]),
                "pos_track_entity_id": np.array([]),
                # Cluster ids of u-p-t tuples.
                "user_cluster_id": np.array([]),
                "playlist_cluster_id": np.array([]),
                "pos_track_cluster_id": np.array([])
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

                    pos_train_tuple["length"] += 1
                    np.append(pos_train_tuple["user_entity_id"], entity_uid)
                    np.append(pos_train_tuple["playlist_entity_id"], entity_pid)
                    np.append(pos_train_tuple["pos_track_entity_id"], entity_tid)
                    np.append(pos_train_tuple["user_cluster_id"], cluster_uid)
                    np.append(pos_train_tuple["playlist_cluster_id"], cluster_pid)
                    np.append(pos_train_tuple["pos_track_cluster_id"], cluster_tid)
        return cluster_pos_train_tuples

    def __gen_cluster_track_ids(self, parts, data_set_num: DatasetNum, global_id_cluster_id_map):
        cluster_track_ids = [{"num": 0, "entity_id": np.array([]), "cluster_id": np.array([])} for _ in range(self.cluster_num)]

        tid_offset = data_set_num.user + data_set_num.playlist
        for entity_tid in range(data_set_num.track):
            global_tid = entity_tid + tid_offset
            cluster_tid = global_id_cluster_id_map[global_tid]
            cluster_number = parts[global_tid]

            cluster_track_ids[cluster_number]["num"] += 1
            np.append(cluster_track_ids[cluster_number]["entity_id"], entity_tid)
            np.append(cluster_track_ids[cluster_number]["cluster_id"], cluster_tid)

        return cluster_track_ids

    def __gen_test_pos_tuples(self, test_data, data_set_num: DatasetNum, global_id_cluster_id_map, parts):
        test_tuples = {
            "length": 0,
            # Entity ids.
            "user_entity_id": np.array([]),
            "playlist_entity_id": np.array([]),
            "pos_track_entity_id": np.array([]),
            # Global ids.
            "user_global_id": np.array([]),
            "playlist_global_id": np.array([]),
            "pos_track_global_id": np.array([])
        }

        for entity_uid, user in test_data.items():
            for entity_pid, entity_tids in user.items():
                for entity_tid in entity_tids:
                    # Get global ids:
                    global_uid, global_pid, global_tid = \
                        self.__map_entity_id_to_global_id(entity_uid, data_set_num, self.ENTITY_USER), \
                        self.__map_entity_id_to_global_id(entity_pid, data_set_num, self.ENTITY_PLAYLIST), \
                        self.__map_entity_id_to_global_id(entity_tid, data_set_num, self.ENTITY_TRACK)

                    test_tuples["length"] += 1
                    np.append(test_tuples["user_entity_id"], entity_uid)
                    np.append(test_tuples["playlist_entity_id"], entity_pid)
                    np.append(test_tuples["pos_track_entity_id"], entity_tid)

                    np.append(test_tuples["user_global_id"], global_uid)
                    np.append(test_tuples["playlist_global_id"], global_pid)
                    np.append(test_tuples["pos_track_global_id"], global_tid)

        return test_tuples

    def __get_cluster_connections(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
        cluster_connections = [{
            "up": [],
            "pt": [],
            "ut": []
        } for _ in range(self.cluster_num)]
        for entity_uid, user in train_data.items():
            for entity_pid, entity_tids in user.items():
                for entity_tid in entity_tids:
                    # Get global ids:
                    global_uid, global_pid, global_tid = \
                        self.__map_entity_id_to_global_id(entity_uid, data_set_num, self.ENTITY_USER), \
                        self.__map_entity_id_to_global_id(entity_pid, data_set_num, self.ENTITY_PLAYLIST), \
                        self.__map_entity_id_to_global_id(entity_tid, data_set_num, self.ENTITY_TRACK)

                    user_cluster_number = parts[global_uid]
                    playlist_cluster_number = parts[global_pid]
                    track_cluster_number = parts[global_tid]

                    cluster_uid, cluster_pid, cluster_tid = \
                        global_id_cluster_id_map[global_uid], global_id_cluster_id_map[global_pid], \
                        global_id_cluster_id_map[global_tid]

                    if user_cluster_number == playlist_cluster_number:
                        cluster_number = user_cluster_number
                        cluster_connections[cluster_number]["up"].append((cluster_uid, cluster_pid))
                    elif playlist_cluster_number == track_cluster_number:
                        cluster_number = playlist_cluster_number
                        cluster_connections[cluster_number]["pt"].append((cluster_pid, cluster_tid))
                    elif user_cluster_number == track_cluster_number:
                        cluster_number = user_cluster_number
                        cluster_connections[cluster_number]["ut"].append((cluster_uid, cluster_tid))
        return cluster_connections

    def __gen_laplacian_matrices(self, cluster_pos_train_tuples, cluster_sizes, cluster_connections, ut_alpha):
        clusters_laplacian_matrices = [dict() for _ in range(self.cluster_num)]

        # Init clusters_laplacian_matrices
        for cluster_number, _ in enumerate(clusters_laplacian_matrices):
            cluster_size = cluster_sizes[cluster_number]
            connections = cluster_connections[cluster_number]
            cluster_total_size = cluster_size["total"]

            # Init laplacian matrix.
            L_matrix = sp.dok_matrix((cluster_total_size, cluster_total_size), dtype=np.float64)
            for cluster_uid, cluster_pid in connections["up"]:
                x = cluster_uid
                y = cluster_pid
                L_matrix[x, y] = 1
                L_matrix[y, x] = 1
            for cluster_pid, cluster_tid in connections["pt"]:
                x = cluster_pid
                y = cluster_tid
                L_matrix[x, y] = 1
                L_matrix[y, x] = 1
            for cluster_uid, cluster_tid in connections["ut"]:
                x = cluster_uid
                y = cluster_tid
                L_matrix[x, y] = 1 * ut_alpha
                L_matrix[y, x] = 1 * ut_alpha

            # Init LI matrix.
            LI_matrix = L_matrix + sp.eye(L_matrix.shape[0])

            clusters_laplacian_matrices[cluster_number] = dict()
            for entity_name, (start, end) in self.cluster_bounds[cluster_number].items():
                clusters_laplacian_matrices[cluster_number][entity_name] = {
                    "L": L_matrix[start: end, :],
                    "LI": LI_matrix[start: end, :]
                }

        return clusters_laplacian_matrices
