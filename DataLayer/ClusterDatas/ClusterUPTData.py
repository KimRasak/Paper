import scipy.sparse as sp
import numpy as np

from DataLayer.ClusterDatas.ClusterData import ClusterData
from Common import DatasetNum


class ClusterUPTData(ClusterData):
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
                self.pt[pid] = tids

                # Add element to ut
                if uid not in self.ut:
                    self.ut[uid] = tids
                else:
                    self.ut[uid] = np.concatenate((self.ut[uid], tids))

    def _get_data_sum(self, data_set_num: DatasetNum):
        return data_set_num.user + data_set_num.playlist + data_set_num.track

    def get_entity_names(self):
        return ["u", "p", "t"]

    def _get_cluster_sizes(self, parts, data_set_num: DatasetNum):
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

        for cluster_number in range(self.cluster_num):
            cluster_size = cluster_sizes[cluster_number]
            if cluster_size["u"] <= 10 or cluster_size["p"] <= 10 or cluster_size["t"] <= 10:
                print("Some entity's num is too small.")

        return cluster_sizes

    def _get_cluster_bounds(self, cluster_sizes):
        return [{
            "u": (0, size["u"]),  # Bound start, bound end.
            "p": (size["u"], size["u"] + size["p"]),
            "t": (size["u"] + size["p"], size["u"] + size["p"] + size["t"])
        } for size in self.cluster_sizes]

    def _gen_global_id_cluster_id_map(self, parts):
        """
        Generate the map from global ids to cluster ids.
        Ids in parts are in the order of user, playlist, track,
        so the cluster id will also be in this order.
        :param parts:
        :return:
        """
        # Init clusters
        cluster_total_sizes = np.zeros((self.cluster_num,), dtype=int)
        global_id_cluster_id_map = np.zeros((len(parts),), dtype=int)

        for global_id, cluster_number in enumerate(parts):
            assert global_id_cluster_id_map[global_id] == 0
            cluster_id = cluster_total_sizes[cluster_number]
            global_id_cluster_id_map[global_id] = cluster_id
            cluster_total_sizes[cluster_number] += 1

        return global_id_cluster_id_map

    def _gen_cluster_id_entity_id_map(self, parts, cluster_sizes, global_id_cluster_id_map):
        """
        Generate the map from cluster id to entity ids.
        """
        cluster_id_entity_id_map = [np.full((cluster_sizes[cluster_no]["total"],), -1)
                                    for cluster_no in range(self.cluster_num)]

        for global_id, cluster_number in enumerate(parts):
            cluster_id = global_id_cluster_id_map[global_id]
            entity_id = self._map_global_id_to_entity_id(global_id, self.data_set_num)

            assert cluster_id_entity_id_map[cluster_number][cluster_id] == -1
            cluster_id_entity_id_map[cluster_number][cluster_id] = entity_id

        for cid_eid_map in cluster_id_entity_id_map:
            for value in cid_eid_map:
                assert value != -1

        return global_id_cluster_id_map

    def _map_entity_id_to_global_id(self, entity_id, data_set_num, entity_name):
        if entity_name not in [self.ENTITY_USER, self.ENTITY_PLAYLIST, self.ENTITY_TRACK]:
            raise Exception("Wrong entity name!")

        if entity_name == self.ENTITY_USER:
            return entity_id
        elif entity_name == self.ENTITY_PLAYLIST:
            return data_set_num.user + entity_id
        elif entity_name == self.ENTITY_TRACK:
            return data_set_num.user + data_set_num.playlist + entity_id
        raise Exception("Wrong entity name!")

    def _map_global_id_to_entity_id(self, global_id, data_set_num: DatasetNum):
        if global_id < data_set_num.user:
            return global_id
        elif data_set_num.user <= global_id < data_set_num.user + data_set_num.playlist:
            return global_id - data_set_num.user
        elif data_set_num.user + data_set_num.playlist <= global_id < self.sum:
            return global_id - data_set_num.user - data_set_num.playlist
        raise Exception("Unexpected global id.")

    def _gen_train_tuples(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
        cluster_pos_train_tuples = [dict() for i in range(self.cluster_num)]
        total_train_num = 0
        same_cluster_num = {1: 0, 2: 0, 3: 0}

        for pos_train_tuple in cluster_pos_train_tuples:
            # The usage of the following ids.
            # Entity ids:
            #   The user and playlist entity ids -> pick negative training tids.
            #   The track entity ids -> search for track biases.
            # Cluster ids:
            #   The u/p/t cluster ids -> search for embeddings,
            pos_train_tuple.update({
                "length": 0,
                # Entity ids of u-p-t tuples.
                "user_entity_id": [],
                "playlist_entity_id": [],
                "pos_track_entity_id": [],
                # Cluster ids of u-p-t tuples.
                "user_cluster_id": [],
                "playlist_cluster_id": [],
                "pos_track_cluster_id": [],
            })

        def ids_are_in_same_cluster(global_uid, global_pid, global_tid, parts):
            # Check if the connection exists.
            user_cluster_number, playlist_cluster_number, track_cluster_number = \
                parts[global_uid], parts[global_pid], parts[global_tid]

            # Check how many entities are in the same cluster.
            same_cluster_num = 0
            if user_cluster_number == playlist_cluster_number:
                if playlist_cluster_number == track_cluster_number or user_cluster_number == track_cluster_number:
                    same_cluster_num = 3
                else:
                    same_cluster_num = 2
            elif playlist_cluster_number == track_cluster_number or user_cluster_number == track_cluster_number:
                same_cluster_num = 2
            else:
                same_cluster_num = 1

            return user_cluster_number == playlist_cluster_number == track_cluster_number, user_cluster_number, same_cluster_num

        for entity_uid, user in train_data.items():
            for entity_pid, entity_tids in user.items():
                for entity_tid in entity_tids:
                    # for each u-p-t tuple, add to train tuples if their connection exists.

                    # Get global ids:
                    global_uid, global_pid, global_tid = \
                        self._map_entity_id_to_global_id(entity_uid, data_set_num, self.ENTITY_USER), \
                        self._map_entity_id_to_global_id(entity_pid, data_set_num, self.ENTITY_PLAYLIST), \
                        self._map_entity_id_to_global_id(entity_tid, data_set_num, self.ENTITY_TRACK)

                    are_same_cluster, cluster_number, same_num = ids_are_in_same_cluster(global_uid, global_pid,
                                                                                         global_tid,
                                                                                         parts)
                    assert same_num != 0
                    same_cluster_num[same_num] += 1

                    if not are_same_cluster:  # The connection exists.
                        continue
                    pos_train_tuple = cluster_pos_train_tuples[cluster_number]

                    cluster_uid, cluster_pid, cluster_tid = \
                        global_id_cluster_id_map[global_uid], global_id_cluster_id_map[global_pid], \
                        global_id_cluster_id_map[global_tid]

                    total_train_num += 1
                    pos_train_tuple["length"] += 1
                    pos_train_tuple["user_entity_id"].append(entity_uid)
                    pos_train_tuple["playlist_entity_id"].append(entity_pid)
                    pos_train_tuple["pos_track_entity_id"].append(entity_tid)
                    pos_train_tuple["user_cluster_id"].append(cluster_uid)
                    pos_train_tuple["playlist_cluster_id"].append(cluster_pid)
                    pos_train_tuple["pos_track_cluster_id"].append(cluster_tid)

        # Convert to numpy arrays.
        for pos_train_tuple in cluster_pos_train_tuples:
            for key in pos_train_tuple.keys():
                if isinstance(pos_train_tuple[key], list):
                    pos_train_tuple[key] = np.array(pos_train_tuple[key], dtype=int)

        same_cluster_sum = 0
        for k, v in same_cluster_num.items():
            # The output shows that many interactions have 2 entities in the same cluster.
            # So it's reasonable to train 2 merged clusters together, rather than vallina cluster-gcn.
            print("same num: {}, num: {}".format(k, v))
            same_cluster_sum += v

        print("There are {} interactions of train set + test set, \n"
              "where {} of them are in the train set."
              "{} interactions of the train set are in the same cluster.".format(self.data_set_num.interaction,
                                                                                 same_cluster_sum, total_train_num))
        return cluster_pos_train_tuples

    def _gen_cluster_track_ids(self, parts, data_set_num: DatasetNum, global_id_cluster_id_map):
        cluster_track_ids = [{
            "num": 0,
            "entity_id": [],
            "cluster_id": []
        }
            for _ in range(self.cluster_num)]

        tid_offset = data_set_num.user + data_set_num.playlist
        for entity_tid in range(data_set_num.track):
            global_tid = entity_tid + tid_offset
            cluster_tid = global_id_cluster_id_map[global_tid]
            cluster_number = parts[global_tid]

            cluster_track_ids[cluster_number]["num"] += 1
            cluster_track_ids[cluster_number]["entity_id"].append(entity_tid)
            cluster_track_ids[cluster_number]["cluster_id"].append(cluster_tid)

        # Convert to numpy arrays.
        for track_ids in cluster_track_ids:
            for key in track_ids.keys():
                if isinstance(track_ids[key], list):
                    track_ids[key] = np.array(track_ids[key], dtype=int)

        return cluster_track_ids

    def _gen_test_pos_tuples(self, test_data, data_set_num: DatasetNum, global_id_cluster_id_map, parts):
        test_tuples = {
            "length": 0,
            # Entity ids.
            "user_entity_id": [],
            "playlist_entity_id": [],
            "track_entity_id": [],
            # Global ids.
            "user_global_id": [],
            "playlist_global_id": [],
            "track_global_id": []
        }

        for entity_uid, user in test_data.items():
            for entity_pid, entity_tids in user.items():
                for entity_tid in entity_tids:
                    # Get global ids:
                    global_uid, global_pid, global_tid = \
                        self._map_entity_id_to_global_id(entity_uid, data_set_num, self.ENTITY_USER), \
                        self._map_entity_id_to_global_id(entity_pid, data_set_num, self.ENTITY_PLAYLIST), \
                        self._map_entity_id_to_global_id(entity_tid, data_set_num, self.ENTITY_TRACK)

                    test_tuples["length"] += 1
                    test_tuples["user_entity_id"].append(entity_uid)
                    test_tuples["playlist_entity_id"].append(entity_pid)
                    test_tuples["track_entity_id"].append(entity_tid)

                    test_tuples["user_global_id"].append(global_uid)
                    test_tuples["playlist_global_id"].append(global_pid)
                    test_tuples["track_global_id"].append(global_tid)

        # Convert to numpy arrays.
        for key in test_tuples.keys():
            if isinstance(test_tuples[key], list):
                test_tuples[key] = np.array(test_tuples[key], dtype=int)

        return test_tuples

    def _get_inter_cluster_connections(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
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
                        self._map_entity_id_to_global_id(entity_uid, data_set_num, self.ENTITY_USER), \
                        self._map_entity_id_to_global_id(entity_pid, data_set_num, self.ENTITY_PLAYLIST), \
                        self._map_entity_id_to_global_id(entity_tid, data_set_num, self.ENTITY_TRACK)

                    user_cluster_number = parts[global_uid]
                    playlist_cluster_number = parts[global_pid]
                    track_cluster_number = parts[global_tid]

                    cluster_uid, cluster_pid, cluster_tid = \
                        global_id_cluster_id_map[global_uid], global_id_cluster_id_map[global_pid], \
                        global_id_cluster_id_map[global_tid]

                    if user_cluster_number == playlist_cluster_number:
                        cluster_number = user_cluster_number
                        cluster_connections[cluster_number]["up"].append((cluster_uid, cluster_pid))
                    if playlist_cluster_number == track_cluster_number:
                        cluster_number = playlist_cluster_number
                        cluster_connections[cluster_number]["pt"].append((cluster_pid, cluster_tid))
                    if user_cluster_number == track_cluster_number:
                        cluster_number = user_cluster_number
                        cluster_connections[cluster_number]["ut"].append((cluster_uid, cluster_tid))
        return cluster_connections

    def _get_all_cluster_connections(self, train_data, data_set_num: DatasetNum, parts, global_id_cluster_id_map):
        cluster_connections = dict()

        def assign_list_or_append(relation: dict, key, list_value):
            if key in relation:
                relation[key].append(list_value)
            else:
                relation[key] = []

        for from_cluster_no in range(self.cluster_num):
            for to_cluster_no in range(self.cluster_num):
                cluster_connections[(from_cluster_no, to_cluster_no)] = {
                    "up": dict(),
                    "pt": dict(),
                    "ut": dict(),
                }

        for entity_uid, user in train_data.items():
            for entity_pid, entity_tids in user.items():
                for entity_tid in entity_tids:
                    # Get global ids:
                    global_uid, global_pid, global_tid = \
                        self._map_entity_id_to_global_id(entity_uid, data_set_num, self.ENTITY_USER), \
                        self._map_entity_id_to_global_id(entity_pid, data_set_num, self.ENTITY_PLAYLIST), \
                        self._map_entity_id_to_global_id(entity_tid, data_set_num, self.ENTITY_TRACK)

                    user_cluster_number = parts[global_uid]
                    playlist_cluster_number = parts[global_pid]
                    track_cluster_number = parts[global_tid]

                    cluster_uid, cluster_pid, cluster_tid = \
                        global_id_cluster_id_map[global_uid], global_id_cluster_id_map[global_pid], \
                        global_id_cluster_id_map[global_tid]

                    assign_list_or_append(cluster_connections[(user_cluster_number, playlist_cluster_number)]["up"],
                                          cluster_uid, cluster_pid)
                    assign_list_or_append(cluster_connections[(user_cluster_number, playlist_cluster_number)]["pt"],
                                          cluster_pid, cluster_tid)
                    assign_list_or_append(cluster_connections[(user_cluster_number, playlist_cluster_number)]["ut"],
                                          cluster_uid, cluster_tid)

        for from_cluster_no in range(self.cluster_num):
            for to_cluster_no in range(self.cluster_num):
                conns = cluster_connections[(from_cluster_no, to_cluster_no)]
                for relation_name, relations in conns.items():
                    for key in relations.keys():
                        if isinstance(relations[key], list):
                            relations[key] = np.array(relations[key])
        return cluster_connections

    def _gen_single_cluster_laplacian_matrices(self, cluster_pos_train_tuples, cluster_sizes,
                                               single_cluster_connections, ut_alpha):
        clusters_laplacian_matrices = [dict() for _ in range(self.cluster_num)]

        # Init clusters_laplacian_matrices
        for cluster_number, _ in enumerate(clusters_laplacian_matrices):
            cluster_size = cluster_sizes[cluster_number]
            connections = single_cluster_connections[cluster_number]
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
                if start == end:
                    raise Exception("No such entities with entity_name: {}".format(entity_name))
                clusters_laplacian_matrices[cluster_number][entity_name] = {
                    "L": L_matrix[start: end, :],
                    "LI": LI_matrix[start: end, :]
                }

        return clusters_laplacian_matrices

    def get_two_cluster_laplacian_matrices(self, small, large):
        """
        The order of cluster ids becomes:
        U1->U2->P1->P2->T1->T2
        """
        assert small < large
        connections = self.all_cluster_connections
        s1: dict = self.cluster_sizes[small]
        s2: dict = self.cluster_sizes[large]
        s_total = {
            key: s1[key] + s2[key]
            for key in s1.keys()
        }

        bounds = {
            "u": (0, s_total["u"]),
            "p": (s_total["u"], s_total["u"] + s_total["p"]),
            "t": (s_total["u"] + s_total["p"], s_total["total"])
        }

        s1_offset = {"u": 0, "p": s2["u"], "t": s2["u"] + s2["p"]}
        s2_offset = {"u": s1["u"], "p": s1["u"] + s1["p"], "t": s1["total"]}
        # Init L matrix.
        L_matrix = sp.dok_matrix((s_total, s_total), dtype=np.float64)

        def fill_matrix(L_matrix, connections, offset1, offset2):
            for cluster_uid, up in connections["up"]:
                for cluster_pid in up:
                    x = cluster_uid + offset1["u"]
                    y = cluster_pid + offset2["p"]
                    L_matrix[x, y] = 1
                    L_matrix[y, x] = 1
            for cluster_pid, pt in connections["pt"]:
                for cluster_tid in pt:
                    x = cluster_pid + offset1["p"]
                    y = cluster_tid + offset2["t"]
                    L_matrix[x, y] = 1
                    L_matrix[y, x] = 1
            for cluster_uid, ut in connections["ut"]:
                for cluster_tid in ut:
                    x = cluster_uid + offset1["u"]
                    y = cluster_tid + offset2["t"]
                    L_matrix[x, y] = 1 * self.ut_alpha
                    L_matrix[y, x] = 1 * self.ut_alpha

        # Build the L matrix from all inter/intra cluster connections.
        fill_matrix(L_matrix, connections[(small, small)], s1_offset, s1_offset)
        fill_matrix(L_matrix, connections[(small, large)], s1_offset, s2_offset)
        fill_matrix(L_matrix, connections[(large, small)], s2_offset, s1_offset)
        fill_matrix(L_matrix, connections[(large, large)], s2_offset, s2_offset)

        # Init LI matrix.
        LI_matrix = L_matrix + sp.eye(L_matrix.shape[0])

        # Fill laplacian matrices.
        laplacian_matrices = dict()
        for entity_name, (start, end) in bounds.items():
            assert start == end
            laplacian_matrices[entity_name] = {
                "L": L_matrix[start: end, :],
                "LI": LI_matrix[start: end, :]
            }

        return laplacian_matrices, bounds

    def get_two_cluster_train_tuples(self, small, large):
        """
        The order of cluster ids becomes:
        U1->U2->P1->P2->T1->T2
        """
        assert small < large
        cluster_sizes = self.cluster_sizes
        s1: dict = self.cluster_sizes[small]
        s2: dict = self.cluster_sizes[large]

        size_total_t = s1["t"] + s2["t"]

        train_tuples = {
            "length": 0,
            # Cluster ids: used for searching the embeddings.
            "user_cluster_id": [],
            "playlist_cluster_id": [],
            "pos_track_cluster_id": [],
            "neg_track_cluster_id": [],
            # Entity ids: used for searching the biases.
            "pos_track_entity_id": [],
            "neg_track_entity_id": [],
        }
        connections = self.all_cluster_connections

        c1_offset = {"u": 0, "p": s2["u"], "t": s2["u"] + s2["p"]}  # Offset of cluster1.
        c2_offset = {"u": s1["u"], "p": s1["u"] + s1["p"], "t": s1["total"]}  # Offset of cluster2.

        def pick_negative_tid(c1_tids, c2_tids):
            while True:
                random_tid = np.random.randint(0, size_total_t)
                if random_tid < s1["t"]:
                    original_cluster_tid = random_tid
                    tids = c1_tids
                elif s1["t"] <= random_tid < size_total_t:
                    original_cluster_tid = random_tid - s1["t"]
                    tids = c2_tids
                else:
                    raise Exception("Unexpected tid %d." % random_tid)
                if original_cluster_tid in tids:  # We've picked an observed tid.
                    continue
                cluster_tid = random_tid
                return cluster_tid, original_cluster_tid

        def add_tuples(up: dict, pt1: dict, pt2: dict, offset: dict):
            for cluster_uid, cluster_pids in up.items():
                for cluster_pid in cluster_pids:
                    # Get uid->pid relation from up
                    c1_tids = pt1[cluster_pid]
                    c2_tids = pt2[cluster_pid]
                    for cluster_pid, c1_tid in c1_tids:
                        # Get pid->tid relation from pt
                        train_tuples["user_cluster_id"].append(cluster_uid + offset["u"])
                        train_tuples["playlist_cluster_id"].append(cluster_pid + offset["p"])
                        train_tuples["pos_track_cluster_id"].append(c1_tid + c1_offset["t"])
                        train_tuples["pos_track_entity_id"].append(self.cluster_id_entity_id_map[small][c1_tid])

                        # Add negative tid.
                        neg_cluster_tid, original_neg_cluster_tid = pick_negative_tid(c1_tids, c2_tids)
                        train_tuples["neg_track_cluster_id"].append(neg_cluster_tid)
                        train_tuples["neg_track_entity_id"].append(self.cluster_id_entity_id_map[small][original_neg_cluster_tid])
                        train_tuples["length"] += 1

                    for cluster_pid, c2_tid in c2_tids:
                        # Get pid->tid relation from pt
                        train_tuples["user_cluster_id"].append(cluster_uid + offset["u"])
                        train_tuples["playlist_cluster_id"].append(cluster_pid + offset["p"])
                        train_tuples["pos_track_cluster_id"].append(c2_tid + c2_offset["t"])
                        train_tuples["pos_track_entity_id"].append(self.cluster_id_entity_id_map[large][c2_tid])

                        # Add negative tid
                        neg_cluster_tid, original_neg_cluster_tid = pick_negative_tid(c1_tids, c2_tids)
                        train_tuples["neg_track_cluster_id"].append(neg_cluster_tid)
                        train_tuples["neg_track_entity_id"].append(self.cluster_id_entity_id_map[large][original_neg_cluster_tid])
                        train_tuples["length"] += 1

        add_tuples(connections[(small, large)]["up"],
                   connections[(large, small)]["pt"], connections[(large, large)]["pt"],
                   {"u": c1_offset["u"], "p": c2_offset["p"]})

        add_tuples(connections[(large, large)]["up"],
                   connections[(large, small)]["pt"], connections[(large, large)]["pt"],
                   {"u": c2_offset["u"], "p": c2_offset["p"]})

        add_tuples(connections[(small, small)]["up"],
                   connections[(small, small)]["pt"], connections[(small, large)]["pt"],
                   {"u": c1_offset["u"], "p": c1_offset["p"]})

        add_tuples(connections[(large, small)]["up"],
                   connections[(small, small)]["pt"], connections[(small, large)]["pt"],
                   {"u": c2_offset["u"], "p": c1_offset["p"]})

        for key in train_tuples.keys():
            if isinstance(train_tuples[key], list):
                train_tuples[key] = np.array(train_tuples[key])
        return train_tuples

    def sample_negative_test_track_ids(self, uid, pid):
        # Sample 100 track id, and ensure each tid doesn't appear in train data and test data.
        track_num = self.data_set_num.track
        negative_test_tids = {
            "num": 0,
            "entity_id": [],
            "global_id": []
        }
        while negative_test_tids["num"] < 100:
            picked_tid = np.random.randint(0, track_num)
            if (picked_tid not in self.train_data[uid][pid]) \
                    and (picked_tid not in self.test_data[uid][pid]):
                global_tid = self._map_entity_id_to_global_id(picked_tid, self.data_set_num, self.ENTITY_TRACK)

                negative_test_tids["num"] += 1
                negative_test_tids["entity_id"].append(picked_tid)
                negative_test_tids["global_id"].append(global_tid)

        # Convert to numpy arrays.
        for key in negative_test_tids.keys():
            if isinstance(negative_test_tids[key], list):
                negative_test_tids[key] = np.array(negative_test_tids[key], dtype=int)
        return negative_test_tids
