from time import time

import numpy as np
import tensorflow as tf

from ModelLayertf2.BaseModel import Loss, MDR
from ModelLayertf2.ClusterModels.ClusterModel import FullGNNSingleCluster, ClusterModel, PureFullGNN
from ModelLayertf2.ClusterModels.ClusterUPTModel import ClusterUPTModel
from ModelLayertf2.Metric import Metrics


class G6_concat_MDR(ClusterUPTModel):
    def _build_model(self):
        self.cluster_initial_ebs = [tf.Variable(self.initializer([size["total"], self.embedding_size]),
                                                name="cluster_{}_ebs".format(cluster_no))
                                    for cluster_no, size in enumerate(self.data.cluster_sizes)]

        Ws = [{
            "W_sides": {
                entity_name: tf.Variable(self.initializer([self.embedding_size, self.embedding_size]),
                                         name="W_side_layer_{}_{}".format(layer_no, entity_name))
                for entity_name in self.data.get_entity_names()
            },
            "W_dots": {
                entity_name: tf.Variable(self.initializer([self.embedding_size, self.embedding_size]),
                                         name="W_dot_layer_{}_{}".format(layer_no, entity_name))
                for entity_name in self.data.get_entity_names()
            }
        } for layer_no in range(self.gnn_layer_num)]

        self.full_single_GNN_layer = FullGNNSingleCluster(self.initializer, self.embedding_size,
                                                          Ws, self.data.single_cluster_laplacian_matrices,
                                                          self.data.cluster_bounds, self.data.get_entity_names(),
                                                          self.cluster_dropout_flag, self.cluster_dropout_ratio,
                                                          self.data.cluster_num, self.gnn_layer_num)
        self.pure_full_GNN_layer = PureFullGNN(Ws, self.data.get_entity_names(),
                 self.cluster_dropout_flag, self.cluster_dropout_ratio,
                 self.gnn_layer_num)

        self.MDR_layer = MDR(self.initializer, self.embedding_size, self.data.data_set_num.track)

        stored = {
            "cluster_ebs": self.cluster_initial_ebs,
            "full_GNN": self.full_single_GNN_layer,
            "MDR": self.MDR_layer
        }

        return stored

    def _same_cluster_train_tuples(self, tape, epoch, pos_cluster_no):
        pos_initial_ebs = self.cluster_initial_ebs[pos_cluster_no]
        pos_train_tuples = self.data.cluster_pos_train_tuples[pos_cluster_no]

        sample_start_t = time()
        if epoch % 3 == 0:
            neg_cluster_no, neg_track_ids = self.same_cluster_strategy.sample_negative_tids(pos_cluster_no)
        else:
            neg_cluster_no, neg_track_ids = self.other_cluster_strategy.sample_negative_tids(pos_cluster_no)
        # print("Sampleing negative track ids used {} seconds".format(time() - sample_start_t))

        gnn_start_t = time()
        gnn_ebs = self.full_single_GNN_layer(pos_initial_ebs, cluster_no=pos_cluster_no, train_flag=True)
        gnn_end_t = time()
        # print("Generating the cluster's GNN embeddings used {} seconds".format(gnn_end_t - gnn_start_t))

        pos_eb_lookup_start_t = time()
        user_ebs = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["user_cluster_id"])
        playlist_ebs = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["playlist_cluster_id"])
        pos_track_ebs = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["pos_track_cluster_id"])
        pos_eb_lookup_end_t = time()
        # print("Positive embeddings lookup used {} seconds.".format(pos_eb_lookup_end_t - pos_eb_lookup_start_t))

        neg_ebs_start_t = time()
        neg_initial_ebs = self.cluster_initial_ebs[neg_cluster_no]
        neg_gnn_ebs = self.full_single_GNN_layer(neg_initial_ebs, cluster_no=neg_cluster_no, train_flag=True)
        neg_track_ebs = tf.nn.embedding_lookup(neg_gnn_ebs, neg_track_ids["cluster_id"])
        # print("Negative gnn embeddings + lookup used {} seconds.".format(time() - neg_ebs_start_t))

        assert len(pos_train_tuples["user_cluster_id"]) == len(pos_train_tuples["playlist_cluster_id"]) \
               == len(pos_train_tuples["pos_track_cluster_id"]) == len(pos_train_tuples["pos_track_entity_id"]) \
               == len(neg_track_ids["cluster_id"])

        pos_scores = self.MDR_layer({
            "user_ebs": user_ebs,
            "playlist_ebs": playlist_ebs,
            "track_ebs": pos_track_ebs,
            "track_entity_ids": pos_train_tuples["pos_track_entity_id"]
        })
        neg_scores = self.MDR_layer({
            "user_ebs": user_ebs,
            "playlist_ebs": playlist_ebs,
            "track_ebs": neg_track_ebs,
            "track_entity_ids": neg_track_ids["entity_id"]
        })

        assert not np.any(np.isnan(pos_scores))
        assert not np.any(np.isnan(neg_scores))

        mf_loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores)))
        reg_loss_B = tf.nn.l2_loss(self.MDR_layer.get_reg_loss())
        reg_loss_W = tf.nn.l2_loss(self.full_single_GNN_layer.get_reg_loss())
        train_tuples_length = pos_train_tuples["length"]
        assert train_tuples_length != 0
        reg_loss_ebs = (tf.nn.l2_loss(user_ebs) + tf.nn.l2_loss(playlist_ebs) +
                        tf.nn.l2_loss(pos_track_ebs) + tf.nn.l2_loss(neg_track_ebs)) / train_tuples_length
        reg_loss = self.reg_loss_ratio * (reg_loss_ebs + reg_loss_B + reg_loss_W)
        assert not np.any(np.isnan(pos_scores)) and not np.any(np.isnan(reg_loss))
        loss = mf_loss + reg_loss

        # Compute and apply gradients.
        update_gradients_start_t = time()
        trainable_variables = []
        trainable_variables.append(self.cluster_initial_ebs[pos_cluster_no])
        trainable_variables.append(self.cluster_initial_ebs[neg_cluster_no])
        trainable_variables.extend(self.MDR_layer.get_trainable_variables())
        trainable_variables.extend(self.full_single_GNN_layer.get_trainable_variables())
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        update_gradients_end_t = time()
        # print("Updating gradients used {} seconds".format(update_gradients_end_t - update_gradients_start_t))
        return reg_loss, mf_loss

    def _two_cluster_train_tuples(self, tape, epoch, cno1):
        small_cno, large_cno = self.data.get_another_random_cno(cno1)
        init_ebs1 = self.cluster_initial_ebs[small_cno]
        init_ebs2 = self.cluster_initial_ebs[large_cno]
        init_ebs = tf.concat([init_ebs1, init_ebs2], 0)
        laplacian_matrices, bounds = self.data.get_two_cluster_laplacian_matrices(small_cno, large_cno)
        train_tuples = self.data.get_two_cluster_train_tuples(small_cno, large_cno)

        gnn_ebs = self.pure_full_GNN_layer(init_ebs, laplacian_matrices=laplacian_matrices, bounds=bounds,
                                 train_flag=True)
        user_ebs = tf.nn.embedding_lookup(gnn_ebs, train_tuples["user_cluster_id"])
        playlist_ebs = tf.nn.embedding_lookup(gnn_ebs, train_tuples["playlist_cluster_id"])
        pos_track_ebs = tf.nn.embedding_lookup(gnn_ebs, train_tuples["pos_track_cluster_id"])
        neg_track_ebs = tf.nn.embedding_lookup(gnn_ebs, train_tuples["neg_track_cluster_id"])

        assert len(train_tuples["user_cluster_id"]) == len(train_tuples["playlist_cluster_id"]) \
               == len(train_tuples["pos_track_cluster_id"]) == len(train_tuples["pos_track_entity_id"]) \
               == len(train_tuples["neg_track_cluster_id"])

        pos_scores = self.MDR_layer({
            "user_ebs": user_ebs,
            "playlist_ebs": playlist_ebs,
            "track_ebs": pos_track_ebs,
            "track_entity_ids": train_tuples["pos_track_entity_id"]
        })
        neg_scores = self.MDR_layer({
            "user_ebs": user_ebs,
            "playlist_ebs": playlist_ebs,
            "track_ebs": neg_track_ebs,
            "track_entity_ids": train_tuples["neg_track_entity_id"]
        })

        assert not np.any(np.isnan(pos_scores))
        assert not np.any(np.isnan(neg_scores))

        mf_loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores)))
        reg_loss_B = tf.nn.l2_loss(self.MDR_layer.get_reg_loss())
        reg_loss_W = tf.nn.l2_loss(self.pure_full_GNN_layer.get_reg_loss())
        train_tuples_length = train_tuples["length"]
        assert train_tuples_length != 0
        reg_loss_ebs = (tf.nn.l2_loss(user_ebs) + tf.nn.l2_loss(playlist_ebs) +
                        tf.nn.l2_loss(pos_track_ebs) + tf.nn.l2_loss(neg_track_ebs)) / train_tuples_length
        reg_loss = self.reg_loss_ratio * (reg_loss_ebs + reg_loss_B + reg_loss_W)
        assert not np.any(np.isnan(pos_scores)) and not np.any(np.isnan(reg_loss))
        loss = mf_loss + reg_loss

        # Compute and apply gradients.
        update_gradients_start_t = time()
        trainable_variables = []
        trainable_variables.append(self.cluster_initial_ebs[small_cno])
        trainable_variables.append(self.cluster_initial_ebs[large_cno])
        trainable_variables.extend(self.MDR_layer.get_trainable_variables())
        trainable_variables.extend(self.pure_full_GNN_layer.get_trainable_variables())
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        update_gradients_end_t = time()
        # print("Updating gradients used {} seconds".format(update_gradients_end_t - update_gradients_start_t))
        return reg_loss, mf_loss

    def _train_cluster(self, epoch, pos_cluster_no):
        train_cluster_start_t = time()
        with tf.GradientTape() as tape:
            reg_loss, mf_loss = self._two_cluster_train_tuples(tape, epoch, pos_cluster_no)
        train_cluster_end_t = time()
        # print("Train cluster {} used {} seconds".format(pos_cluster_no, train_cluster_end_t - train_cluster_start_t))
        return Loss(reg_loss, mf_loss)

    def _test(self, epoch):
        test_start_time = time()
        # Compute gnn processed embeddings.
        cluster_gnn_ebs = [self.full_single_GNN_layer(initial_ebs, cluster_no=cluster_no, train_flag=False)
                           for cluster_no, initial_ebs in enumerate(self.cluster_initial_ebs)]
        global_gnn_ebs = tf.concat(cluster_gnn_ebs, 0)

        # For each group compute metrices.
        test_pos_tuples = self.data.test_pos_tuples
        metrics = Metrics(self.max_K)
        for i in range(test_pos_tuples["length"]):
            user_entity_id = test_pos_tuples["user_entity_id"][i]
            playlist_entity_id = test_pos_tuples["playlist_entity_id"][i]
            pos_track_entity_id = test_pos_tuples["track_entity_id"][i]
            user_global_id = test_pos_tuples["user_global_id"][i]
            playlist_global_id = test_pos_tuples["playlist_global_id"][i]
            pos_track_global_id = test_pos_tuples["track_global_id"][i]

            neg_tids = self.data.sample_negative_test_track_ids(user_entity_id, playlist_entity_id)

            # Used by track embeddings.
            track_global_ids = [pos_track_global_id]
            track_global_ids.extend(neg_tids["global_id"])
            track_global_ids = np.array(track_global_ids, dtype=int)

            # Used by track biases.
            track_entity_ids = [pos_track_entity_id]
            track_entity_ids.extend(neg_tids["entity_id"])
            track_entity_ids = np.array(track_entity_ids, dtype=int)

            user_ebs = tf.nn.embedding_lookup(global_gnn_ebs, user_global_id)
            playlist_ebs = tf.nn.embedding_lookup(global_gnn_ebs, playlist_global_id)
            tracks_ebs = tf.nn.embedding_lookup(global_gnn_ebs, track_global_ids)

            scores = self.MDR_layer({
                "user_ebs": user_ebs,
                "playlist_ebs": playlist_ebs,
                "track_ebs": tracks_ebs,
                "track_entity_ids": track_entity_ids
            })

            metrics.add_metrics(track_entity_ids, scores, pos_track_entity_id)
        metrics.cal_avg_metrics()
        test_end_time = time()
        return metrics, test_end_time - test_start_time
