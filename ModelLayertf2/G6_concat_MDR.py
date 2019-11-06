import tensorflow as tf

from ModelLayertf2 import Metric
from ModelLayertf2.ClusterModel import ClusterModel, MDR, FullGNN
from ModelLayertf2.ClusterUPTModel import ClusterUPTModel
from ModelLayertf2.Metric import Metrics


class G6_concat_MDR(ClusterUPTModel):
    def _build_model(self):
        self.cluster_initial_ebs = [tf.Variable(self.initializer([size["total"], self.embedding_size]),
                                                name="cluster_{}_ebs".format(cluster_no))
                                    for cluster_no, size in enumerate(self.data.cluster_sizes)]

        self.full_GNN_layer = FullGNN(self.initializer, self.embedding_size,
                                      self.data.clusters_laplacian_matrices, self.data.cluster_bounds,
                                      self.data.get_entity_names(),
                                      self.cluster_dropout_flag, self.cluster_dropout_ratio,
                                      self.data.cluster_num, self.gnn_layer_num)

        self.MDR_layer = MDR(self.initializer, self.embedding_size, self.data.data_set_num.track)

    def _train_cluster(self, pos_cluster_no):
        with tf.GradientTape() as tape:
            pos_initial_ebs = self.cluster_initial_ebs[pos_cluster_no]
            pos_train_tuples = self.data.cluster_pos_train_tuples[pos_cluster_no]

            gnn_ebs = self.full_GNN_layer(pos_initial_ebs, cluster_no=pos_cluster_no, train_flag=True)

            user_ebs = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["user_cluster_id"])
            playlist_ebs = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["playlist_cluster_id"])
            pos_track_ebs = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["pos_track_cluster_id"])
            pos_track_entity_ids = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["pos_track_entity_id"])

            neg_cluster_no, neg_track_ids = self.neg_sample_strategy.sample_negative_tids(pos_cluster_no)
            neg_initial_ebs = self.cluster_initial_ebs[neg_cluster_no]
            neg_gnn_ebs = self.full_GNN_layer(neg_initial_ebs, cluster_no=neg_cluster_no, train_flag=True)
            neg_track_ebs = tf.nn.embedding_lookup(neg_gnn_ebs, neg_track_ids["cluster_id"])

            pos_scores = self.MDR_layer({
                "user_ebs": user_ebs,
                "playlist_ebs": playlist_ebs,
                "track_ebs": pos_track_ebs,
                "track_entity_ids": pos_track_entity_ids
            })
            neg_scores = self.MDR_layer({
                "user_ebs": user_ebs,
                "playlist_ebs": playlist_ebs,
                "track_ebs": neg_track_ebs,
                "track_entity_ids": neg_track_ids["entity_id"]
            })

            mf_loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores)))
            reg_loss_B = tf.nn.l2_loss(self.MDR_layer.get_reg_loss())
            reg_loss_W = tf.nn.l2_loss(self.full_GNN_layer.get_reg_loss())
            reg_loss_emb = tf.nn.l2_loss(user_ebs) + tf.nn.l2_loss(playlist_ebs) + tf.nn.l2_loss(
                pos_track_ebs) + tf.nn.l2_loss(neg_track_ebs)
            reg_loss = self.reg_loss_ratio * (reg_loss_emb + reg_loss_B + reg_loss_W)
            loss = mf_loss + reg_loss

            # Compute and apply gradients.
            trainable_variables = []
            trainable_variables.append(self.cluster_initial_ebs[pos_cluster_no])
            trainable_variables.append(self.cluster_initial_ebs[neg_cluster_no])
            trainable_variables.extend(self.MDR_layer.get_trainable_variables())
            trainable_variables.extend(self.full_GNN_layer.get_trainable_variables())
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def _test(self, epoch):
        # Compute gnn processed embeddings.
        cluster_gnn_ebs = [self.full_GNN_layer(initial_ebs, cluster_no=cluster_no, train_flag=False)
                           for cluster_no, initial_ebs in enumerate(self.cluster_initial_ebs)]
        gnn_ebs = tf.concat(cluster_gnn_ebs, 0)

        # For each group compute metrices.
        test_pos_tuples = self.data.test_pos_tuples
        metrics = Metrics(20)
        for i in range(test_pos_tuples["length"]):
            user_entity_id = test_pos_tuples["user_entity_id"][i]
            playlist_entity_id = test_pos_tuples["playlist_entity_id"][i]
            pos_track_entity_id = test_pos_tuples["track_entity_id"][i]
            neg_tids = self.data.sample_negative_test_track_ids(user_entity_id, playlist_entity_id)

            track_entity_ids = [pos_track_entity_id]
            track_entity_ids.extend(neg_tids["entity_id"])

            user_ebs = tf.nn.embedding_lookup(gnn_ebs, user_entity_id)
            playlist_ebs = tf.nn.embedding_lookup(gnn_ebs, playlist_entity_id)
            tracks_ebs = tf.nn.embedding_lookup(gnn_ebs, track_entity_ids)

            scores = self.MDR_layer({
                "user_ebs": user_ebs,
                "playlist_ebs": playlist_ebs,
                "track_ebs": tracks_ebs,
                "track_entity_ids": neg_tids["global_id"]
            })

            metrics.add_metrics(track_entity_ids, scores, pos_track_entity_id)

        return metrics
