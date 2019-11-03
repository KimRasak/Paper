from ModelLayertf2.ClusterModel import ClusterModel
import tensorflow as tf


class ClusterUPTModel(ClusterModel):
    def __train_epoch(self, epoch):
        for cluster_no in range(self.cluster_num):
            self.__train_cluster(cluster_no)

        pass

    def __train_cluster(self, pos_cluster_no):
        pos_initial_ebs = self.cluster_initial_ebs[pos_cluster_no]
        pos_train_tuples = self.data.cluster_pos_train_tuples[pos_cluster_no]

        gnn_ebs = self.full_GNN_layer(pos_initial_ebs, cluster_no=pos_cluster_no)

        user_ebs = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["user_cluster_id"])
        playlist_ebs = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["playlist_cluster_id"])
        pos_track_ebs = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["pos_track_cluster_id"])
        pos_track_entity_ids = tf.nn.embedding_lookup(gnn_ebs, pos_train_tuples["pos_track_entity_id"])

        pos_scores = self.MDR_layer({
            "user_ebs": user_ebs,
            "playlist_ebs": playlist_ebs,
            "track_ebs": pos_track_ebs,
            "track_entity_ids": pos_track_entity_ids
        })
        neg_scores = self.MDR_layer({
            "user_ebs": user_ebs,
            "playlist_ebs": playlist_ebs,
            "track_ebs": pos_track_ebs,
            "track_entity_ids": neg_track_entity_ids
        })

    def __test(self, epoch):
        pass
