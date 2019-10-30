import tensorflow as tf

from ModelLayer.ClusterUPTModel import ClusterUPTModel

"""
This model consists of 3 parts:
1. Pass initial embeddings through some GNN layers, each layer having 6 weight matrices.
2. Concat the embedding matrices horizontally.
3. Send them to the MDR layer.
"""


def get_output(delta, B):
    B_delta = tf.multiply(B, delta)
    square = tf.square(B_delta)
    print("square:", square, len(square.shape) - 1)
    return tf.reduce_sum(square, axis=len(square.shape) - 1)


def MDR_layer(embed_user, embed_playlist, embed_track, B1, B2):
    delta_ut = embed_user - embed_track
    delta_pt = embed_playlist - embed_track

    o1 = get_output(delta_ut, B1)
    o2 = get_output(delta_pt, B2)
    print("shape of o1/o2:", o1.shape, o2.shape)

    return o1 + o2


class G6_concat_MDR(ClusterUPTModel):
    def __get_init_embeddings(self):
        cluster_sizes = self.data.cluster_sizes
        cluster_init_ebs = [tf.Variable(self.initializer([cluster_size["total"], self.embedding_size])) for cluster_size
                            in cluster_sizes]
        return cluster_init_ebs

    def __get_bias_embeddings(self):
        track_num = self.data.data_set_num.track
        return tf.Variable(self.initializer([track_num]))

    def __build_network(self):
        cluster_init_ebs = self.__get_init_embeddings()
        track_bias_ebs = self.__get_bias_embeddings()

        # Pass embedding matrices through GNN layers.
        layer1_ebs, layer1_weight_loss = self._cluster_GNN_layer(cluster_init_ebs, self.embedding_size)
        layer2_ebs, layer2_weight_loss  = self._cluster_GNN_layer(layer1_ebs, self.embedding_size)
        layer3_ebs, layer3_weight_loss  = self._cluster_GNN_layer(layer2_ebs, self.embedding_size)
        ebs_list = [cluster_init_ebs, layer1_ebs, layer2_ebs, layer3_ebs]
        weight_loss = layer1_weight_loss + layer2_weight_loss + layer3_weight_loss

        # For each cluster, concatenate the cluster embeddings of each layer.
        clusters_concat_ebs = []
        for cluster_number in range(self.cluster_num):
            concat_ebs = tf.concat([cluster_ebs[cluster_number] for cluster_ebs in ebs_list], 1)
            clusters_concat_ebs.append(concat_ebs)

        # For each cluster, define tensors for training.
        self.train_input_tensors = []
        self.train_output_tensors = []
        for cluster_number in range(self.cluster_num):
            cluster_ebs = clusters_concat_ebs[cluster_number]
            input_tensors = {
                # Input Entity ids.
                "pos_track_entity_id": tf.placeholder(tf.int32, shape=[None]),
                "neg_track_entity_id": tf.placeholder(tf.int32, shape=[None]),
                # Input Cluster ids.
                "user_cluster_id": tf.placeholder(tf.int32, shape=[None]),
                "playlist_cluster_id": tf.placeholder(tf.int32, shape=[None]),
                "pos_track_cluster_id": tf.placeholder(tf.int32, shape=[None]),
                "neg_track_cluster_id": tf.placeholder(tf.int32, shape=[None]),
            }
            user_ebs = tf.nn.embedding_lookup(cluster_ebs, input_tensors["user_cluster_id"])
            playlist_ebs = tf.nn.embedding_lookup(cluster_ebs, input_tensors["playlist_cluster_id"])
            pos_track_ebs = tf.nn.embedding_lookup(cluster_ebs, input_tensors["pos_track_cluster_id"])
            neg_track_ebs = tf.nn.embedding_lookup(cluster_ebs, input_tensors["neg_track_cluster_id"])

            B1 = tf.Variable(self.initializer([self.embedding_size * 4]))
            B2 = tf.Variable(self.initializer([self.embedding_size * 4]))

            # Define the score and bias.
            pos_score = MDR_layer(user_ebs, playlist_ebs, pos_track_ebs, B1, B2)
            neg_score = MDR_layer(user_ebs, playlist_ebs, neg_track_ebs, B1, B2)
            pos_bias = tf.nn.embedding_lookup(track_bias_ebs, input_tensors["pos_track_entity_id"])
            neg_bias = tf.nn.embedding_lookup(track_bias_ebs, input_tensors["neg_track_entity_id"])

            # Define the losses.
            mf_loss = tf.reduce_mean(
                -tf.log(tf.nn.sigmoid(neg_score + pos_bias + pos_score - neg_bias)))
            reg_loss_B = tf.nn.l2_loss(B1) + tf.nn.l2_loss(B2)
            reg_loss_emb = tf.nn.l2_loss(user_ebs) + tf.nn.l2_loss(playlist_ebs) + \
                           tf.nn.l2_loss(pos_track_ebs) + tf.nn.l2_loss(neg_track_ebs) + \
                           tf.nn.l2_loss(pos_bias) + tf.nn.l2_loss(neg_bias)
            reg_loss = self.reg_loss_ratio * (reg_loss_emb / self.data.batch_size + reg_loss_B + weight_loss)
            loss = mf_loss + reg_loss

            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

            output_tensor = {
                "mf_loss": mf_loss,
                "reg_loss": reg_loss,
                "loss": loss,
                "opt": opt
            }

            self.train_input_tensors.append(input_tensors)
            self.train_output_tensors.append(output_tensor)

        self.test_input_tensors = {
            # Input Entity ids.
            "track_entity_id": tf.placeholder(tf.int32, shape=[None]),
            # Input Cluster ids.
            "user_cluster_id": tf.placeholder(tf.int32, shape=[None]),
            "playlist_cluster_id": tf.placeholder(tf.int32, shape=[None]),
            "track_cluster_id": tf.placeholder(tf.int32, shape=[None]),
        }
