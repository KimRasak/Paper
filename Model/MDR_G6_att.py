from time import time
import numpy as np
import tensorflow as tf

from Model.ModelUPT import ModelUPT


class MDR_G6_att(ModelUPT):
    def get_init_embeddings(self):
        return tf.Variable(self.initializer([self.data.n_user + self.data.n_playlist + self.data.n_track, self.embedding_size]))

    def build_graph_layers(self, embeddings):
        embeddings1 = self.build_graph_UPT(embeddings, self.embedding_size, self.embedding_size)
        embeddings2 = self.build_graph_UPT(embeddings1, self.embedding_size, self.embedding_size)
        embeddings3 = self.build_graph_UPT(embeddings2, self.embedding_size, self.embedding_size)
        return embeddings1, embeddings2, embeddings3

    def MDR_layer(self, embed_user, embed_playlist, embed_track, B1, B2):
        delta_ut = embed_user - embed_track
        delta_pt = embed_playlist - embed_track

        def get_output(delta, B):
            B_delta = tf.multiply(B, delta)
            square = tf.square(B_delta)
            print("square:", square, len(square.shape) - 1)
            return tf.reduce_sum(square, axis=len(square.shape) - 1)

        o1 = get_output(delta_ut, B1)
        o2 = get_output(delta_pt, B2)
        print("o1:", o1)

        return o1 + o2

    def get_attentive_scores(self, raw_scores):
        softmax_scores = tf.nn.softmax(raw_scores, axis=1)
        scores = tf.reduce_sum(tf.multiply(softmax_scores, raw_scores), axis=len(raw_scores.shape) - 1)
        print("raw scores:", raw_scores)
        print("softmax_scores:", softmax_scores)
        print("scores:", scores)
        return scores

    def get_layers_scores(self, ebs_list, B1, B2):
        raw_scores_pos = None
        raw_scores_neg = None
        raw_scores_predict = None
        reg_loss_emb = 0
        track_bias = tf.Variable(self.initializer([self.data.n_track]))
        bias_pos = tf.nn.embedding_lookup(track_bias, self.X_pos_item)
        bias_neg = tf.nn.embedding_lookup(track_bias, self.X_neg_item)
        bias_predict_embedding = tf.nn.embedding_lookup(track_bias, self.X_items_predict)
        for ebs in ebs_list:
            user_embedding = ebs[:self.data.n_user, :]
            playlist_embedding = ebs[self.data.n_user:self.data.n_user + self.data.n_playlist, :]
            track_embedding = ebs[self.data.n_user + self.data.n_playlist:, :]

            # Embedding for training.
            embed_user = tf.nn.embedding_lookup(user_embedding, self.X_user)
            embed_playlist = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist)
            embed_pos_item = tf.nn.embedding_lookup(track_embedding, self.X_pos_item)
            embed_neg_item = tf.nn.embedding_lookup(track_embedding, self.X_neg_item)

            # Output for testing/predicting
            predict_user_embed = tf.nn.embedding_lookup(user_embedding, self.X_user_predict)
            predict_playlist_embed = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist_predict)
            items_predict_embeddings = tf.nn.embedding_lookup(track_embedding, self.X_items_predict)

            pos_score = self.MDR_layer(embed_user, embed_playlist, embed_pos_item, B1, B2)
            neg_score = self.MDR_layer(embed_user, embed_playlist, embed_neg_item, B1, B2)
            predict_score = self.MDR_layer(predict_user_embed, predict_playlist_embed, items_predict_embeddings, B1, B2)

            raw_scores_pos = pos_score if raw_scores_pos is None else tf.concat([raw_scores_pos, pos_score], axis=1)
            raw_scores_neg = neg_score if raw_scores_neg is None else tf.concat([raw_scores_neg, neg_score], axis=1)

            expand_predict_score = tf.expand_dims(predict_score, -1)
            print("pos_score:", pos_score)
            print("raw_scores_pos:", raw_scores_pos)
            print("predict_score:", predict_score)
            print("expand_predict_score:", expand_predict_score)
            raw_scores_predict = expand_predict_score if raw_scores_predict is None else tf.concat([raw_scores_predict, expand_predict_score], axis=1)
            print("raw_scores_predict:", raw_scores_predict)

            reg_loss_emb += tf.nn.l2_loss(ebs)

            print("embed_pos_item", embed_pos_item)
            print("pos_score:", pos_score)
            print("bias_pos", bias_pos)

        reg_loss_bias = tf.nn.l2_loss(track_bias)

        oo = self.get_attentive_scores(raw_scores_pos)
        scores_pos = oo + tf.squeeze(bias_pos)
        scores_neg = self.get_attentive_scores(raw_scores_neg) + tf.squeeze(bias_neg)
        print("self.get_attentive_scores(raw_scores_predict):", self.get_attentive_scores(raw_scores_predict))
        print("bias_predict_embedding:", bias_predict_embedding)
        scores_predict = self.get_attentive_scores(raw_scores_predict) + bias_predict_embedding
        print("bias_pos", bias_pos)
        print("oo", oo)
        print("scores_pos: ", scores_pos)
        print("scores_neg:", scores_neg)
        print("scores_predict: ", scores_predict)
        return scores_pos, scores_neg, scores_predict, reg_loss_emb, reg_loss_bias

    def build_model(self):
        super().build_model()

        # An input for training is a traid (playlist id, positive_item id, negative_item id)
        batch_size = self.data.batch_size
        self.X_user = tf.placeholder(tf.int32, shape=(batch_size, 1))
        self.X_playlist = tf.placeholder(tf.int32, shape=(batch_size, 1))
        self.X_pos_item = tf.placeholder(tf.int32, shape=(batch_size, 1))
        self.X_neg_item = tf.placeholder(tf.int32, shape=(batch_size, 1))

        # An input for testing/predicting is only the playlist id
        self.X_user_predict = tf.placeholder(tf.int32, shape=(1), name="x_user_predict")
        self.X_playlist_predict = tf.placeholder(tf.int32, shape=(1), name="x_playlist_predict")
        self.X_items_predict = tf.placeholder(tf.int32, shape=(101), name="x_items_predict")

        # Loss, optimizer definition for training.
        ebs0 = self.get_init_embeddings()
        ebs1, ebs2, ebs3 = self.build_graph_layers(ebs0)
        ebs_list = [ebs0, ebs1, ebs2, ebs3]

        B1 = tf.Variable(self.initializer([self.embedding_size]))
        B2 = tf.Variable(self.initializer([self.embedding_size]))
        self.t_pos_score, self.t_neg_score, self.t_predict, reg_loss_emb, reg_loss_bias = self.get_layers_scores(ebs_list, B1, B2)
        print("t_predict", self.t_predict)
        self.t_mf_loss = tf.reduce_sum(-tf.log(tf.nn.sigmoid(self.t_pos_score - self.t_neg_score) + 1e-8))

        reg_loss_B = tf.nn.l2_loss(B1) + tf.nn.l2_loss(B2)
        self.t_reg_loss = self.reg_rate * (reg_loss_emb + reg_loss_bias + reg_loss_B + self.t_weight_loss)
        self.t_loss = self.t_mf_loss + self.t_reg_loss
        self.t_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.t_loss)

    def train_batch(self, batch):
        for key, batch_value in batch.items():
            batch[key] = np.array(batch_value).reshape(-1, 1)
        opt, loss, mf_loss, reg_loss, pos_score, neg_score = self.sess.run([self.t_opt, self.t_loss,
                                                                                  self.t_mf_loss, self.t_reg_loss, self.t_pos_score, self.t_neg_score], feed_dict={
            self.X_user: batch["users"],
            self.X_playlist: batch["playlists"],
            self.X_pos_item: batch["pos_tracks"],
            self.X_neg_item: batch["neg_tracks"]
        })

        return {
            "loss": loss,
            "mf_loss": mf_loss,
            "reg_loss": reg_loss,
        }