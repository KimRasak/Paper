from time import time
import numpy as np
import tensorflow as tf

from Model.ModelUPT import ModelUPT


class MDR(ModelUPT):
    def get_init_embeddings(self):
        return tf.Variable(tf.truncated_normal(shape=[self.data.n_user+self.data.n_playlist+self.data.n_track, self.embedding_size], mean=0.0,
                                stddev=0.5))

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
        embeddings = self.get_init_embeddings()
        user_embedding = embeddings[:self.data.n_user, :]
        playlist_embedding = embeddings[self.data.n_user:self.data.n_user+self.data.n_playlist, :]
        track_embedding = embeddings[self.data.n_user+self.data.n_playlist:, :]
        # user_embedding = tf.Variable(tf.truncated_normal(shape=[self.data.n_user, self.embedding_size], mean=0.0,
        #                         stddev=0.5))
        # playlist_embedding = tf.Variable(tf.truncated_normal(shape=[self.data.n_playlist, self.embedding_size], mean=0.0,
        #                         stddev=0.5))
        # track_embedding = tf.Variable(tf.truncated_normal(shape=[self.data.n_track, self.embedding_size], mean=0.0,
        #                         stddev=0.5))
        track_bias = tf.Variable(tf.truncated_normal(shape=[self.data.n_track], mean=0.0,
                                stddev=0.5))

        embed_user = tf.nn.embedding_lookup(user_embedding, self.X_user)
        embed_playlist = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist)
        embed_pos_item = tf.nn.embedding_lookup(track_embedding, self.X_pos_item)
        embed_neg_item = tf.nn.embedding_lookup(track_embedding, self.X_neg_item)
        bias_pos = tf.nn.embedding_lookup(track_bias, self.X_pos_item)
        bias_neg = tf.nn.embedding_lookup(track_bias, self.X_neg_item)

        B1 = tf.Variable(tf.truncated_normal(shape=[self.embedding_size], mean=0.0, stddev=0.5))
        B2 = tf.Variable(tf.truncated_normal(shape=[self.embedding_size], mean=0.0, stddev=0.5))
        # B = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.embedding_size], mean=0.0, stddev=0.01))

        self.t_pos_score = self.MDR_layer(embed_user, embed_playlist, embed_pos_item, B1, B2)
        self.t_neg_score = self.MDR_layer(embed_user, embed_playlist, embed_neg_item, B1, B2)

        print("embed_pos_item", embed_pos_item)
        print("t_pos_score:", self.t_pos_score)
        print("t_neg_score:", self.t_neg_score)
        print("bias_pos", bias_pos)

        self.t_mf_loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(self.t_pos_score + bias_pos - self.t_neg_score - bias_neg)))

        reg_loss_B = tf.nn.l2_loss(B1) + tf.nn.l2_loss(B2)
        reg_loss_emb = tf.nn.l2_loss(user_embedding) + tf.nn.l2_loss(playlist_embedding) + tf.nn.l2_loss(track_embedding) + tf.nn.l2_loss(track_bias)
        self.t_reg_loss = self.reg_rate * (reg_loss_emb + reg_loss_B)
        self.t_loss = self.t_mf_loss + self.t_reg_loss
        self.t_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.t_loss)
        # self.print_loss = tf.print("loss: ", self.loss, output_stream=sys.stdout)

        # Output for testing/predicting
        predict_user_embed = tf.nn.embedding_lookup(user_embedding, self.X_user_predict)
        predict_playlist_embed = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist_predict)
        items_predict_embeddings = tf.nn.embedding_lookup(track_embedding, self.X_items_predict)
        bias_predict_embedding = tf.nn.embedding_lookup(track_bias, self.X_items_predict)
        self.t_predict = self.MDR_layer(predict_user_embed, predict_playlist_embed, items_predict_embeddings, B1, B2) + bias_predict_embedding
        print("t_predict", self.t_predict)

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