import numpy as np
import tensorflow as tf
from Model.ModelPT import ModelPT
from Model.utility.data_helper import Data


class BPR_PT(ModelPT):
    def get_init_embeddings(self):
        return None

    def build_model(self):
        super().build_model()

        # An input for training is a traid (playlist id, positive_item id, negative_item id)
        batch_size = self.data.batch_size
        self.X_playlist = tf.placeholder(tf.int32, shape=(batch_size, 1))
        self.X_pos_item = tf.placeholder(tf.int32, shape=(batch_size, 1))
        self.X_neg_item = tf.placeholder(tf.int32, shape=(batch_size, 1))

        # An input for testing/predicting is only the playlist id
        self.X_playlist_predict = tf.placeholder(tf.int32, shape=(1), name="x_playlist_predict")
        self.X_items_predict = tf.placeholder(tf.int32, shape=(101), name="x_items_predict")

        # Loss, optimizer definition for training.
        playlist_embedding = tf.Variable(self.initializer([self.data.n_playlist, self.embedding_size]))
        track_embedding = tf.Variable(self.initializer([self.data.n_track, self.embedding_size]))

        embed_playlist = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist)
        embed_pos_item = tf.nn.embedding_lookup(track_embedding, self.X_pos_item)
        embed_neg_item = tf.nn.embedding_lookup(track_embedding, self.X_neg_item)

        self.t_pos_score = tf.matmul(embed_playlist, embed_pos_item, transpose_b=True)
        self.t_neg_score = tf.matmul(embed_playlist, embed_neg_item, transpose_b=True)

        self.t_mf_loss = tf.reduce_sum(-tf.log(tf.nn.sigmoid(self.t_pos_score - self.t_neg_score)))

        self.t_reg_loss = tf.nn.l2_loss(playlist_embedding) + tf.nn.l2_loss(track_embedding)
        self.t_loss = self.t_mf_loss + self.reg_rate * self.t_reg_loss
        self.t_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.t_loss)

        # Output for testing/predicting
        predict_playlist_embed = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist_predict)
        items_predict_embeddings = tf.nn.embedding_lookup(track_embedding, self.X_items_predict)
        self.t_predict = tf.matmul(predict_playlist_embed, items_predict_embeddings, transpose_b=True)

    def train_batch(self, batch):
        for key, batch_value in batch.items():
            batch[key] = np.array(batch_value).reshape(-1, 1)

        opt, loss, pos_score, neg_score = self.sess.run([self.t_opt, self.t_loss, self.t_pos_score, self.t_neg_score], feed_dict={
            self.X_playlist: batch["playlists"],
            self.X_pos_item: batch["pos_tracks"],
            self.X_neg_item: batch["neg_tracks"]
        })

        return {
            "loss": loss,
            "pos_score": neg_score,
            "neg_score": neg_score
        }


