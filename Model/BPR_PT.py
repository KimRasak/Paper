import numpy as np
import tensorflow as tf
from Model.ModelPT import ModelPT
from Model.utility.data_helper import Data


class BPR_PT(ModelPT):
    def __init__(self, num_epoch, data: Data):
        super().__init__(num_epoch, data)

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
        playlist_embedding = tf.Variable(tf.truncated_normal(shape=[self.data.n_playlist, self.embedding_size], mean=0.0, stddev=0.5))
        track_embedding = tf.Variable(tf.truncated_normal(shape=[self.data.n_track, self.embedding_size], mean=0.0, stddev=0.5))

        embed_playlist = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist)
        embed_pos_item = tf.nn.embedding_lookup(track_embedding, self.X_pos_item)
        embed_neg_item = tf.nn.embedding_lookup(track_embedding, self.X_neg_item)

        self.t_pos_score = tf.matmul(embed_playlist, embed_pos_item, transpose_b=True)
        self.t_neg_score = tf.matmul(embed_playlist, embed_neg_item, transpose_b=True)

        self.t_loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(self.t_pos_score - self.t_neg_score)))
        self.t_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.t_loss)
        # self.print_loss = tf.print("loss: ", self.loss, output_stream=sys.stdout)

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
            "score_loss": loss,
            "pos_score": neg_score,
            "neg_score": neg_score
        }


