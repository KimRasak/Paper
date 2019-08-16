import tensorflow as tf
import numpy as np
from time import time
from Model.ModelPT import ModelPT
from Model.utility.data_helper import Data

"""
Neural Graph Collaborative Filtering on playlist-track recommendation.
NGCF_PT only uses playlist and track embeddings to model the recommendation, 
but may train user, playlist, track embeddings. (depending on the implementation)  
"""
class NGCF_PT(ModelPT):
    def __init__(self, num_epoch, data: Data):
        super().__init__(num_epoch, data)

    def build_graph_layers(self, embeddings):
        embeddings1 = self.build_graph_PT(embeddings, self.embedding_size, self.embedding_size, num_weight=2)
        embeddings2 = self.build_graph_PT(embeddings1, self.embedding_size, self.embedding_size, num_weight=2)
        embeddings3 = self.build_graph_PT(embeddings2, self.embedding_size, self.embedding_size, num_weight=2)
        return embeddings1, embeddings2, embeddings3

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
        ebs0 = self.get_init_embeddings()
        ebs1, ebs2, ebs3 = self.build_graph_layers(ebs0)
        ebs_list = [ebs0, ebs1, ebs2, ebs3]

        embed_playlist = self.get_concat_embedding(ebs_list, self.X_playlist)
        embed_pos_item = self.get_concat_embedding(ebs_list, self.data.n_playlist + self.X_pos_item)
        embed_neg_item = self.get_concat_embedding(ebs_list, self.data.n_playlist + self.X_neg_item)
        self.t_eb_playlist = embed_playlist
        self.t_eb_pos_item = embed_pos_item
        self.t_eb_neg_item = embed_neg_item

        # Get embeddings loss.
        self.t_embed_loss = tf.nn.l2_loss(ebs_list[0])
        for ebs in ebs_list[1:]:
            e_loss = tf.nn.l2_loss(ebs)
            self.t_embed_loss = self.t_embed_loss + e_loss

        self.t_pos_score = tf.reduce_sum(tf.multiply(embed_playlist, embed_pos_item), axis=1)
        self.t_neg_score = tf.reduce_sum(tf.multiply(embed_playlist, embed_neg_item), axis=1)

        # self.t_reg_loss =
        self.t_temp = tf.nn.sigmoid(self.t_pos_score - self.t_neg_score)
        self.t_mf_loss = tf.negative(tf.reduce_mean(tf.log(self.t_temp)))
        self.t_reg_loss = self.reg_rate * (self.t_embed_loss + self.t_weight_loss)
        self.t_loss = self.t_mf_loss + self.t_reg_loss
        self.t_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.t_loss)

        # Output for testing/predicting
        predict_playlist_embed = self.get_concat_embedding(ebs_list, self.X_playlist_predict)
        items_predict_embeddings = self.get_concat_embedding(ebs_list, self.data.n_playlist + self.X_items_predict)
        self.t_predict = tf.matmul(predict_playlist_embed, items_predict_embeddings, transpose_b=True)

    def get_concat_embedding(self, ebs_list, index):
        num = 1
        embed = tf.nn.embedding_lookup(ebs_list[0], index)
        rank = len(embed.shape)
        for ebs in ebs_list[1:]:
            embed_l = tf.nn.embedding_lookup(ebs, index)
            embed = tf.concat([embed, embed_l], rank - 1)
            num += 1
        return tf.reshape(embed, [embed.shape[0], embed.shape[-1]])

    def train_batch(self, batch):
        t1 = time()
        for key, batch_value in batch.items():
            batch[key] = np.array(batch_value).reshape(-1, 1)
        opt, temp, loss, mf_loss, reg_loss, pos_score, neg_score, eb_p, eb_pos, eb_neg = self.sess.run([self.t_opt, self.t_temp, self.t_loss,
                                                                                  self.t_mf_loss, self.t_reg_loss, self.t_pos_score, self.t_neg_score,
                                                                                  self.t_eb_playlist, self.t_eb_pos_item, self.t_eb_neg_item], feed_dict={
            self.X_playlist: batch["playlists"],
            self.X_pos_item: batch["pos_tracks"],
            self.X_neg_item: batch["neg_tracks"]
        })
        batch_time = time() - t1

        return {
            "loss": loss,
            "temp": temp,
            "mf_loss": mf_loss,
            "reg_loss": reg_loss,
            "pos_score": neg_score,
            "neg_score": neg_score,
            "batch_time": batch_time
        }