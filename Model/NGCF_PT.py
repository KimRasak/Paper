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
        ebs_list = [ebs0, ebs1, ebs2, ebs3 ]
        # ebs_list = [ebs0]
        ebs = tf.concat(ebs_list, 1)
        print("ebs1:", ebs1.shape)
        print("ebs:", ebs)

        embed_playlist = tf.nn.embedding_lookup(ebs, self.X_playlist)
        embed_pos_item = tf.nn.embedding_lookup(ebs, self.data.n_playlist + self.X_pos_item)
        embed_neg_item = tf.nn.embedding_lookup(ebs, self.data.n_playlist + self.X_neg_item)
        print("embed_playlist:", embed_playlist)
        print("embed_pos_item", embed_pos_item)
        print("embed_neg_item:", embed_neg_item)

        self.t_eb_playlist = embed_playlist
        self.t_eb_pos_item = embed_pos_item
        self.t_eb_neg_item = embed_neg_item

        # Get embeddings loss.
        self.t_embed_loss = tf.nn.l2_loss(ebs)

        self.t_pos_score = tf.matmul(embed_playlist, embed_pos_item, transpose_b=True)
        self.t_neg_score = tf.matmul(embed_playlist, embed_neg_item, transpose_b=True)
        print("t_pos_score:", self.t_pos_score)
        print("t_neg_score:", self.t_neg_score)
        # self.t_reg_loss =
        self.t_temp = tf.nn.sigmoid(self.t_pos_score - self.t_neg_score)
        self.t_mf_loss = tf.negative(tf.reduce_mean(tf.log(self.t_temp)))
        print("t_mf_loss:", self.t_mf_loss)
        self.t_reg_loss = self.reg_rate * (self.t_embed_loss + self.t_weight_loss) / self.data.batch_size
        self.t_loss = self.t_mf_loss + self.t_reg_loss
        print("t_reg_loss:", self.t_reg_loss)
        print("t_loss:", self.t_loss)
        self.t_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.t_loss)

        # Output for testing/predicting
        predict_playlist_embed = tf.nn.embedding_lookup(ebs, self.X_playlist_predict)
        print("predict_playlist_embed:", predict_playlist_embed)
        items_predict_embeddings = tf.nn.embedding_lookup(ebs, self.data.n_playlist + self.X_items_predict)
        print("items_predict_embeddings:", items_predict_embeddings)
        self.t_predict = tf.matmul(predict_playlist_embed, items_predict_embeddings, transpose_b=True)
        print("t_predict:", self.t_predict)