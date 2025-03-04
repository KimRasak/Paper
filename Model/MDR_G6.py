from time import time
import numpy as np
import tensorflow as tf

from Model.ModelUPT import ModelUPT
from tensorflow.contrib.layers import xavier_initializer


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
    print("o1:", o1)

    return o1 + o2

class MDR_G6(ModelUPT):
    def get_init_embeddings(self):
        return tf.Variable(self.initializer([self.data.n_user + self.data.n_playlist + self.data.n_track, self.embedding_size]))

    def build_graph_layers(self, embeddings):
        embeddings1 = self.build_graph_UPT(embeddings, self.embedding_size, self.embedding_size)
        embeddings2 = self.build_graph_UPT(embeddings1, self.embedding_size, self.embedding_size)
        embeddings3 = self.build_graph_UPT(embeddings2, self.embedding_size, self.embedding_size)
        return embeddings1, embeddings2, embeddings3

    def concat_eb(self, eb0, eb1):
        return eb1 if eb0 is None else tf.concat([eb0, eb1], axis=len(eb0.shape) - 1)

    # def get_layers_scores(self, ebs_list, B1, B2):
    #     track_bias = tf.Variable(self.initializer([self.data.n_track]))
    #     bias_pos = tf.nn.embedding_lookup(track_bias, self.X_pos_item)
    #     bias_neg = tf.nn.embedding_lookup(track_bias, self.X_neg_item)
    #     bias_predict_embedding = tf.nn.embedding_lookup(track_bias, self.X_items_predict)
    #
    #     self.t_embed_user = None
    #     self.t_embed_playlist = None
    #     self.t_embed_pos_item = None
    #     embed_neg_item = None
    #     predict_user_embed = None
    #     predict_playlist_embed = None
    #     predict_items_embeddings = None
    #
    #     for ebs in ebs_list:
    #         user_embedding = ebs[:self.data.n_user, :]
    #         playlist_embedding = ebs[self.data.n_user:self.data.n_user + self.data.n_playlist, :]
    #         track_embedding = ebs[self.data.n_user + self.data.n_playlist:, :]
    #
    #         layer_embed_user = tf.nn.embedding_lookup(user_embedding, self.X_user)
    #         layer_embed_playlist = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist)
    #         layer_embed_pos_item = tf.nn.embedding_lookup(track_embedding, self.X_pos_item)
    #         layer_embed_neg_item = tf.nn.embedding_lookup(track_embedding, self.X_neg_item)
    #
    #         layer_predict_user_embed = tf.nn.embedding_lookup(user_embedding, self.X_user_predict)
    #         layer_predict_playlist_embed = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist_predict)
    #         layer_predict_items_embeddings = tf.nn.embedding_lookup(track_embedding, self.X_items_predict)
    #
    #         self.t_embed_user = self.concat_eb(self.t_embed_user, layer_embed_user)
    #         self.t_embed_playlist = self.concat_eb(self.t_embed_playlist, layer_embed_playlist)
    #         self.t_embed_pos_item = self.concat_eb(self.t_embed_pos_item, layer_embed_pos_item)
    #         embed_neg_item = self.concat_eb(embed_neg_item, layer_embed_neg_item)
    #
    #         predict_user_embed = self.concat_eb(predict_user_embed, layer_predict_user_embed)
    #         predict_playlist_embed = self.concat_eb(predict_playlist_embed, layer_predict_playlist_embed)
    #         predict_items_embeddings = self.concat_eb(predict_items_embeddings, layer_predict_items_embeddings)
    #
    #     print("embed_user:", self.t_embed_user)
    #     self.t_pos_score = MDR_layer(self.t_embed_user, self.t_embed_playlist, self.t_embed_pos_item, B1, B2)
    #     self.t_neg_score = MDR_layer(self.t_embed_user, self.t_embed_playlist, embed_neg_item, B1, B2)
    #
    #     self.delt = self.t_pos_score + bias_pos - self.t_neg_score - bias_neg
    #
    #     self.t_mf_loss = tf.reduce_sum(-tf.log(tf.nn.sigmoid(self.t_pos_score + bias_pos - self.t_neg_score - bias_neg)))
    #     reg_loss_emb = tf.nn.l2_loss(self.t_embed_user) + tf.nn.l2_loss(self.t_embed_playlist) + tf.nn.l2_loss(self.t_embed_pos_item) + tf.nn.l2_loss(embed_neg_item) + tf.nn.l2_loss(bias_pos) + tf.nn.l2_loss(bias_neg)
    #     reg_loss_B = tf.nn.l2_loss(B1) + tf.nn.l2_loss(B2)
    #     self.t_reg_loss = self.reg_rate * (reg_loss_emb + reg_loss_B + self.t_weight_loss)
    #     self.t_loss = self.t_mf_loss + self.t_reg_loss
    #     self.t_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.t_loss)
    #
    #     self.t_predict = MDR_layer(predict_user_embed, predict_playlist_embed, predict_items_embeddings, B1, B2) + bias_predict_embedding

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

        # embeddings
        ebs0 = self.get_init_embeddings()
        ebs1, ebs2, ebs3 = self.build_graph_layers(ebs0)
        self.ebs0, self.ebs1, self.ebs2, self.ebs3 = ebs0, ebs1, ebs2, ebs3
        ebs_list = [ebs0, ebs1, ebs2, ebs3]

        # B1 = tf.Variable(self.initializer([self.embedding_size * 4]))
        # B2 = tf.Variable(self.initializer([self.embedding_size * 4]))
        # self.get_layers_scores(ebs_list, B1, B2)
        ebs = tf.concat(ebs_list, 1)
        user_embedding = ebs[:self.data.n_user, :]
        playlist_embedding = ebs[self.data.n_user:self.data.n_user + self.data.n_playlist, :]
        track_embedding = ebs[self.data.n_user + self.data.n_playlist:, :]
        track_bias = tf.Variable(self.initializer([self.data.n_track]))

        embed_user = tf.nn.embedding_lookup(user_embedding, self.X_user)
        embed_playlist = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist)
        embed_pos_item = tf.nn.embedding_lookup(track_embedding, self.X_pos_item)
        embed_neg_item = tf.nn.embedding_lookup(track_embedding, self.X_neg_item)
        bias_pos = tf.nn.embedding_lookup(track_bias, self.X_pos_item)
        bias_neg = tf.nn.embedding_lookup(track_bias, self.X_neg_item)
        print("embed_pos_item", embed_pos_item.shape)
        print("embed_playlist:", embed_playlist)

        self.t_eb_user = embed_user
        self.t_eb_playlist = embed_playlist
        self.t_eb_pos_item = embed_pos_item
        self.t_eb_neg_item = embed_neg_item

        B1 = tf.Variable(self.initializer([self.embedding_size * 4]))
        B2 = tf.Variable(self.initializer([self.embedding_size * 4]))

        self.t_pos_score = MDR_layer(embed_user, embed_playlist, embed_pos_item, B1, B2)
        self.t_neg_score = MDR_layer(embed_user, embed_playlist, embed_neg_item, B1, B2)
        print("t_pos_score:", self.t_pos_score)

        self.delt = self.t_pos_score + bias_pos - self.t_neg_score - bias_neg
        self.t_mf_loss = tf.reduce_sum(-tf.log(tf.nn.sigmoid(self.t_pos_score + bias_pos - self.t_neg_score - bias_neg)))

        reg_loss_B = tf.nn.l2_loss(B1) + tf.nn.l2_loss(B2)
        # reg_loss_emb = tf.nn.l2_loss(ebs) + tf.nn.l2_loss(track_bias)
        reg_loss_emb = tf.nn.l2_loss(embed_user) + tf.nn.l2_loss(embed_pos_item) + tf.nn.l2_loss(embed_neg_item) + tf.nn.l2_loss(bias_pos) + tf.nn.l2_loss(bias_neg)
        self.t_reg_loss = self.reg_rate * (reg_loss_emb + reg_loss_B + self.t_weight_loss)
        self.t_loss = self.t_mf_loss + self.t_reg_loss
        self.t_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.t_loss)

        # Output for testing/predicting
        predict_user_embed = tf.nn.embedding_lookup(user_embedding, self.X_user_predict)
        predict_playlist_embed = tf.nn.embedding_lookup(playlist_embedding, self.X_playlist_predict)
        items_predict_embeddings = tf.nn.embedding_lookup(track_embedding, self.X_items_predict)
        bias_predict_embedding = tf.nn.embedding_lookup(track_bias, self.X_items_predict)
        self.t_predict = MDR_layer(predict_user_embed, predict_playlist_embed, items_predict_embeddings, B1, B2) + bias_predict_embedding
        print("t_predict", self.t_predict)

    def train_batch(self, batch):

        for i, t in enumerate(batch['pos_tracks']):
            assert t < self.data.n_track  # 采样的id在合理范围
            assert t != batch['neg_tracks'][i]  # 正负采样不同
            pl = batch['playlists'][i]
            u = batch['users'][i]
            assert self.data.R_up[u, pl] > 0
            assert self.data.R_pt[pl, t] > 0 and self.data.R_ut[u, t] >= 0, "%r %r" % (t in self.data.R_pt[pl], t in self.data.R_ut[u])  # 正采样
            assert pl in self.data.up[u] and t in self.data.pt[pl]
            assert batch['neg_tracks'][i] not in self.data.pt[pl]  # 负采样
        for i in batch['neg_tracks']:
            assert i < self.data.n_track
        for i in batch['users']:
            assert i < self.data.n_user
        for i in batch['playlists']:
            assert i < self.data.n_playlist

        for key, batch_value in batch.items():
            batch[key] = np.array(batch_value).reshape(-1, 1)
        opt, \
        loss, mf_loss, reg_loss, \
        pos_score, neg_score, \
        eb_user, eb_playlist, eb_pos_item, \
        ebs0, ebs1, ebs2, ebs3,\
        delt = self.sess.run([self.t_opt, self.t_loss, self.t_mf_loss, self.t_reg_loss,
                                                                            self.t_pos_score, self.t_neg_score,
                                                                            self.t_embed_user, self.t_embed_playlist, self.t_embed_pos_item,
                                                                            self.ebs0, self.ebs1, self.ebs2, self.ebs3,
                                                                            self.delt], feed_dict={
            self.X_user: batch["users"],
            self.X_playlist: batch["playlists"],
            self.X_pos_item: batch["pos_tracks"],
            self.X_neg_item: batch["neg_tracks"]
        })

        # for d in delt:
        #     if d == 0 or np.isnan(d):
        #         print("found...")
        #         break
        #
        # if np.isnan(loss) or np.isnan(reg_loss) or np.isnan(mf_loss):
        #     print("loss")

        return {
            "loss": loss,
            "mf_loss": mf_loss,
            "reg_loss": reg_loss,
        }