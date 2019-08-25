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
def convert_sp_mat_to_sp_tensor(X):
    if X.getnnz() == 0:
        print("add one.", X.shape)
        X[0, 0] = 1
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()

    return tf.sparse.SparseTensor(indices, coo.data, dense_shape=X.shape)

class NGCF_PT_PT2(ModelPT):
    def get_init_embeddings(self):
        return tf.Variable(self.initializer([self.data.n_playlist + self.data.n_track, self.embedding_size]))

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.data.n_playlist + self.data.n_track) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.data.n_playlist + self.data.n_track
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

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

        self.n_fold = 100
        LI_fold_hat = self._split_A_hat(self.data.LI)
        L_fold_hat = self._split_A_hat(self.data.L)

        ego_embeddings = self.get_init_embeddings()
        all_embeddings = [ego_embeddings]
        for k in range(0, 3):
            eb_size = self.embedding_size
            W1 = tf.Variable(self.initializer([eb_size, eb_size]))
            W2 = tf.Variable(self.initializer([eb_size, eb_size]))

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(LI_fold_hat[f], ego_embeddings))
            side_LI_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.matmul(side_LI_embeddings, W1)

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(L_fold_hat[f], ego_embeddings))
            side_L_embeddings = tf.concat(temp_embed, 0)
            bi_embeddings = tf.matmul(tf.multiply(side_L_embeddings, ego_embeddings), W2)

            ego_embeddings = tf.nn.leaky_relu(sum_embeddings + bi_embeddings)
            self.t_weight_loss += tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)

            all_embeddings += [ego_embeddings]

        ebs = tf.concat(all_embeddings, 1)

        embed_playlist = tf.nn.embedding_lookup(ebs, self.X_playlist)
        embed_pos_item = tf.nn.embedding_lookup(ebs, self.data.n_playlist + self.X_pos_item)
        embed_neg_item = tf.nn.embedding_lookup(ebs, self.data.n_playlist + self.X_neg_item)
        print("embed_pos_item", embed_pos_item.shape)
        print("embed_playlist:", embed_playlist)

        self.t_eb_playlist = embed_playlist
        self.t_eb_pos_item = embed_pos_item
        self.t_eb_neg_item = embed_neg_item

        # Get embeddings loss.
        self.t_embed_loss = tf.nn.l2_loss(ebs)

        self.t_pos_score = tf.matmul(embed_playlist, embed_pos_item, transpose_b=True)
        self.t_neg_score = tf.matmul(embed_playlist, embed_neg_item, transpose_b=True)
        print("t_pos_score:", self.t_pos_score)
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

    def train_batch(self, batch):
        for key, batch_value in batch.items():
            batch[key] = np.array(batch_value).reshape(-1, 1)
        opt, temp, loss, mf_loss, reg_loss, pos_score, neg_score, eb_p, eb_pos, eb_neg = self.sess.run([self.t_opt, self.t_temp, self.t_loss,
                                                                                  self.t_mf_loss, self.t_reg_loss, self.t_pos_score, self.t_neg_score,
                                                                                  self.t_eb_playlist, self.t_eb_pos_item, self.t_eb_neg_item], feed_dict={
            self.X_playlist: batch["playlists"],
            self.X_pos_item: batch["pos_tracks"],
            self.X_neg_item: batch["neg_tracks"]
        })

        return {
            "loss": loss,
            "temp": temp,
            "mf_loss": mf_loss,
            "reg_loss": reg_loss
        }