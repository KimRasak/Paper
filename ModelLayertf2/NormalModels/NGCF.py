from time import time

import numpy as np
import tensorflow as tf

from DataLayer.NormalDatas.NormalData import NormalData
from DataLayer.NormalDatas.NormalPTData import NormalPTData
from DataLayer.NormalDatas.RatingData import RatingData
from ModelLayertf2 import Metric
from ModelLayertf2.BaseModel import Loss
from ModelLayertf2.Metric import Metrics
from ModelLayertf2.NormalModels.NormalModel import FullNGCFLayer
from ModelLayertf2.NormalModels.NormalPTModel import NormalPTModel


class RatingMetric:
    def __init__(self, RMSE, MAE):
        self.RMSE = RMSE
        self.MAE = MAE


class NGCF(NormalPTModel):
    def __init__(self, data: RatingData, epoch_num, embedding_size=64, learning_rate=2e-4, reg_loss_ratio=5e-5, gnn_layer_num=3,
                 dropout_flag=True, dropout_ratio=0.1, layer_num=3):
        super().__init__(data, epoch_num, embedding_size, learning_rate, reg_loss_ratio)
        self.gnn_layer_num = gnn_layer_num
        self.dropout_flag = dropout_flag
        self.dropout_ratio = dropout_ratio
        self.layer_num = layer_num

        self.user_num = data.data_set_num.playlist
        self.item_num = data.data_set_num.track
        self.data_sum = self.user_num + self.item_num

    def _build_model(self):
        self.total_ebs = tf.Variable(self.initializer([self.data_sum, self.embedding_size]), name="total_ebs")

        self.user_ebs = self.total_ebs[:self.user_num]
        self.track_ebs = self.total_ebs[self.user_num:]

        W_sides = [tf.Variable(self.initializer([self.embedding_size, self.embedding_size]),
                               name="W_side_layer_{}".format(layer_no)) for layer_no in range(self.gnn_layer_num)]
        W_dots = [tf.Variable(self.initializer([self.embedding_size, self.embedding_size]),
                              name="W_dot_layer_{}".format(layer_no)) for layer_no in range(self.gnn_layer_num)]

        laplacian_matrices = self.data.laplacian_matrices
        self.full_ngcf_layer = FullNGCFLayer(W_sides, W_dots, laplacian_matrices["LI"], laplacian_matrices["L"],
                                             self.dropout_flag, self.dropout_ratio, self.layer_num)
        stored = {
            "full_ngcf_layer": self.full_ngcf_layer
        }

        return stored

    def _train_batch(self, batch_no, batch):
        with tf.GradientTape() as tape:
            batch_size = batch["size"]
            user_ids = batch["user_ids"]
            item_ids = batch["item_ids"]
            scores = batch["scores"]

            gnn_total_ebs = self.full_ngcf_layer(self.total_ebs, train_flag=True)
            gnn_user_ebs = gnn_total_ebs[:self.user_num]
            gnn_item_ebs = gnn_total_ebs[self.user_num:]

            user_ebs = tf.nn.embedding_lookup(gnn_user_ebs, user_ids)
            item_ebs = tf.nn.embedding_lookup(gnn_item_ebs, item_ids)

            assert batch_size == len(user_ids) == len(item_ids) == len(scores)

            pred_scores = tf.reduce_sum(tf.multiply(user_ebs, item_ebs), 1)
            assert len(pred_scores.shape) == 1 and pred_scores.shape[0] == batch_size

            mf_loss = tf.reduce_mean(tf.pow(pred_scores - scores, 2))
            reg_loss_W = tf.nn.l2_loss(self.full_ngcf_layer.get_reg_loss())
            assert batch_size != 0
            reg_loss_ebs = (tf.nn.l2_loss(user_ebs) + tf.nn.l2_loss(item_ebs)) / batch_size
            reg_loss = self.reg_loss_ratio * (reg_loss_ebs + reg_loss_W)
            assert not np.any(np.isnan(pred_scores)) and not np.any(np.isnan(scores))
            loss = mf_loss + reg_loss

            # Compute and apply gradients.
            update_gradients_start_t = time()
            trainable_variables = []
            trainable_variables.append(self.total_ebs)
            trainable_variables.extend(self.full_ngcf_layer.get_trainable_variables())
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))
            return Loss(reg_loss, mf_loss)

    def _test(self, epoch):
        test_start_time = time()
        # Compute gnn processed embeddings.
        gnn_total_ebs = self.full_ngcf_layer(self.total_ebs, train_flag=False)
        gnn_user_ebs = gnn_total_ebs[:self.user_num]
        gnn_item_ebs = gnn_total_ebs[self.user_num:]

        user_ids = []
        item_ids = []
        scores = []
        test_data = self.data.test_data
        for user_id in test_data:
            # Add element to ui
            for item_id, score in test_data[user_id].items():
                user_ids.append(user_id)
                item_ids.append(item_id)
                scores.append(score)

        user_ebs = tf.nn.embedding_lookup(gnn_user_ebs, user_ids)
        item_ebs = tf.nn.embedding_lookup(gnn_item_ebs, item_ids)
        pred_scores = tf.reduce_sum(tf.multiply(user_ebs, item_ebs), 1)

        RMSE = tf.sqrt(tf.reduce_mean(tf.pow(pred_scores - scores, 2)))
        MAE = tf.reduce_mean(tf.abs(pred_scores - scores))

        test_end_time = time()
        return RatingMetric(RMSE, MAE), test_end_time - test_start_time

    def _output_test_result(self, epoch, test_time, metrics: RatingMetric):
        log_str = "RMSE: {}, MAE: {}".format(metrics.RMSE, metrics.MAE)
        print(log_str)
        self.log_manager.write("Test in epoch {} used {} seconds.\n".format(epoch, test_time))
        self.log_manager.write(log_str)