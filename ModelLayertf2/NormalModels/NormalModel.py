from abc import ABC, abstractmethod
from time import time

from DataLayer.NormalDatas.NormalData import NormalData
from ModelLayertf2 import Metric
from ModelLayertf2.BaseModel import BaseModel, Loss, _convert_sp_mat_to_sp_tensor

import tensorflow as tf
from tensorflow.keras import layers


class NGCFLayer(tf.keras.layers.Layer):
    def __init__(self, W_side, W_dot,
                 LI, L, dropout_flag: bool, drop_out_ratio: float, **kwargs):
        super().__init__(**kwargs)

        self.W_side = W_side
        self.W_dot = W_dot
        self.LI = _convert_sp_mat_to_sp_tensor(LI)
        self.L = _convert_sp_mat_to_sp_tensor(L)
        self.dropout_flag = dropout_flag
        self.drop_out_ratio = drop_out_ratio

    def call(self, ebs, **kwargs):
        train_flag = kwargs["train_flag"]
        folds = []
        W_side = self.W_side
        W_dot = self.W_dot

        LI_side_embed = tf.sparse.sparse_dense_matmul(self.LI, ebs)
        L_side_embed = tf.sparse.sparse_dense_matmul(self.L, ebs)

        sum_embed = tf.matmul(LI_side_embed, W_side)
        dot_embed = tf.matmul(tf.multiply(L_side_embed, ebs), W_dot)

        fold = tf.nn.leaky_relu(sum_embed + dot_embed)
        folds.append(fold)
        agg = tf.concat(folds, axis=0)
        if train_flag and self.dropout_flag:
            dropout_embed = tf.nn.dropout(agg, self.drop_out_ratio)
            return dropout_embed
        else:
            return agg

    def get_trainable_variables(self):
        return [self.W_dot, self.W_side]


class FullNGCFLayer(tf.keras.layers.Layer):
    def __init__(self, W_sides, W_dots,
                 LI, L, dropout_flag: bool, drop_out_ratio: float, layer_num: int,
                 **kwargs):
        self.W_sides = W_sides
        self.W_dots = W_dots
        self.LI = LI
        self.L = L
        self.dropout_flag = dropout_flag
        self.drop_out_ratio = drop_out_ratio
        self.layer_num = layer_num

        self.ngcf_layers = []
        self.Ws = []
        for layer_no in range(layer_num):
            W_side = W_sides[layer_no]
            W_dot = W_dots[layer_no]
            ngcfLayer = NGCFLayer(W_side, W_dot,
                           LI, L, dropout_flag, drop_out_ratio)
            self.ngcf_layers.append(ngcfLayer)
            self.Ws.extend([W_side, W_dot])
        super().__init__(**kwargs)

    def call(self, initial_ebs, **kwargs):
        train_flag = kwargs["train_flag"]

        # initial_ebs = self.cluster_initial_ebs[cluster_no]
        old_ebs = initial_ebs
        layers_ebs = []
        for layer_no in range(self.layer_num):
            NGCF_layer = self.ngcf_layers[layer_no]
            new_ebs = NGCF_layer(old_ebs, train_flag=train_flag)
            layers_ebs.append(new_ebs)

            # Update variable.
            old_ebs = new_ebs
        agg = tf.concat(layers_ebs, axis=0)
        return agg

    def get_reg_loss(self):
        reg_loss = 0
        for W in self.Ws:
            reg_loss += tf.nn.l2_loss(W)
        return reg_loss

    def get_trainable_variables(self):
        return self.Ws


class NormalModel(BaseModel, ABC):
    def __init__(self, data: NormalData, epoch_num, embedding_size=64, learning_rate=2e-4, reg_loss_ratio=5e-5):
        super().__init__(epoch_num, embedding_size,
                         learning_rate, reg_loss_ratio)
        self.data = data

    def _train_epoch(self, epoch):
        epoch_loss = Loss()
        epoch_start_time = time()
        for batch_no, batch in enumerate(self.data.get_batches()):
            batch_loss: Loss = self._train_batch(batch_no, batch)
            epoch_loss.add_loss(batch_loss)
        epoch_end_time = time()
        return epoch_loss, epoch_end_time - epoch_start_time

    @abstractmethod
    def _train_batch(self, batch_no, batch):
        pass

    def _output_test_result(self, epoch, test_time, metrics: Metric):
        print("hrs_10: {}, ndcgs_10: {}".format(metrics.get_avg_hr(10), metrics.get_avg_ndcg(10)))
        self.log_manager.write("Test in epoch {} used {} seconds.\n".format(epoch, test_time))
        self.log_manager.write(metrics.to_string())
