from abc import abstractmethod, ABC
from time import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from DataLayer.ClusterDatas.ClusterData import ClusterData
from ModelLayertf2.BaseModel import BaseModel, Loss
from ModelLayertf2.Strategy.NegativeTrainSampleStrategy import OtherClusterStrategyTrain


class GNN(tf.keras.layers.Layer):
    def __init__(self,
                 W_sides, W_dots,
                 LIs: dict, Ls: dict,
                 eb_size: int, bounds: dict, dropout_flag: bool, drop_out_ratio: float):
        super(GNN, self).__init__()
        self.bounds = bounds
        self.dropout_flag = dropout_flag
        self.dropout_ratio = drop_out_ratio

        # Tensor variables.
        self.W_sides = W_sides
        self.W_dots = W_dots

        # Constant sparse laplacian matrix tensors.
        self.LIs = {
            entity_name: GNN.__convert_sp_mat_to_sp_tensor(LI)
            for entity_name, LI in LIs.items()
        }

        self.Ls = {
            entity_name: GNN.__convert_sp_mat_to_sp_tensor(L)
            for entity_name, L in Ls.items()
        }

    @staticmethod
    def __convert_sp_mat_to_sp_tensor(X):
        if X.getnnz() == 0:
            print("add one.", X.shape)
            return 0
            # X[0, 0] = 1
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()

        return tf.sparse.SparseTensor(indices, coo.data, dense_shape=X.shape)

    def call(self, ebs, **kwargs):
        train_flag = kwargs["train_flag"]
        folds = []
        for entity_name, (start, end) in self.bounds.items():
            W_side = self.W_sides[entity_name]
            W_dot = self.W_dots[entity_name]
            entity_emb = ebs[start:end]

            LI_side_embed = tf.sparse.sparse_dense_matmul(self.LIs[entity_name], ebs)
            L_side_embed = tf.sparse.sparse_dense_matmul(self.Ls[entity_name], ebs)

            sum_embed = tf.matmul(LI_side_embed, W_side)
            dot_embed = tf.matmul(tf.multiply(L_side_embed, entity_emb), W_dot)

            fold = tf.nn.leaky_relu(sum_embed + dot_embed)
            folds.append(fold)
        agg = tf.concat(folds, axis=0)
        if train_flag and self.dropout_flag:
            dropout_embed = tf.nn.dropout(agg, self.dropout_ratio)
            return dropout_embed
        else:
            return agg

    def get_trainable_variables(self):
        variables = []
        variables.extend(list(self.W_sides.values()))
        variables.extend(list(self.W_dots.values()))
        return variables


class FullGNN(layers.Layer):
    def __init__(self, initializer, eb_size,
                 cluster_laplacian_matrices, cluster_bounds, entity_names,
                 dropout_flag, drop_out_ratio,
                 cluster_num, layer_num):
        super(FullGNN, self).__init__()
        self.layer_num = layer_num
        self.multi_GNN_layers = []
        self.Ws = []
        for layer_no in range(layer_num):
            GNN_layers = []

            W_sides = {
                entity_name: tf.Variable(initializer([eb_size, eb_size]),
                                         name="W_side_layer_{}_{}".format(layer_no, entity_name))
                for entity_name in entity_names
            }

            W_dots = {
                entity_name: tf.Variable(initializer([eb_size, eb_size]),
                                         name="W_dot_layer_{}_{}".format(layer_no, entity_name))
                for entity_name in entity_names
            }

            self.Ws.extend(list(W_sides.values()))
            self.Ws.extend(list(W_dots.values()))

            for cluster_no in range(cluster_num):
                LIs = {
                    entity_name: cluster_laplacian_matrices[cluster_no][entity_name]["LI"]
                    for entity_name in entity_names
                }
                Ls = {
                    entity_name: cluster_laplacian_matrices[cluster_no][entity_name]["L"]
                    for entity_name in entity_names
                }
                bounds = cluster_bounds[cluster_no]
                GNN_layers.append(GNN(W_sides, W_dots, LIs, Ls, eb_size, bounds, dropout_flag, drop_out_ratio))
            self.multi_GNN_layers.append(GNN_layers)

    def call(self, initial_ebs, **kwargs):
        cluster_no = kwargs["cluster_no"]
        train_flag = kwargs["train_flag"]

        # initial_ebs = self.cluster_initial_ebs[cluster_no]
        old_ebs = initial_ebs
        layers_ebs = []
        for layer_no in range(self.layer_num):
            GNN_layer = self.multi_GNN_layers[layer_no][cluster_no]
            new_ebs = GNN_layer(old_ebs, train_flag=train_flag)
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


class ClusterModel(BaseModel, ABC):
    def __init__(self, data: ClusterData, epoch_num, embedding_size=64, learning_rate=2e-4, reg_loss_ratio=5e-5,
                 cluster_dropout_flag=True, cluster_dropout_ratio=0.1, gnn_layer_num=3):
        super().__init__(epoch_num,
                         embedding_size, learning_rate, reg_loss_ratio)
        self.data = data
        self.cluster_num = data.cluster_num
        self.cluster_dropout_flag = cluster_dropout_flag
        self.cluster_dropout_ratio = cluster_dropout_ratio
        self.gnn_layer_num = gnn_layer_num
        self.neg_sample_strategy = OtherClusterStrategyTrain(self.data.cluster_track_ids)

    def _train_epoch(self, epoch):
        epoch_loss = Loss()
        epoch_start_time = time()
        for cluster_no in range(self.cluster_num):
            cluster_loss: Loss = self._train_cluster(cluster_no)
            epoch_loss.add_loss(cluster_loss)
        epoch_end_time = time()
        return epoch_loss, epoch_end_time - epoch_start_time

    @abstractmethod
    def _train_cluster(self, cluster_no):
        pass
