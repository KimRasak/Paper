from abc import abstractmethod
from time import time

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from DataLayer.ClusterData import ClusterData
from DataLayer.Data import Data
from ModelLayer.Model import Model, Loss


def convert_sp_mat_to_sp_tensor(X: sp.spmatrix):
    """
    Convert scipy sparse matrix to tensorflow sparse tensor.
    :param X: The scipy sparse matrix.
    :return: The tensorflow sparse tensor.
    """
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()

    return tf.sparse.SparseTensor(indices, coo.data, dense_shape=coo.shape)


def dropout_sparse(X: tf.sparse.SparseTensor, keep_prob: tf.Tensor, n_nonzero_elems: int):
    """
    Add dropout for tensorflow sparse tensor.
    :param X: The tensorflow sparse tensor.
    """
    noise_shape = [n_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(X, dropout_mask)

    return pre_out * tf.div(1., keep_prob)


class ClusterModel(Model):
    data: ClusterData  # The data source.
    cluster_num: int  # Number of clusters.

    cluster_dropout_flag: bool  # If use dropout tensors for nodes in the cluster.
    cluster_dropout_ratio: float  # The ratio of nodes dropped out in a cluster. It is only useful if node_dropout_flag

    def __init__(self, epoch_num, data: ClusterData, n_save_batch_loss=300, embedding_size=64,
                 learning_rate=2e-4, reg_loss_ratio=5e-5,
                 cluster_num=100, cluster_dropout_flag=True, node_dropout_ratio=0.1):
        super().__init__(epoch_num, n_save_batch_loss, embedding_size, learning_rate, reg_loss_ratio)

        self.data = data
        self.cluster_num = cluster_num
        self.cluster_dropout_flag = cluster_dropout_flag
        self.cluster_dropout_ratio = node_dropout_ratio
        self.cluster_dropout_tensor = tf.placeholder(tf.float32)

    def __build_model(self):
        self.__build_laplacian_tensors()
        self.__build_network()

    def __build_laplacian_tensors(self):
        """
        Build laplacian matrix tensors from the data's laplacian matrices.
        """
        self.clusters_laplacian_matrices = []
        for data_cluster_matrices in self.data.clusters_laplacian_matrices:
            cluster_matrices = dict()
            for entity_name, entity_matrices in data_cluster_matrices.items():
                # entity_name: "u", "p", "t"...
                # entity_matrices: {"L": ..., "LI": ...}
                cluster_matrices[entity_name] = {
                    "L": self.__sp_to_tensor_fold(entity_matrices["L"]),
                    "LI": self.__sp_to_tensor_fold(entity_matrices["LI"])
                }
            self.clusters_laplacian_matrices.append(cluster_matrices)

    @abstractmethod
    def __build_network(self):
        pass

    def __sp_to_tensor_fold(self, X: sp.spmatrix):
        tensor_fold = convert_sp_mat_to_sp_tensor(X)

        if self.cluster_dropout_flag:
            n_nonzero_fold = X.count_nonzero()
            dropout_tensor_fold = dropout_sparse(tensor_fold, 1 - self.cluster_dropout_tensor, n_nonzero_fold)
            return dropout_tensor_fold
        else:
            return tensor_fold

    @abstractmethod
    def __cluster_GNN_layer(self, cluster_ebs, eb_size):
        pass

    @staticmethod
    def __build_weight_graph(ebs, LI, L, W_side, W_dot, entity_ebs):
        LI_side_embed = tf.sparse_tensor_dense_matmul(LI, ebs)
        L_side_embed = tf.sparse_tensor_dense_matmul(L, ebs)

        sum_embed = tf.matmul(LI_side_embed, W_side)
        dot_embed = tf.matmul(tf.multiply(L_side_embed, entity_ebs), W_dot)

        return tf.nn.leaky_relu(sum_embed + dot_embed)

    def fit(self, restore_model_path=None):
        self._create_session()
        self.save_manager.restore_model(self.sess, model_path=restore_model_path)

        self.__test(-1)  # Test once before training.
        for epoch in range(self.epoch_num):
            self.__epoch_init()
            epoch_loss: Loss = self.__train_epoch(epoch)
            self.__output_epoch_loss(epoch, epoch_loss)
            self.__test(epoch)
            self.__save_model(epoch, epoch_loss)

    def __train_epoch(self, epoch):
        epoch_loss = Loss()
        for cluster_number in range(self.cluster_num):
            cluster_start_time = time()
            pos_train_tuples = self.data.cluster_pos_train_tuples[cluster_number]

            # Pick and merge negative samples.
            neg_tids = self.__get_negative_samples(pos_train_tuples, cluster_number)

            # Build train batch.
            train_batch = dict()
            train_batch.update(pos_train_tuples)
            train_batch.update(neg_tids)

            # Train the batch, output cluster loss, and add to epoch loss.
            cluster_loss = self.__train_cluster(train_batch)
            cluster_end_time = time()
            self.__output_cluster_loss(cluster_number, cluster_loss, cluster_end_time - cluster_start_time)
            epoch_loss.add_loss(cluster_loss)

        return epoch_loss

    @abstractmethod
    def __train_cluster(self, next_batch):
        pass

    def __output_epoch_loss(self, epoch, epoch_loss: Loss):
        log = "Epoch %d complete. %s" % (epoch, epoch_loss.to_string())
        self.log_manager.print_and_write(log)

    def __output_cluster_loss(self, cluster_number, cluster_loss: Loss, cluster_used_time):
        log = "Cluster [%d/%d] used %f seconds. %s" % (cluster_number, self.cluster_num, cluster_used_time, cluster_loss.to_string())
        self.log_manager.print_and_write(log)

    def __get_negative_samples(self, pos_train_tuples, train_cluster_number):
        pick_num = pos_train_tuples["length"]
        neg_tids = self.data.pick_negative_tids(train_cluster_number, pick_num)
        return {
            "neg_track_entity_id": neg_tids["entity_id"],
            "neg_track_cluster_id": neg_tids["cluster_id"]
        }
