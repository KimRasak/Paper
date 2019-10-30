from abc import ABC

import numpy as np
import tensorflow as tf

from ModelLayer.ClusterModel import ClusterModel


class ClusterUPTModel(ClusterModel, ABC):
    def _cluster_GNN_layer(self, clusters_ebs, eb_size):
        entity_names = ["u", "p", "t"]
        W_sides = {
            entity_name: tf.Variable(self.initializer([eb_size, eb_size])) for entity_name in entity_names
        }
        W_dots = {
            entity_name: tf.Variable(self.initializer([eb_size, eb_size])) for entity_name in entity_names
        }

        cluster_aggs = []
        for cluster_number in range(self.cluster_num):  # For each cluster.
            cluster_ebs = clusters_ebs[cluster_number]
            cluster_L_matrices = self.clusters_laplacian_matrices[cluster_number]
            cluster_size = self.data.cluster_sizes[cluster_number]

            entities_ebs = {
                "u": cluster_ebs[: cluster_size["u"]],
                "p": cluster_ebs[cluster_size["u"]: cluster_size["u"] + cluster_size["p"]],
                "t": cluster_ebs[cluster_size["u"] + cluster_size["p"]:],
            }

            entity_folds = []
            for entity_name in entity_names:  # For each entity.
                L = cluster_L_matrices[entity_name]["L"]
                LI = cluster_L_matrices[entity_name]["LI"]
                entity_ebs = entities_ebs[entity_name]

                entity_fold = self.__build_weight_graph(cluster_ebs, L, LI, W_sides[entity_name], W_dots[entity_name], entity_ebs)
                entity_folds.append(entity_fold)
            cluster_aggs.append(tf.concat(entity_folds, axis=0))

        weight_loss = 0
        for entity_name in entity_names:
            weight_loss += tf.nn.l2_loss(W_sides[entity_name]) + tf.nn.l2_loss(W_dots[entity_name])
        return cluster_aggs, weight_loss

    def __train_cluster(self, next_batch):
        pass

    def __epoch_init(self):
        pass

    def __test(self, epoch):
        pass
