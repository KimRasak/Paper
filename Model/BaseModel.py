import os
from abc import ABCMeta, abstractmethod
from time import time

import tensorflow as tf
import numpy as np
from Model.utility.data_helper import Data
from Model.utility.batch_test import test


def hr_k(predicts, top_k, idx_test_tid):
    sorted_idx = np.argsort(predicts)[-top_k:]
    return idx_test_tid in sorted_idx


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


class BaseModel(metaclass=ABCMeta):
    def __init__(self, num_epoch, data: Data, output_path="../output.txt",
                 n_save_batch_loss=100, embedding_size=64, learning_rate=0.001, reg_rate=5e-5):
        self.num_epoch = num_epoch
        self.data = data
        self.output_path = output_path
        self.n_save_batch_loss = n_save_batch_loss
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate

        self.t_weight_loss = None

        self.class_name = self.__class__.__name__
        self.build_model()
        self.create_session()
        self.restore_model()
        self.show_graph()

    def build_model(self):
        laplacian_mode = self.data.laplacian_mode
        if laplacian_mode == "PT":
            self.L = convert_sp_mat_to_sp_tensor(self.data.L)  # Normalized laplacian matrix of A. (n+l * n+l)
            self.L_p = convert_sp_mat_to_sp_tensor(self.data.L_p)
            self.L_t = convert_sp_mat_to_sp_tensor(self.data.L_t)

            print("data.L: shape", self.L.shape)
            self.LI = convert_sp_mat_to_sp_tensor(self.data.LI)  # L + I. where I is the identity matrix.
            self.LI_p = convert_sp_mat_to_sp_tensor(self.data.LI_p)
            self.LI_t = convert_sp_mat_to_sp_tensor(self.data.LI_t)

        elif laplacian_mode == "UT":
            self.L = convert_sp_mat_to_sp_tensor(self.data.L)  # Normalized laplacian matrix of A. (m+n * m+n)
            self.L_u = convert_sp_mat_to_sp_tensor(self.data.L_u)
            self.L_t = convert_sp_mat_to_sp_tensor(self.data.L_t)

            # self.LI = self.L + tf.eye(self.L.shape[0])  # L + I. where I is the identity matrix.
            self.LI_u = convert_sp_mat_to_sp_tensor(self.data.LI_u)
            self.LI_t = convert_sp_mat_to_sp_tensor(self.data.LI_t)
        elif laplacian_mode == "UPT":
            # self.L = convert_sp_mat_to_sp_tensor(self.data.L)  # Normalized laplacian matrix of A. (m+n+l * m+n+l)
            self.L_u = convert_sp_mat_to_sp_tensor(self.data.L_u)
            self.L_p = convert_sp_mat_to_sp_tensor(self.data.L_p)
            self.L_t = convert_sp_mat_to_sp_tensor(self.data.L_t)

            # self.LI = self.L + tf.eye(self.L.shape[0])  # L + I. where I is the identity matrix.
            self.LI_u = convert_sp_mat_to_sp_tensor(self.data.LI_u)
            self.LI_p = convert_sp_mat_to_sp_tensor(self.data.LI_p)
            self.LI_t = convert_sp_mat_to_sp_tensor(self.data.LI_t)

    def fit(self):
        self.global_init()
        for i_epoch in range(self.num_epoch):
            self.epoch_init()
            self.test(i_epoch)
            self.train_epoch(i_epoch)
            self.save_model(i_epoch)
        self.save_output()

    def train_epoch(self, i_epoch):
        for i_batch in range(self.data.n_batch):
            batch: dict = self.next_batch(i_batch)
            result: dict = self.train_batch(batch)
            self.print_result_and_add_loss(i_epoch, i_batch, result)
        self.output_total_loss()

    def create_session(self):
        # Config: Create session and use fixed fraction of gpu.
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Config: Create session and allow growth.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Create session
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore_model(self):
        # Restore model.
        self.model_folder_path = os.path.join('cpkt/', self.class_name)
        ckpt = tf.train.get_checkpoint_state(self.model_folder_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model checkpoint found on path %r. Using model." % self.model_folder_path)
        else:
            print('No model on path %r. Not using model.' % self.model_folder_path)

    def save_model(self, i_epoch):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
        model_path = os.path.join(self.model_folder_path, "model%d.cpkt" % i_epoch)
        self.saver.save(self.sess, model_path)

    @abstractmethod
    def next_batch(self, i_batch: int) -> dict:
        pass

    def test(self, i_epoch):
        hr_10s = []
        t1 = time()
        for uid, user in self.data.test_set.items():
            for pid, tids in user.items():
                for tid in tids:
                    input_tids = [tid]
                    hundred_neg_tids = self.data.sample_hundred_negative_item(pid)
                    input_tids.extend(hundred_neg_tids)

                    predicts = self.test_predict(uid, pid, input_tids)
                    hr_10s.append(hr_k(predicts, 10, 0))
        test_time = time() - t1
        output_str = "Epoch %d complete. Used %d seconds, hr_10: %f" % (i_epoch, test_time, np.average(hr_10s))
        self.print_and_append_record(output_str)

    @abstractmethod
    def test_predict(self, uid, pid, tids):
        pass

    def save_output(self):
        with open(self.output_path, 'w') as f:
            for line in self.output:
                f.write("%s" % line)

    def global_init(self):
        self.output = []

    @abstractmethod
    def train_batch(self, batch: dict) -> dict:
        pass

    def print_result_and_add_loss(self, i_epoch, i_batch, result):
        if i_batch % self.n_save_batch_loss != 0:
            return
        output_str = "[epoch%d-batch%d] complete, " % (i_epoch, i_batch)
        for result_key, result_value in result.items():
            if not result_key.endswith("loss"):
                continue
            if result_key not in self.total_loss:
                self.total_loss[result_key] = result_value
            else:
                self.total_loss[result_key] += result_value
            output_str += "%s: %r " % (result_key, result_value)

        if "batch_time" in result:
            print("Batch session used %d seconds." % result["batch_time"])

        self.print_and_append_record(output_str)

    def epoch_init(self):
        self.total_loss = {}

    def output_total_loss(self):
        output_str = "Epoch complete. Print total loss. "
        for loss_name, loss_value in self.total_loss.items():
            output_str += "%s: %f " % (loss_name, loss_value)
        self.print_and_append_record(output_str)

    def print_and_append_record(self, output_str):
        print(output_str)
        output_str += "\n"
        self.output.append(output_str)

    @abstractmethod
    def get_init_embeddings(self):
        pass

    # graph_PT
    def build_graph_PT(self, embeddings: tf.Tensor, eb_size1, eb_size2, num_weight=2):
        assert self.data.laplacian_mode == "PT"
        if num_weight not in [2, 4]:
            raise Exception("Wrong number of layer weight.")
        if num_weight == 2:
            W1 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))
            W2 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))
            aggregate = tf.matmul(tf.sparse_tensor_dense_matmul(self.LI, embeddings), W1) + \
                        tf.matmul(tf.multiply(tf.sparse_tensor_dense_matmul(self.L, embeddings), embeddings), W2)

            w_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
            self.t_weight_loss = w_loss if self.t_weight_loss is None else self.t_weight_loss + w_loss
            new_embeddings = tf.nn.leaky_relu(aggregate)
        else:
            W1 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))
            W2 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))
            W3 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))
            W4 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))

            aggregate1 = tf.matmul(tf.sparse_tensor_dense_matmul(self.LI_p, embeddings), W1) + tf.matmul(
                tf.multiply(tf.sparse_tensor_dense_matmul(self.L_p, embeddings), embeddings[:self.data.n_playlist, :]), W2)
            aggregate2 = tf.matmul(tf.sparse_tensor_dense_matmul(self.LI_t, embeddings), W3) + tf.matmul(
                tf.multiply(tf.sparse_tensor_dense_matmul(self.L_t, embeddings), embeddings[self.data.n_playlist:, :]), W4)

            w_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)
            self.t_weight_loss = w_loss if self.t_weight_loss is None else self.t_weight_loss + w_loss
            new_embeddings = tf.nn.leaky_relu(tf.concat([aggregate1, aggregate2], axis=0))
            print("new_embeddings:", new_embeddings)

        return new_embeddings

    # graph_UT

    # graph_UPT
    def build_graph_UPT(self, embeddings, eb_size1, eb_size2):
        W1 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))
        W2 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))
        W3 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))
        W4 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))
        W5 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))
        W6 = tf.Variable(tf.truncated_normal(shape=[eb_size1, eb_size2], mean=0.0, stddev=0.001))

        aggregate1 = tf.matmul(tf.sparse_tensor_dense_matmul(self.LI_u, embeddings), W1) + tf.matmul(
            tf.multiply(tf.sparse_tensor_dense_matmul(self.L_u, embeddings), embeddings[:self.data.n_user]), W2)
        aggregate2 = tf.matmul(tf.sparse_tensor_dense_matmul(self.LI_p, embeddings), W3) + tf.matmul(
            tf.multiply(tf.sparse_tensor_dense_matmul(self.L_p, embeddings), embeddings[self.data.n_user:self.data.n_user+self.data.n_playlist, :]), W4)
        aggregate3 = tf.matmul(tf.sparse_tensor_dense_matmul(self.LI_t, embeddings), W5) + tf.matmul(
            tf.multiply(tf.sparse_tensor_dense_matmul(self.L_t, embeddings), embeddings[self.data.n_user+self.data.n_playlist:, :]), W6)

        w_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6)
        self.t_weight_loss = w_loss if self.t_weight_loss is None else self.t_weight_loss + w_loss
        new_embeddings = tf.nn.leaky_relu(tf.concat([aggregate1, aggregate2, aggregate3], axis=0))
        return new_embeddings

    def show_graph(self):
        tensorboard_dir = 'tensorboard/'  # 保存目录
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(self.sess.graph)


