import os
from abc import ABCMeta, abstractmethod
from time import time

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
from Model.utility.data_helper import Data
from Model.utility.batch_test import test


def get_metric(ranklist, gt_item):
    return get_hit_ratio(ranklist, gt_item), get_ndcg(ranklist, gt_item)

def get_hit_ratio(ranklist, gt_item):
    for item in ranklist:
        if item == gt_item:
            return 1
    return 0

def get_ndcg(ranklist, gt_item):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gt_item:
            return np.log(2) / np.log(i + 2)
    return 0

def get_base_index(i):
    a = i
    index = 0
    while a < 1:
        a *= 10
        index += 1
    base = a
    return base, index

def convert_sp_mat_to_sp_tensor(X):
    if X.getnnz() == 0:
        print("add one.", X.shape)
        X[0, 0] = 1
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()

    return tf.sparse.SparseTensor(indices, coo.data, dense_shape=coo.shape)

class BaseModel(metaclass=ABCMeta):
    def __init__(self, num_epoch, data: Data, output_path="./output.txt",
                 n_save_batch_loss=300, embedding_size=64, learning_rate=2e-4, reg_rate=5e-5, ngcf2=False):
        self.num_epoch = num_epoch
        self.data = data
        self.output_path = output_path
        self.n_save_batch_loss = n_save_batch_loss
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.ngcf2 = ngcf2

        self.t_weight_loss = 0
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.class_name = self.__class__.__name__
        self.model_name = self.get_model_name()
        self.build_model()
        self.create_session()
        self.restore_model()

    def build_model(self):
        if self.ngcf2:
            return
        laplacian_mode = self.data.laplacian_mode
        if laplacian_mode == "PT":
            self.L = convert_sp_mat_to_sp_tensor(self.data.L)  # Normalized laplacian matrix of A. (n+l * n+l)
            self.L_p = convert_sp_mat_to_sp_tensor(self.data.L_p)
            self.L_t = convert_sp_mat_to_sp_tensor(self.data.L_t)

            self.LI = convert_sp_mat_to_sp_tensor(self.data.LI)  # L + I. where I is the identity matrix.
            self.LI_p = convert_sp_mat_to_sp_tensor(self.data.LI_p)
            self.LI_t = convert_sp_mat_to_sp_tensor(self.data.LI_t)
            print("data.L: shape", self.L.shape)
            print("data.LI: shape", self.LI.shape)

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
        elif laplacian_mode == "TestPT" or laplacian_mode == "TestUPT":
            self.L = convert_sp_mat_to_sp_tensor(self.data.L)  # Normalized laplacian matrix of A. (n+l * n+l)
            if hasattr(self.data, "L_u"):
                self.L_u = convert_sp_mat_to_sp_tensor(self.data.L_u)
            self.L_p = convert_sp_mat_to_sp_tensor(self.data.L_p)
            self.L_t = convert_sp_mat_to_sp_tensor(self.data.L_t)

            print("data.L: shape", self.L.shape)
            self.LI = convert_sp_mat_to_sp_tensor(self.data.LI)  # L + I. where I is the identity matrix.
            if hasattr(self.data, "LI_u"):
                self.LI_u = convert_sp_mat_to_sp_tensor(self.data.LI_u)
            self.LI_p = convert_sp_mat_to_sp_tensor(self.data.LI_p)
            self.LI_t = convert_sp_mat_to_sp_tensor(self.data.LI_t)

    def fit(self):
        self.global_init()
        self.test(0)
        for i_epoch in range(self.num_epoch):
            self.epoch_init()
            self.train_epoch(i_epoch)
            self.test(i_epoch)
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

    def get_model_name(self):
        base_lr, index_lr = get_base_index(self.learning_rate)
        base_rr, index_rr = get_base_index(self.reg_rate)
        eb_size = self.embedding_size

        model_name = "%s_%s_eb%d_lr%de-%d_rr%de-%d" % (self.class_name, self.data.dataset_name, eb_size, base_lr, index_lr, base_rr, index_rr)
        # file_path = "./%s_eb%d_lr%de-%d_rr%de-%d.txt" % (self.class_name, eb_size, base_lr, index_lr, base_rr, index_rr)
        # self.output_file = open(file_path, "a+")
        return model_name

    def restore_model(self):
        # Restore model.
        self.model_folder_path = os.path.join('cpkt/', self.model_name)
        ckpt = tf.train.get_checkpoint_state(self.model_folder_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model checkpoint found on path %r. Using model." % self.model_folder_path)
        else:
            print('No model on path %r. Not using model.' % self.model_folder_path)

    def save_model(self, i_epoch):
        if np.isnan(self.total_loss["loss"]) :
            return

        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
        model_path = os.path.join(self.model_folder_path, "model%d.cpkt" % i_epoch)
        self.saver.save(self.sess, model_path)

    @abstractmethod
    def next_batch(self, i_batch: int) -> dict:
        pass

    def test(self, i_epoch):
        max_K = 20
        hrs = {i: [] for i in range(1, max_K+1)}
        ndcgs = {i: [] for i in range(1, max_K+1)}
        t1 = time()
        num_tested = 0
        test_t0 = time()
        delta_t0 = 0
        delta_t1 = 0
        delta_t2 = 0
        for uid, user in self.data.test_set.items():
            for pid, tids in user.items():
                for tid in tids:
                    test_t1 = time()
                    input_tids = [tid]
                    hundred_neg_tids = self.data.sample_hundred_negative_item(pid)
                    input_tids.extend(hundred_neg_tids)
                    np.random.shuffle(input_tids)
                    index_tid = input_tids.index(tid)
                    test_t2 = time()

                    predicts = self.test_predict(uid, pid, input_tids)
                    pos_item_score = predicts[index_tid]
                    test_t3 = time()
                    sorted_idx = np.argsort(-predicts)
                    for k in range(1, max_K+1):
                        indices = sorted_idx[:k]  # indices of items with highest scores
                        ranklist = np.array(input_tids)[indices]
                        hr_k, ndcg_k = get_metric(ranklist, tid)
                        hrs[k].append(hr_k)
                        ndcgs[k].append(ndcg_k)
                    test_t4 = time()

                    delta_t0 += test_t2 - test_t1
                    delta_t1 += test_t3 - test_t2
                    delta_t2 += test_t4 - test_t3

                    num_tested += 1
                    if num_tested % 2000 == 0:
                        print("Tested %d pairs. Used %d seconds. hr_10: %f, hr_20: %f" % (num_tested, time() - test_t0, np.average(hrs[10]), np.average(hrs[20])))
                        print("Test time use: %d %d %d" % (delta_t0, delta_t1, delta_t2))
                        test_t0 = time()
                        delta_t0 = 0
                        delta_t1 = 0
                        delta_t2 = 0
        test_time = time() - t1
        output_str = "Epoch %d complete. Testing used %d seconds, hr_10: %f, hr_20: %f" % (i_epoch, test_time, np.average(hrs[10]), np.average(hrs[20]))
        self.print_and_append_record(output_str)
        self.append_metric_record(hrs, ndcgs, max_K)

    @abstractmethod
    def test_predict(self, uid, pid, tids):
        pass

    def save_output(self):
        with open(self.output_path, 'w') as f:
            for line in self.output:
                f.write("%s" % line)

    def global_init(self):
        self.output = []
        self.output_file = open("%s.txt" % self.model_name, "a+")


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

        if i_batch == 0:
            self.print_and_append_record(output_str)
        else:
            self.print_and_append_record(output_str, write_file=False)


    def epoch_init(self):
        self.total_loss = {}
        self.epoch_t0 = time()

    def output_total_loss(self):
        output_str = "Epoch complete. Used %d seconds. Print total loss. " % (time() - self.epoch_t0)
        for loss_name, loss_value in self.total_loss.items():
            output_str += "%s: %f " % (loss_name, loss_value)
        self.print_and_append_record(output_str)

    def append_metric_record(self, hrs, ndcgs, max_K):
        self.output_file.write("hr_k: ")
        for i in range(1, max_K+1):
            hr_k = np.average(hrs[i])
            self.output_file.write("%f " % hr_k)
        self.output_file.write("\n")

        self.output_file.write("ndcg_k: ")
        for i in range(1, max_K+1):
            ndcg_k = np.average(ndcgs[i])
            self.output_file.write("%f " % ndcg_k)
        self.output_file.write("\n")


    def print_and_append_record(self, output_str, write_file=True):
        print(output_str)
        output_str += "\n"
        self.output.append(output_str)
        if write_file:
            self.output_file.write(output_str)
            self.output_file.flush()

    @abstractmethod
    def get_init_embeddings(self):
        pass

    # graph_PT
    def build_graph_PT(self, embeddings: tf.Tensor, eb_size1, eb_size2, num_weight=2):
        # assert self.data.laplacian_mode == "PT"
        if num_weight not in [2, 4]:
            raise Exception("Wrong number of layer weight.")
        if num_weight == 2:
            W1 = tf.Variable(tf.truncated_normal([eb_size1, eb_size2], dtype=tf.float64))
            W2 = tf.Variable(tf.truncated_normal([eb_size1, eb_size2], dtype=tf.float64))
            aggregate = tf.matmul(tf.sparse_tensor_dense_matmul(self.LI, embeddings), W1) + \
                        tf.matmul(tf.multiply(tf.sparse_tensor_dense_matmul(self.L, embeddings), embeddings), W2)

            w_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
            self.t_weight_loss += w_loss
            new_embeddings = tf.nn.leaky_relu(aggregate)
            print("graph PT 2. new_embeddings:", new_embeddings)
        else:
            W1 = tf.Variable(self.initializer([eb_size1, eb_size2]))
            W2 = tf.Variable(self.initializer([eb_size1, eb_size2]))
            W3 = tf.Variable(self.initializer([eb_size1, eb_size2]))
            W4 = tf.Variable(self.initializer([eb_size1, eb_size2]))

            print(self.LI_p)

            LI_P_emb = tf.sparse_tensor_dense_matmul(self.LI_p, embeddings)
            L_P_emb = tf.sparse_tensor_dense_matmul(self.L_p, embeddings)
            L_P_emb_mul_emb = tf.multiply(L_P_emb, embeddings[:self.data.n_playlist, :])
            aggregate1 = tf.matmul(LI_P_emb, W1) + tf.matmul(L_P_emb_mul_emb, W2)

            LI_T_emb = tf.sparse_tensor_dense_matmul(self.LI_t, embeddings)
            L_T_emb = tf.sparse_tensor_dense_matmul(self.L_t, embeddings)
            L_T_emb_mul_emb = tf.multiply(L_T_emb, embeddings[self.data.n_playlist:, :])
            aggregate2 = tf.matmul(LI_T_emb, W3) + tf.matmul(L_T_emb_mul_emb, W4)

            w_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)
            self.t_weight_loss += w_loss
            new_embeddings = tf.nn.selu(tf.concat([aggregate1, aggregate2], axis=0))
            print("graph PT 4. new_embeddings:", new_embeddings)

        return new_embeddings

    # graph_UT

    # graph_UPT
    def build_graph_UPT(self, embeddings, eb_size1, eb_size2):
        W1 = tf.Variable(self.initializer([eb_size1, eb_size2]))
        W2 = tf.Variable(self.initializer([eb_size1, eb_size2]))
        W3 = tf.Variable(self.initializer([eb_size1, eb_size2]))
        W4 = tf.Variable(self.initializer([eb_size1, eb_size2]))
        W5 = tf.Variable(self.initializer([eb_size1, eb_size2]))
        W6 = tf.Variable(self.initializer([eb_size1, eb_size2]))

        user_embeddings = embeddings[:self.data.n_user, :]
        playlist_embeddings = embeddings[self.data.n_user:self.data.n_user+self.data.n_playlist, :]
        track_embeddings = embeddings[self.data.n_user+self.data.n_playlist:, :]

        LI_u_emb = tf.sparse_tensor_dense_matmul(self.LI_u, embeddings)
        emb_u_W1 = tf.matmul(LI_u_emb, W1)

        L_u_emb = tf.sparse_tensor_dense_matmul(self.L_u, embeddings)
        emb_multiply = tf.multiply(L_u_emb, user_embeddings)
        emb_u_W2 = tf.matmul(emb_multiply, W2)

        aggregate1 = emb_u_W1 + emb_u_W2
        aggregate2 = tf.matmul(tf.sparse_tensor_dense_matmul(self.LI_p, embeddings), W3) + tf.matmul(
            tf.multiply(tf.sparse_tensor_dense_matmul(self.L_p, embeddings), playlist_embeddings), W4)
        aggregate3 = tf.matmul(tf.sparse_tensor_dense_matmul(self.LI_t, embeddings), W5) + tf.matmul(
            tf.multiply(tf.sparse_tensor_dense_matmul(self.L_t, embeddings), track_embeddings), W6)

        w_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6)
        self.t_weight_loss += w_loss
        new_embeddings = tf.nn.selu(tf.concat([aggregate1, aggregate2, aggregate3], axis=0))
        return new_embeddings

    def show_graph(self):
        tensorboard_dir = 'tensorboard/'  # 保存目录
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(self.sess.graph)


