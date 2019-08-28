import os
from abc import ABCMeta, abstractmethod
from time import time

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from tensorflow.contrib.layers import xavier_initializer
from Model.utility.data_helper import Data
from Model.utility.batch_test import get_metric


def get_base_index(i):
    a = i
    index = 0
    while a < 1:
        a *= 10
        index += 1
    base = a
    return base, index


def convert_sp_mat_to_sp_tensor(X):
    if X.getnnz() == 0:  # Testing
        X[0, 0] = 1
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()

    return tf.sparse.SparseTensor(indices, coo.data, dense_shape=coo.shape)


def dropout_sparse(X, keep_prob, n_nonzero_elems):
    """
    Dropout for sparse tensors.
    """
    noise_shape = [n_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(X, dropout_mask)

    return pre_out * tf.div(1., keep_prob)

class BaseModel(metaclass=ABCMeta):
    def __init__(self, num_epoch, data: Data, output_path="./output.txt",
                 n_save_batch_loss=300, embedding_size=64, learning_rate=2e-4, reg_rate=5e-5,
                 node_dropout_flag=True, node_dropout=0.1, message_dropout=0.1, n_fold=100,
                 test_batch_size=150):
        self.num_epoch = num_epoch
        self.data = data
        self.output_path = output_path
        self.n_save_batch_loss = n_save_batch_loss
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate

        self.node_dropout_flag = node_dropout_flag
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.t_node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.t_message_dropout = tf.placeholder(tf.float32, shape=[None])
        self.n_fold = n_fold

        self.test_batch_size = test_batch_size

        self.t_weight_loss = 0
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.class_name = self.__class__.__name__
        self.model_name = self.get_model_name()
        self.build_model()
        self.create_session()
        self.restore_model()



    def build_model(self):
        laplacian_mode = self.data.laplacian_mode
        self.L_folds = dict()
        self.LI_folds = dict()
        if laplacian_mode == "PT2":
            entity_names = ["complete"]
        elif laplacian_mode == "PT4":
            entity_names = ["p", "t"]
        elif laplacian_mode == "TestPT":
            entity_names = ["complete", "p", "t"]
        elif laplacian_mode == "UT":
            entity_names = ["u", "t"]
        elif laplacian_mode in ["TestUPT", "UPT"]:
            entity_names = ["u", "p", "t"]
        else:
            entity_names = []

        for entity_name in entity_names:
            self.L_folds[entity_name] = self.sparse_matrix_to_tensor_folds(self.data.L[entity_name], self.node_dropout_flag)
            self.LI_folds[entity_name] = self.sparse_matrix_to_tensor_folds(self.data.LI[entity_name], self.node_dropout_flag)

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
        t1 = time()
        max_k = 20

        hrs = {i: [] for i in range(1, max_k + 1)}
        ndcgs = {i: [] for i in range(1, max_k + 1)}
        time_2000 = 0
        for test_i, test_tuple in enumerate(self.data.test_set):
            t1_batch = time()
            pid = test_tuple[1]
            tid = test_tuple[2]

            # change test_tuple[2] from int to id list.
            hundred_neg_tids: list = self.data.sample_hundred_negative_item(pid)
            tids = hundred_neg_tids
            tids.append(tid)
            test_tuple[2] = tids
            assert len(tids) == 101

            scores = self.test_predict(test_tuple)
            sorted_idx = np.argsort(-scores)
            assert len(scores) == 101
            for k in range(1, max_k + 1):
                indices = sorted_idx[:k]  # indices of items with highest scores
                ranklist = np.array(tids)[indices]
                hr_k, ndcg_k = get_metric(ranklist, tid)
                hrs[k].append(hr_k)
                ndcgs[k].append(ndcg_k)

            time_2000 += time() - t1_batch
            if (test_i + 1) % 2000 == 0:
                if i_epoch <= 30:
                    break
                print("test_batch[%d] cost %d seconds. hr_10: %f, hr_20: %f" %
                      (test_i + 1, time_2000, np.average(hrs[10]), np.average(hrs[20])))
                time_2000 = 0
        test_time = time() - t1
        output_str = "Epoch %d complete. Testing used %d seconds, hr_10: %f, hr_20: %f" % (i_epoch, test_time, np.average(hrs[10]), np.average(hrs[20]))
        self.print_and_append_record(output_str)
        self.append_metric_record(hrs, ndcgs, max_k)

    @abstractmethod
    def test_predict(self, test_data: list) -> np.ndarray:
        pass

    @abstractmethod
    def train_batch(self, batch: dict) -> dict:
        pass

    def save_output(self):
        with open(self.output_path, 'w') as f:
            for line in self.output:
                f.write("%s" % line)

    def global_init(self):
        self.output = []
        self.output_file = open("%s.txt" % self.model_name, "a+")

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

    def append_metric_record(self, hrs, ndcgs, max_k):
        self.output_file.write("hr_k: ")
        for i in range(1, max_k+1):
            hr_k = np.average(hrs[i])
            self.output_file.write("%f " % hr_k)
        self.output_file.write("\n")

        self.output_file.write("ndcg_k: ")
        for i in range(1, max_k+1):
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


    def sparse_tensor_folds_mul_embed(self, tensor_folds, embed):
        temp_embed = []
        for tensor_fold in tensor_folds:
            temp_embed.append(tf.sparse_tensor_dense_matmul(tensor_fold, embed))
        side_embeddings = tf.concat(temp_embed, 0)
        return side_embeddings

    def build_weight_graph(self, embeddings, entity_names, W_sides, W_dots, entity_embs):
        agg_folds = []
        for entity_name in entity_names:
            W_side = W_sides[entity_name]
            W_dot = W_dots[entity_name]
            entity_emb = entity_embs[entity_name]

            LI_side_embed = self.sparse_tensor_folds_mul_embed(self.LI_folds[entity_name], embeddings)
            L_side_embed = self.sparse_tensor_folds_mul_embed(self.L_folds[entity_name], embeddings)
            sum_embed = tf.matmul(LI_side_embed, W_side)
            dot_embed = tf.matmul(tf.multiply(L_side_embed, entity_emb), W_dot)

            agg_fold = tf.nn.leaky_relu(sum_embed + dot_embed)
            agg_folds.append(agg_fold)

            self.t_weight_loss += tf.nn.l2_loss(W_side) + tf.nn.l2_loss(W_dot)
        agg = tf.concat(agg_folds, axis=0)
        dropout_embed = tf.nn.dropout(agg, 1 - self.t_message_dropout[0])
        print("entity_names:", entity_names)
        print("dropout_embed:", dropout_embed)
        return dropout_embed

    # graph_PT
    def build_graph_PT(self, embeddings: tf.Tensor, eb_size1, eb_size2, num_weight=2):
        # assert self.data.laplacian_mode == "PT"
        if num_weight not in [2, 4]:
            raise Exception("Wrong number of layer weight.")
        if num_weight == 2:
            entity_names = ["complete"]
            W_sides = {"complete": tf.Variable(self.initializer([eb_size1, eb_size2]))}
            W_dots = {"complete": tf.Variable(self.initializer([eb_size1, eb_size2]))}
            entity_embs = {"complete": embeddings}

        else:
            entity_names = ["p", "t"]

            W_sides = {
                "p": tf.Variable(self.initializer([eb_size1, eb_size2])),
                "t": tf.Variable(self.initializer([eb_size1, eb_size2]))
            }
            W_dots = {
                "p": tf.Variable(self.initializer([eb_size1, eb_size2])),
                "t": tf.Variable(self.initializer([eb_size1, eb_size2]))
            }

            entity_embs = {
                "p": embeddings[:self.data.n_playlist, :],
                "t": embeddings[self.data.n_playlist:, :]
            }
        dropout_embed = self.build_weight_graph(embeddings, entity_names, W_sides, W_dots, entity_embs)
        return dropout_embed

    # graph_UT

    # graph_UPT
    def build_graph_UPT(self, embeddings, eb_size1, eb_size2):
        entity_names = ["u", "p", "t"]

        W_sides = {
            "u": tf.Variable(self.initializer([eb_size1, eb_size2])),
            "p": tf.Variable(self.initializer([eb_size1, eb_size2])),
            "t": tf.Variable(self.initializer([eb_size1, eb_size2]))
        }
        W_dots = {
            "u": tf.Variable(self.initializer([eb_size1, eb_size2])),
            "p": tf.Variable(self.initializer([eb_size1, eb_size2])),
            "t": tf.Variable(self.initializer([eb_size1, eb_size2]))
        }

        entity_embs = {
            "u": embeddings[:self.data.n_user, :],
            "p": embeddings[self.data.n_user:self.data.n_user+self.data.n_playlist, :],
            "t": embeddings[self.data.n_user+self.data.n_playlist:, :]
        }

        dropout_embed = self.build_weight_graph(embeddings, entity_names, W_sides, W_dots, entity_embs)
        return dropout_embed

    def show_graph(self):
        tensorboard_dir = 'tensorboard/'  # 保存目录
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(self.sess.graph)

    def sparse_matrix_to_tensor_folds(self, X: sp.spmatrix, drop_out):
        tensor_folds = []

        height = X.shape[0]
        fold_len = height // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = height
            else:
                end = (i_fold + 1) * fold_len

            fold = X[start:end]
            tensor_fold = convert_sp_mat_to_sp_tensor(fold)
            if drop_out:
                n_nonzero_fold = fold.count_nonzero()
                dropout_tensor_fold = dropout_sparse(tensor_fold, 1 - self.t_node_dropout[0], n_nonzero_fold)
                tensor_folds.append(dropout_tensor_fold)
            else:
                tensor_folds.append(tensor_fold)
        return tensor_folds


