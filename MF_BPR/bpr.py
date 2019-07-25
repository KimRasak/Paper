import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import sys


class BPR():
    """
    MF_BPR implementation.
    The model preprocess the data, so the input data should be preprocessd already.
    """

    def __init__(self, train_data, test_data, interaction_data: sp.csr_matrix,
                 n_epochs=100, batch_size=256, embedding_k=64, top_k=10, learning_rate=0.0001):
        """
        Init function.
        :param train_data: The train data.
        :param test_data: The test data.
        :param interaction_data: The user-track interaction data.
        :param n_epochs: Train epochs.
        :param batch_size: Train batch size.
        :param embedding_k: The length of user/track embedding vector.
        :param top_k: In top-k recommendation, Recommend top_k tracks for each user.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.interaction_data: sp.csr_matrix = interaction_data

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.embedding_k = embedding_k
        self.top_k = top_k
        self.learning_rate = learning_rate

        self.num_user = interaction_data.get_shape()[0]
        self.num_item = interaction_data.get_shape()[1]

        # build TF graph
        self.build_model()

        # create session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # ckpt = tf.train.get_checkpoint_state('../cpkt/')  # checkpoint存在的目录
        # if ckpt and ckpt.model_checkpoint_path:
        #     self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # 自动恢复model_checkpoint_path保存模型一般是最新
        #     print("Model restored...")
        # else:
        #     print('No Model')
        return

    def build_model(self):
        # An input for training is a traid (user id, positive_item id, negative_item id)
        self.X_user = tf.placeholder(tf.int32, shape=(None, 1))
        self.X_pos_item = tf.placeholder(tf.int32, shape=(None, 1))
        self.X_neg_item = tf.placeholder(tf.int32, shape=(None, 1))

        # An input for testing/predicting is only  the user id
        self.X_predict = tf.placeholder(tf.int32, shape=(1))

        # Loss, optimizer definition for training.
        user_embedding = tf.Variable(tf.truncated_normal(shape=[self.num_user, self.embedding_k], mean=0.0, stddev=0.5))
        item_embedding = tf.Variable(tf.truncated_normal(shape=[self.num_item, self.embedding_k], mean=0.0, stddev=0.5))

        embed_user = tf.nn.embedding_lookup(user_embedding, self.X_user)
        embed_pos_item = tf.nn.embedding_lookup(item_embedding, self.X_pos_item)
        embed_neg_item = tf.nn.embedding_lookup(item_embedding, self.X_neg_item)

        pos_score = tf.matmul(embed_user, embed_pos_item, transpose_b=True)
        neg_score = tf.matmul(embed_user, embed_neg_item, transpose_b=True)

        self.loss = tf.reduce_sum(-tf.log(tf.nn.sigmoid(pos_score - neg_score)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.print_loss = tf.print("loss: ", self.loss, output_stream=sys.stdout)

        # Output for testing/predicting
        predict_user_embed = tf.nn.embedding_lookup(user_embedding, self.X_predict)
        self.predict = tf.matmul(predict_user_embed, item_embedding, transpose_b=True)

    def fit(self):
        self.outputs = []

        for epoch in range(self.n_epochs):
            # shuffle the input along the first index
            np.random.shuffle(self.train_data)

            # Epoch total loss.
            total_loss = 0

            # Train in batch.
            batch_num = len(self.train_data) // self.batch_size
            for i in range(0, len(self.train_data), self.batch_size):
                # Get the batch.
                start = i
                end = min(i + self.batch_size, len(self.train_data) + 1)
                data_batch = self.train_data[start:end]

                # Initialize batch input
                uid_batch = []
                tid_batch = []  # Positive track ids batch.
                neg_batch = []  # Negative track ids batch.

                # Append training batch input.
                for traid in data_batch:
                    uid_batch.append(traid[0])
                    tid_batch.append(traid[1])
                    neg_batch.append(traid[2])

                # Resize.
                uid_batch = np.array(uid_batch).reshape(-1, 1)
                tid_batch = np.array(tid_batch).reshape(-1, 1)
                neg_batch = np.array(neg_batch).reshape(-1, 1)

                _, loss = self.sess.run([self.optimizer, self.loss, self.print_loss],
                                        feed_dict={self.X_user: uid_batch, self.X_pos_item: tid_batch,
                                                   self.X_neg_item: neg_batch}
                                        )
                batch_no = i // self.batch_size
                total_loss += loss
                if batch_no % 200 == 0:
                    print("Epoch %d, batch %d/%d, loss %f" % (epoch,
                                                                batch_no, batch_num,
                                                                loss))

            # Test
            # Calculate the hr@k
            num_hit = 0
            for uid, tid in self.test_data:
                # Rank the test item agains 100 unobserved items.
                # Following《Signed Distance-based Deep Memory Recommender》and《Neural Collaborative Filtering》

                # Predict possibilities for the user on each track.
                result = self.sess.run(self.predict, feed_dict={self.X_predict: [uid]})
                result: np.ndarray = result.reshape(-1)

                # The interactions of the user.
                uis = self.interaction_data.getrow(uid).indptr

                # Recommend tracks that the user hasn't interacted.
                result = np.delete(result, uis)

                # Get tracks with largest values.
                recommended_tids = np.argsort(result)[-self.top_k:]

                # If tid hit the top-k recommendation.
                if tid in recommended_tids:
                    num_hit += 1
            hr_k = num_hit / len(self.test_data)
            print("-----Epoch %d complete, total loss %f, hr@%d %f-----" % (epoch, total_loss, self.top_k, hr_k))
            self.saver.save(self.sess, "../cpkt/bpr_30music_epoch%d.cpkt" % epoch)
