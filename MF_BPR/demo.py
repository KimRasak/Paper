from preprocess.gen_30music_dataset import *
import tensorflow as tf
import numpy as np

# Test filter and compact
# temp = {0: [0, 1, 2, 3], 1: [0, 2], 2: [0, 2], 3: [0, 3]}
# num_user = len(temp)
# num_track = max([len(row) for row in temp.values()])
# res = filter_and_compact(temp, min_ui_count=2, min_iu_count=2)
# print(res)


# dict表示下，矩阵的行/列优先转换
# dcf = { 0: [0, 3, 5], 1: [0, 1, 2, 3, 4] }
# print(len(dcf))
# drf = transfer_row_column(dcf)
# for k in list(drf.keys()):
#     print(k)

# Test tensorflow
# X_user_predict = tf.placeholder(tf.int32, shape=(1))
# X_items_predict = tf.placeholder(tf.int32, shape=(None))
#
# user_embedding = tf.Variable(tf.truncated_normal(shape=[5, 3], mean=0.0, stddev=0.5))
# item_embedding = tf.Variable(tf.truncated_normal(shape=[4, 3], mean=0.0, stddev=0.5))
#
#
# embed_user = tf.nn.embedding_lookup(user_embedding, X_user_predict)
# items_predict_embeddings = tf.nn.embedding_lookup(item_embedding, X_items_predict)
# print(items_predict_embeddings.shape)
# neg_score = tf.matmul(embed_user, items_predict_embeddings, transpose_b=True)
#
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# res, ebs = sess.run([neg_score, items_predict_embeddings], feed_dict={X_user_predict:[2], X_items_predict:[1, 2]})
# print(ebs.shape)
# print(np.array(res).shape)
# print(res)

# Test python yield
def gen_samples():
    for i in range(100):
        yield i, i+1, i+2

for i, (a, b, c) in enumerate(gen_samples()):
    print(i, a, b, c)
