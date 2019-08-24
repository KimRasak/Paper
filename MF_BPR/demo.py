import numpy as np
import scipy.sparse as sp

# num_user = 3
# num_item = 4
# num_all = num_user + num_item
# R: sp.dok_matrix = sp.dok_matrix((num_user, num_item), dtype=np.float32).tolil()
# R[0, 1] = 1
# R[0, 2] = 1
# R[0, 3] = 1
# R[1, 2] = 1
# R[1, 3] = 1
# A = sp.dok_matrix((num_all, num_all), dtype=np.float32)
# A[:num_user, num_user:] = R
# A[num_user:, :num_user] = R.T
#
# print("A:", A)
# print("A.shape:", R.shape[0], R.shape[1])
#
# def get_D(adj: sp.spmatrix, power=-0.5):  # Get degree diagonal matrix, where a node's degree is the number of edges starting from this node.
#     rowsum = np.sum(adj, axis=1).flatten().A[0]
#     print("rowsum: ", rowsum)
#
#     # Get inverse( x^(-1) ) of every element, but zero remains zero.
#     with np.errstate(divide='ignore'):
#         d_inv = np.float_power(rowsum, power)
#     d_inv[np.isinf(d_inv)] = 0
#     print("d_inv: ", d_inv)
#     d_mat_inv = sp.diags(d_inv)
#     print("d_mat_inv.shape:", d_mat_inv.shape)
#     return d_mat_inv
# # print(sp.csgraph.laplacian(adj, normed=True))
# D = get_D(A)
# print("D:\n", D)
# print("D.dot(A)", D.dot(A))
# L = D.dot(A).dot(D)
# print("L: ", L)

# print(1 / (3 ** 0.5))
# print(1 / (2 ** 0.5))

# mul = np.multiply(B, D)
# print(mul)
# print(mul.T.dot(mul))
# print(np.sum(np.square(mul), axis=0))


# import tensorflow as tf
# sess = tf.Session()
# W0 = tf.constant(1, shape=[1, 2])
# W1 = tf.constant(3, shape=[2, 2])
# # a = 0 + tf.square(W0 - W1)
# eb = tf.Variable(tf.truncated_normal(shape=[4, 7]))
# t1 = tf.nn.embedding_lookup(eb, [2])
# t2 = tf.nn.embedding_lookup(eb, [3])
# sess.run(tf.global_variables_initializer())
# print(sess.run([t1, t2]))
# print(sess.run(eb[2:]))
# print(sess.run(eb[2:, :]))


num_user = 3
num_item = 4
num_all = num_user + num_item
R: sp.dok_matrix = sp.dok_matrix((num_user, num_item), dtype=np.float32).tolil()
R[0, 1] = 1
R[0, 2] = 1
R[0, 3] = 1
R[1, 2] = 1
R[1, 3] = 1
A = sp.dok_matrix((num_all, num_all), dtype=np.float32)
A[:num_user, num_user:] = R
A[num_user:, :num_user] = R.T
print("A", A)

B = sp.dok_matrix((num_all, num_all), dtype=np.float32)
print(B)
cx = R.tocoo()
print("cx", cx)
for i, j, v in zip(cx.row, cx.col, cx.data):
    print(i, j, num_user, j + num_user, v)
    B[i, j + num_user] = v
    B[j + num_user, i] = v
    # B[i, j + num_user] = cx.data
    # B[j + num_user, i] = cx.data
print(B)