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
# print("A part:", A[:2, :])
# print("A:", A)
# print(R.shape[0], R.shape[1])
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
# L = D.dot(A).dot(D)
# print("L: ", L)
#
# print(1 / (3 ** 0.5))
# print(1 / (6 ** 0.5))
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

a = np.array([8.244136, 0.3552748])
b = np.array([2.711251, 0.598261])
c = -np.log(sigmoid(a - b))
print(c)
print(np.mean(c))
# [0.00394676 0.82200248]
# 0.4129746235682125
# loss 0.41297466
import tensorflow as tf

W1 = tf.get_variable(tf.truncated_normal(shape=[2, 3], mean=0.0, stddev=0.5))
a = 0 + W1