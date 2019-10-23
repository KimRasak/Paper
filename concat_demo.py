from time import time

import tensorflow as tf

initializer = tf.contrib.layers.xavier_initializer()

EB_SIZE = 100
SMALL_EB_LENGTH = 10
LARGE_EB_LENGTH = SMALL_EB_LENGTH * 100000

TRANSFORM_SIZE1 = TRANSFORM_SIZE2 = EB_SIZE

TRANSFORM_TIME = 100


def get_concat_outputs():
    full_ebs = tf.Variable(initializer([SMALL_EB_LENGTH + LARGE_EB_LENGTH, EB_SIZE]))
    small_ebs = full_ebs[:SMALL_EB_LENGTH]
    large_ebs = full_ebs[SMALL_EB_LENGTH:]

    m = tf.Variable(initializer([TRANSFORM_SIZE1, TRANSFORM_SIZE2]))

    small_output = small_ebs
    large_output = large_ebs
    for i in range(TRANSFORM_TIME):
        small_output = tf.matmul(small_output, m)
        large_output = tf.matmul(large_output, m)

    full_output = tf.concat([small_output, large_output], axis=0)
    small_output = full_output[:SMALL_EB_LENGTH]
    large_output = full_output[SMALL_EB_LENGTH:]
    return small_output, large_output


def get_divided_outputs():
    small_output = tf.Variable(initializer([SMALL_EB_LENGTH, EB_SIZE]))
    large_output = tf.Variable(initializer([LARGE_EB_LENGTH, EB_SIZE]))

    m = tf.Variable(initializer([TRANSFORM_SIZE1, TRANSFORM_SIZE2]))

    for i in range(TRANSFORM_TIME):
        small_output = tf.matmul(small_output, m)
        large_output = tf.matmul(large_output, m)
    return small_output, large_output


small_output, large_output = get_concat_outputs()
small_eb_X = tf.placeholder(tf.int32)
large_eb_X = tf.placeholder(tf.int32)
small_eb = small_output[small_eb_X]
large_eb = large_output[large_eb_X]

sess = tf.Session()
writer = tf.summary.FileWriter("logs", sess.graph)
sess.run(tf.global_variables_initializer())
session_t_sum = 0
for i in range(10):
    session_start_t = time()
    sess.run(small_eb, feed_dict={small_eb_X: 1})
    session_end_t = time()

    session_t_sum += session_end_t - session_start_t
print("sum of session: ", session_t_sum)
