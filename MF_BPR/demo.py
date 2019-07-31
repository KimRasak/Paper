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
# def gen_samples():
#     for i in range(100):
#         yield i, i+1, i+2
#
# for i, (a, b, c) in enumerate(gen_samples()):
#     print(i, a, b, c)
# i： 0， 1， 2， 3...

a = {2, 3}
a.add(3)

import re
import json
playlist_info_pattern = re.compile('{"ID":.+?}')  # Containing Playlist info
line = 'playlist	381	1357156475	{"ID":10985280,"Title":"2012 m. "Radiocentro" Top 100","numtracks":66,"duration":13712}	{"subjects":[{"type":"user","id":43580}],"objects":[{"type":"track","id":2374504},{"type":"track","id":633023},{"type":"track","id":2205687},{"type":"track","id":2056701},{"type":"track","id":122518},{"type":"track","id":2733092},{"type":"track","id":2514711},{"type":"track","id":686532},{"type":"track","id":1736577},{"type":"track","id":3241885},{"type":"track","id":2026968},{"type":"track","id":2552800},{"type":"track","id":1203212},{"type":"track","id":1590256},{"type":"track","id":3618565},{"type":"track","id":568987},{"type":"track","id":1748173},{"type":"track","id":1047128},{"type":"track","id":2369248},{"type":"track","id":3830462},{"type":"track","id":3745106},{"type":"track","id":2108300},{"type":"track","id":3722051},{"type":"track","id":254917},{"type":"track","id":3755741},{"type":"track","id":2262949},{"type":"track","id":157377},{"type":"track","id":2965522},{"type":"track","id":2077722},{"type":"track","id":3618497},{"type":"track","id":2001013},{"type":"track","id":329291},{"type":"track","id":2061887},{"type":"track","id":1839570},{"type":"track","id":3087182},{"type":"track","id":281859},{"type":"track","id":1748143},{"type":"track","id":1439479},{"type":"track","id":74380},{"type":"track","id":2000939},{"type":"track","id":507542},{"type":"track","id":1162946},{"type":"track","id":3250318},{"type":"track","id":281865},{"type":"track","id":1534087},{"type":"track","id":2363712},{"type":"track","id":548909},{"type":"track","id":1999885},{"type":"track","id":2477501},{"type":"track","id":2680405},{"type":"track","id":1995264},{"type":"track","id":1080951},{"type":"track","id":1350447},{"type":"track","id":1202197},{"type":"track","id":1921162},{"type":"track","id":1969506},{"type":"track","id":2735049},{"type":"track","id":1629375},{"type":"track","id":3003704},{"type":"track","id":3150106},{"type":"track","id":2752874},{"type":"track","id":1203012},{"type":"track","id":3755554},{"type":"track","id":1688864},{"type":"track","id":2077594},{"type":"track","id":3763545}]}'
new_line = ""
for i, c in enumerate(line):
    if c == '"':
        if line[i-1] == '{' or line[i-1] == ',' or line[i-1] == ':' or line[i+1] == ':' or line[i+1] == ',':
            new_line += c
        else:
            new_line += '\\' + c
    else:
        new_line += c

playlist_info = json.loads(re.findall(playlist_info_pattern, new_line)[0])
print(playlist_info)

line2 = ' playlist	4760	1246144347	{"ID":5234695,"Title":""sd,"}","","numtracks":2,"duration":564}	{"subjects":[{"type":"user","id":44558}],"objects":[{"type":"track","id":1973029},{"type":"track","id":1805266}]}'
t = ',"Title":"'
a = line2.find(t)
b = line2.find('{"subjects"')

title_a_pattern = re.compile('"ID":\d+,"Title":"')
title_b_pattern = re.compile('","numtracks":\d+,"duration":\d+}')
st = re.search(title_a_pattern, line2).end()
ed = re.search(title_b_pattern, line2).start()


print("----")
print(line2[st:ed])
c = line2.find('","numtracks":')
print("---------")
print(line2[a + len(t):c])
print(re.findall(playlist_info_pattern, line2)[0])