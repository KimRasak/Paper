import tensorflow as tf
from Model.NGCF_PT import NGCF_PT
from tensorflow.contrib.layers import xavier_initializer

"""
MENGCF_PT_bi_PT only uses playlist and track embeddings to model the recommendation, 
and trains playlist, track embeddings.
Each graph layer uses 4 weight matrices. Different matrices are considered in Bi-direct links. (That's why 
the class name contains "bi". 
"""
class MENGCF_PT_bi_PT(NGCF_PT):

    def get_init_embeddings(self):
        return tf.Variable(self.initializer([self.data.n_playlist + self.data.n_track, self.embedding_size]))

    def build_graph_layers(self, embeddings):
        embeddings1 = self.build_graph_PT(embeddings, self.embedding_size, self.embedding_size, num_weight=4)
        embeddings2 = self.build_graph_PT(embeddings1, self.embedding_size, self.embedding_size, num_weight=4)
        embeddings3 = self.build_graph_PT(embeddings2, self.embedding_size, self.embedding_size, num_weight=4)
        return embeddings1, embeddings2, embeddings3