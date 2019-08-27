import tensorflow as tf
from Model.NGCF_PT import NGCF_PT

"""
MENGCF_UPT_PT only uses playlist and track embeddings to model the recommendation, 
and trains user, playlist, track embeddings. (That is, the Model trains user, playlist and track embeddings,
but only uses the playlist and track embeddings to model the recommendation.
Each graph layer uses 6 weight matrices. (6 = 3 Entities * 2 directions) Different matrices are considered in Bi-direct links.
"""
class NGCF_UPT_PT(NGCF_PT):
    def get_init_embeddings(self):
        return tf.Variable(self.initializer([self.data.n_user + self.data.n_playlist + self.data.n_track, self.embedding_size]))

    def build_graph_layers(self, embeddings):
        embeddings1 = self.build_graph_UPT(embeddings, self.embedding_size, self.embedding_size)
        embeddings2 = self.build_graph_UPT(embeddings1, self.embedding_size, self.embedding_size)
        embeddings3 = self.build_graph_UPT(embeddings2, self.embedding_size, self.embedding_size)
        return embeddings1, embeddings2, embeddings3