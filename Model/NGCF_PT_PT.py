import tensorflow as tf
from Model.NGCF_PT import NGCF_PT

"""
NGCF_PT_PT only uses playlist and track embeddings to model the recommendation, 
and only train playlist, track embeddings. 
Each graph layer only uses 2 weight matrices.
Default implementation of NGCF_PT is NGCF_PT_PT, so there's no need to change functions.
"""
class NGCF_PT_PT(NGCF_PT):
    def get_init_embeddings(self):
        return tf.Variable(tf.truncated_normal(shape=[self.data.n_playlist + self.data.n_track, self.embedding_size], mean=0.0,
                                stddev=0.5))
