from time import time

import numpy as np
from Model.BaseModel import BaseModel


# Model Trained of playlist-track data.
class ModelPT(BaseModel):
    def next_batch(self, i_batch: int) -> dict:
        batch = self.data.next_batch_pt()
        return batch

    def test_predict(self, uid, pid, tids):
        predicts = self.sess.run(self.t_predict, feed_dict={
            self.X_playlist_predict: [pid],
            self.X_items_predict: tids
        })
        assert len(predicts[0]) == 101
        return predicts[0]
