from time import time

import numpy as np
from Model.BaseModel import BaseModel


# Model Trained of playlist-track data.
class ModelPT(BaseModel):
    def next_batch(self, i_batch: int) -> dict:
        t1 = time()
        batch = self.data.next_batch_pt()
        print("Read next batch used %d seconds." % (time() - t1))
        return batch

    def test_predict(self, uid, pid, tids):
        predicts = self.sess.run(self.t_predict, feed_dict={
            self.X_playlist_predict: [pid],
            self.X_items_predict: tids
        })
        return predicts[0]
