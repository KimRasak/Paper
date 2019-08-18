from Model.BaseModel import BaseModel


class ModelUPT(BaseModel):
    def next_batch(self, i_batch: int) -> dict:
        batch = self.data.next_batch_upt()
        return batch

    def test_predict(self, uid, pid, tids):
        predicts = self.sess.run(self.t_predict, feed_dict={
            self.X_user_predict: [uid],
            self.X_playlist_predict: [pid],
            self.X_items_predict: tids
        })

        return predicts
