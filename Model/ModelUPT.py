from Model.BaseModel import BaseModel
import numpy as np

class ModelUPT(BaseModel):
    def next_batch(self, i_batch: int) -> dict:
        batch = self.data.next_batch_upt()
        return batch

    def train_batch(self, batch):
        for key, batch_value in batch.items():
            batch[key] = np.array(batch_value).reshape(-1, 1)

        feed_dict = {
            self.X_user: batch["users"],
            self.X_playlist: batch["playlists"],
            self.X_pos_item: batch["pos_tracks"],
            self.X_neg_item: batch["neg_tracks"],
            self.t_message_dropout: [self.message_dropout],
            self.t_node_dropout: [self.node_dropout]
        }

        if "cluster" in self.data.laplacian_mode:
            feed_dict[self.X_pos_item_bias] = batch["pos_track_biases"]
            feed_dict[self.X_neg_item_bias] = batch["neg_track_biases"]
        opt, loss, mf_loss, reg_loss = self.sess.run([self.t_opt, self.t_loss, self.t_mf_loss, self.t_reg_loss], feed_dict=feed_dict)

        return {
            "loss": loss,
            "mf_loss": mf_loss,
            "reg_loss": reg_loss,
        }

    def test_predict(self, test_data: list):
        uid = test_data[0]
        pid = test_data[1]
        tids = test_data[2]

        feed_dict = {
            self.X_user_predict: [uid],
            self.X_playlist_predict: [pid],
            self.X_items_predict: tids,
            self.t_message_dropout: [0.],
            self.t_node_dropout: [0.]
        }

        if "cluster" in self.data.laplacian_mode:
            tid_biases = self.data.cluster_id_map_reverse[tids] - self.data.t_offset
            feed_dict[self.X_items_predict_bias] = tid_biases
        predicts = self.sess.run(self.t_predict, feed_dict=feed_dict)
        predicts = np.squeeze(predicts)
        if len(predicts) == 101:
            return predicts
        else:
            raise Exception("Wrong len of predict")