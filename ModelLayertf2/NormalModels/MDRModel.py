from time import time

import numpy as np
import tensorflow as tf

from ModelLayertf2.BaseModel import MDR, Loss
from ModelLayertf2.Metric import Metrics
from ModelLayertf2.NormalModels.NormalUPTModel import NormalUPTModel


class MDRModel(NormalUPTModel):
    def _build_model(self):
        data_set_num = self.data.data_set_num
        self.user_ebs = tf.Variable(self.initializer([data_set_num.user, self.embedding_size]),
                                    name="user_ebs")
        self.playlist_ebs = tf.Variable(self.initializer([data_set_num.playlist, self.embedding_size]),
                                        name="user_ebs")
        self.track_ebs = tf.Variable(self.initializer([data_set_num.track, self.embedding_size]),
                                     name="user_ebs")

        self.MDR_layer = MDR(self.initializer, self.embedding_size, data_set_num.track)

    def _train_batch(self, batch_no, batch):
        with tf.GradientTape() as tape:
            user_ebs = tf.nn.embedding_lookup(self.user_ebs, batch["uids"])
            playlist_ebs = tf.nn.embedding_lookup(self.playlist_ebs, batch["pids"])
            pos_track_ebs = tf.nn.embedding_lookup(self.track_ebs, batch["pos_tids"])
            neg_track_ebs = tf.nn.embedding_lookup(self.track_ebs, batch["neg_tids"])

            pos_scores = self.MDR_layer({
                "uesr_ebs": user_ebs,
                "playlist_ebs": playlist_ebs,
                "track_ebs": pos_track_ebs,
                "track_entity_ids": batch["pos_tids"]
            })

            neg_scores = self.MDR_layer({
                "uesr_ebs": user_ebs,
                "playlist_ebs": playlist_ebs,
                "track_ebs": neg_track_ebs,
                "track_entity_ids": batch["neg_tids"]
            })

            mf_loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores)))
            reg_loss_B = tf.nn.l2_loss(self.MDR_layer.get_reg_loss())
            reg_loss_ebs = (tf.nn.l2_loss(user_ebs) + tf.nn.l2_loss(playlist_ebs) +
                            tf.nn.l2_loss(pos_track_ebs) + tf.nn.l2_loss(neg_track_ebs)) / batch["size"]
            reg_loss = self.reg_loss_ratio * (reg_loss_ebs + reg_loss_B)
            loss = mf_loss + reg_loss

            # Compute and apply gradients.
            trainable_variables = [self.user_ebs, self.playlist_ebs, self.track_ebs]
            trainable_variables.extend(self.MDR_layer.get_trainable_variables())
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return Loss(mf_loss, reg_loss)

    def _test(self, epoch):
        test_start_time = time()
        # Compute gnn processed embeddings.

        # For each group compute metrices.
        test_pos_tuples = self.data.test_pos_tuples
        metrics = Metrics(self.max_K)
        for uid, user in self.data.test_data.items():
            for pid, tids in user.items():
                for pos_tid in tids:
                    neg_tids = self.data.sample_negative_test_track_ids(uid, pid)

                    tids = [pos_tid]
                    tids.extend(neg_tids)
                    tids = np.array(tids)

                    user_eb = tf.nn.embedding_lookup(self.user_ebs, uid)
                    playlist_eb = tf.nn.embedding_lookup(self.playlist_ebs, pid)
                    tracks_ebs = tf.nn.embedding_lookup(self.track_ebs, tids)

                    scores = self.MDR_layer({
                        "user_ebs": user_eb,
                        "playlist_ebs": playlist_eb,
                        "track_ebs": tracks_ebs,
                        "track_entity_ids": tids
                    })
                    metrics.add_metrics(tids, scores, pos_tid)
        metrics.cal_avg_metrics()
        test_end_time = time()
        return metrics, test_end_time - test_start_time
