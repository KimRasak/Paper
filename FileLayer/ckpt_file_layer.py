# Restore model.
import os

import tensorflow as tf


class SaveManager:
    CHECKPOINT_DIR = "ckpt"

    def __init__(self, model_name):
        self.model_dir = os.path.join(SaveManager.CHECKPOINT_DIR, model_name)
        self.saver = tf.train.Saver()

    def restore_model(self, sess, model_path=None):
        if model_path:
            # Restore the model on the given path.
            self.saver.restore(sess, model_path)
            return

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restore the checkpoint model.
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model checkpoint found on dir %r. Using model." % model_path)
        else:
            print('No model on dir %r. Not using model.' % model_path)

    def save_model(self, sess, epoch):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, "model%d.cpkt" % epoch)
        self.saver.save(sess, model_path)
