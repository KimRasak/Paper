# Restore model.
import os

import tensorflow as tf


class SaveManager:
    TRAIN_STEP_VARIABLE_NAME = "train_step"
    CHECKPOINT_BASE_DIR = "ckpt"

    def __init__(self, model_name, optimizer, nets):
        self.__model_dir = os.path.join(SaveManager.CHECKPOINT_BASE_DIR, model_name)
        self.__train_step = tf.Variable(0, name=SaveManager.TRAIN_STEP_VARIABLE_NAME)
        self.__ckpt = tf.train.Checkpoint(train_step=self.__train_step, optimizer=optimizer)
        self.__ckpt.nets = nets
        self.__manager = tf.train.CheckpointManager(self.__ckpt, self.__model_dir, max_to_keep=3)

    def restore_model(self):
        self.__manager.restore()
        if self.__manager.latest_checkpoint:
            print("Restored from {}".format(self.__manager.latest_checkpoint))
        else:
            print("No checkpoint on {}, Initializing from scratch.".format(self.__model_dir))

    def save_model(self):
        if not os.path.exists(self.__model_dir):
            os.makedirs(self.__model_dir)
        print("Saving model of epoch {}".format(self.__train_step.numpy()))
        self.__manager.save()

        # Update variable
        self.__train_step.assign_add(1)

    def get_train_step(self):
        return self.__train_step.numpy()
