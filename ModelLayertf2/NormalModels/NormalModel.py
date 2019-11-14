from abc import ABC, abstractmethod
from time import time

from DataLayer.NormalDatas.NormalData import NormalData
from ModelLayertf2.BaseModel import BaseModel, Loss


class NormalModel(BaseModel, ABC):
    def __init__(self, data: NormalData, epoch_num, embedding_size=64, learning_rate=2e-4, reg_loss_ratio=5e-5):
        super().__init__(epoch_num, embedding_size,
                         learning_rate, reg_loss_ratio)
        self.data = data

    def _train_epoch(self, epoch):
        epoch_loss = Loss()
        epoch_start_time = time()
        for batch_no, batch in enumerate(self.data.get_batches()):
            batch_loss: Loss = self._train_batch(batch_no, batch)
            epoch_loss.add_loss(batch_loss)
        epoch_end_time = time()
        return epoch_loss, epoch_end_time - epoch_start_time

    @abstractmethod
    def _train_batch(self, batch_no, batch):
        pass
