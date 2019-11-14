from ModelLayertf2.NormalModels.NormalPTModel import NormalPTModel


class PT_BPR(NormalPTModel):
    def _train_batch(self, batch_no, batch):
        tids = batch["tids"]

    def _test(self, epoch):
        pass

    def _build_model(self):
        pass
