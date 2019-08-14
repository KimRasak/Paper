from Model.BaseModel import BaseModel
from Model.utility.data_helper import Data


class ME_NGCF(BaseModel):
    def __init__(self, num_epoch, data: Data):
        super().__init__(num_epoch, data)
        # self.t_opt, self.t_predict_score, self.t_loss = self.build_model()

    def build_model(self):
        print("1, 2, 3")
        return 1, 2, 3

