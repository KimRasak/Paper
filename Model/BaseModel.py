

class BaseModel:
    def __init__(self, num_epoch):
        self.num_epoch = num_epoch
        pass

    def fit(self):
        for i_epoch in range(self.num_epoch):
            self.train_epoch()
            self.test()

    def train_epoch(self):
        pass

    def test(self):
        pass