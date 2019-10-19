from DataLayer.Data import Data


class ClusterData(Data):
    def __init__(self, data_set_name, use_picked_data=True,
                 batch_size=256, epoch_times=4, is_test_mode=False,
                 num_cluster=50):
        super().__init__(data_set_name, use_picked_data=True, batch_size=256, epoch_times=4, is_test_mode=False)
        self.num_cluster = num_cluster