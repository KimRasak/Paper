from DataLayer.data_set_layer.Data import Data


class ClusterData(Data):
    def __init__(base_data_path, num_cluster=50):
        super().__init__(base_data_path)