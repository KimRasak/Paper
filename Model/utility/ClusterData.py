from Model.utility.Data import Data


class ClusterData(Data):
    def __init__(data_base_path, num_cluster=50):
        super().__init__(data_base_path)