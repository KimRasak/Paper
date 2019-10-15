from DataLayer.NormalData import NormalData


class NormalDataUPT(NormalData):
    def __init__(self, base_data_path, use_reductive_ut_data=True, alpha_ut=1):
        super().__init__(base_data_path)