from Model.utility.NormalData import NormalData


class NormalDataUPT(NormalData):
    def __init__(self, data_base_path, use_reductive_ut_data=True, alpha_ut=1):
        super().__init__(data_base_path)