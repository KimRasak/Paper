from abc import ABC, abstractmethod

from DataLayer.Data import Data


class ClusterData(Data, ABC):
    def __init__(self, data_set_name, use_picked_data=True,
                 batch_size=256, epoch_times=4, is_test_mode=False,
                 num_cluster=50):
        super().__init__(data_set_name, use_picked_data=use_picked_data, batch_size=batch_size, epoch_times=epoch_times, is_test_mode=is_test_mode)
        self.num_cluster = num_cluster

    def __init_relation_data(self, train_data: dict):
        """
        This function only needs to init dicts storing the relationships.
        The ids in the dicts are local to its entity.
        """
        self.__init_relation_dict(train_data)

    # @abstractmethod
    # def __gen_global_id_pairs(self):
    #     """
    #     Generate a list that stores all global id connection pairs.
    #     :return: A list of pairs. Each pair is in the form of (from_id, to_id), indicating that there's a
    #     connection between from_id and to_id. Note that only user->playlist / playlist->track / user->track connections
    #     are valid.
    #     """
    #     pass