from Model.utility.data_helper import Data


def test(data: Data):
    for uid, user in data.test_set.items():
        for pid, tids in user:
            pass