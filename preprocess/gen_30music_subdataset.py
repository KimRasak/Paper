# coding=utf-8
import scipy as spy
import scipy.sparse as sp
import numpy as np
from scipy.io import mmwrite, mmread
from surprise.model_selection import LeaveOneOut
from preprocess.gen_30music_dataset import filter_and_compact, convert_to_sparse_matrix, transfer_row_column
import random
from preprocess.util import random_pick_subset, filter_dataset

import preprocess.examine_30music_events as ex

interaction_data_path = "30music_interaction.mtx"
filter_subset_data_path = "30music_interaction_subset_min10x10.mtx"
percent_subset_data_path = "30music_interaction_subset_min10x10_percent.mtx"

# def filter_data_subset(mtx_data_path):
#     # Read file.
#     data, max_uid, max_tid = ex.read_file_events(ex.file_path_events)
#     print("Read file complete.")
#
#     # Filter data
#     filtered_data, num_user, num_track = filter_and_compact(data, min_ui_count=10, min_iu_count=10)
#     print("Filter data complete.")
#
#     # Save the sub-dataset for later processing.
#     print("Saving the sparse matrix.")
#     sparse_matrix = convert_to_sparse_matrix(filtered_data, num_user, num_track)
#     mmwrite(mtx_data_path, sparse_matrix)
#
#     # Now, each user/track in the generated sub-dataset has at least 10 interactions.
#
#
# def mtx_file_to_dict(mtx_file_path):
#     """
#     Read matrix from .mtx file and store it in the form of dict.
#     :return: Dict. Containing the matrix.
#     """
#     mtx_data = mmread(mtx_file_path).tocsr()
#     data = {}
#
#     num_rows = mtx_data.get_shape()[0]
#     print("matrix file read. shape is", mtx_data.get_shape())
#     for row in range(num_rows):
#         if row not in data:
#             data[row] = []
#
#         for col in mtx_data.getrow(row).indices:
#             data[row].append(col)
#
#     return data
#
#
# def random_pick_rows(data: dict, row_percent: int):
#     """
#     Random pick {row_percent} of rows in {data}.
#     :param data: Dict. Storing the 2-d matrix.
#     """
#     # Uniformly pick some rows.
#     if row_percent == 1:
#         print("[Random pick rows] %d rows remained." % len(data.keys()))
#         return data
#
#     keys = list(data.keys())
#     num_rows = len(data.keys())
#     num_remain = int(num_rows * row_percent)
#     remaining_rows = random.sample(keys, num_remain)
#
#     # Delete other rows and
#     # compact the row keys to make them continuous.
#     compact_data = {}
#     compact_key = 0
#     for key in keys:
#         if key not in remaining_rows:
#             del data[key]
#         else:
#             compact_data[compact_key] = data[key]
#             compact_key += 1
#
#     print("[Random pick rows] %d rows remained." % compact_key)
#     return compact_data
#
# def gen_random_pick_subset(mtx_file_path, gen_mtx_file_path, row_percent, col_percent):
#     """
#     Randomly pick sub percent of rows and cols.
#     The function first uniformly throw away 1-{row_percent} of rows,
#     then throw away 1-{col_percent} of cols.
#     :param mtx_file_path: The path to .mtx file.
#     :param row_percent: The remaining percent of users.
#     :param col_percent: The remaining percent of tracks.
#     :return: The remaining data.
#     """
#     data = mtx_file_to_dict(mtx_file_path)
#
#     data = random_pick_rows(data, row_percent)
#
#     # Randomly pick som rows
#     T_data = transfer_row_column(data)
#     T_data = random_pick_rows(T_data, col_percent)
#     num_cols = len(T_data.keys())
#     data = transfer_row_column(T_data)
#
#     num_rows = len(data.keys())
#     print("Remaining sub-dataset matrix: (%d * %d)" % (num_rows, num_cols))
#     sparse_matrix = convert_to_sparse_matrix(data, num_rows, num_cols)
#     mmwrite(gen_mtx_file_path, sparse_matrix)
#

if __name__ == '__main__':
    print("Reading .mtx file...")
    mtx_data: sp.csr_matrix = mmread(interaction_data_path).tocsr()
    print("Read .mtx file complete.")
    mtx_data = filter_dataset(mtx_data)
    print("filtered_data: ", mtx_data.get_shape())
    mmwrite(filter_subset_data_path, mtx_data)

    mtx_data = random_pick_subset(mtx_data, 0.5, 0.9)
    # filtered_data:  (41916, 269824)
    # percent_data (20958, 242476) 10*10 (0.5, 0.9) epoch: 52 loss:2035 hr@10: 0.61
    # epoch: 78 loss: 1084 hr@10: 0.70
    print("percent_data", mtx_data.get_shape())
    mmwrite(percent_subset_data_path, mtx_data)

