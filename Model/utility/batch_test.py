import multiprocessing
from time import time

import numpy as np

cores = multiprocessing.cpu_count() // 2

def get_metric(ranklist, gt_item):
    return get_hit_ratio(ranklist, gt_item), get_ndcg(ranklist, gt_item)

def get_hit_ratio(ranklist, gt_item):
    for item in ranklist:
        if item == gt_item:
            return 1
    return 0

def get_ndcg(ranklist, gt_item):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gt_item:
            return np.log(2) / np.log(i + 2)
    return 0


# def test(func_predict, func_sample_hundred_neg, test_set, test_batch_size, max_k):
#     pool = multiprocessing.Pool(cores)
#     n_test = len(test_set)
#
#     hrs = {i: [] for i in range(1, max_k + 1)}
#     ndcgs = {i: [] for i in range(1, max_k + 1)}
#
#     for start in range(0, n_test, test_batch_size):
#         # Generate test data.
#         t1_batch = time()
#         end = min(start + test_batch_size, n_test)
#         test_tuples: list = test_set[start:end]
#         for test_tuple in test_tuples:
#             pid = test_tuple[1]
#             tid = test_tuple[2]
#
#             # change test_tuple[2] from int to id list.
#             hundred_neg_tids: list = func_sample_hundred_neg(pid)
#             tids = hundred_neg_tids
#             tids.append(tid)
#             test_tuple[2] = tids
#
#         # Map predict.
#         scores_group = pool.map(func_predict, test_tuples)
#
#         # Add metrics.
#         for i, scores in enumerate(scores_group):
#             assert len(scores_group) == 101
#             sorted_idx = np.argsort(-scores)
#             input_tids = test_tuples[i][2]
#             assert len(input_tids) == 101
#             tid = input_tids[100]
#
#             for k in range(1, max_k + 1):
#                 indices = sorted_idx[:k]  # indices of items with highest scores
#                 ranklist = np.array(input_tids)[indices]
#                 hr_k, ndcg_k = get_metric(ranklist, tid)
#                 hrs[k].append(hr_k)
#                 ndcgs[k].append(ndcg_k)
#
#         print("test_batch[%d] cost %d seconds. hr_10: %f, hr_20: %f" %
#               (start / test_batch_size, time() - t1_batch, np.average(hrs[10]), np.average(hrs[20])))
#     return hrs, ndcgs