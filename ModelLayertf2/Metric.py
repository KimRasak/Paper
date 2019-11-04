import numpy as np



class Metrics:
    """
    Remember to call cal_avg_metrics before get_avg_hr, get_avg_ndcg and to_string.
    """
    hrs: dict  # Hit ratios.
    ndcgs: dict  # NDCGs.
    avg_hrs: dict
    avg_ndcgs: dict

    def __init__(self, max_k):
        self.hrs = {k: [] for k in range(1, max_k + 1)}
        self.ndcgs = {k: [] for k in range(1, max_k + 1)}
        self.max_k = max_k

    @staticmethod
    def __get_metric(ranklist, gt_item):
        return Metrics.__get_hit_ratio(ranklist, gt_item), Metrics.__get_ndcg(ranklist, gt_item)

    @staticmethod
    def __get_hit_ratio(ranks, gt_item):
        for item in ranks:
            if item == gt_item:
                return 1
        return 0

    @staticmethod
    def __get_ndcg(ranks, gt_item):
        for i in range(len(ranks)):
            item = ranks[i]
            if item == gt_item:
                return np.log(2) / np.log(i + 2)
        return 0

    def add_metrics(self, test_tids, tid_scores, tid):
        sorted_indices = np.argsort(-tid_scores)
        test_tids = np.array(test_tids)
        for k in range(1, self.max_k + 1):
            indices = sorted_indices[:k]  # indices of items with highest scores
            ranklist = test_tids[indices]
            hr_k, ndcg_k = Metrics.__get_metric(ranklist, tid)
            self.hrs[k].append(hr_k)
            self.ndcgs[k].append(ndcg_k)

    def cal_avg_metrics(self):
        for k in range(1, self.max_k + 1):
            self.avg_hrs[k] = np.average(self.hrs[k])
            self.avg_ndcgs[k] = np.average(self.ndcgs[k])

    def get_avg_hr(self, num):
        return self.avg_hrs[num]

    def get_avg_ndcg(self, num):
        return self.avg_ndcgs[num]

    def to_string(self):
        hrs_str = ", ".join(["%f" % self.avg_hrs[k] for k in range(1, self.max_k + 1)])
        ndcgs_str = ", ".join(["%f" % self.avg_ndcgs[k] for k in range(1, self.max_k + 1)])
        return "hr_k: {}\n ndcg_k: {}\n".format(hrs_str, ndcgs_str)