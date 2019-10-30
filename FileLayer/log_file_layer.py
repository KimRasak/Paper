import os

import numpy as np


class LogManager:
    OUTPUT_DIR = "output/"

    def __init__(self, model_name):
        self.output_file = self.__open_output_file(model_name)

    def __get_output_file_path(self, model_name):
        output_file_name = "%s.txt" % model_name
        output_file_path = os.path.join(LogManager.OUTPUT_DIR, output_file_name)
        return output_file_path

    def __open_output_file(self, model_name):
        output_file_path = self.__get_output_file_path(model_name)
        output_file = open(output_file_path, "a+")
        return output_file

    def write_metric_log(self, hrs, ndcgs, max_k):
        """
        Write mean hr@k and mean ndcg@k to the log file.
        :param model_name: The model's name.
        :param hrs: dict of lists.
        :param ndcgs: dict of lists.
        :param max_k: max observed k of hr@k and ndcg@k.
        :return: None
        """
        # Write mean hr@k.
        self.output_file.write("hr_k: ")
        for i in range(1, max_k+1):
            hr_k = np.average(hrs[i])
            self.output_file.write("%f " % hr_k)
        self.output_file.write("\n")

        # Write mean ndcg@k.
        self.output_file.write("ndcg_k: ")
        for i in range(1, max_k+1):
            ndcg_k = np.average(ndcgs[i])
            self.output_file.write("%f " % ndcg_k)
        self.output_file.write("\n")

    def print_and_write(self, output_str):
        print(output_str)

        self.output_file.write(output_str)
        self.output_file.flush()