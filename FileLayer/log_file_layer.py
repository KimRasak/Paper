import os

import numpy as np


class LogManager:
    OUTPUT_DIR = os.path.abspath("./output/")

    def __init__(self, model_name):
        self.output_file = self.__open_output_file(model_name)

    def __get_output_file_path(self, model_name):
        output_file_name = "%s.txt" % model_name
        output_file_path = os.path.join(LogManager.OUTPUT_DIR, output_file_name)
        return output_file_path

    def __open_output_file(self, model_name):
        output_file_path = self.__get_output_file_path(model_name)
        if not os.path.isfile(output_file_path):
            with open(output_file_path, "w+"): pass
        output_file = open(output_file_path, "a+")
        return output_file

    def write(self, output_str):
        self.output_file.write(output_str)
        self.output_file.flush()

    def print_and_write(self, output_str):
        print(output_str)

        self.output_file.write(output_str)
        self.output_file.flush()
