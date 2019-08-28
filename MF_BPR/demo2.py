import multiprocessing
from abc import ABCMeta, abstractmethod

import tensorflow as tf

import numpy as np


a = [1, 2]
b = a[0]
a[0] = 3
print(b, a)