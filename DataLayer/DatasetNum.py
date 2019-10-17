# Define namedtuple for numbers of entities in a data set.
from collections import namedtuple

DatasetNum = namedtuple("DatasetNum", ["user", "playlist", "track", "interaction"])