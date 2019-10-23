from collections import namedtuple

# Define namedtuple for numbers of entities in a data set.
DatasetNum = namedtuple("DatasetNum", ["user", "playlist", "track", "interaction"])