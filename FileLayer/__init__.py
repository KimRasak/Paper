"""
This layer defines how to read/write the raw-data/data-set files.
Most functions in Data Layer will depend on this layer.
"""
import os
from collections import namedtuple


# Define namedtuple for numbers of entities in a data set.
DatasetNum = namedtuple("DatasetNum", ["user", "playlist", "track", "interaction"])


# Define names of data sets.
class DatasetName:
    THIRTY_MUSIC = "30music"
    AOTM = "aotm"


# Define paths of raw data.
RAW_DATA_BASE_PATH = "../raw-data"
RAW_THIRTY_MUSIC_PATH = os.path.join(RAW_DATA_BASE_PATH, DatasetName.THIRTY_MUSIC)
RAW_AOTM_PATH = os.path.join(RAW_DATA_BASE_PATH, DatasetName.AOTM)

RAW_PLAYLIST_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(RAW_THIRTY_MUSIC_PATH, "entities/playlist.idomaar"),
    DatasetName.AOTM: os.path.join(RAW_AOTM_PATH, "aotm2011_playlists.json")
}

# Define paths of data sets.
# A data set usually contains a playlist file containing the playlist data
# and a count file containing the number of ids.
DATA_BASE_PATH = "../data"
THIRTY_MUSIC_PATH = os.path.join(DATA_BASE_PATH, DatasetName.THIRTY_MUSIC)
AOTM_PATH = os.path.join(DATA_BASE_PATH, DatasetName.AOTM)

PLAYLIST_FILE_NAME = "playlist.txt"
COUNT_FILE_NAME = "count.txt"

PICK_PLAYLIST_FILE_NAME = "pick_playlist.txt"
PICK_COUNT_FILE_NAME = "pick_count.txt"

WHOLE_PLAYLIST_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, PLAYLIST_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, PLAYLIST_FILE_NAME)
}

WHOLE_COUNT_FILE_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, COUNT_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, COUNT_FILE_NAME)
}

PICK_PLAYLIST_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, PICK_PLAYLIST_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, PICK_PLAYLIST_FILE_NAME)
}

PICK_COUNT_FILE_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, PICK_COUNT_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, PICK_COUNT_FILE_NAME)
}

TRAIN_FILE_NAME = "train.txt"
TEST_FILE_NAME = "test.txt"

PICK_TRAIN_FILE_NAME = "pick_train.txt"
PICK_TEST_FILE_NAME = "pick_test.txt"

TRAIN_FILE_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, TRAIN_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, TRAIN_FILE_NAME)
}


TEST_FILE_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, TEST_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, TEST_FILE_NAME)
}


PICK_TRAIN_FILE_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, PICK_TRAIN_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, PICK_TRAIN_FILE_NAME)
}


PICK_TEST_FILE_PATH = {
    DatasetName.THIRTY_MUSIC: os.path.join(THIRTY_MUSIC_PATH, PICK_TEST_FILE_NAME),
    DatasetName.AOTM: os.path.join(AOTM_PATH, PICK_TEST_FILE_NAME)
}