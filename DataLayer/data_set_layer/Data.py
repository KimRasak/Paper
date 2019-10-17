import os
from time import time


class Data:
    # File names of data set.
    TRAIN_FILE_NAME = "train.txt"
    TEST_FILE_NAME = "test.txt"
    EVENTS_FILE_NAME = "event.txt"

    # File names of picked data set.
    PREFIX_PICK_FILE = "pick"
    PICK_TRAIN_FILE_NAME = PREFIX_PICK_FILE + "_" + TRAIN_FILE_NAME
    PICK_TEST_FILE_NAME = PREFIX_PICK_FILE + "_" + TEST_FILE_NAME
    PICK_EVENTS_FILE_NAME = PREFIX_PICK_FILE + "_" + EVENTS_FILE_NAME

    def __init__(self, base_data_path, use_picked_data=True,
                 batch_size=256, epoch_times=4, is_test_mode=False):
        t_all_start = time()
        self.data_base_path = base_data_path
        self.use_picked_data = use_picked_data

        self.batch_size = batch_size

        # Define data set paths.
        if use_picked_data:
            print("{pick} == %r, Using picked playlist data. That is, you're using a sub-dataset" % use_picked_data)
            train_filepath = os.path.join(base_data_path, Data.PICK_TRAIN_FILE_NAME)
            test_filepath = os.path.join(base_data_path, Data.PICK_TRAIN_FILE_NAME)
            event_filepath = os.path.join(base_data_path, Data.PICK_EVENTS_FILE_NAME)
            # cluster_map_filepath = data_base_path + "/pick_cluster_%d.txt" % (num_cluster)
        else:
            print("{pick} == %r, Using complete playlist data. That is, you're using a complete dataset" % use_picked_data)
            train_filepath = os.path.join(base_data_path, Data.TRAIN_FILE_NAME)
            test_filepath = os.path.join(base_data_path, Data.TEST_FILE_NAME)
            event_filepath = os.path.join(base_data_path, Data.EVENTS_FILE_NAME)
            # cluster_map_filepath = data_base_path + "/cluster_%s_%d.txt" % (laplacian_mode, num_cluster)

        # 验证laplacian模式
        laplacian_modes = ["PT2", "PT4", "UT", "UPT", "None", "TestPT", "TestUPT",
                           "clusterPT2", "clusterPT4", "clusterUT", "clusterUPT"]
        if laplacian_mode not in laplacian_modes:
            raise Exception("Wrong laplacian mode. Expected one of %r, got %r" % (laplacian_modes, laplacian_mode))
        self.laplacian_mode = laplacian_mode

        # Read and print statistics.
        self.read_entity_num(train_filepath)

        if "cluster" not in laplacian_mode:
            self.R_up = sp.dok_matrix((self.n_user, self.n_playlist), dtype=np.float64)
            self.R_ut = sp.dok_matrix((self.n_user, self.n_track), dtype=np.float64)
            self.R_pt = sp.dok_matrix((self.n_playlist, self.n_track), dtype=np.float64)
        self.pt = {}  # Storing playlist-track relationship of training set.
        self.up = {}  # Storing user-playlist relationship of training set.
        self.ut = {}  # Storing user-track relationship of training set.
        self.test_set = []  # Storing user-playlist-track test set.


        # Print time used for reading and pre-processing data.
        t_all_end = time()
        print("Reading data used %d seconds in all." % (t_all_end - t_all_start))

    def get_dataset_name(self):
        return self.data_base_path.split("/")[-1]