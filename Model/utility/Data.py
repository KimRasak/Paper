from time import time


class Data:
    def __init__(self, data_base_path, use_picked_data=True, batch_size=256, use_reductive_ut_data=True, alpha=1,
                 epoch_times=4, num_cluster=50, is_test_mode=False):
        t_all_start = time()
        self.data_base_path = data_base_path

        self.batch_size = batch_size
        self.alpha = alpha
        self.num_cluster = num_cluster

        if use_picked_data is True:
            print("{pick} == %r, Using picked playlist data. That is, you're using a sub-dataset" % use_picked_data)
            train_filepath = data_base_path + "/pick_train.txt"
            test_filepath = data_base_path + "/pick_test.txt"
            event_filepath = data_base_path + "/pick_events.txt"
            cluster_map_filepath = data_base_path + "/pick_cluster_%d.txt" % (num_cluster)
        else:
            print("{pick} == %r, Using complete playlist data. That is, you're using a complete dataset" % use_picked_data)
            train_filepath = data_base_path + "/train.txt"
            test_filepath = data_base_path + "/test.txt"
            event_filepath = data_base_path + "/events.txt"
            cluster_map_filepath = data_base_path + "/cluster_%s_%d.txt" % (laplacian_mode, num_cluster)

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

        if laplacian_mode != "Test":
            # Initialize R_ut
            if use_reductive_ut_data:
                print("Using Reductive R_ut, not reading events data.")
            else:
                self.read_events_file(event_filepath)

            # Initialize matrix R_up, matrix R_pt, dict up and dict pt.
            self.read_train_file(train_filepath, reductive_ut=use_reductive_ut_data)

            # Initialize test_set
            self.read_test_file(test_filepath)

        self.n_batch = int(np.ceil(self.n_train / self.batch_size)) * epoch_times
        print("There are %d training instances. Each epoch has %d batches with batch size of %d." % (self.n_train, self.n_batch, self.batch_size))

        self.set_offset()
        # 读取cluster映射
        if "cluster" in laplacian_mode:
            print("laplacian_mode=%r, loading clustered laplacian matrix." % laplacian_mode)
            self.read_cluster_laplacian_matrix(num_cluster, cluster_map_filepath)
            self.map_ids()
        elif laplacian_mode != "None":
            print("laplacian_mode=%r, loading laplacian matrix." % laplacian_mode)
            self.gen_normal_laplacian_matrix()
        elif laplacian_mode == "None":
            print("laplacian_mode=%r. Don't load laplacian matrix." % laplacian_mode)

        t_all_end = time()
        print("Read data used %d seconds in all." % (t_all_end - t_all_start))

    def get_dataset_name(self):
        return self.data_base_path.split("/")[-1]