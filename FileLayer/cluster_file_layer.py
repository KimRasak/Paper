"""
Provide read/write API of cluster files.
"""


def read_cluster_file(cluster_file_path):
    """
    Read cluster file and return an array that maps entity id to cluster id.
    :param cluster_file_path: Path of the cluster file.
    :return: An array that maps entity id to cluster id.
    """
    parts = []
    with open(cluster_file_path) as f:
        line = f.readline()
        while line:
            ids = [int(s) for s in line.split() if s.isdigit()]
            assert len(ids) == 2

            id = ids[0]  # Not used.
            map_id = ids[1]
            parts.append(map_id)

            line = f.readline()
    return parts


def write_cluster_file(cluster_file_path, parts):
    """
    Write the mappings to the cluster file.
    :param cluster_file_path: Path of the cluster file.
    :param parts: An array that maps entity id to cluster id.
    :return: null
    """
    with open(cluster_file_path, "w") as f:
        for id, part in enumerate(parts):
            f.write("%d %d\n" % (id, part))
    print("Generated cluster file...")