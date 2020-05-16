import os
from functools import partial


raw_base_path = "raw-data/rating"
raw_train_path = os.path.join(raw_base_path, "rating_data_train_ab.txt")
raw_test_path = os.path.join(raw_base_path, "rating_data_test.txt")

data_base_path = "data/ratings"
train_path = os.path.join(data_base_path, "train.txt")
test_path = os.path.join(data_base_path, "test.txt")
count_path = os.path.join(data_base_path, "count.txt")


def iter_rating(file_path, func):
    with open(file_path) as f:
        line = f.readline()
        while line:
            items = line.split("\t")
            user = items[0]
            item = items[1]
            score = int(float(items[2]))

            func(user, item, score)
            line = f.readline()


def get_tags_map():
    user_map = dict()
    item_map = dict()

    def add_to_set(user_map: dict, item_map:dict, user, item, score):
        if user not in user_map:
            user_map[user] = int(len(user_map))
        if item not in item_map:
            item_map[item] = int(len(item_map))

    iter_rating(raw_train_path, partial(add_to_set, user_map, item_map))
    iter_rating(raw_test_path, partial(add_to_set, user_map, item_map))
    return user_map, item_map


def check_all_ids_in_map(user_map, item_map):
    def check_in_map(user, item, score):
        assert user in user_map and item in item_map

    iter_rating(raw_train_path, check_in_map)
    iter_rating(raw_test_path, check_in_map)


def get_data(user_map, item_map):
    def add_to_data(data: dict, user_tag, item_tag, score):
        user_id = user_map[user_tag]
        item_id = item_map[item_tag]
        if user_id not in data:
            data[user_id] = dict()
        data[user_id][item_id] = score
    train_data = dict()
    iter_rating(raw_train_path, partial(add_to_data, train_data))
    test_data = dict()
    iter_rating(raw_test_path, partial(add_to_data, test_data))
    return train_data, test_data


def save_playlist_count_file(user_num, item_num):
    if not os.path.exists(data_base_path):
        os.makedirs(data_base_path)

    with open(count_path, "w") as f:
        f.write("number of user\n")
        f.write("%d\n" % user_num)
        f.write("number of item\n")
        f.write("%d\n" % item_num)
        f.write("number of interactions\n")
        f.write("1\n")  # No use, just fill in a dummy,


def write_data(data_path, data):
    if not os.path.exists(data_base_path):
        os.makedirs(data_base_path)

    with open(data_path, "w") as f:
        f.write("user_id track_ids\n")
        for user_id in data:
            f.write("{} ".format(user_id))
            for item_id, score in data[user_id].items():
                f.write("(%d,%d) " % (item_id, score))
            f.write("\n")


def main():
    user_map, item_map = get_tags_map()
    check_all_ids_in_map(user_map, item_map)
    print("All ids are in map.")
    train_data, test_data = get_data(user_map, item_map)
    save_playlist_count_file(len(user_map), len(item_map))
    print("Generate data.")
    write_data(train_path, train_data)
    write_data(test_path, test_data)


if __name__ == '__main__':
    main()