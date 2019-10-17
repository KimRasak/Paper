
def write_playlist_data(playlist_data: dict, playlist_data_path):
    # Write playlist data to file.
    with open(playlist_data_path, 'w') as f:
        f.write("user_id playlist_id track_ids\n")
        for uid, user in playlist_data.items():
            for pid, tids in user.items():
                f.write("%d %d " % (uid, pid))
                for tid in tids:
                    f.write("%d " % tid)
                f.write("\n")


def write_count_file(dataset_num, count_file_path):
    # Write the number of user/playlist/track into the file.
    num_user = dataset_num.user
    num_playlist = dataset_num.playlist
    num_track = dataset_num.track

    with open(count_file_path, 'w') as f:
        f.write("number of user\n")
        f.write(num_user + "\n")
        f.write("number of playlist\n")
        f.write(num_playlist + "\n")
        f.write("number of track\n")
        f.write(num_track + "\n")