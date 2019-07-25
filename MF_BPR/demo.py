from preprocess.gen_30music_dataset import filter_and_compact

temp = {5: [1, 2], 1: [1, 3, 100], 3: [1, 5, 100]}
res = filter_and_compact(temp, min_ui_count=2, min_iu_count=1)
print(res)