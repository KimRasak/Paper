import numpy as np

arr = [1, 3, 5, 66, 77, 88]
np.random.shuffle(arr)
arr = np.array(arr)
print(arr)
idx = [3, 4]
print(type(arr))
print(arr[idx])