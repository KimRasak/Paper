import os

if __name__ == '__main__':
    strs = []
    with open("../data/movielens/test.txt") as f:
        lines = f.readlines()
        for line in lines:
           strs.append(line[:-1] + ' \n')

    with open("../data/movielens/test.txt", 'w') as f:
        f.writelines(strs)
