import random

data = []
with open("train_haze4k.txt", "r") as file:
    for line in file:
        data.append(line.strip().split("\t"))

random.shuffle(data)

with open("train_haze4k_shuffle.txt", "w") as file:
    for item in data:
        file.write("\t".join(item) + "\n")

print('shuffle done')
