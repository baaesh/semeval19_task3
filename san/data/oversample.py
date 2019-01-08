import os
import io
import random

dataFilePath = 'raw/train.txt'
others = []
emotional = []
head = ''
with io.open(dataFilePath, encoding="utf8") as finput:
    finput.readline()
    for line in finput:
        line_split = line.strip().split('\t')
        label = line_split[4]

        if label == 'others':
            others.append(line)
        else:
            emotional.append(line)

train_set = others * 5 + emotional
random.shuffle(train_set)

with open('train_oversampled.txt', 'w', encoding='utf-8') as f:
    f.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
    for line in train_set:
        f.write(line)