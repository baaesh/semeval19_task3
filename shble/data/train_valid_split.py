import os
import io
import random

dataFilePath = 'raw/train.txt'
others = []
happy = []
angry = []
sad = []

with io.open(dataFilePath, encoding="utf8") as finput:
    finput.readline()
    for line in finput:
        line_split = line.strip().split('\t')
        label = line_split[4]

        if label == 'others':
            others.append(line)
        elif label == 'happy':
            happy.append(line)
        elif label == 'angry':
            angry.append(line)
        else:
            sad.append(line)

total_len = len(others) + len(happy) + len(angry) + len(sad)
print(f'others: {len(others)/total_len} '
      f'happy: {len(happy)/total_len} '
      f'angry: {len(angry)/total_len} '
      f'sad: {len(sad)/total_len}')

dev_ratio = 0.1
dev_num = total_len * dev_ratio
others_num = int(dev_num * 0.86)
happy_num = int((dev_num - others_num)/3)
angry_num = happy_num
sad_num = happy_num

random.shuffle(others)
random.shuffle(happy)
random.shuffle(angry)
random.shuffle(sad)

dev_set = others[:others_num] + happy[:happy_num] + angry[:angry_num] + sad[:sad_num]
train_set = others[others_num:] + happy[happy_num:] + angry[angry_num:] + sad[sad_num:]

random.shuffle(dev_set)
random.shuffle(train_set)

with open('train_split.txt', 'w', encoding='utf-8') as f:
    for line in train_set:
        f.write(line)

with open('valid_split.txt', 'w', encoding='utf-8') as f:
    for line in dev_set:
        f.write(line)