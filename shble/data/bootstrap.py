import os
import io
import random

trainFilePath = 'data/train_emoji.txt'
validFilePath = 'data/dev_emoji.txt'
others = []
happy = []
angry = []
sad = []

with io.open(trainFilePath, encoding="utf8") as finput:
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

with io.open(validFilePath, encoding="utf8") as finput:
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

happy_num = int(dev_num * 0.05)
sad_num = int(dev_num * 0.05)
angry_num = int(dev_num * 0.05)
others_num = int(dev_num - happy_num - angry_num - sad_num)

random.shuffle(others)
random.shuffle(happy)
random.shuffle(angry)
random.shuffle(sad)

dev_set = others[:others_num] + happy[:happy_num] + angry[:angry_num] + sad[:sad_num]
train_set = others[others_num:] + happy[happy_num:] + angry[angry_num:] + sad[sad_num:]

random.shuffle(dev_set)
random.shuffle(train_set)

with open('data/train_emoji_split.txt', 'w', encoding='utf-8') as f:
    for line in train_set:
        line = line.strip() + '\n'
        f.write(line)

with open('data/dev_emoji_split.txt', 'w', encoding='utf-8') as f:
    for line in dev_set:
        line = line.strip() + '\n'
        f.write(line)