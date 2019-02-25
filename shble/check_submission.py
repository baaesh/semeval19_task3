import os
import io
import random

dataFilePath = 'experiments/best.txt'

with io.open(dataFilePath, encoding="utf8") as finput:
    finput.readline()
    for line in finput:
        line_split = line.strip().split('\t')
        try:
            label = line_split[4]
        except:
            print(line_split)

        if label == 'others':
            pass
        elif label == 'happy':
            pass
        elif label == 'angry':
            pass
        elif label == 'sad':
            pass
        else:
            print(line_split[0])
            print(label)