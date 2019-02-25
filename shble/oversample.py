import random
import copy

def oversampling(lines):
    others_max = 102337
    happy_max = 6210
    sad_max = 5463
    angry_max = 6560

    others = []
    happy = []
    sad = []
    angry = []

    for line in lines:
        line_split = line.strip().split('\t')
        label = line_split[4]

        if label == 'others':
            others.append(line)
        elif label == 'happy':
            happy.append(line)
        elif label == 'sad':
            sad.append(line)
        elif label == 'angry':
            angry.append(line)
        else:
            print("exception: " + label)

    others_oversampled = copy.deepcopy(others)
    happy_oversampled = copy.deepcopy(happy)
    sad_oversampled = copy.deepcopy(sad)
    angry_oversampled = copy.deepcopy(angry)

    # others oversampling
    for i in range(others_max - len(others)):
        randi = random.randrange(0, len(others))
        others_oversampled.append(others[randi])

    # happy oversampling
    for i in range(happy_max - len(happy)):
        randi = random.randrange(0, len(happy))
        happy_oversampled.append(happy[randi])

    # angry oversampling
    for i in range(angry_max - len(angry)):
        randi = random.randrange(0, len(angry))
        angry_oversampled.append(angry[randi])

    data_set = others_oversampled + happy_oversampled + sad_oversampled + angry_oversampled
    random.shuffle(data_set)

    return data_set