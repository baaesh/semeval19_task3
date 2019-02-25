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


def undersampling(lines):
    others_min = 14948
    happy_min = 907
    sad_min = 798
    angry_min = 958

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

    others_undersampled = copy.deepcopy(others)
    happy_undersampled = copy.deepcopy(happy)
    sad_undersampled = copy.deepcopy(sad)
    angry_undersampled = copy.deepcopy(angry)

    # happy undersampling
    for i in range(len(happy) - happy_min):
        randi = random.randrange(0, len(happy_undersampled))
        del happy_undersampled[randi]

    # sad undersampling
    for i in range(len(sad) - sad_min):
        randi = random.randrange(0, len(sad_undersampled))
        del sad_undersampled[randi]

    # angry undersampling
    for i in range(len(angry) - angry_min):
        randi = random.randrange(0, len(angry_undersampled))
        del angry_undersampled[randi]

    data_set = others_undersampled + happy_undersampled + sad_undersampled + angry_undersampled
    random.shuffle(data_set)

    return data_set