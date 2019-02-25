import io
import argparse


def getStatistics(dataFilePath, mode='train'):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    count = 0
    happy = 0
    sad = 0
    angry = 0
    others = 0
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            line = line.strip().split('\t')
            label = line[4]
            if label == 'others':
                others += 1
            elif label == 'happy':
                happy += 1
            elif label == 'sad':
                sad += 1
            elif label == 'angry':
                angry += 1
            else:
                print('error: ' + label)
            count += 1

    print(mode + ' class distribution')
    print('ohers: ' + str(others/count))
    print(others)
    print('happy: ' + str(happy/count))
    print(happy)
    print('sad: ' + str(sad/count))
    print(sad)
    print('angry: ' + str(angry/count))
    print(angry)
    return [others/count, happy/count, sad/count, angry/count]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='test.txt')
    args = parser.parse_args()

    getStatistics(args.name, 'train')