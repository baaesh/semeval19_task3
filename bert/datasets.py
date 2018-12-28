import os
import io
import re
import random

from torchtext import data
from nltk.tokenize import TweetTokenizer

class EMO(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, turn1_field, turn2_field, turn3_field, label_field, examples=None, path=None, mode='train', **kwargs):
        """Create an dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        if mode == 'train':
            fields = [('turn1', turn1_field), ('turn2', turn2_field), ('turn3', turn3_field), ('label', label_field)]
        else:
            fields = [('turn1', turn1_field), ('turn2', turn2_field), ('turn3', turn3_field)]

        if examples is None:
            dataset = self.preprocessData(path, mode)
            examples = []
            if mode == 'train':
                examples += [
                    data.Example.fromlist([line[0], line[1], line[2], line[3]], fields) for line in dataset]
            else:
                examples += [
                    data.Example.fromlist([line[0], line[1], line[2]], fields) for line in dataset]
        super(EMO, self).__init__(examples, fields, **kwargs)


    def preprocessData(self, dataFilePath, mode):
        """Load data from a file, process and return indices, conversations and labels in separate lists
        Input:
            dataFilePath : Path to train/test file to be processed
            mode : "train" mode returns labels. "test" mode doesn't return labels.
        Output:
            indices : Unique conversation ID list
            conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
            labels : [Only available in "train" mode] List of labels
        """
        turn1s = []
        turn2s = []
        turn3s = []
        labels = []
        with io.open(dataFilePath, encoding="utf8") as finput:
            finput.readline()
            for line in finput:
                repeatedChars = ['.', '?', '!', ',']
                for c in repeatedChars:
                    lineSplit = line.split(c)
                    while True:
                        try:
                            lineSplit.remove('')
                        except:
                            break
                    cSpace = ' ' + c + ' '
                    line = cSpace.join(lineSplit)

                line = line.strip().split('\t')
                if mode == "train":
                    # Train data contains id, 3 turns and label
                    label = line[4]
                    labels.append(label)

                turn1s.append(line[1])
                turn2s.append(line[2])
                turn3s.append(line[3])

        assert len(turn1s) == len(turn2s)
        assert len(turn2s) == len(turn3s)

        examples = []
        if mode == 'train':
            for i in range(len(turn1s)):
                examples.append([turn1s[i], turn2s[i], turn3s[i], labels[i]])
        else:
            for i in range(len(turn1s)):
                examples.append([turn1s[i], turn2s[i], turn3s[i]])
        return examples


    @classmethod
    def splits(cls, turn1_field, turn2_field, turn3_field, label_field, root='.', **kwargs):
        """Create dataset objects for splits of the dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        trainDataPath = './data/train_split.txt'
        validDataPath = './data/valid_split.txt'
        train_examples = cls(turn1_field, turn2_field, turn3_field, label_field, path=trainDataPath, mode='train', **kwargs).examples
        valid_examples = cls(turn1_field, turn2_field, turn3_field, label_field, path=validDataPath, mode='train', **kwargs).examples

        return (cls(turn1_field, turn2_field, turn3_field, label_field, examples=train_examples),
                cls(turn1_field, turn2_field, turn3_field, label_field, examples=valid_examples))


    @classmethod
    def getTestData(cls, turn1_field, turn2_field, turn3_field):
        testDataPath = './data/devwithoutlabels.txt'
        return cls(turn1_field, turn2_field, turn3_field, None, path=testDataPath, mode='test')
