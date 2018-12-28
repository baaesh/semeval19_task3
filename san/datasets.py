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

    def __init__(self, raw_field, text_field, label_field, examples=None, path=None, mode='train', **kwargs):
        """Create an dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        text_field.preprocessing = data.Pipeline(self.clean_str)

        if mode == 'train':
            fields = [('raw', raw_field), ('text', text_field), ('label', label_field)]
        else:
            fields = [('raw', raw_field), ('text', text_field)]

        if examples is None:
            dataset = self.preprocessData(path, mode)
            examples = []
            if mode == 'train':
                examples += [
                    data.Example.fromlist([line[0], line[1], line[2]], fields) for line in dataset]
            else:
                examples += [
                    data.Example.fromlist([line[0], line[1]], fields) for line in dataset]
        super(EMO, self).__init__(examples, fields, **kwargs)


    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()


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
        indices = []
        raws = []
        conversations = []
        labels = []
        tokenizer = TweetTokenizer()
        with io.open(dataFilePath, encoding="utf8") as finput:
            finput.readline()
            for line in finput:
                # Convert multiple instances of . ? ! , to single instance
                # okay...sure -> okay . sure
                # okay???sure -> okay ? sure
                # Add whitespace around such punctuation
                # okay!sure -> okay ! sure
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

                conv = ' <eos> '.join(line[1:4])

                # Remove any duplicate spaces
                duplicateSpacePattern = re.compile(r'\ +')
                conv = re.sub(duplicateSpacePattern, ' ', conv)

                indices.append(int(line[0]))
                raws.append(tokenizer.tokenize(conv))
                conversations.append(conv)

        examples = []
        if mode == 'train':
            for i in range(len(conversations)):
                examples.append([raws[i], conversations[i], labels[i]])
        else:
            for i in range(len(conversations)):
                examples.append([raws[i], conversations[i]])
        return examples


    @classmethod
    def splits(cls, raw_field, text_field, label_field, root='.', **kwargs):
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
        train_examples = cls(raw_field, text_field, label_field, path=trainDataPath, mode='train', **kwargs).examples
        valid_examples = cls(raw_field, text_field, label_field, path=validDataPath, mode='train', **kwargs).examples

        return (cls(raw_field, text_field, label_field, examples=train_examples),
                cls(raw_field, text_field, label_field, examples=valid_examples))


    @classmethod
    def getTestData(cls, raw_field, text_field):
        testDataPath = './data/devwithoutlabels.txt'
        return cls(raw_field, text_field, None, path=testDataPath, mode='test')
