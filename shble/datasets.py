import os
import io
import re
import random

from torchtext import data
from nltk.tokenize import TweetTokenizer
from ekphrasis.classes.tokenizer import SocialTokenizer
from oversample import oversampling
from undersample import undersampling
from preprocess import remove_duplicated_emojis

class EMO(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, args, raw_field, text_field, label_field, examples=None, path=None, mode='train', **kwargs):
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
            fields = [('raw', raw_field),
                      ('raw_c', raw_field),
                      ('raw_s', raw_field),
                      ('raw_turn1', raw_field),
                      ('raw_turn2', raw_field),
                      ('raw_turn3', raw_field),
                      ('text', text_field),
                      ('context', text_field),
                      ('sent', text_field),
                      ('turn1', text_field),
                      ('turn2', text_field),
                      ('turn3', text_field),
                      ('label', label_field)]
        else:
            fields = [('raw', raw_field),
                      ('raw_c', raw_field),
                      ('raw_s', raw_field),
                      ('raw_turn1', raw_field),
                      ('raw_turn2', raw_field),
                      ('raw_turn3', raw_field),
                      ('text', text_field),
                      ('context', text_field),
                      ('sent', text_field),
                      ('turn1', text_field),
                      ('turn2', text_field),
                      ('turn3', text_field)]

        if examples is None:
            dataset = self.preprocessData(args, path, mode)
            examples = []
            if mode == 'train':
                examples += [
                    data.Example.fromlist([line[0],
                                           line[1],
                                           line[2],
                                           line[3],
                                           line[4],
                                           line[5],
                                           line[6],
                                           line[7],
                                           line[8],
                                           line[9],
                                           line[10],
                                           line[11],
                                           line[12]],
                                          fields) for line in dataset]
            else:
                examples += [
                    data.Example.fromlist([line[0],
                                           line[1],
                                           line[2],
                                           line[3],
                                           line[4],
                                           line[5],
                                           line[6],
                                           line[7],
                                           line[8],
                                           line[9],
                                           line[10],
                                           line[11]],
                                          fields) for line in dataset]
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


    def preprocessData(self, args, dataFilePath, mode):
        """Load data from a file, process and return indices, conversations and labels in separate lists
        Input:
            dataFilePath : Path to train/test file to be processed
            mode : "train" mode returns labels. "test" mode doesn't return labels.
        Output:
            indices : Unique conversation ID list
            conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
            labels : [Only available in "train" mode] List of labels
        """
        binary = False

        indices = []
        raws = []
        raws_c = []
        raws_s = []
        raws_turn1 = []
        raws_turn2 = []
        raws_turn3 = []
        conversations = []
        contexts = []
        sents = []
        labels = []
        turn1s = []
        turn2s = []
        turn3s = []

        if args.datastories:
            tokenizer = SocialTokenizer(lowercase=True)
        else:
            tokenizer = TweetTokenizer()
            
        with io.open(dataFilePath, encoding="utf8") as finput:
            finput.readline()
            lines = finput.readlines()

            if args.remove_duplicated_emojis:
                lines = remove_duplicated_emojis(lines)

            if args.oversampling:
                lines = oversampling(lines)
            elif args.undersampling:
                lines = undersampling(lines)

            for line in lines:
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
                    # binary classification
                    if binary and label != 'others':
                        label = 'emotional'
                    labels.append(label)

                #conv = ' <eos> '.join(line[1:4])
                conv = line[1] + ' <eos1> ' + line[2] + ' <eos2> ' + line[3]
                context = ' <eos> '.join(line[1:3])
                sent = line[3]
                turn1 = line[1]
                turn2 = line[2]
                turn3 = line[3]

                # Remove any duplicate spaces
                duplicateSpacePattern = re.compile(r'\ +')
                conv = re.sub(duplicateSpacePattern, ' ', conv)
                context = re.sub(duplicateSpacePattern, ' ', context)
                sent = re.sub(duplicateSpacePattern, ' ', sent)
                turn1 = re.sub(duplicateSpacePattern, ' ', turn1)
                turn2 = re.sub(duplicateSpacePattern, ' ', turn2)
                turn3 = re.sub(duplicateSpacePattern, ' ', turn3)

                indices.append(int(line[0]))
                raws.append(tokenizer.tokenize(conv))
                raws_c.append(tokenizer.tokenize(context))
                raws_s.append(tokenizer.tokenize(sent))
                raws_turn1.append(tokenizer.tokenize(turn1))
                raws_turn2.append(tokenizer.tokenize(turn2))
                raws_turn3.append(tokenizer.tokenize(turn3))
                conversations.append(conv)
                contexts.append(context)
                sents.append(sent)
                turn1s.append(turn1)
                turn2s.append(turn2)
                turn3s.append(turn3)

        examples = []
        if mode == 'train':
            for i in range(len(conversations)):
                examples.append([raws[i],
                                 raws_c[i],
                                 raws_s[i],
                                 raws_turn1[i],
                                 raws_turn2[i],
                                 raws_turn3[i],
                                 conversations[i],
                                 contexts[i],
                                 sents[i],
                                 turn1s[i],
                                 turn2s[i],
                                 turn3s[i],
                                 labels[i]])
        else:
            for i in range(len(conversations)):
                examples.append([raws[i],
                                 raws_c[i],
                                 raws_s[i],
                                 raws_turn1[i],
                                 raws_turn2[i],
                                 raws_turn3[i],
                                 conversations[i],
                                 contexts[i],
                                 sents[i],
                                 turn1s[i],
                                 turn2s[i],
                                 turn3s[i]])
        return examples


    @classmethod
    def splits(cls, args, raw_field, text_field, label_field, trainDataPath, validDataPath, testDataPath, root='.', **kwargs):
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
        return (cls(args, raw_field, text_field, label_field, path=trainDataPath, mode='train', **kwargs),
                cls(args, raw_field, text_field, label_field, path=validDataPath, mode='train', **kwargs),
                cls(args, raw_field, text_field, label_field, path=testDataPath, mode='train', **kwargs))


    @classmethod
    def getTestData(cls, args, raw_field, text_field, testDataPath):
        return cls(args, raw_field, text_field, None, path=testDataPath, mode='test')
