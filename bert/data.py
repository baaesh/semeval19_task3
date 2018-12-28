import io
import re
import pickle
import torch

from torchtext import data
import datasets
from torchtext.vocab import GloVe
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert import BertTokenizer

class EMO():
    def __init__(self, args):
        self.RAW = data.RawField()
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev = datasets.EMO.splits(self.RAW, self.LABEL)

        self.LABEL.build_vocab(self.train)

        self.train_iter, self.dev_iter = \
            data.BucketIterator.splits((self.train, self.dev),
                                       batch_size=args.batch_size,
                                       device=args.device,
                                       repeat=False)

        filehandler = open('./data/label.obj', 'wb')
        pickle.dump(self.LABEL.vocab, filehandler)


class EMO_test():
    def __init__(self, args):
        self.RAW = data.RawField()
        self.LABEL = data.Field(sequential=False, unk_token=None)

        filehandler = open('./data/label.obj', 'rb')
        self.LABEL.vocab = pickle.load(filehandler)

        self.test = datasets.EMO.getTestData(self.RAW)

        self.test_iter = \
            data.Iterator(self.test,
                          batch_size=args.batch_size,
                          device=args.device,
                          shuffle=False,
                          sort=False,
                          repeat=False)


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    examples = []
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

            conv = ' <eos> '.join(line[1:4])

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            if mode == "train":
                # Train data contains id, 3 turns and label
                label = line[4]
                example = {'index': int(line[0]),
                           'conversation': conv.lower(),
                           'turn1': line[1],
                           'turn2': line[2],
                           'turn3': line[3],
                           'label': label}
            else:
                example = {'index': int(line[0]),
                           'conversation': conv.lower(),
                           'turn1': line[1],
                           'turn2': line[2],
                           'turn3': line[3]}
            examples.append(example)

    return examples


def arglongest(len1, len2, len3):
    Max = len1
    max_idx = 1
    if len2 > Max:
        Max = len2
        max_idx = 2
    if len3 > Max:
        Max = len3
        max_idx = 3
    return max_idx


def _truncate_seq_pair(turn1_tokens, turn2_tokens, turn3_tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(turn1_tokens) + len(turn2_tokens) + len(turn3_tokens)
        if total_length <= max_length:
            break
        max_idx = arglongest(turn1_tokens, turn2_tokens, turn3_tokens)
        if max_idx == 1:
            turn1_tokens.pop()
        elif max_idx == 2:
            turn2_tokens.pop()
        else:
            turn3_tokens.pop()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        turn1_tokens = tokenizer.tokenize(example['turn1'])
        turn2_tokens = tokenizer.tokenize(example['turn2'])
        turn3_tokens = tokenizer.tokenize(example['turn3'])

        _truncate_seq_pair(turn1_tokens, turn2_tokens, turn3_tokens, max_seq_length - 4)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        #
        # In this project, there are 3 turns in a conversation.
        #   tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP] Ok, got it.
        #   type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1 0 0 0
        tokens = ["[CLS]"] + turn1_tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        tokens += turn2_tokens + ["[SEP]"]
        segment_ids += [1] * (len(turn2_tokens) + 1)

        tokens += turn3_tokens + ["[SEP]"]
        segment_ids += [0] * (len(turn3_tokens) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        #print(len(segment_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example['label']]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def getDataLoaders(args):
    trainExamples = preprocessData(args.train_data_path, mode='train')
    validExamples = preprocessData(args.valid_data_path, mode='train')
    label_list = ['others', 'happy', 'sad', 'angry']
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_features = convert_examples_to_features(trainExamples,
                                                  label_list,
                                                  args.max_seq_length,
                                                  tokenizer)
    valid_features = convert_examples_to_features(validExamples,
                                                  label_list,
                                                  args.max_seq_length,
                                                  tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
    valid_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)

    return train_dataloader, valid_dataloader, len(trainExamples)





















