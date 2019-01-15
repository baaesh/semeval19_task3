import copy
import pickle
from torchtext import data
from torchtext.vocab import GloVe, FastText
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer

import datasets

class EMO():

    def __init__(self, args):
        tokenizer = TweetTokenizer()
        self.RAW = data.RawField()
        self.TEXT = data.Field(batch_first=True, include_lengths=True,
                               lower=True, tokenize=tokenizer.tokenize)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev = datasets.EMO.splits(self.RAW, self.TEXT, self.LABEL,
                                                   args.train_data_path, args.valid_data_path)

        self.TEXT.build_vocab(self.train, self.dev, vectors=GloVe(name='840B', dim=300))

        if args.fasttext:
            self.FASTTEXT = data.Field(batch_first=True, include_lengths=True,
                                       lower=True, tokenize=tokenizer.tokenize)
            self.FASTTEXT.vocab = copy.deepcopy(self.TEXT.vocab)
            self.FASTTEXT.vocab.set_vectors(self.FASTTEXT.vocab.stoi,
                                            vectors=FastText(language='en'), dim=300)
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.dev_iter = \
            data.BucketIterator.splits((self.train, self.dev),
                                       batch_size=args.batch_size,
                                       device=args.device,
                                       repeat=False)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        # for <pad>
        self.char_vocab = {'': 0}
        # for <unk> and <pad>
        self.characterized_words = [[0] * self.max_word_len, [0] * self.max_word_len]

        if args.char_emb:
            self.build_char_vocab()

        filehandler = open('./data/vocab.obj', 'wb')
        pickle.dump(self.TEXT.vocab, filehandler)
        filehandler = open('./data/label.obj', 'wb')
        pickle.dump(self.LABEL.vocab, filehandler)


    def build_char_vocab(self):
        # for normal words
        for word in self.TEXT.vocab.itos[2:]:
            chars = []
            for c in list(word):
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

                chars.append(self.char_vocab[c])

            chars.extend([0] * (self.max_word_len - len(word)))
            self.characterized_words.append(chars)


    def characterize(self, batch):
        """
    	:param batch: Pytorch Variable with shape (batch, seq_len)
    	:return: Pytorch Variable with shape (batch, seq_len, max_word_len)
    	"""
        batch = batch.data.cpu().numpy().astype(int).tolist()
        return [[self.characterized_words[w] for w in words] for words in batch]


class EMO_test():

    def __init__(self, args):
        tokenizer = TweetTokenizer()
        self.RAW = data.RawField()
        self.TEXT = data.Field(batch_first=True, include_lengths=True,
                               lower=True, tokenize=tokenizer.tokenize)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        filehandler = open('./data/vocab.obj', 'rb')
        self.TEXT.vocab = pickle.load(filehandler)
        filehandler = open('./data/label.obj', 'rb')
        self.LABEL.vocab = pickle.load(filehandler)

        self.test = datasets.EMO.getTestData(self.RAW, self.TEXT)

        self.test_iter = \
            data.Iterator(self.test,
                          batch_size=args.batch_size,
                          device=args.device,
                          shuffle=False,
                          sort=False,
                          repeat=False)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        # for <pad>
        self.char_vocab = {'': 0}
        # for <unk> and <pad>
        self.characterized_words = [[0] * self.max_word_len, [0] * self.max_word_len]

        if args.char_emb:
            self.build_char_vocab()


    def build_char_vocab(self):
        # for normal words
        for word in self.TEXT.vocab.itos[2:]:
            chars = []
            for c in list(word):
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

                chars.append(self.char_vocab[c])

            chars.extend([0] * (self.max_word_len - len(word)))
            self.characterized_words.append(chars)


    def characterize(self, batch):
        """
    	:param batch: Pytorch Variable with shape (batch, seq_len)
    	:return: Pytorch Variable with shape (batch, seq_len, max_word_len)
    	"""
        batch = batch.data.cpu().numpy().astype(int).tolist()
        return [[self.characterized_words[w] for w in words] for words in batch]