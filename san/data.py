import pickle
from torchtext import data
import datasets
from torchtext.vocab import GloVe
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer


class EMO():
    def __init__(self, args):
        tokenizer = Tokenizer(num_words=20000)
        self.RAW = data.RawField(is_target=False)
        self.TEXT = data.Field(batch_first=True, include_lengths=True, lower=False)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev = datasets.EMO.splits(self.RAW, self.TEXT, self.LABEL)

        self.TEXT.build_vocab(self.train, self.dev, vectors=GloVe(name='840B', dim=300))
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.dev_iter = \
            data.BucketIterator.splits((self.train, self.dev),
                                       batch_size=args.batch_size,
                                       device=args.device)

        filehandler = open('./data/vocab.obj', 'wb')
        pickle.dump(self.TEXT.vocab, filehandler)
        filehandler = open('./data/label.obj', 'wb')
        pickle.dump(self.LABEL.vocab, filehandler)


class EMO_test():
    def __init__(self, args):
        self.TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        filehandler = open('./data/vocab.obj', 'rb')
        self.TEXT.vocab = pickle.load(filehandler)
        filehandler = open('./data/label.obj', 'rb')
        self.LABEL.vocab = pickle.load(filehandler)

        self.test = datasets.EMO.getTestData(self.TEXT)

        #for i in range(100):
        #    print(self.test.examples[i].text)
        self.test_iter = \
            data.Iterator(self.test,
                          batch_size=args.batch_size,
                          device=args.device,
                          shuffle=False,
                          sort=False)