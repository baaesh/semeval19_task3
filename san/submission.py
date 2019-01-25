import os
import io
from time import gmtime, strftime

import torch
from torch import nn
import numpy as np

from data import EMO_test
from config import set_args
from model import NN4EMO, NN4EMO_FUSION, NN4EMO_ENSEMBLE, NN4EMO_SEPERATE


def predict(model, args, data):
    iterator = data.test_iter
    model.eval()
    preds = []
    softmax = nn.Softmax(dim=1)
    for batch in iter(iterator):
        if args.char_emb:
            if args.fusion:
                char_c = torch.LongTensor(data.characterize(batch.context[0])).to(args.device)
                char_s = torch.LongTensor(data.characterize(batch.sent[0])).to(args.device)
                setattr(batch, 'char_c', char_c)
                setattr(batch, 'char_s', char_s)
            elif args.seperate:
                char_turn1 = torch.LongTensor(data.characterize(batch.turn1[0])).to(args.device)
                char_turn2 = torch.LongTensor(data.characterize(batch.turn2[0])).to(args.device)
                char_turn3 = torch.LongTensor(data.characterize(batch.turn3[0])).to(args.device)
                setattr(batch, 'char_turn1', char_turn1)
                setattr(batch, 'char_turn2', char_turn2)
                setattr(batch, 'char_turn3', char_turn3)
            else:
                char = torch.LongTensor(data.characterize(batch.text[0])).to(args.device)
                setattr(batch, 'char', char)
        pred = model(batch)
        pred = softmax(pred)
        preds.append(pred.detach().cpu().numpy())
    preds = np.concatenate(preds)

    return preds


def submission():
    model_name = 'SAN4EMO_07:39:55_0.7630.pt'
    args = set_args()

    # loading EmoContext data
    print("loading data")
    data = EMO_test(args)
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)
    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'FILTER_SIZES', [1, 3, 5])
    print(args.char_vocab_size)

    device = torch.device(args.device)
    if args.fusion:
        model = NN4EMO_FUSION(args, data).to(device)
    elif args.ensemble:
        model = NN4EMO_ENSEMBLE(args, data).to(device)
    elif args.seperate:
        model = NN4EMO_SEPERATE(args, data).to(device)
    else:
        model = NN4EMO(args, data).to(device)

    model.load_state_dict(torch.load('./saved_models/' + model_name))

    preds = predict(model, args, data)

    maxs = preds.max(axis=1)
    print(maxs)
    preds = preds.argmax(axis=1)

    if not os.path.exists('final_submissions'):
        os.makedirs('final_submissions')

    solutionPath = './final_submissions/' + model_name + '.txt'
    testDataPath = './data/raw/testwithoutlabels.txt'
    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(data.LABEL.vocab.itos[preds[lineNum]] + '\n')


if __name__ == '__main__':
    submission()
