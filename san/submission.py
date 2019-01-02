import os
import io

import torch
from torch import nn
import numpy as np

from data import EMO_test
from config import set_args
from model import NN4EMO


def predict(model, data):
    iterator = data.test_iter
    model.eval()
    preds = []
    softmax = nn.Softmax(dim=1)
    for batch in iter(iterator):
        pred = model(batch.text, batch.raw)
        pred = softmax(pred)
        preds.append(pred.detach().cpu().numpy())
    preds = np.concatenate(preds)

    return preds


def main():
    args = set_args()
    data = EMO_test(args)

    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', 4)

    model_name = 'SAN4EMO_08:40:18_0.7735.pt'

    device = torch.device(args.device)
    model = NN4EMO(args, data).to(device)
    model.load_state_dict(torch.load('./saved_models/' + model_name))

    preds = predict(model, data)

    maxs = preds.max(axis=1)
    print(maxs)
    preds = preds.argmax(axis=1)

    if not os.path.exists('submissions'):
        os.makedirs('submissions')

    solutionPath = './submissions/' + model_name + '.txt'
    testDataPath = './data/raw/devwithoutlabels.txt'
    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(data.LABEL.vocab.itos[preds[lineNum]] + '\n')


if __name__ == '__main__':
    main()
