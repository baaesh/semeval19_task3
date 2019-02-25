import os
import io

import torch
import numpy as np

from data import EMO_test
from model import *


def predict(model, args, data):
    iterator = data.test_iter
    model.eval()
    preds = []

    if args.thresholding:
        others_idx = data.LABEL.vocab.stoi['others']
        happy_idx = data.LABEL.vocab.stoi['happy']
        sad_idx = data.LABEL.vocab.stoi['sad']
        angry_idx = data.LABEL.vocab.stoi['angry']
        train_dist = getStatistics(args.train_data_path, 'train')
        valid_dist = getStatistics(args.valid_data_path, 'valid')
        reverse_prior = [1.0] * args.class_size
        reverse_prior[others_idx] = valid_dist[0] / train_dist[0]
        reverse_prior[happy_idx] = valid_dist[1] / train_dist[1]
        reverse_prior[sad_idx] = valid_dist[2] / train_dist[2]
        reverse_prior[angry_idx] = train_dist[3] / train_dist[3]
        reverse_prior = torch.tensor(reverse_prior)

    for batch in iter(iterator):
        if args.char_emb:
            if args.fusion:
                char_c = torch.LongTensor(data.characterize(batch.context[0])).to(args.device)
                char_s = torch.LongTensor(data.characterize(batch.sent[0])).to(args.device)
                setattr(batch, 'char_c', char_c)
                setattr(batch, 'char_s', char_s)
            elif args.separate:
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

        pred = torch.softmax(pred.detach(), dim=1)
        if args.thresholding:
            if reverse_prior.type() != pred.data.type():
                reverse_prior = reverse_prior.type_as(pred.data).to(pred.get_device())
            at = reverse_prior.repeat(pred.size()[0], 1)
            pred = pred * at

        preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds)

    return preds


def submission(args, model_name):
    data = EMO_test(args)

    device = torch.device(args.device)
    if args.fusion:
        model = NN4EMO_FUSION(args, data).to(device)
    elif args.separate:
        model = NN4EMO_SEPARATE(args, data).to(device)
    else:
        model = NN4EMO_SEMI_HIERARCHICAL(args, data).to(device)

    model.load_state_dict(torch.load('./saved_models/' + model_name))

    preds = predict(model, args, data)

    maxs = preds.max(axis=1)
    print(maxs)
    preds = preds.argmax(axis=1)

    if not os.path.exists('experiments'):
        os.makedirs('experiments')

    solutionPath = './experiments/' + model_name + '.txt'
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
