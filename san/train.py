import io
import os
import copy
import pickle
import numpy as np
import torch
from time import gmtime, strftime

from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from torchtext import data

from config import set_args
from data import EMO, EMO_test
from model import NN4EMO
from test import test
from loss import FocalLoss


def train(args, data):
    if args.ss_emb:
        ss_vectors = torch.load(args.ss_vector_path)
    else:
        ss_vectors = None

    device = torch.device(args.device)
    model = NN4EMO(args, data, ss_vectors).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=args.lr_gamma)
    if args.fl_loss:
        others_idx = data.LABEL.vocab.stoi['others']
        alpha = [(1.-args.fl_alpha)/3.] * args.class_size
        alpha[others_idx] = args.fl_alpha
        criterion = FocalLoss(gamma=args.fl_gamma,
                               alpha=alpha, size_average=True).to(device)
    else:
        criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()

    acc, loss, size, last_epoch = 0, 0, 0, -1
    max_dev_acc = 0
    max_dev_f1 = 0
    best_model = None

    print("tarining start")
    iterator = data.train_iter
    for epoch in range(args.max_epoch):
        print('epoch: ', epoch + 1)
        scheduler.step()
        for i, batch in enumerate(iterator):
            pred = model(batch.text, batch.raw)

            optimizer.zero_grad()

            batch_loss = criterion(pred, batch.label)
            loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

            _, pred = pred.max(dim=1)
            acc += (pred == batch.label).sum().float().cpu().item()
            size += len(pred)

            if (i + 1) % args.print_every == 0:
                acc = acc / size
                c = (i + 1) // args.print_every
                writer.add_scalar('loss/train', loss, c)
                writer.add_scalar('acc/train', acc, c)
                print(f'{i+1} steps - train loss: {loss:.3f} / train acc: {acc:.3f}')
                acc, loss, size = 0, 0, 0

            if (i + 1) % args.validate_every == 0:
                c = (i + 1) // args.validate_every
                dev_loss, dev_acc, dev_f1 = test(model, data, criterion, args)
                if dev_acc > max_dev_acc:
                    max_dev_acc = dev_acc
                if dev_f1 > max_dev_f1:
                    max_dev_f1 = dev_f1
                    best_model = copy.deepcopy(model.state_dict())
                writer.add_scalar('loss/dev', dev_loss, c)
                writer.add_scalar('acc/dev', dev_acc, c)
                writer.add_scalar('f1/dev', dev_f1, c)
                print(f'dev loss: {dev_loss:.4f} / dev acc: {dev_acc:.4f} / dev f1: {dev_f1:.4f} '
                      f'(max dev acc: {max_dev_acc:.4f} / max dev f1: {max_dev_f1:.4f})')
                model.train()

    writer.close()
    return best_model, max_dev_f1


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


def submission(model_name):
    args = set_args()
    data = EMO_test(args)

    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', 4)

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


def build_sswe_vectors():
    vector_path = 'data/sswe/Twitter_07:09:28.pt'
    vocab_path = 'data/sswe/Twitter_vocab.txt'
    TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)

    filehandler = open('data/vocab.obj', 'rb')
    TEXT.vocab = pickle.load(filehandler)

    embedding = torch.load(vector_path)

    itos = []
    stoi = {}
    idx = 0
    with open(vocab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            itos.append(line)
            stoi[line] = idx
            idx += 1

    print('Vocab Size: ' + str(len(TEXT.vocab)))
    vectors = []
    for i in range(len(TEXT.vocab)):
        try:
            index = stoi[TEXT.vocab.itos[i]]
        except:
            index = 0 # <unk> index
        vectors.append(embedding[index])

    vectors = torch.stack(vectors)

    torch.save(vectors, 'data/sswe/sswe.pt')


def main():
    args = set_args()

    # loading EmoContext data
    print("loading data")
    data = EMO(args)
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    setattr(args, 'class_size', len(data.LABEL.vocab))

    build_sswe_vectors()

    best_model, max_dev_f1 = train(args, data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model_name = f'SAN4EMO_{args.model_time}_{max_dev_f1:.4f}.pt'
    torch.save(best_model, 'saved_models/' + model_name)

    print('training finished!')

    submission(model_name)


if __name__ == '__main__':
    main()