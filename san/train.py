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
from model import *
from test import test
from loss import *
from statistics import getStatistics


def train(args, data):
    if args.ss_emb:
        ss_vectors = torch.load(args.ss_vector_path)
    else:
        ss_vectors = None

    device = torch.device(args.device)
    if args.fusion:
        model = NN4EMO_FUSION(args, data, ss_vectors).to(device)
    elif args.ensemble:
        model = NN4EMO_ENSEMBLE(args, data, ss_vectors).to(device)
    elif args.separate:
        model = NN4EMO_SEPARATE(args, data, ss_vectors).to(device)
    else:
        model = NN4EMO_SEMI_HIERARCHICAL(args, data, ss_vectors).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=args.lr_gamma)
    if args.wce_loss:
        others_idx = data.LABEL.vocab.stoi['others']
        happy_idx = data.LABEL.vocab.stoi['happy']
        sad_idx = data.LABEL.vocab.stoi['sad']
        angry_idx = data.LABEL.vocab.stoi['angry']
        train_dist = getStatistics(args.train_data_path, 'train')
        valid_dist = getStatistics(args.valid_data_path, 'valid')
        alpha = [1.0] * args.class_size
        alpha[others_idx] = valid_dist[0] / train_dist[0]
        alpha[happy_idx] = valid_dist[1] / train_dist[1]
        alpha[sad_idx] = valid_dist[2] / train_dist[2]
        alpha[angry_idx] = valid_dist[3] / train_dist[3]

        criterion = WeightedCrossEntropyLoss(alpha=alpha,
                                             size_average=True).to(device)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.mfe_loss:
        others_idx = data.LABEL.vocab.stoi['others']
        mfe_loss = AMFELoss(args, others_idx)

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()

    acc, loss, size, last_epoch = 0, 0, 0, -1
    max_dev_acc, max_test_acc = 0, 0
    max_dev_f1, max_test_f1 = 0, 0
    best_model = None

    print("tarining start")
    iterator = data.train_iter
    for epoch in range(args.max_epoch):
        print('epoch: ', epoch + 1)
        #scheduler.step()
        for i, batch in enumerate(iterator):
            if args.char_emb:
                if args.fusion:
                    char_c = torch.LongTensor(data.characterize(batch.context[0])).to(device)
                    char_s = torch.LongTensor(data.characterize(batch.sent[0])).to(device)
                    setattr(batch, 'char_c', char_c)
                    setattr(batch, 'char_s', char_s)
                elif args.separate:
                    char_turn1 = torch.LongTensor(data.characterize(batch.turn1[0])).to(device)
                    char_turn2 = torch.LongTensor(data.characterize(batch.turn2[0])).to(device)
                    char_turn3 = torch.LongTensor(data.characterize(batch.turn3[0])).to(device)
                    setattr(batch, 'char_turn1', char_turn1)
                    setattr(batch, 'char_turn2', char_turn2)
                    setattr(batch, 'char_turn3', char_turn3)
                else:
                    char = torch.LongTensor(data.characterize(batch.text[0])).to(device)
                    setattr(batch, 'char', char)

            pred = model(batch)

            optimizer.zero_grad()

            batch_loss = criterion(pred, batch.label)
            if args.mfe_loss:
                #print(batch_loss)
                batch_loss += mfe_loss(pred, batch.label)

            loss += batch_loss.item()

            batch_loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=args.norm_limit)
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
                dev_loss, dev_acc, dev_f1 = test(model, data, data.dev_iter, criterion, args)
                test_loss, test_acc, test_f1 = test(model, data, data.test_iter, criterion, args)
                if dev_acc > max_dev_acc:
                    max_dev_acc = dev_acc
                if dev_f1 > max_dev_f1:
                    max_dev_f1 = dev_f1
                    max_test_acc = test_acc
                    max_test_f1 = test_f1
                    best_model = copy.deepcopy(model.state_dict())
                writer.add_scalar('loss/dev', dev_loss, c)
                writer.add_scalar('acc/dev', dev_acc, c)
                writer.add_scalar('f1/dev', dev_f1, c)
                writer.add_scalar('acc/test', test_acc, c)
                writer.add_scalar('f1/test', test_f1, c)
                print(f'dev loss: {dev_loss:.4f} / dev acc: {dev_acc:.4f} / dev f1: {dev_f1:.4f} / '
                      f'test acc: {test_acc:.4f} / test f1: {test_f1:.4f}')
                print(f'(max dev acc: {max_dev_acc:.4f} / max dev f1: {max_dev_f1:.4f} / '
                      f'max test acc: {max_test_acc:.4f} / max test f1: {max_test_f1:.4f})')
                model.train()

    writer.close()
    return best_model, max_test_acc, max_test_f1


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
    elif args.ensemble:
        model = NN4EMO_ENSEMBLE(args, data).to(device)
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


    vectors = []
    for i in range(len(TEXT.vocab)):
        try:
            index = stoi[TEXT.vocab.itos[i]]
        except:
            index = 0 # <unk> index
        vectors.append(embedding[index])

    vectors = torch.stack(vectors)

    torch.save(vectors, 'data/sswe/sswe.pt')


def build_datastories_vectors(data):
    vector_path = 'data/datastories/datastories.twitter.300d.txt'

    if os.path.exists(vector_path):
        print('Indexing file datastories.twitter.300d.txt ...')
        embeddings_dict = {}
        with open(vector_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                values = line.split()
                word = values[0]
                if word in data.TEXT.vocab.stoi:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_dict[word] = coefs

        print('Found %s word vectors.' % len(embeddings_dict))


def main():
    args = set_args()

    # loading EmoContext data
    print("loading data")
    data = EMO(args)
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)
    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'FILTER_SIZES', [1, 3, 5])
    print(args.char_vocab_size)

    print('Vocab Size: ' + str(len(data.TEXT.vocab)))

    build_sswe_vectors()

    best_model, max_test_acc, max_test_f1 = train(args, data)

    model_name_str = 'NN4EMO_' + args.name_tag
    
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model_name = f'{model_name_str}_{args.model_time}_{max_test_acc:.4f}_{max_test_f1:.4f}.pt'
    torch.save(best_model, 'saved_models/' + model_name)

    print('training finished!')

    submission(args, model_name)


if __name__ == '__main__':
    main()
