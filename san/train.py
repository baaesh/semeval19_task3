import os
import copy
import torch
from time import gmtime, strftime

from torch import nn, optim
from tensorboardX import SummaryWriter

from config import set_args
from data import EMO
from model import NN4EMO
from test import test
from loss import FocalLoss


def train(args, data):
    device = torch.device(args.device)
    model = NN4EMO(args, data).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    if args.fl_loss:
        others_idx = data.LABEL.vocab.stoi['others']
        alpha = [0.04] * args.class_size
        alpha[others_idx] = 0.88
        criterion = FocalLoss(gamma=args.fl_gamma,
                               alpha=alpha, size_average=True)
    else:
        criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()

    acc, loss, size, last_epoch = 0, 0, 0, -1
    fl_loss = 0
    max_dev_acc = 0
    max_dev_f1 = 0
    best_model = None

    print("tarining start")
    iterator = data.train_iter
    for epoch in range(args.max_epoch):
        print('epoch: ', epoch + 1)
        for i, batch in enumerate(iterator):
            pred = model(batch.text)

            optimizer.zero_grad()

            if args.fl_loss:
                batch_loss = criterion(pred, batch.label)
                loss += batch_loss.item()
            else:
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
                print(f'train loss: {loss:.3f} / train acc: {acc:.3f}')
                acc, loss, size = 0, 0, 0

            if (i + 1) % args.validate_every == 0:
                c = (i + 1) // args.validate_every
                dev_loss, dev_acc, dev_f1 = test(model, data, criterion, args)
                if dev_acc > max_dev_acc:
                    max_dev_acc = dev_acc
                if dev_f1 > max_dev_f1:
                    max_dev_f1 = dev_f1
                    #best_model = copy.deepcopy(model)
                writer.add_scalar('loss/dev', dev_loss, c)
                writer.add_scalar('acc/dev', dev_acc, c)
                writer.add_scalar('f1/dev', dev_f1, c)
                print(f'dev loss: {dev_loss:.4f} / dev acc: {dev_acc:.4f} / dev f1: {dev_f1:.4f} '
                      f'(max dev acc: {max_dev_acc:.4f} / max dev f1: {max_dev_f1:.4f})')
                model.train()

    writer.close()
    return best_model, max_dev_f1


def main():
    args = set_args()

    # loading EmoContext data
    print("loading data")
    data = EMO(args)
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    #print(args.class_size)

    best_model, max_dev_f1 = train(args, data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(), f'saved_models/SAN4EMO_{args.model_time}_{max_dev_f1}.pt')

    print('training finished!')


if __name__ == '__main__':
    main()