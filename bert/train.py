import os
import copy
import torch
from time import gmtime, strftime

from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from pytorch_pretrained_bert import BertTokenizer,\
    BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
    
from config import set_args
from data import getDataLoaders
from test import test
from loss import FocalLoss


def train(args, train_dataloader, valid_dataloader, num_train_examples):
    device = torch.device(args.device)
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
              #cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1),
              num_labels=args.class_size).to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_steps = int(
        num_train_examples / args.batch_size / 1 * args.max_epoch)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    if args.fl_loss:
        others_idx = 0
        alpha = [(1.-args.fl_alpha)/3.] * args.class_size
        alpha[others_idx] = args.fl_alpha
        criterion = FocalLoss(gamma=args.fl_gamma,
                               alpha=alpha, size_average=True)
    else:
        criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()

    acc, loss, size, last_epoch = 0, 0, 0, -1
    max_dev_acc = 0
    max_dev_f1 = 0
    best_model = None

    print("tarining start")
    for epoch in range(args.max_epoch):
        print('epoch: ', epoch + 1)
        for i, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            pred = model(input_ids, segment_ids, input_mask)

            optimizer.zero_grad()

            batch_loss = criterion(pred, label_ids)
            loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

            _, pred = pred.max(dim=1)
            acc += (pred == label_ids).sum().float().cpu().item()
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
                dev_loss, dev_acc, dev_f1 = test(model, valid_dataloader, criterion, args, device)
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


def main():
    args = set_args()
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    setattr(args, 'class_size', 4)

    # loading EmoContext data
    print("loading data")
    train_dataloader, valid_dataloader, num_train_examples = getDataLoaders(args)

    best_model, max_dev_f1 = train(args, train_dataloader, valid_dataloader, num_train_examples)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model, f'saved_models/BERT_{args.model_time}_{max_dev_f1}.pt')

    print('training finished!')


if __name__ == '__main__':
    main()