import argparse
import torch


def model_config(parser):
    parser.add_argument('--bert_model', default='bert-base-uncased')

    return parser


def data_config(parser):
    parser.add_argument('--train_data_path', default='./data/train_split.txt')
    parser.add_argument('--valid_data_path', default='./data/valid_split.txt')
    parser.add_argument('--do_lower_case', default=False, action='store_true')
    parser.add_argument('--max_seq_length', type=int, default=128)

    return parser


def train_config(parser):
    parser.add_argument('--device',
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--validate_every', type=int, default=100)
    parser.add_argument('--no_fl_loss', dest='fl_loss', action='store_false')
    parser.add_argument('--fl_gamma', type=float, default=0.0)
    parser.add_argument('--fl_alpha', type=float, default=0.7)

    return parser


def set_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    return args
