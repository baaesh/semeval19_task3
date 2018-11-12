import argparse
import torch


def model_config(parser):
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--d_e', default=300, type=int)
    parser.add_argument('--num_heads', default=5, type=int)
    parser.add_argument('--d_ff', default=300 * 4, type=int)

    parser.add_argument('--dist_mask', default=False, action='store_true')
    parser.add_argument('--alpha', default=1.5, type=float)

    return parser


def data_config(parser):
    parser.add_argument('--class_size', default=4, type=int)
    return parser


def train_config(parser):
    parser.add_argument('--device',
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tune_embeddings', default=False, action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--validate_every', type=int, default=100)

    return parser


def set_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    return args
