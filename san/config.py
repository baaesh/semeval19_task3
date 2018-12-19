import argparse
import torch


def model_config(parser):
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--d_e', default=300, type=int)
    parser.add_argument('--num_heads', default=5, type=int)
    parser.add_argument('--d_ff', default=300 * 4, type=int)

    parser.add_argument('--dist_mask', default=False, action='store_true')
    parser.add_argument('--alpha', default=1.5, type=float)
    parser.add_argument('--seg_emb', default=False, action='store_true')

    parser.add_argument('--elmo_num', type=int, default=1)
    parser.add_argument('--no_elmo_feed_forward', dest='elmo_feed_forward', action='store_false')
    parser.add_argument('--elmo_dim', type=int, default=1024)

    return parser


def data_config(parser):
    #parser.add_argument('--class_size', default=4, type=int)
    parser.add_argument('--elmo_option_path', default='data/elmo/elmo_options.json')
    parser.add_argument('--elmo_weight_path', default='data/elmo/elmo_weights.hdf5')

    return parser


def train_config(parser):
    parser.add_argument('--device',
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tune_embeddings', default=False, action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_gamma', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--validate_every', type=int, default=100)
    parser.add_argument('--fl_loss', default=False, action='store_true')
    parser.add_argument('--fl_gamma', type=float, default=2.0)
    parser.add_argument('--fl_alpha', type=float, default=0.5)

    return parser


def set_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    return args
