import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from allennlp.modules.elmo import Elmo, batch_to_ids


# Masked softmax
def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec - max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros = (masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps / (masked_sums + 1e-20)


# Directional mask
def get_direct_mask_tile(direction, sentence_len, device):
    mask = torch.FloatTensor(sentence_len, sentence_len).to(torch.device(device))
    mask.data.fill_(1)
    if direction == 'fw':
        mask = torch.tril(mask, diagonal=-1)
    else:
        mask = torch.triu(mask, diagonal=1)
    mask.unsqueeze_(0)
    return mask


# Representation mask for sentences of variable lengths
def get_rep_mask_tile(rep_mask):
    batch_size, sentence_len, _ = rep_mask.size()

    m1 = rep_mask.view(batch_size, sentence_len, 1)
    m2 = rep_mask.view(batch_size, 1, sentence_len)
    mask = torch.mul(m1, m2)

    return mask


# Distance mask
def get_dist_mask_tile(sentence_len, device):
    mask = torch.FloatTensor(sentence_len, sentence_len).to(torch.device(device))
    for i in range(sentence_len):
        for j in range(sentence_len):
            mask[i, j] = -abs(i-j)
    mask.unsqueeze_(0)
    return mask


class Attention(nn.Module):

    def __init__(self, d_model, direction=None, alpha=1.0, dist_mask=False, device='cuda:0'):
        super(Attention, self).__init__()

        self.direction = direction
        self.device = device
        self.alpha = alpha
        self.dist_mask = dist_mask

        self.scaling_factor = Variable(torch.Tensor([math.pow(d_model, 0.5)]),
                                       requires_grad=False).to(device)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, q, k, v, rep_mask):
        batch_size, seq_len, d_model = q.size()
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scaling_factor

        rep_mask_tile = get_rep_mask_tile(rep_mask)
        mask = rep_mask_tile
        if self.direction is not None:
            direct_mask_tile = get_direct_mask_tile(self.direction, seq_len, self.device)
            mask = rep_mask_tile * direct_mask_tile
        if self.dist_mask:
            dist_mask_tile = get_dist_mask_tile(seq_len, self.device)
            attn += self.alpha * dist_mask_tile

        attn = masked_softmax(attn, mask, dim=2)
        out = torch.bmm(attn, v)

        return out, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, args, direction):
        super(MultiHeadAttention, self).__init__()

        self.n_head = args.num_heads
        self.d_k = int(args.d_e / args.num_heads)
        self.d_v = int(args.d_e / args.num_heads)
        self.d_model = args.d_e

        self.w_qs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_v))

        self.attention = Attention(self.d_model, direction=direction, alpha=args.alpha,
                                   dist_mask=args.dist_mask, device=args.device)
        self.layer_norm = nn.LayerNorm(int(self.d_k))
        self.layer_norm2 = nn.LayerNorm(self.d_model)

        self.proj = nn.Linear(self.n_head * self.d_v, self.d_model)

        self.dropout = nn.Dropout(args.dropout)

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)


    def forward(self, q, k, v, rep_mask):
        n_head = self.n_head

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)

        q_s = self.layer_norm(torch.bmm(q_s, self.w_qs).view(-1, len_q, self.d_k))
        k_s = self.layer_norm(torch.bmm(k_s, self.w_ks).view(-1, len_k, self.d_k))
        v_s = self.layer_norm(torch.bmm(v_s, self.w_vs).view(-1, len_v, self.d_v))

        rep_mask = rep_mask.repeat(n_head, 1, 1).view(-1, len_q, 1)
        outs, attns = self.attention(q_s, k_s, v_s, rep_mask)

        outs = torch.cat(torch.split(outs, mb_size, dim=0), dim=-1)

        outs = self.layer_norm2(self.proj(outs))
        outs = self.dropout(outs)

        return outs


class FusionGate(nn.Module):

    def __init__(self, d_e, dropout=0.1):
        super(FusionGate, self).__init__()

        self.w_s = nn.Parameter(torch.FloatTensor(d_e, d_e))
        self.w_h = nn.Parameter(torch.FloatTensor(d_e, d_e))
        self.b = nn.Parameter(torch.FloatTensor(d_e))

        init.xavier_normal_(self.w_s)
        init.xavier_normal_(self.w_h)
        init.constant_(self.b, 0)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_e)


    def forward(self, s, h):
        s_f = self.layer_norm(torch.matmul(s, self.w_s))
        h_f = self.layer_norm(torch.matmul(h, self.w_h))

        f = self.sigmoid(self.dropout(s_f + h_f + self.b))

        outs = f * s_f + (1 - f) * h_f

        return self.layer_norm(outs)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_h, d_in_h, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_h, d_in_h, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_in_h, d_h, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_h)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.relu(self.w_1(x.transpose(1, 2)))
        out = self.w_2(out).transpose(2, 1)
        out = self.dropout(out)
        return self.layer_norm(out + x)


class LayerBlock(nn.Module):

    def __init__(self, args, direction):
        super(LayerBlock, self).__init__()

        self.attn_layer = MultiHeadAttention(args, direction)
        self.fusion_gate = FusionGate(args.d_e, args.dropout)
        self.feed_forward = PositionwiseFeedForward(args.d_e, args.d_ff, dropout=args.dropout)


    def forward(self, x, rep_mask):
        outs = self.attn_layer(x, x, x, rep_mask)
        outs = self.fusion_gate(x, outs)
        outs = self.feed_forward(outs)

        return outs


class Source2Token(nn.Module):

    def __init__(self, d_h, dropout=0.1):
        super(Source2Token, self).__init__()

        self.d_h = d_h
        self.dropout_rate = dropout

        self.fc1 = nn.Linear(d_h, d_h)
        self.fc2 = nn.Linear(d_h, d_h)

        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(d_h)


    def forward(self, x, rep_mask):
        out = self.elu(self.layer_norm(self.fc1(x)))
        out = self.layer_norm(self.fc2(out))

        out = masked_softmax(out, rep_mask, dim=1)
        out = torch.sum(torch.mul(x, out), dim=1)

        return out


class PositionwiseFeedForwardReduce(nn.Module):

    def __init__(self, d_i, d_h, dropout=0):
        super(PositionwiseFeedForwardReduce, self).__init__()
        self.w_0 = nn.Conv1d(d_i, d_h, 1)
        self.w_1 = nn.Conv1d(d_h, d_h, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_0(x.transpose(1, 2)))
        output = self.dropout(output)
        output = self.w_1(output)
        output = self.dropout(output).transpose(2, 1)
        return output


class ELMo(nn.Module):

    def __init__(self, args):
        super(ELMo, self).__init__()
        options_file = args.elmo_option_path
        weight_file = args.elmo_weight_path

        self.device = torch.device(args.device)
        self.num_emb = args.elmo_num
        self.feed_forward = args.elmo_feed_forward
        self.dropout = nn.Dropout(args.dropout)

        self.elmo = Elmo(options_file, weight_file, self.num_emb, dropout=0)

        if self.feed_forward:
            for i in range(self.num_emb):
                pwff = PositionwiseFeedForwardReduce(
                    args.elmo_dim, args.d_e, dropout=args.dropout)
                setattr(self, 'pwff_' + str(i), pwff)

    def forward(self, x_plain):
        x = batch_to_ids(x_plain).to(self.device)

        elmo_embs = self.elmo(x)['elmo_representations']

        embs = []
        if self.feed_forward:
            for i in range(self.num_emb):
                emb = elmo_embs[i]
                pwff = getattr(self, 'pwff_' + str(i))
                emb = pwff(self.dropout(emb))
                embs.append(emb)
        else:
            embs = elmo_embs
        return embs


class LSTMEncoder(nn.Module):

    def __init__(self, args, input_dim=None, last_hidden=False):
        super(LSTMEncoder, self).__init__()

        self.args = args
        self.emb_dim = input_dim if input_dim is not None else args.word_dim
        self.last_hidden=last_hidden

        for i in range(args.lstm_num_layers):
            if i == 0:
                lstm_input_dim = self.emb_dim
            else:
                lstm_input_dim = self.emb_dim + 2 * args.lstm_hidden_dim
            lstm_layer = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=args.lstm_hidden_dim,
                bidirectional=args.lstm_bidirection,
                batch_first=True
            )
            setattr(self, f'lstm_layer_{i}', lstm_layer)


    def get_lstm_layer(self, i):
        return getattr(self, f'lstm_layer_{i}')


    def forward(self, x, lengths):
        lens, indices = torch.sort(lengths, 0, True)

        x_sorted = x[indices]

        for i in range(self.args.lstm_num_layers):
            if i == 0:
                lstm_in = pack(x_sorted, lens.tolist(), batch_first=True)
            else:
                lstm_in = pack(torch.cat([x_sorted, lstm_out], dim=-1), lens.tolist(), batch_first=True)
            lstm_layer = self.get_lstm_layer(i)
            lstm_out, hid = lstm_layer(lstm_in)
            lstm_out = unpack(lstm_out, batch_first=True)[0]

        _, _indices = torch.sort(indices, 0)

        if self.last_hidden:
            hid = hid[0]
            hid.transpose_(0, 1)
            hid = hid[_indices]
            out = hid.view(hid.size()[0], -1)
        else:
            out = lstm_out[_indices]

        return out


class CharCNN(nn.Module):

    def __init__(self, args):
        super(CharCNN, self).__init__()

        self.args = args
        self.FILTER_SIZES = args.FILTER_SIZES

        for filter_size in args.FILTER_SIZES:
            conv = nn.Conv1d(1, args.num_feature_maps, args.char_dim * filter_size, stride=args.char_dim)
            setattr(self, 'conv_' + str(filter_size), conv)


    def forward(self, x):
        batch_seq_len, max_word_len, char_dim = x.size()

        # (batch * seq_len, 1, max_word_len * char_dim)
        x = x.view(batch_seq_len, 1, -1)

        conv_result = [
            F.max_pool1d(F.relu(getattr(self, 'conv_' + str(filter_size))(x)), max_word_len - filter_size + 1).view(-1,
                                                                                                                    self.args.num_feature_maps)
            for filter_size in self.FILTER_SIZES]

        out = torch.cat(conv_result, 1)

        return out


class BiAttention(nn.Module):

    def __init__(self, args):
        super(BiAttention, self).__init__()
        self.args = args

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        # x: (batch size) x (sequence length x) x (dim)
        # y: (batch size) x (sequence length y) x (dim)
        # xy: (batch size) x (sequence length x) x (sequence length y)
        xy = torch.bmm(x, y.transpose(1, 2))

        # a_x: (batch size) x (sequence length x) x (sequence length y)
        # a_y: (batch size) x (sequence length y) x (sequence length x)
        a_x = self.softmax(xy)
        a_y = self.softmax(xy.transpose(1, 2))

        # c_x: (batch size) x (sequence length y) x (dim)
        # c_y: (batch_size) x (sequence length x) x (dim)
        c_x = torch.bmm(a_x.transpose(1, 2), x)
        c_y = torch.bmm(a_y.transpose(1, 2), y)

        # x_out: (batch size) x (sequence length x) x (3 * dim)
        # y_out: (batch size) x (sequence length y) x (3 * dim)
        x_out = torch.cat([x, x - c_y, x * c_y], dim=-1)
        y_out = torch.cat([y, y - c_x, y * c_x], dim=-1)

        return x_out, y_out


class BiAttentionEncoder(nn.Module):

    def __init__(self, args, data):
        super(BiAttentionEncoder, self).__init__()
        self.args = args

        self.lstm1 = LSTMEncoder(args, input_dim = args.word_dim * 2)
        self.lstm2 = LSTMEncoder(args, input_dim = args.word_dim * 2)

        self.c_integrate = LSTMEncoder(args, input_dim = args.word_dim * 2 * 3, last_hidden=True)
        self.s_integrate = LSTMEncoder(args, input_dim=args.word_dim * 2 * 3, last_hidden=True)

        self.elmo = ELMo(args)

        # character embedding
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=0)
        self.charCNN = CharCNN(args)

        self.biattention = BiAttention(args)

        self.s2t_SA = Source2Token(d_h=2 * args.d_e, dropout=args.dropout)

    def forward(self, c_in1, c_in2, s_in1, s_in2,
                c_in_char, s_in_char, c_in_raw, s_in_raw,
                c_rep_mask, s_rep_mask, c_lengths, s_lengths):

        c_elmo = self.elmo(c_in_raw)[0]
        s_elmo = self.elmo(s_in_raw)[0]
        c_in1 = torch.cat([c_in1, c_elmo], dim=-1)
        s_in1 = torch.cat([s_in1, s_elmo], dim=-1)

        # character embedding
        if self.args.char_emb:
            # (batch, seq_len, max_word_len)
            c_char = c_in_char
            s_char = s_in_char
            batch_size, c_seq_len, _ = c_char.size()
            batch_size, s_seq_len, _ = s_char.size()

            # (batch * seq_len, max_word_len)
            c_char = c_char.view(-1, self.args.max_word_len)
            s_char = s_char.view(-1, self.args.max_word_len)

            # (batch * seq_len, max_word_len, char_dim)
            c_char = self.char_emb(c_char)
            s_char = self.char_emb(s_char)

            # (batch, seq_len, len(FILTER_SIZES) * num_feature_maps)
            c_char = self.charCNN(c_char).view(batch_size, c_seq_len, -1)
            s_char = self.charCNN(s_char).view(batch_size, s_seq_len, -1)

            c_in2 = torch.cat([c_in2, c_char], dim=-1)
            s_in2 = torch.cat([s_in2, s_char], dim=-1)

        c_u_1 = self.lstm1(c_in1, c_lengths)
        c_u_2 = self.lstm2(c_in2, c_lengths)
        s_u_1 = self.lstm1(s_in1, s_lengths)
        s_u_2 = self.lstm2(s_in2, s_lengths)

        c_u = torch.cat([c_u_1, c_u_2], dim=-1)
        s_u = torch.cat([s_u_1, s_u_2], dim=-1)

        c_u, s_u = self.biattention(c_u, s_u)

        c_out = self.c_integrate(c_u, c_lengths)
        s_out = self.s_integrate(s_u, s_lengths)

        return c_out, s_out


class SentenceEncoder(nn.Module):

    def __init__(self, args, data):
        super(SentenceEncoder, self).__init__()
        self.args = args
        self.device = args.device
        # forward and backward transformer block
        #self.glove_block = LayerBlock(args, direction=None)
        #self.ss_block = LayerBlock(args, direction=None)

        self.glove_lstm = LSTMEncoder(args, input_dim=args.word_dim * 2)
        self.ss_lstm = LSTMEncoder(args, input_dim=args.word_dim * 2)

        self.elmo = ELMo(args)

        # character embedding
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=0)
        self.charCNN = CharCNN(args)

        if args.seg_emb:
            self.seg_emb_g = nn.Embedding(3, args.word_dim * 2)
            self.seg_emb_s = nn.Embedding(3, args.word_dim)
            self.seg_emb_e = nn.Embedding(3, args.word_dim)
            self.eos_idx = data.TEXT.vocab.stoi['<eos>']

        # Multi-dimensional source2token self-attention
        self.s2t_SA = Source2Token(d_h=2 * args.d_e, dropout=args.dropout)
        # vector-based multi-head attention
        #for i in range(args.num_heads):
        #    s2t = Source2Token(d_h=2 * args.d_e, dropout=args.dropout)
        #    setattr(self, f's2tSA_{i}', s2t)


    def get_s2tSA(self, i):
        return getattr(self, f's2tSA_{i}')


    def seg_seq(self, seq):
        seg_idx_batch = []
        for i in range(len(seq)):
            idx = 0
            seg_idx = []
            for j in range(len(seq[i])):
                seg_idx.append(idx)
                if (seq[i][j] == self.eos_idx):
                    if self.args.seg_emb_share:
                        idx = 1 - idx
                    else:
                        idx = idx + 1
            seg_idx_batch.append(torch.tensor(seg_idx))
        seg_idx_batch = torch.stack(seg_idx_batch).to(self.device)
        return seg_idx_batch


    def forward(self, inputs, inputs_ss, inputs_char, batch_raw, rep_mask, lengths, seq):
        batch, seq_len, d_e = inputs.size()

        elmo_emb_w, elmo_emb = self.elmo(batch_raw)
        inputs = torch.cat([inputs, elmo_emb_w], dim=-1)

        # character embedding
        if self.args.char_emb:
            # (batch, seq_len, max_word_len)
            char = inputs_char
            batch_size, seq_len, _ = char.size()

            # (batch * seq_len, max_word_len)
            char = char.view(-1, self.args.max_word_len)

            # (batch * seq_len, max_word_len, char_dim)
            char = self.char_emb(char)

            # (batch, seq_len, len(FILTER_SIZES) * num_feature_maps)
            char = self.charCNN(char).view(batch_size, seq_len, -1)

            inputs_ss = torch.cat([inputs_ss, char], dim=-1)

        if self.args.seg_emb:
            seg_idx = self.seg_seq(seq)
            inputs = inputs + self.seg_emb_g(seg_idx)
            inputs_ss = inputs_ss + self.seg_emb_s(seg_idx)

        #u_g = self.glove_block(inputs, rep_mask)
        #u_s = self.ss_block(inputs_ss, rep_mask)
        u_g = self.glove_lstm(inputs, lengths)
        u_s = self.ss_lstm(inputs_ss, lengths)
        u_e = elmo_emb

        u = torch.cat([u_g, u_s], dim=-1)

        pooling = nn.MaxPool2d((seq_len, 1), stride=1)
        pool_s = pooling(u * rep_mask).view(batch, -1)
        s2t_s = self.s2t_SA(u, rep_mask)

        #s2t_s = []
        #for i in range(self.args.num_heads):
        #    s2tSA = self.get_s2tSA(i)
        #    s2t = s2tSA(u, rep_mask)
        #    s2t_s.append(s2t)

        outs = torch.cat([s2t_s, pool_s], dim=-1)
        #outs = torch.cat(s2t_s, dim=-1)

        return outs


class SimpleEncoder(nn.Module):

    def __init__(self, args, data):
        super(SimpleEncoder, self).__init__()
        self.args = args
        self.device = args.device

        self.glove_lstm = LSTMEncoder(args, input_dim=args.word_dim * 2, last_hidden=True)
        self.ss_lstm = LSTMEncoder(args, input_dim=args.word_dim * 2, last_hidden=True)

        self.elmo = ELMo(args)

        # character embedding
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=0)
        self.charCNN = CharCNN(args)


    def forward(self, inputs, inputs_ss, inputs_char, batch_raw, rep_mask, lengths, seq):

        elmo_emb = self.elmo(batch_raw)[0]
        inputs = torch.cat([inputs, elmo_emb], dim=-1)

        # character embedding
        if self.args.char_emb:
            # (batch, seq_len, max_word_len)
            char = inputs_char
            batch_size, seq_len, _ = char.size()

            # (batch * seq_len, max_word_len)
            char = char.view(-1, self.args.max_word_len)

            # (batch * seq_len, max_word_len, char_dim)
            char = self.char_emb(char)

            # (batch, seq_len, len(FILTER_SIZES) * num_feature_maps)
            char = self.charCNN(char).view(batch_size, seq_len, -1)

            inputs_ss = torch.cat([inputs_ss, char], dim=-1)

        h_g = self.glove_lstm(inputs, lengths)
        h_s = self.ss_lstm(inputs_ss, lengths)

        outs = torch.cat([h_g, h_s], dim=-1)

        return outs