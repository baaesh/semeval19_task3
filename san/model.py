import pickle
import torch
import torch.nn as nn
import emoji
import gensim.models as gsm

from module import *


def get_rep_mask(lengths, device):
    batch_size = len(lengths)
    seq_len = torch.max(lengths)
    rep_mask = torch.FloatTensor(batch_size, seq_len).to(torch.device(device))
    rep_mask.data.fill_(1)
    for i in range(batch_size):
        rep_mask[i, lengths[i]:] = 0

    return rep_mask.unsqueeze_(-1)


class NN4EMO(nn.Module):

    def __init__(self, args, data, ss_vectors=None):
        super(NN4EMO, self).__init__()

        self.args = args
        self.class_size = args.class_size
        self.dropout = args.dropout
        self.d_e = args.d_e
        self.d_ff = args.d_ff
        self.device = args.device

        # GloVe embedding
        self.glove_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        # initialize word embedding with GloVe
        self.glove_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        # emojis init
        with open('data/emoji/emoji-vectors.pkl', 'rb') as f:
            emoji_vectors = pickle.load(f)
            for i in range(args.word_vocab_size):
                word = data.TEXT.vocab.itos[i]
                if word in emoji_vectors:
                    self.glove_emb.weight.data[i] = torch.tensor(emoji_vectors[word])
        # fine-tune the word embedding
        if not args.tune_embeddings:
            self.glove_emb.weight.requires_grad = False
        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.glove_emb.weight.data[0], -0.05, 0.05)

        # sentiment specific embedding
        self.ss_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        if args.ss_emb:
            self.ss_emb.weight.data.copy_(ss_vectors)
            if not args.ss_emb_tune:
                self.ss_emb.weight.requires_grad = False
        if args.fasttext:
            self.ss_emb.weight.data.copy_(data.FASTTEXT.vocab.vectors)
            if not args.fasttext_tune:
                self.ss_emb.weight.requires_grad = False
        if args.word2vec:
            word2vec = gsm.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
            emoji2vec = gsm.KeyedVectors.load_word2vec_format('data/emoji/emoji2vec.bin', binary=True)
            for i in range(args.word_vocab_size):
                word = data.TEXT.vocab.itos[i]
                if word in emoji.UNICODE_EMOJI and word in emoji2vec.vocab:
                    self.ss_emb.weight.data[i] = torch.tensor(emoji2vec[word])
                elif word in word2vec.vocab:
                    self.ss_emb.weight.data[i] = torch.tensor(word2vec[word])
                else:
                    nn.init.uniform_(self.ss_emb.weight.data[i], -0.05, 0.05)
            if not args.word2vec_tune:
                self.ss_emb.weight.requires_grad = False

        if args.pos_emb:
            self.pos_emb = nn.Parameter(torch.rand(512, args.word_dim))

        if args.simple_encoder:
            self.sentence_encoder = SimpleEncoder(args, data)
            self.fc_dim = 2 * args.d_e
        else:
            self.sentence_encoder = SentenceEncoder(args, data)
            self.fc_dim = 2 * args.d_e * 2

        self.fc1 = nn.Linear(self.fc_dim, args.d_e)
        self.fc2 = nn.Linear(self.fc_dim + args.d_e, args.d_e)
        self.fc_out = nn.Linear(args.d_e, args.class_size)

        self.layer_norm = nn.LayerNorm(args.d_e)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, batch):
        batch_raw = batch.raw
        seq, lens = batch.text
        if self.args.char_emb:
            batch_char = batch.char
        else:
            batch_char = None

        # (batch, seq_len, word_dim)
        x_g = self.glove_emb(seq)
        x_s = self.ss_emb(seq)

        if self.args.pos_emb:
            batch_size, seq_len, _ = x_g.size()
            pos_emb = self.pos_emb[:seq_len]
            pos_emb_batch = torch.stack([pos_emb] * batch_size).to(self.device)
            x_g = x_g + pos_emb_batch
            x_s = x_s + pos_emb_batch

        # (batch, seq_len, 1)
        rep_mask = get_rep_mask(lens, self.device)

        # (batch, seq_len, 4 * d_e)
        s = self.sentence_encoder(x_g, x_s, batch_char, batch_raw,
                                  rep_mask, lens, seq)

        outputs = self.fc1(s)
        outputs = self.relu(outputs)
        outputs = self.fc2(torch.cat([s, outputs], dim=-1))
        outputs = self.relu(outputs)
        outputs = self.fc_out(outputs)

        return outputs


class NN4EMO_FUSION(nn.Module):

    def __init__(self, args, data, ss_vectors=None):
        super(NN4EMO_FUSION, self).__init__()

        self.args = args
        self.class_size = args.class_size
        self.dropout = args.dropout
        self.d_e = args.d_e
        self.d_ff = args.d_ff
        self.device = args.device

        # GloVe embedding
        self.glove_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        # initialize word embedding with GloVe
        self.glove_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        # emojis init
        with open('data/emoji/emoji-vectors.pkl', 'rb') as f:
            emoji_vectors = pickle.load(f)
            for i in range(args.word_vocab_size):
                word = data.TEXT.vocab.itos[i]
                if word in emoji_vectors:
                    self.glove_emb.weight.data[i] = torch.tensor(emoji_vectors[word])
        # fine-tune the word embedding
        if not args.tune_embeddings:
            self.glove_emb.weight.requires_grad = False
        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.glove_emb.weight.data[0], -0.05, 0.05)

        # sentiment specific embedding
        self.ss_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        if args.ss_emb:
            self.ss_emb.weight.data.copy_(ss_vectors)
            if not args.ss_emb_tune:
                self.ss_emb.weight.requires_grad = False
        if args.fasttext:
            self.ss_emb.weight.data.copy_(data.FASTTEXT.vocab.vectors)
            if not args.fasttext_tune:
                self.ss_emb.weight.requires_grad = False
        if args.word2vec:
            word2vec = gsm.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
            emoji2vec = gsm.KeyedVectors.load_word2vec_format('data/emoji/emoji2vec.bin', binary=True)
            for i in range(args.word_vocab_size):
                word = data.TEXT.vocab.itos[i]
                if word in emoji.UNICODE_EMOJI and word in emoji2vec.vocab:
                    self.ss_emb.weight.data[i] = torch.tensor(emoji2vec[word])
                elif word in word2vec.vocab:
                    self.ss_emb.weight.data[i] = torch.tensor(word2vec[word])
                else:
                    nn.init.uniform_(self.ss_emb.weight.data[i], -0.05, 0.05)
            if not args.word2vec_tune:
                self.ss_emb.weight.requires_grad = False

        if args.biattention:
            self.biattention_encoder = BiAttentionEncoder(args, data)
            self.fc_dim = 4 * args.d_e
        else:
            if args.simple_encoder:
                self.sentence_encoder_c = SimpleEncoder(args, data)
                self.fc_dim = 4 * 2 * args.d_e
            else:
                self.sentence_encoder_c = SentenceEncoder(args, data)
                self.fc_dim = 4 * 2 * args.d_e * 2

            if args.share_encoder:
                self.sentence_encoder_s = self.sentence_encoder_c
            else:
                if args.simple_encoder:
                    self.sentence_encoder_s = SimpleEncoder(args, data)
                else:
                    self.sentence_encoder_s = SentenceEncoder(args, data)

        self.fc1 = nn.Linear(self.fc_dim, args.d_e)
        self.fc2 = nn.Linear(self.fc_dim + args.d_e, args.d_e)
        self.fc_out = nn.Linear(args.d_e, args.class_size)

        self.layer_norm = nn.LayerNorm(args.d_e)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()


    def forward(self, batch):
        batch_raw_c = batch.raw_c
        batch_raw_s = batch.raw_s
        batch_c = batch.context
        batch_s = batch.sent

        seq_c, lens_c = batch_c
        seq_s, lens_s = batch_s
        if self.args.char_emb:
            batch_char_c = batch.char_c
            batch_char_s = batch.char_s
        else:
            batch_char_c = None
            batch_char_s = None

        # (batch, seq_len, word_dim)
        x_c_g = self.glove_emb(seq_c)
        x_c_s = self.ss_emb(seq_c)
        x_s_g = self.glove_emb(seq_s)
        x_s_s = self.ss_emb(seq_s)

        # (batch, seq_len, 1)
        rep_mask_c = get_rep_mask(lens_c, self.device)
        rep_mask_s = get_rep_mask(lens_s, self.device)

        if self.args.biattention:
            s_c, s_s = self.biattention_encoder(x_c_g, x_c_s, x_s_g, x_s_s,
                                                batch_char_c, batch_char_s,
                                                batch_raw_c, batch_raw_s,
                                                rep_mask_c, rep_mask_s,
                                                lens_c, lens_s)
        else:
            # (batch, seq_len, 4 * d_e)
            s_c = self.sentence_encoder_c(x_c_g, x_c_s, batch_char_c, batch_raw_c,
                                          rep_mask_c, lens_c, seq_c)
            s_s = self.sentence_encoder_s(x_s_g, x_s_s, batch_char_s, batch_raw_s,
                                          rep_mask_s, lens_s, seq_s)

        # fusion
        s = torch.cat([s_c, s_s, s_c - s_s, s_c * s_s], dim=-1)

        outputs = self.fc1(s)
        outputs = self.relu(outputs)
        #outputs = self.fc2(torch.cat([s, outputs], dim=-1))
        #outputs = self.relu(outputs)
        outputs = self.fc_out(outputs)

        return outputs


class NN4EMO_ENSEMBLE(nn.Module):

    def __init__(self, args, data, ss_vectors=None):
        super(NN4EMO_ENSEMBLE, self).__init__()
        self.nn4emo = NN4EMO(args, data, ss_vectors)
        args.ss_emb_tune = True
        self.nn4emo_fusion = NN4EMO_FUSION(args, data, ss_vectors)

    def forward(self, batch):
        out1 = self.nn4emo(batch)
        out2 = self.nn4emo_fusion(batch)

        return (out1 + out2) / 2