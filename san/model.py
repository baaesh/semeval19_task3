import pickle
import torch
import torch.nn as nn
import emoji
import gensim.models as gsm

from data import build_datastories_vectors
from module import *


def get_rep_mask(lengths, device):
    batch_size = len(lengths)
    seq_len = torch.max(lengths)
    rep_mask = torch.FloatTensor(batch_size, seq_len).to(torch.device(device))
    rep_mask.data.fill_(1)
    for i in range(batch_size):
        rep_mask[i, lengths[i]:] = 0

    return rep_mask.unsqueeze_(-1)


class NN4EMO_SEMI_HIERARCHICAL(nn.Module):

    def __init__(self, args, data, ss_vectors=None):
        super(NN4EMO_SEMI_HIERARCHICAL, self).__init__()

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
        if args.datastories:
            embeddings_dict = build_datastories_vectors(data)
            for word in embeddings_dict:
                index = data.TEXT.vocab.stoi[word]
                self.glove_emb.weight.data[index] = torch.tensor(embeddings_dict[word])
        # fine-tune the word embedding
        if not args.tune_embeddings:
            self.glove_emb.weight.requires_grad = False
        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.glove_emb.weight.data[0], -0.05, 0.05)

        # word2vec + emoji2vec embeddings
        self.word2vec_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        word2vec = gsm.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
        emoji2vec = gsm.KeyedVectors.load_word2vec_format('data/emoji/emoji2vec.bin', binary=True)
        for i in range(args.word_vocab_size):
            word = data.TEXT.vocab.itos[i]
            if word in emoji.UNICODE_EMOJI and word in emoji2vec.vocab:
                self.word2vec_emb.weight.data[i] = torch.tensor(emoji2vec[word])
            elif word in word2vec.vocab:
                self.word2vec_emb.weight.data[i] = torch.tensor(word2vec[word])
            else:
                nn.init.uniform_(self.word2vec_emb.weight.data[i], -0.05, 0.05)
        if not args.word2vec_tune:
            self.word2vec_emb.weight.requires_grad = False

        # character embedding
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=0)
        self.charCNN = CharCNN(args)

        # utterance encoders
        self.utterance_encoder_turn1 = SentenceEncoder(args, data)
        if args.share_encoder:
            self.utterance_encoder_turn2 = self.utterance_encoder_turn1
            self.utterance_encoder_turn3 = self.utterance_encoder_turn1
        else:
            self.utterance_encoder_turn2 = SentenceEncoder(args, data)
            self.utterance_encoder_turn3 = SentenceEncoder(args, data)

        # hierarchical LSTM encoder
        self.lstm_input_dim = 2 * 2 * args.d_e
        self.hierarchical_lstm = LSTMEncoder(args, input_dim=self.lstm_input_dim,
                                             last_hidden=True)

        # feed-forward layers
        # u1, u2, u3, u1 - u2 + u3 and output of lstm
        self.fc_dim = 2 * 2 * args.d_e * 4 + 2 * args.lstm_hidden_dim
        self.fc1 = nn.Linear(self.fc_dim, args.d_e)
        self.fc2 = nn.Linear(self.fc_dim + args.d_e, args.d_e)
        self.fc_out = nn.Linear(args.d_e, args.class_size)

        self.layer_norm = nn.LayerNorm(args.d_e)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, batch):
        batch_turn1 = batch.turn1
        batch_turn2 = batch.turn2
        batch_turn3 = batch.turn3
        seq_turn1, lens_turn1 = batch_turn1
        seq_turn2, lens_turn2 = batch_turn2
        seq_turn3, lens_turn3 = batch_turn3

        # raw inputs for ELMo
        batch_raw_turn1 = batch.raw_turn1
        batch_raw_turn2 = batch.raw_turn2
        batch_raw_turn3 = batch.raw_turn3

        # character inputs for charCNN
        batch_char_turn1 = batch.char_turn1
        batch_char_turn2 = batch.char_turn2
        batch_char_turn3 = batch.char_turn3

        # (batch, seq_len, word_dim)
        x_turn1_1 = self.glove_emb(seq_turn1)
        x_turn1_2 = self.word2vec_emb(seq_turn1)
        x_turn2_1 = self.glove_emb(seq_turn2)
        x_turn2_2 = self.word2vec_emb(seq_turn2)
        x_turn3_1 = self.glove_emb(seq_turn3)
        x_turn3_2 = self.word2vec_emb(seq_turn3)

        # character embedding
        # (batch, seq_len, max_word_len)
        batch_size, seq_len_turn1, _ = batch_char_turn1.size()
        batch_size, seq_len_turn2, _ = batch_char_turn2.size()
        batch_size, seq_len_turn3, _ = batch_char_turn3.size()

        # (batch * seq_len, max_word_len)
        char_turn1 = batch_char_turn1.view(-1, self.args.max_word_len)
        char_turn2 = batch_char_turn2.view(-1, self.args.max_word_len)
        char_turn3 = batch_char_turn3.view(-1, self.args.max_word_len)

        # (batch * seq_len, max_word_len, char_dim)
        char_turn1 = self.char_emb(char_turn1)
        char_turn2 = self.char_emb(char_turn2)
        char_turn3 = self.char_emb(char_turn3)

        # (batch, seq_len, len(FILTER_SIZES) * num_feature_maps)
        char_turn1 = self.charCNN(char_turn1).view(batch_size, seq_len_turn1, -1)
        char_turn2 = self.charCNN(char_turn2).view(batch_size, seq_len_turn2, -1)
        char_turn3 = self.charCNN(char_turn3).view(batch_size, seq_len_turn3, -1)

        # (batch, seq_len, 1)
        rep_mask_turn1 = get_rep_mask(lens_turn1, self.device)
        rep_mask_turn2 = get_rep_mask(lens_turn2, self.device)
        rep_mask_turn3 = get_rep_mask(lens_turn3, self.device)

        # (batch, seq_len, 4 * d_e)
        u_turn1 = self.utterance_encoder_turn1(x_turn1_1, x_turn1_2,
                                               char_turn1,
                                               batch_raw_turn1,
                                               rep_mask_turn1,
                                               lens_turn1, seq_turn1)
        u_turn2 = self.utterance_encoder_turn2(x_turn2_1, x_turn2_2,
                                               char_turn2,
                                               batch_raw_turn2,
                                               rep_mask_turn2,
                                               lens_turn2, seq_turn2)
        u_turn3 = self.utterance_encoder_turn3(x_turn3_1, x_turn3_2,
                                               char_turn3,
                                               batch_raw_turn3,
                                               rep_mask_turn3,
                                               lens_turn3, seq_turn3)

        hierarchical_lstm_in = torch.stack([u_turn1, u_turn2, u_turn3], dim=1)
        hierarchical_lstm_lens = torch.LongTensor([3]*hierarchical_lstm_in.size()[0])
        hierarchical_lstm_out = self.hierarchical_lstm(hierarchical_lstm_in,
                                                       hierarchical_lstm_lens)

        u = torch.cat([u_turn1, u_turn2, u_turn3, u_turn1 - u_turn2 + u_turn3,
                       hierarchical_lstm_out], dim=-1)

        outputs = self.fc1(u)
        outputs = self.relu(outputs)
        outputs = self.fc2(torch.cat([u, outputs], dim=-1))
        outputs = self.relu(outputs)
        outputs = self.fc_out(outputs)

        return outputs


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
        if args.datastories:
            embeddings_dict = build_datastories_vectors(data)
            for word in embeddings_dict:
                index = data.TEXT.vocab.stoi[word]
                self.glove_emb.weight.data[index] = torch.tensor(embeddings_dict[word])
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
        if args.datastories:
            embeddings_dict = build_datastories_vectors(data)
            for word in embeddings_dict:
                index = data.TEXT.vocab.stoi[word]
                self.glove_emb.weight.data[index] = torch.tensor(embeddings_dict[word])
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
        outputs = self.fc2(torch.cat([s, outputs], dim=-1))
        outputs = self.relu(outputs)
        outputs = self.fc_out(outputs)

        return outputs


class NN4EMO_ENSEMBLE(nn.Module):

    def __init__(self, args, data, ss_vectors=None):
        super(NN4EMO_ENSEMBLE, self).__init__()
        self.nn4emo = NN4EMO(args, data, ss_vectors)
        self.nn4emo_fusion = NN4EMO_FUSION(args, data, ss_vectors)
        self.nn4emo_seperate = NN4EMO_SEPERATE(args, data, ss_vectors)

    def forward(self, batch):
        out1 = self.nn4emo(batch)
        out2 = self.nn4emo_fusion(batch)
        out3 = self.nn4emo_seperate(batch)

        return (out1 + out2 + out3) / 3


class NN4EMO_SEPARATE(nn.Module):

    def __init__(self, args, data, ss_vectors=None):
        super(NN4EMO_SEPARATE, self).__init__()

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
        if args.datastories:
            embeddings_dict = build_datastories_vectors(data)
            for word in embeddings_dict:
                index = data.TEXT.vocab.stoi[word]
                self.glove_emb.weight.data[index] = torch.tensor(embeddings_dict[word])
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

        # character embedding
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=0)
        self.charCNN = CharCNN(args)

        if args.uni_encoder:
            self.sentence_encoder_turn1 = UniEncoder(args, data)
            if args.simple_encoder:
                self.fc_dim = 2 * args.lstm_hidden_dim * 5
                self.lstm_dim = 2 * args.lstm_hidden_dim
            else:
                self.fc_dim = 2 * args.lstm_hidden_dim * 4 * 2 + 2 * args.lstm_hidden_dim
                self.lstm_dim = 2 * args.lstm_hidden_dim * 2
        elif args.simple_encoder:
            self.sentence_encoder_turn1 = SimpleEncoder(args, data)
            self.fc_dim = 2 * args.d_e * 4 + 2 * args.lstm_hidden_dim
            self.lstm_dim = 2 * args.d_e
        else:
            self.sentence_encoder_turn1 = SentenceEncoder(args, data)
            self.fc_dim = 2 * args.d_e * 2 * 4 + 2 * args.lstm_hidden_dim
            self.lstm_dim = 2 * args.d_e * 2

        if args.share_encoder:
            self.sentence_encoder_turn3 = self.sentence_encoder_turn1
            if args.turn2:
                if args.uni_encoder:
                    self.sentence_encoder_turn2 = UniEncoder(args, data)
                elif args.simple_encoder:
                    self.sentence_encoder_turn2 = SimpleEncoder(args, data)
                else:
                    self.sentence_encoder_turn2 = SentenceEncoder(args, data)
            else:
                self.sentence_encoder_turn2 = self.sentence_encoder_turn1
        else:
            if args.uni_encoder:
                self.sentence_encoder_turn2 = UniEncoder(args, data)
                self.sentence_encoder_turn3 = UniEncoder(args, data)
            elif args.simple_encoder:
                self.sentence_encoder_turn2 = SimpleEncoder(args, data)
                self.sentence_encoder_turn3 = SimpleEncoder(args, data)
            else:
                self.sentence_encoder_turn2 = SentenceEncoder(args, data)
                self.sentence_encoder_turn3 = SentenceEncoder(args, data)

        self.lstm = LSTMEncoder(args, input_dim=self.lstm_dim, last_hidden=True)

        if args.no_turn2:
            if args.simple_encoder:
                self.fc_dim = 2 * args.d_e * 4
            else:
                self.fc_dim = 2 * args.d_e * 2 * 4

        self.fc1 = nn.Linear(self.fc_dim, args.d_e)
        self.fc2 = nn.Linear(self.fc_dim + args.d_e, args.d_e)
        self.fc_out = nn.Linear(args.d_e, args.class_size)

        self.layer_norm = nn.LayerNorm(args.d_e)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, batch):
        batch_raw_turn1 = batch.raw_turn1
        batch_raw_turn2 = batch.raw_turn2
        batch_raw_turn3 = batch.raw_turn3
        batch_turn1 = batch.turn1
        batch_turn2 = batch.turn2
        batch_turn3 = batch.turn3

        seq_turn1, lens_turn1 = batch_turn1
        seq_turn2, lens_turn2 = batch_turn2
        seq_turn3, lens_turn3 = batch_turn3
        if self.args.char_emb:
            batch_char_turn1 = batch.char_turn1
            batch_char_turn2 = batch.char_turn2
            batch_char_turn3 = batch.char_turn3
        else:
            batch_char_turn1 = None
            batch_char_turn2 = None
            batch_char_turn3 = None

        # (batch, seq_len, word_dim)
        x_turn1_1 = self.glove_emb(seq_turn1)
        x_turn1_2 = self.ss_emb(seq_turn1)
        x_turn2_1 = self.glove_emb(seq_turn2)
        x_turn2_2 = self.ss_emb(seq_turn2)
        x_turn3_1 = self.glove_emb(seq_turn3)
        x_turn3_2 = self.ss_emb(seq_turn3)

        # character embedding
        # (batch, seq_len, max_word_len)
        batch_size, seq_len_turn1, _ = batch_char_turn1.size()
        batch_size, seq_len_turn2, _ = batch_char_turn2.size()
        batch_size, seq_len_turn3, _ = batch_char_turn3.size()

        # (batch * seq_len, max_word_len)
        char_turn1 = batch_char_turn1.view(-1, self.args.max_word_len)
        char_turn2 = batch_char_turn2.view(-1, self.args.max_word_len)
        char_turn3 = batch_char_turn3.view(-1, self.args.max_word_len)

        # (batch * seq_len, max_word_len, char_dim)
        char_turn1 = self.char_emb(char_turn1)
        char_turn2 = self.char_emb(char_turn2)
        char_turn3 = self.char_emb(char_turn3)

        # (batch, seq_len, len(FILTER_SIZES) * num_feature_maps)
        char_turn1 = self.charCNN(char_turn1).view(batch_size, seq_len_turn1, -1)
        char_turn2 = self.charCNN(char_turn2).view(batch_size, seq_len_turn2, -1)
        char_turn3 = self.charCNN(char_turn3).view(batch_size, seq_len_turn3, -1)

        # (batch, seq_len, 1)
        rep_mask_turn1 = get_rep_mask(lens_turn1, self.device)
        rep_mask_turn2 = get_rep_mask(lens_turn2, self.device)
        rep_mask_turn3 = get_rep_mask(lens_turn3, self.device)

        # (batch, seq_len, 4 * d_e)
        s_turn1 = self.sentence_encoder_turn1(x_turn1_1, x_turn1_2,
                                              char_turn1,
                                              batch_raw_turn1,
                                              rep_mask_turn1,
                                              lens_turn1, seq_turn1)
        s_turn2 = self.sentence_encoder_turn2(x_turn2_1, x_turn2_2,
                                              char_turn2,
                                              batch_raw_turn2,
                                              rep_mask_turn2,
                                              lens_turn2, seq_turn2)
        s_turn3 = self.sentence_encoder_turn3(x_turn3_1, x_turn3_2,
                                              char_turn3,
                                              batch_raw_turn3,
                                              rep_mask_turn3,
                                              lens_turn3, seq_turn3)
        if self.args.no_turn2:
            s = torch.cat([s_turn1, s_turn3, s_turn1 - s_turn3, s_turn1 * s_turn3], dim=-1)
        else:
            s_lstm_in = torch.stack([s_turn1, s_turn2, s_turn3], dim=1)
            s_lstm_out = self.lstm(s_lstm_in, torch.LongTensor([3]*s_lstm_in.size()[0]))

            s = torch.cat([s_turn1, s_turn2, s_turn3, s_lstm_out,
                           s_turn1 - s_turn2 + s_turn3], dim=-1)

        outputs = self.fc1(s)
        outputs = self.relu(outputs)
        outputs = self.fc2(torch.cat([s, outputs], dim=-1))
        outputs = self.relu(outputs)
        outputs = self.fc_out(outputs)

        return outputs