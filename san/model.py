import torch
import torch.nn as nn

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

		if args.seg_emb:
			self.seg_emb = nn.Embedding(3, args.word_dim)
			self.eos_idx = data.TEXT.vocab.stoi['<eos>']

		if args.pos_emb:
			self.pos_emb = nn.Parameter(torch.rand(512, args.word_dim))

		self.sentence_encoder = SentenceEncoder(args)

		self.fc = nn.Linear(args.d_e * 6, args.d_e)
		self.fc_out = nn.Linear(args.d_e, args.class_size)

		self.layer_norm = nn.LayerNorm(args.d_e)
		self.dropout = nn.Dropout(args.dropout)
		self.relu = nn.ReLU()

	def forward(self, batch, batch_raw):
		seq, lens = batch

		# (batch, seq_len, word_dim)
		x_g = self.glove_emb(seq)
		x_s = self.ss_emb(seq)

		if self.args.seg_emb:
			seg_emb = self.seg_seq(seq)
			x_g = x_g + seg_emb
			x_s = x_s + seg_emb

		if self.args.pos_emb:
			batch_size, seq_len, _ = x_g.size()
			pos_emb = self.pos_emb[:seq_len]
			pos_emb_batch = torch.stack([pos_emb] * batch_size).to(self.device)
			x_g = x_g + pos_emb_batch
			x_s = x_s + pos_emb_batch

		# (batch, seq_len, 1)
		rep_mask = get_rep_mask(lens, self.device)

		# (batch, seq_len, 4 * d_e)
		s = self.sentence_encoder(x_g, x_s, batch_raw, rep_mask, lens)

		s = self.dropout(s)
		outputs = self.relu(self.layer_norm(self.fc(s)))
		outputs = self.dropout(outputs)
		outputs = self.fc_out(outputs)

		return outputs

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
		return self.seg_emb(seg_idx_batch)
