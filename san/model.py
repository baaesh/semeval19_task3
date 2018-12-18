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

	def __init__(self, args, data):
		super(NN4EMO, self).__init__()

		self.args = args
		self.class_size = args.class_size
		self.dropout = args.dropout
		self.d_e = args.d_e
		self.d_ff = args.d_ff
		self.device = args.device
		self.data = data

		self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
		# initialize word embedding with GloVe
		self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
		# fine-tune the word embedding
		self.word_emb.weight.requires_grad = False
		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.05, 0.05)

		if args.seg_emb:
			self.seg_emb = nn.Embedding(3, args.word_dim)

		self.sentence_encoder = SentenceEncoder(args)

		self.fc = nn.Linear(args.d_e * 4, args.d_e)
		self.fc_out = nn.Linear(args.d_e, args.class_size)

		self.layer_norm = nn.LayerNorm(args.d_e)
		self.dropout = nn.Dropout(args.dropout)
		self.relu = nn.ReLU()

	def forward(self, batch):
		seq, lens = batch

		# (batch, seq_len, word_dim)
		x = self.word_emb(seq)

		if self.args.seg_emb:
			seg_emb = self.seg_seq(seq)

			x = x + seg_emb

		# (batch, seq_len, 1)
		rep_mask = get_rep_mask(lens, self.device)

		# (batch, seq_len, 4 * d_e)
		s = self.sentence_encoder(x, rep_mask)

		s = self.dropout(s)
		outputs = self.relu(self.layer_norm(self.fc(s)))
		outputs = self.dropout(outputs)
		outputs = self.fc_out(outputs)

		return outputs

	def seg_seq(self, seq):
		seg_batch = []
		for i in range(len(seq)):
			idx = torch.tensor(0).to(self.device)
			seg = []
			for j in range(len(seq[i])):
				seg.append(self.seg_emb(idx))
				if (seq[i][j] == self.data.TEXT.vocab.stoi['<eos>']):
					idx = idx + 1
			seg_batch.append(torch.stack(seg))
		return torch.stack(seg_batch)
