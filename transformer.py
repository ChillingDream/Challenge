import torch
import torch.nn.functional as F
from torch import nn

from config import drop_rate
from data_process import word_embeddings

class MutiheadAttention(nn.Module):
	def __init__(self, num_in, num_out, num_head, use_layer_norm=True, use_dropout=True):
		super().__init__()
		self.num_out = num_out
		self.num_head = num_head
		self.queries = nn.Linear(num_in, num_out * num_head, bias=False)
		self.keys = nn.Linear(num_in, num_out * num_head, bias=False)
		self.values = nn.Linear(num_in, num_out * num_head, bias=False)
		self.weights_dropout = nn.Dropout(drop_rate) if use_dropout else None
		self.res_layer = nn.Linear(num_in, num_out * num_head, bias=False)
		self.layer_norm = nn.LayerNorm([num_out * num_head]) if use_layer_norm else None

	def forward(self, x, mask=None):
		batch = x.size()[0]
		queries = torch.cat(self.queries(x).split(self.num_out, -1), 0)
		keys = torch.cat(self.keys(x).split(self.num_out, -1), 0)
		values = torch.cat(self.values(x).split(self.num_out, -1), 0)
		weights = queries.matmul(keys.transpose(-2, -1)) / self.num_out ** 0.5  # QK^T/sqrt(d)
		if mask is not None:
			weights = weights.masked_fill((mask == 0).unsqueeze(1).repeat(self.num_head, 1, 1), -1e9)
		weights = F.softmax(weights, -1)
		if self.weights_dropout:
			weights = self.weights_dropout(weights)
		outputs = weights.matmul(values)
		outputs = torch.cat(outputs.split(batch, 0), 2)
		outputs = F.relu(outputs + self.res_layer(x))
		if self.layer_norm:
			outputs = self.layer_norm(outputs)
		return outputs

class PositionalEncoder(nn.Module):
	def __init__(self, maxlen, emb_dim):
		super().__init__()
		self.embedding = torch.zeros([maxlen, emb_dim])
		position = torch.arange(maxlen).unsqueeze(1)
		div_term = 10000 ** (torch.arange(0, emb_dim, 2, dtype=torch.float32) / emb_dim)
		term = position.float() / div_term
		self.embedding[:, 0::2] = torch.sin(term)
		self.embedding[:, 1::2] = torch.cos(term)
		self.embedding = nn.Parameter(self.embedding, requires_grad=False)

	def forward(self, x):
		return x + self.embedding

class TransformerEncoder(nn.Module):
	def __init__(self, maxlen, emb_dim):
		super().__init__()
		self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=True)
		self.pos_enc = PositionalEncoder(maxlen, emb_dim)
		self.att_layer = MutiheadAttention(emb_dim, 64, 4)

	def forward(self, x):
		mask = x != 0
		emb = self.word_embedding(x)
		emb = self.pos_enc(emb)
		out = self.att_layer(emb, mask)
		return torch.sum(out * mask.unsqueeze(2).float(), 1) / mask.sum(1, keepdim=True).float()

if __name__ == '__main__':
	p = PositionalEncoder(3, 20)
