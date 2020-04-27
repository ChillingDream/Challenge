import torch
import torch.nn.functional as F
from torch import nn

from config import device, drop_rate
from data_process import field_dims, word_embeddings, onehot_idx, multihot_idx

class Flatten(nn.Module):
	def __init__(self, start_dim=1, end_dim=-1):
		super().__init__()
		self.start_dim = start_dim
		self.end_dim = end_dim

	def forward(self, x):
		return torch.flatten(x, self.start_dim, self.end_dim)

class MutiheadAttention(nn.Module):
	def __init__(self, num_in, num_out, num_head, use_layer_norm=True, use_dropout=True):
		super().__init__()
		self.num_out = num_out
		self.queries = nn.Linear(num_in, num_out * num_head, bias=False)
		self.keys = nn.Linear(num_in, num_out * num_head, bias=False)
		self.values = nn.Linear(num_in, num_out * num_head, bias=False)
		self.weights_dropout = nn.Dropout(drop_rate) if use_dropout else None
		self.res_layer = nn.Linear(num_in, num_out * num_head, bias=False)
		self.layer_norm = nn.LayerNorm([num_out * num_head]) if use_layer_norm else None

	def forward(self, x):
		batch = x.size()[0]
		queries = torch.cat(self.queries(x).split(self.num_out, -1), 0)
		keys = torch.cat(self.keys(x).split(self.num_out, -1), 0)
		values = torch.cat(self.values(x).split(self.num_out, -1), 0)
		weights = queries.matmul(keys.permute([0, 2, 1]))  # QK^T
		weights = F.softmax(weights / self.num_out ** 0.5, -1)
		if self.weights_dropout:
			weights = self.weights_dropout(weights)
		outputs = weights.matmul(values)
		outputs = torch.cat(outputs.split(batch, 0), 2)
		outputs = F.relu(outputs + self.res_layer(x))
		if self.layer_norm:
			outputs = self.layer_norm(outputs)
		return outputs

class AutoInt(nn.Module):
	def __init__(self, emb_length, num_units, num_heads):
		super().__init__()
		self.token_emb_layer = nn.EmbeddingBag.from_pretrained(word_embeddings, freeze=True, mode='mean')
		self.emb_layers = nn.ModuleList([nn.Linear(word_embeddings.size()[1], emb_length, bias=False)] +
										[nn.EmbeddingBag(field_dims[i], emb_length, mode='mean') for i in
										 multihot_idx] +
										[nn.Embedding(field_dims[i], emb_length) for i in onehot_idx])
		layers = []
		for i in range(len(num_units)):
			if i == 0:
				layers.append(MutiheadAttention(emb_length, num_units[0], num_heads[0]))
			else:
				layers.append(MutiheadAttention(num_units[i - 1] * num_heads[i - 1], num_units[i], num_heads[i]))
		layers.append(Flatten())
		layers.append(nn.Linear(len(field_dims) * num_units[-1] * num_heads[-1], 1, bias=True))
		layers.append(nn.Sigmoid())
		self.layers = nn.Sequential(*layers)

		self.loss_function = nn.BCELoss()

	def forward(self, x):
		for i in range(len(x)):
			for j in range(len(x[i])):
				x[i][j] = x[i][j].to(device)
		x = [[self.token_emb_layer(*x[0])]] + x[1:]
		emb = torch.stack([layer(*field) for (field, layer) in zip(x, self.emb_layers)], 1)
		return self.layers(emb)

	def loss(self, logits, labels):
		return self.loss_function(logits, labels.float().to(device))

	@staticmethod
	def transform(x):
		return x

if __name__ == '__main__':
	m = AutoInt(32, [10], [8])
	for name, param in m.named_parameters():
		print(name, param.size())
