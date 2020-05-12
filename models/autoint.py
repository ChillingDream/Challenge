import torch
from torch import nn

from config import device, drop_rate
from data_process import field_dims, word_embeddings, onehot_idx, multihot_idx, max_length
from transformer import MutiheadAttention, TransformerEncoder

class Flatten(nn.Module):
	def __init__(self, start_dim=1, end_dim=-1):
		super().__init__()
		self.start_dim = start_dim
		self.end_dim = end_dim

	def forward(self, x):
		return torch.flatten(x, self.start_dim, self.end_dim)


class AutoInt(nn.Module):
	def __init__(self, emb_length, num_units, num_heads, dnn_units):
		super().__init__()
		# self.token_emb_layer = nn.EmbeddingBag.from_pretrained(word_embeddings, freeze=True, mode='mean')
		self.token_emb_layer = TransformerEncoder(max_length, word_embeddings.size()[1])
		self.emb_layers = nn.ModuleList([nn.Linear(256, emb_length, bias=False)])
		for i in range(1, len(field_dims)):
			if i in multihot_idx:
				self.emb_layers.append(nn.EmbeddingBag(field_dims[i], emb_length, mode='mean'))
			elif i in onehot_idx:
				self.emb_layers.append(nn.Embedding(field_dims[i], emb_length))
		self.dnn_layer = None
		if dnn_units:
			dnn = [Flatten()]
			dnn_units = [len(field_dims) * emb_length] + dnn_units
			for i in range(len(dnn_units) - 2):
				dnn.append(nn.Linear(dnn_units[i], dnn_units[i + 1], bias=False))
				dnn.append(nn.BatchNorm1d(dnn_units[i + 1]))
				dnn.append(nn.ReLU(True))
				dnn.append(nn.Dropout(drop_rate))
			dnn.append(nn.Linear(dnn_units[-2], dnn_units[-1]))
			self.dnn_layer = nn.Sequential(*dnn)
		for layer in self.emb_layers[1:]:
			nn.init.normal_(layer.weight, 0, emb_length ** -0.5)
		layers = []
		for i in range(len(num_units)):
			if i == 0:
				layers.append(MutiheadAttention(emb_length, num_units[0], num_heads[0]))
			else:
				layers.append(MutiheadAttention(num_units[i - 1] * num_heads[i - 1], num_units[i], num_heads[i]))
		layers.append(Flatten())
		self.att_layers = nn.Sequential(*layers)
		self.last_layer = nn.Linear(
			len(field_dims) * num_units[-1] * num_heads[-1] + (dnn_units[-1] if dnn_units else 0),
			1, bias=True)

		self.loss_function = nn.BCELoss()

	def forward(self, x):
		x = [[self.token_emb_layer(*x[0])]] + x[1:]
		emb = torch.stack([layer(*field) for (field, layer) in zip(x, self.emb_layers)], 1)
		out = self.att_layers(emb)
		if self.dnn_layer:
			out = torch.cat([out, self.dnn_layer(emb)], 1)
		out = self.last_layer(out)
		return torch.sigmoid(out)

	def loss(self, logits, labels):
		return self.loss_function(logits, labels.float().to(device))

	@staticmethod
	def transform(x):
		return x

if __name__ == '__main__':
	m = AutoInt(32, [10], [8])
	for name, param in m.named_parameters():
		print(name, param.size())
