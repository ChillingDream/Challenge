# to be updated
import torch
from torch import nn

from config import device
from data_process import field_dims, word_embeddings, multihot_idx, onehot_idx

class WideDeep(nn.Module):
	def __init__(self, emb_length, hidden_units):
		super().__init__()
		self.token_emb_layer = nn.EmbeddingBag.from_pretrained(word_embeddings, freeze=True, mode='mean')
		self.emb_layers = nn.ModuleList([nn.EmbeddingBag(field_dims[i], emb_length, mode='mean') for i in
										 multihot_idx] +
										[nn.Embedding(field_dims[i], emb_length) for i in onehot_idx])
		for layer in self.emb_layers:
			nn.init.normal_(layer.weight, 0, layer.weight.size(0))
		hidden_units = [field_dims[0] + (len(field_dims) - 1) * emb_length] + hidden_units
		layers = []
		for i in range(len(hidden_units) - 1):
			layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
			layers.append(nn.ReLU(True))
			if i != len(hidden_units) - 2:
				layers.append(nn.Dropout(0.5))
		self.mid_layers = nn.Sequential(*layers)
		self.last_layer_deep = nn.Linear(hidden_units[-1], 1)
		nn.init.normal_(self.last_layer_deep.weight, 0, (hidden_units[-1] + sum(field_dims[1:])) ** -0.5)
		self.last_layer_wide = nn.ModuleList([nn.EmbeddingBag(field_dims[i], 1, mode='sum') for i in
											  multihot_idx] +
											 [nn.Embedding(field_dims[i], 1) for i in onehot_idx])
		for layer in self.last_layer_wide:
			nn.init.normal_(layer.weight, 0, (hidden_units[-1] + sum(field_dims[1:])) ** -0.5)
		self.loss_function = nn.BCELoss()

	def forward(self, x):
		token_emb = self.token_emb_layer(*x[0])  # batch * 768
		emb = torch.cat([token_emb] + [layer(*field) for (field, layer) in zip(x[1:], self.emb_layers)], 1)
		wide = [layer(*field) for field, layer in zip(x[1:], self.last_layer_wide)]  # 11 * batch * 1
		wide = torch.sum(torch.cat(wide, 1), 1, keepdim=True)  # batch * 1
		deep = self.mid_layers(emb)
		out = self.last_layer_deep(deep) + wide
		out = torch.sigmoid(out)
		return out

	def loss(self, logits, labels):
		return self.loss_function(logits, labels.float().to(device))

	@staticmethod
	def transform(x):
		return x

if __name__ == '__main__':
	m = WideDeep(32, [10, 10, 10])
	for name, param in m.named_parameters():
		print(name, param.size())
