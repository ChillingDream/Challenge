import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import device
from data_process import cont_idx, field_dims

class Flatten(nn.Module):
	def __init__(self, start_dim=1, end_dim=-1):
		super().__init__()
		self.start_dim = start_dim
		self.end_dim = end_dim

	def forward(self, x):
		return torch.flatten(x, self.start_dim, self.end_dim)

class MutiheadAttention(nn.Module):
	def __init__(self, num_in, num_out, num_head):
		super().__init__()
		self.num_out = num_out
		self.queries = nn.Linear(num_in, num_out * num_head, bias=False)
		self.keys = nn.Linear(num_in, num_out * num_head, bias=False)
		self.values = nn.Linear(num_in, num_out * num_head, bias=False)
		self.res_layer = nn.Linear(num_in, num_out * num_head, bias=False)

	def forward(self, x):
		batch = x.size()[0]
		queries = torch.cat(self.queries(x).split(self.num_out, -1), 0)
		keys = torch.cat(self.keys(x).split(self.num_out, -1), 0)
		values = torch.cat(self.values(x).split(self.num_out, -1), 0)
		weights = queries.matmul(keys.permute([0, 2, 1]))  # QK^T
		weights = F.softmax(weights / self.num_out ** 0.5, -1)
		outputs = weights.matmul(values)
		outputs = torch.cat(outputs.split(batch, 0), 2)
		outputs = F.relu(outputs + self.res_layer(x))
		return outputs

class AutoInt(nn.Module):
	def __init__(self, emb_length, num_units, num_heads):
		super().__init__()
		self.emb_layers = nn.ModuleList(
			[nn.Linear(field_dims[i], emb_length, bias=False) for i in range(len(field_dims))])
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
		emb = torch.stack([layer(field.to(device)) for (field, layer) in zip(x, self.emb_layers)], 1)
		return self.layers(emb)

	def loss(self, logits, labels):
		return self.loss_function(logits, labels.float().to(device))

	@staticmethod
	def transform(x):
		return [torch.tensor(x[i] if i in cont_idx else x[i] / max((np.sum(x[i]), 1.))).float() for i in
				range(len(field_dims))]

if __name__ == '__main__':
	m = AutoInt(32, [10], [8])
	for name, param in m.named_parameters():
		print(name, param.size())
