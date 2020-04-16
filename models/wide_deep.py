import numpy as np
import torch
from torch import nn

from config import device
from data_process import cate_idx, cont_idx, cont_n, cate_n, field_dims

class WideDeep(nn.Module):
	def __init__(self, emb_length, hidden_units):
		super().__init__()
		self.emb_layers = nn.ModuleList([nn.Linear(field_dims[i], emb_length, bias=False) for i in cate_idx])
		hidden_units = [cont_n + len(cate_idx) * emb_length] + hidden_units
		layers = []
		for i in range(len(hidden_units) - 1):
			layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
			layers.append(nn.ReLU(True))
			if i != len(hidden_units) - 2:
				layers.append(nn.Dropout(0.5))
		self.mid_layers = nn.Sequential(*layers)
		self.last_layer = nn.Linear(hidden_units[-1] + cate_n, 1)
		self.loss_function = nn.BCELoss()

	def forward(self, x):
		cont_x, cate_x = x
		cont_x = cont_x.to(device)
		emb = torch.cat([layer(cx.to(device)) for (cx, layer) in zip(cate_x, self.emb_layers)], 1)
		cate_x = torch.cat(cate_x, 1).to(device)
		net = torch.cat([cont_x, emb], 1)
		net = self.mid_layers(net)
		net = torch.cat([cate_x, net], 1)
		net = torch.sigmoid(self.last_layer(net))
		return net

	def loss(self, logits, labels):
		return self.loss_function(logits, labels.float().to(device))

	@staticmethod
	def transform(x):
		cate_x = [torch.tensor(x[i]).float() for i in cate_idx]
		cont_x = [x[i] for i in cont_idx]
		cont_x = torch.tensor(np.concatenate(cont_x)).float()
		return cont_x, cate_x

if __name__ == '__main__':
	m = WideDeep(32, [10, 10, 10])
	for name, param in m.named_parameters():
		print(name, param.size())
