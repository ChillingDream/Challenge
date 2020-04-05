import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange

from data_process import process

class TwitterDataset(Dataset):

	def __init__(self, path, from_raw=True):
		self.X = []
		self.Y = []
		if from_raw:
			with open(path, encoding="utf-8") as file:
				lines = file.readlines()
				lines = [line.split('\x01') for line in lines]
				stride = 100
				for i in trange(0, len(lines), stride):
					data = process(lines[i:i + stride])
					for entry in data:
						self.X.append(torch.tensor(np.concatenate(entry[:-4])).float())
						self.Y.append(torch.tensor(entry[-4:]))
		else:
			data = np.load(path, allow_pickle=True)
			for entry in data:
				self.X.append(torch.tensor(np.concatenate(entry[:-4])).float())
				self.Y.append(torch.tensor(list(entry[-4:])))

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, item):
		return self.X[item], self.Y[item][0]

