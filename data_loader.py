import torch
from torch.utils.data import Dataset

from data_process import process
from tqdm import tqdm

class TwitterDataset(Dataset):

	def __init__(self, path):
		self.X = []
		self.Y = []
		with open(path, encoding="utf-8") as file:
			lines = file.readlines()
			for line in tqdm(lines):
				entry = process(line.split('\x01'))
				self.X.append(torch.cat(entry[:-4]).float())
				self.Y.append(torch.tensor(entry[-4:]))

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, item):
		return self.X[item], self.Y[item][0]

