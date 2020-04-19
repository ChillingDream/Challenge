import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange, tqdm

from config import device
from data_process import process

class DataPrefetcher():
	'''
	useless at present
	'''
	def __init__(self, loader):
		self.loader = loader
		self.stream = torch.cuda.Stream()
		self.preload()

	def preload(self):
		try:
			self.batch = next(self.loader)
		except StopIteration:
			self.batch = None
			return
		with torch.cuda.stream(self.stream):
			for k in self.batch:
				if k != 'meta':
					self.batch[k] = self.batch[k].to(device=device, non_blocking=True)

	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		batch = self.batch
		self.preload()
		return batch

class TwitterDataset(Dataset):

	def __init__(self, path, trans_func, max_entries=None, token_embedding_level='sentence'):
		'''
		read data from file
		:param path: path of the data file. The raw file ends with .tsv while processed file ends with .npy.
		'''
		self.X = []
		self.Y = []
		self.transform = trans_func
		if os.path.splitext(path)[1] == '.tsv':
			with open(path, encoding="utf-8") as file:
				if max_entries:
					lines = file.readlines()[:max_entries]
				else:
					lines = file.readlines()
				lines = [line.split('\x01') for line in lines]
				stride = 100
				for i in trange(0, len(lines), stride):
					data = process(lines[i:i + stride], token_embedding_level)
					for entry in data:
						self.X.append(self.transform(entry[:-4]))
						self.Y.append(torch.tensor(entry[-4:]))
		else:
			data = np.load(path, allow_pickle=True)
			if max_entries:
				data = data[:max_entries]
			for entry in tqdm(data):
				self.X.append(self.transform(entry[:-4]))
				self.Y.append(torch.tensor(list(entry[-4:])))

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, item):
		return self.X[item], self.Y[item][0]  # only predict retweet at present
