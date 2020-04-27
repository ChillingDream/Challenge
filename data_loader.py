from itertools import islice

import numpy as np
import torch

from config import device, batch_size
from data_process import process
from models.autoint import AutoInt

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

class TwitterDataset():

	def __init__(self, path, trans_func, max_lines=None, token_embedding_level=None, cache_size=1000000, shuffle=False,
				 load_all=False):
		'''
		read data from file
		:param path: path of the data file. The .npy is abolished.
		'''
		self.data = []
		self.current_line = 0
		self.index = 0
		self.transform = trans_func
		self.file = open(path, encoding="utf-8")
		self.max_lines = max_lines
		self.cache_size = cache_size
		self.shuffle = shuffle
		self.token_embedding_level = token_embedding_level
		self.load_all = load_all
		if load_all:
			self._load()

	def _load(self):
		if self.max_lines:
			lines = list(islice(self.file, min([self.cache_size, self.max_lines - self.current_line])))
		else:
			lines = list(islice(self.file, self.cache_size))
		if not lines:
			return False
		self.current_line += len(lines)
		if self.shuffle:
			permuation = np.random.permutation(len(lines))
			lines = [lines[i].split('\x01') for i in permuation]
		else:
			lines = [line.split('\x01') for line in lines]
		self.data = [(self.transform(process(lines[i:i + batch_size], self.token_embedding_level)))
					 for i in range(0, len(lines), batch_size)]
		return True

	def __iter__(self):
		self.index = 0
		if not self.load_all:
			self.current_line = 0
			self.file.seek(0)
			self._load()
		return self

	def __next__(self):
		if self.index >= len(self.data):
			if self.load_all or not self._load():
				raise StopIteration
			self.index = 0
		data = self.data[self.index]
		self.index += 1
		return data

if __name__ == '__main__':
	d = TwitterDataset('data/reduced_training.tsv', AutoInt.transform)
	iter(d)
