from itertools import islice

import numpy as np
import torch
import torch.multiprocessing

from config import device, batch_size, val_size
from data_process import process
from models.autoint import AutoInt

def load_data(file_path, queue, offset, stride, cache_size, use_user_info, shuffle=True, drop_val=True):
	f = open(file_path, encoding="utf-8")
	list(islice(f, offset * cache_size))
	while True:
		lines = list(islice(f, cache_size))
		if drop_val and len(lines) < cache_size:
			lines = lines[:-val_size]
		if not lines:
			f.seek(0)
			list(islice(f, offset * cache_size))
			lines = list(islice(f, cache_size))
		if shuffle:
			permuation = np.random.permutation(len(lines))
			lines = [lines[i].strip().split('\x01') for i in permuation]
		else:
			lines = [line.strip().split('\x01') for line in lines]
		for i in range(0, len(lines), batch_size):
			queue.put(process(lines[i:i + batch_size], None, use_user_info))
		list(islice(f, (stride - 1) * cache_size))
		use_user_info ^= 1

class TwitterDataset():

	def __init__(self, path, max_lines=None, token_embedding_level=None, cache_size=1000000, use_user_info=True,
				 shuffle=False, drop_val=False, n_workers=0):
		'''
		read the data from the disk file
		:param path: the full path of the file
		:param max_lines: if set, dataset will only read the first max_lines lines of the file.
		:param token_embedding_level: the token_embedding_level for process
		:param cache_size: how many lines will one load read in
		:param shuffle: whether to random shuffle a cache
		:param n_workers: how many processes used to process data. if set 0, dataset will not use multiprocessing
		'''
		self.data = []
		self.current_line = 0
		self.index = 0
		self.file = open(path, encoding="utf-8")
		self.max_lines = max_lines
		self.cache_size = cache_size
		self.shuffle = shuffle
		self.token_embedding_level = token_embedding_level
		self.load_all = False
		self.drop_val = drop_val
		self.use_user_info = use_user_info
		self.n_workers = n_workers
		self.processors = []
		with open(path, encoding='utf-8') as f:
			self.__len = sum(1 for _ in f)
		if self.__len <= cache_size * max([1, n_workers]):
			self.load_all = True
		if self.load_all:
			self._load(True)
		elif n_workers > 0:
			self.queue = torch.multiprocessing.Manager().Queue(100)
			for i in range(n_workers):
				self.processors.append(torch.multiprocessing.Process(target=load_data, args=(path,
																							 self.queue,
																							 i,
																							 n_workers,
																							 cache_size,
																							 shuffle,
																							 drop_val)))
				self.processors[-1].daemon = True
				self.processors[-1].start()

	def __len__(self):
		return self.__len

	def _load(self, use_user_info):
		'''
		load cache_size lines from the last ending
		:return: whether the file ends
		'''
		if self.max_lines:
			lines = list(islice(self.file, min([self.cache_size, self.max_lines - self.current_line])))
		else:
			lines = list(islice(self.file, self.cache_size))
			if self.drop_val and len(lines) < self.cache_size:
				lines = lines[:-val_size]
		if not lines:
			return False
		self.current_line += len(lines)
		if self.shuffle:
			permuation = np.random.permutation(len(lines))
			lines = [lines[i].strip().split('\x01') for i in permuation]
		else:
			lines = [line.strip().split('\x01') for line in lines]
		self.data = [(process(lines[i:i + batch_size], self.token_embedding_level, use_user_info))
					 for i in range(0, len(lines), batch_size)]
		return True

	def __iter__(self):
		self.index = 0
		if not self.load_all and self.n_workers == 0:
			self.current_line = 0
			self.file.seek(0)
			self._load(self.use_user_info)
		return self

	def __next__(self):
		if self.n_workers > 0 and not self.load_all:
			x, y = self.queue.get()
		else:
			if self.index >= len(self.data):
				if self.load_all or not self._load(self.use_user_info):
					raise StopIteration
				self.index = 0
			x, y = self.data[self.index]
			self.index += 1
		for i in range(len(x)):
			for j in range(len(x[i])):
				x[i][j] = x[i][j].to(device)
		return x, y

	def close(self):
		if self.n_workers > 0:
			for p in self.processors:
				p.terminate()
				p.join()


if __name__ == '__main__':
	d = TwitterDataset('data/reduced_training.tsv', AutoInt.transform)
	iter(d)
