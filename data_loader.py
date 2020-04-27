import time
from itertools import islice

import numpy as np
import torch
import torch.multiprocessing

from config import batch_size
from data_process import process
from models.autoint import AutoInt

torch.multiprocessing.set_sharing_strategy('file_system')

def load_data(file_path, queue, offset, stride, cache_size, shuffle=True):
	f = open(file_path, encoding="utf-8")
	list(islice(f, offset * cache_size))
	while True:
		lines = list(islice(f, cache_size))
		if not lines:
			f.seek(0)
			list(islice(f, offset * cache_size))
			lines = list(islice(f, cache_size))
		if shuffle:
			permuation = np.random.permutation(len(lines))
			lines = [lines[i].split('\x01') for i in permuation]
		else:
			lines = [line.split('\x01') for line in lines]
		for i in range(0, len(lines), batch_size):
			queue.put(process(lines[i:i + batch_size], None))
		list(islice(f, (stride - 1) * cache_size))

class TwitterDataset():

	def __init__(self, path, trans_func, max_lines=None, token_embedding_level=None, cache_size=100000, shuffle=False,
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
		self.queue = torch.multiprocessing.Manager().Queue(1000)
		if load_all:
			self._load()
		else:
			n_workers = 4
			self.processors = []
			for i in range(n_workers):
				self.processors.append(torch.multiprocessing.Process(target=load_data, args=(path,
																							 self.queue,
																							 i,
																							 n_workers,
																							 cache_size)))
				self.processors[-1].daemon = True
				self.processors[-1].start()

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
		for i in range(0, len(lines), batch_size):
			self.data.append(self.transform(process(lines[i:i + batch_size], self.token_embedding_level)))
		return True

	def __iter__(self):
		self.index = 0
		return self

	def __next__(self):
		if not self.load_all:
			return self.queue.get()
		if self.index >= len(self.data):
			if self.load_all or not self._load():
				raise StopIteration
			self.index = 0
		data = self.data[self.index]
		self.index += 1
		return data

	def close(self):
		for p in self.processors:
			p.terminate()
			p.join()

if __name__ == '__main__':
	d = TwitterDataset('data/reduced_training.tsv', AutoInt.transform)
	time.sleep(5)
	print(d.queue.qsize())
	st = time.time()
	for i in range(100):
		d.queue.get()
	print(time.time() - st)
	d.close()
