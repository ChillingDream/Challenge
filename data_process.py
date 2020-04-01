import os
from itertools import islice

data_path = '/home2/swp/data/twitter/'
N = 0
max_lines = 10000000
with open(os.path.join(data_path, "training.tsv"), encoding="utf-8") as f:
	while True:
		lines = list(islice(f, max_lines))
		N += len(lines)
		if not lines:
			break
print(N)
