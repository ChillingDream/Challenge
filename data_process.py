import os

data_path = '/home2/swp/data/twitter/'
N = 0
max_lines = 100000
with open(os.path.join(data_path, "training.tsv"), encoding="utf-8") as f:
	while True:
		lines = f.readlines(max_lines)
		N += len(lines)
		if not N:
			break
print(N)
