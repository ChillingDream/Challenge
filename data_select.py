from data_count import features_to_idx
from data_process import all_user_language

path = '/hom2/swp/data/twitter/val.tsv'
# path = 'data/reduced_val.tsv'
out1 = 'data/val1.tsv'
out2 = 'data/val2.tsv'
with open(path, encoding='utf-8') as f1, open(out1, 'w') as f2, open(out2, 'w') as f3:
	lines = f1.readlines()
	for line in lines:
		features = line.strip().split('\x01')
		if features[features_to_idx['engaging_user_id']] in all_user_language:
			f2.write(line)
		else:
			f3.write(line)
