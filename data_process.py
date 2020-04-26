from itertools import islice

import numpy as np
from pytorch_pretrained_bert import BertModel
from torch import sparse
from tqdm import trange

from config import *

all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                 "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
                "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified",
                "enaging_user_account_creation", "engagee_follows_engager"]
features_to_idx = dict(zip(all_features, range(len(all_features))))
feature_idx = [features_to_idx['text_tokens'], features_to_idx['hashtags'], features_to_idx['present_media'],
			   features_to_idx['tweet_type'], features_to_idx['language'],
			   features_to_idx['engaged_with_user_follower_count'],
			   features_to_idx['engaged_with_user_following_count'], features_to_idx['engaged_with_user_is_verified'],
			   features_to_idx['engaging_user_follower_count'], features_to_idx['engaging_user_following_count'],
			   features_to_idx['engaging_user_is_verified'], features_to_idx['engagee_follows_engager']]
labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23};
follow_intervals = 5
cate_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
cont_idx = [0]
field_dims = [768, 480, 3, 4, 66, follow_intervals, follow_intervals, 2, follow_intervals, follow_intervals, 2, 2]
cate_n = sum([field_dims[i] for i in cate_idx])
cont_n = sum([field_dims[i] for i in cont_idx])

def data_count(path, val_path=None):
	'''
	collect the statistic of the full data described in the doc.
	'''
	language = {}
	hashtag = {}
	hashtag_count = {}
	engaged_user_count = {}
	engaging_user_count = {}
	M_fer = 0
	M_fng = 0
	N = 0
	max_lines = 1000000
	with open(path, encoding="utf-8") as f:
		while True:
			lines = list(islice(f, max_lines))
			if not lines:
				break
			N += len(lines)
			for line in lines:
				features = line.split('\x01')
				language[features[features_to_idx['language']]] = \
					language.get(features[features_to_idx['language']], len(language))
				for tag in features[features_to_idx['hashtags']].split():
					hashtag_count[tag] = hashtag_count.get(tag, 0) + 1
				engaged_user_count[features[features_to_idx['engaged_with_user_id']]] = \
					engaged_user_count.get(features[features_to_idx['engaged_with_user_id']], 0) + 1
				engaging_user_count[features[features_to_idx['engaging_user_id']]] = \
					engaging_user_count.get(features[features_to_idx['engaging_user_id']], 0) + 1
				M_fer = max(M_fer, int(features[features_to_idx['engaged_with_user_follower_count']]))
				M_fer = max(M_fer, int(features[features_to_idx['engaging_user_follower_count']]))
				M_fng = max(M_fng, int(features[features_to_idx['engaged_with_user_following_count']]))
				M_fng = max(M_fng, int(features[features_to_idx['engaging_user_following_count']]))
			print(N)
	print(len(language))
	print(len(hashtag_count))

	if val_path:
		tag_recall = 0
		engaged_engaged_recall = 0
		engaged_engaging_recall = 0
		engaging_engaged_recall = 0
		engaging_engaging_recall = 0
		with open(val_path, encoding='utf-8') as f:
			lines = f.readlines()
			for line in lines:
				features = line.split('\x01')
				for tag in features[features_to_idx['hashtags']].split():
					if tag in hashtag_count:
						tag_recall += 1
				if features[features_to_idx['engaged_with_user_id']] in engaged_user_count:
					engaged_engaged_recall += 1
				if features[features_to_idx['engaged_with_user_id']] in engaging_user_count:
					engaged_engaging_recall += 1
				if features[features_to_idx['engaging_user_id']] in engaged_user_count:
					engaging_engaged_recall += 1
				if features[features_to_idx['engaging_user_id']] in engaging_user_count:
					engaging_engaging_recall += 1
		print(tag_recall)
		print(engaged_engaged_recall, engaged_engaging_recall)
		print(engaging_engaged_recall, engaging_engaging_recall)

	hashtag_count = sorted(hashtag_count.items(), key=lambda x:x[1], reverse=True)
	engaged_user_count = sorted(engaged_user_count.items(), key=lambda x:x[1], reverse=True)
	engaging_user_count = sorted(engaging_user_count.items(), key=lambda x:x[1], reverse=True)
	with open('hashtag_count.txt', 'w') as f:
		f.writelines(['%s %d\n' % (tag, count) for tag, count in hashtag_count])
	with open('engaged_user_count.txt', 'w') as f:
		f.writelines(['%s %d\n' % (id, count) for id, count in engaged_user_count])
	with open('engaging_user_count.txt', 'w') as f:
		f.writelines(['%s %d\n' % (id, count) for id, count in engaging_user_count])
	for tag, _ in hashtag_count[:480]:
		hashtag[tag] = hashtag.get(tag, len(hashtag))
	np.savez('statistic.npz', N=N, hashtag=hashtag, language=language, M_fer=M_fer, M_fng=M_fng)

bert = BertModel.from_pretrained('./bert-base-multilingual-cased')
word_embeddings = bert.embeddings.word_embeddings.weight.data.to(device)
statistic = np.load('statistic.npz', allow_pickle=True)
all_hashtag = statistic['hashtag'][()]
all_language = statistic['language'][()]
LM_fer = np.log(statistic['M_fer'] + 1) + 1
LM_fng = np.log(statistic['M_fng'] + 1) + 1

def process(entries, token_embedding_level):
	'''
	process multiple lines including token embedding computing and average pooling, onehot encoding and numerical
	normalization.
	:param entries: the lines to be processed
	:return: a list of processed line, each item as a list consisting of 16 numpy array described in the doc.
	'''

	def to_sparse(indices, values, size):
		if indices:
			return sparse.FloatTensor(torch.tensor(indices).t(), torch.tensor(values, dtype=torch.float32),
									  torch.Size(size))
		return sparse.FloatTensor(torch.Size(size))

	batch = len(entries)
	features = [[] for i in range(len(feature_idx))]
	for line in entries:
		for i in range(len(feature_idx)):
			features[i].append(line[feature_idx[i]])
	tokens = features[0]
	tokens = [[int(token) for token in line.split()] for line in tokens]
	if token_embedding_level == 'sentence':
		max_length = max([len(line) for line in tokens])
		attention_mask = []
		weight_mask = []
		for line in tokens:
			l = len(line)
			line += [0] * (max_length - l)
			attention_mask.append([1] * l + [0] * (max_length - l))
			weight_mask.append(np.array(attention_mask[-1]) / l)

		tokens = torch.tensor(tokens).to(device)
		segments = torch.ones_like(tokens).to(device)
		attention_mask = torch.tensor(attention_mask).to(device)
		weight_mask = torch.tensor(weight_mask, dtype=torch.float32).to(device)
		bert.to(device)
		bert.eval()
		with torch.no_grad():
			layers, _ = bert(tokens, segments, attention_mask)
		sentence_embedding = (torch.sum(layers[11] * weight_mask.unsqueeze(2), 1)).cpu()
	elif token_embedding_level == 'word':
		sentence_embedding = torch.stack([torch.mean(word_embeddings[line], 0) for line in tokens])
	elif token_embedding_level == None:
		cur_tokens = 0
		indices = []
		sentence_embedding = []
		for line in tokens:
			indices.append(cur_tokens)
			cur_tokens += len(line)
			sentence_embedding.extend(line)
		sentence_embedding = (torch.tensor(sentence_embedding), torch.tensor(indices))
	else:
		raise Exception('wrong token embedding level')

	indices = []
	values = []
	for i, tags in enumerate(features[1]):
		tags = tags.split()
		tags = [tag for tag in tags if tag in all_hashtag]
		for tag in tags:
			indices.append([i, all_hashtag[tag]])
			values.append(1 / len(tags))
	hashtags = to_sparse(indices, values, [batch, len(all_hashtag)])

	indices = []
	values = []
	for i, media in enumerate(features[2]):
		media = media.split()
		cnt = 0
		for j, m in enumerate(['Photo', 'Video', 'Gif']):
			if m in media:
				cnt += 1
				indices.append([i, j])
		if cnt:
			values += [1 / cnt] * cnt
	medias = to_sparse(indices, values, [batch, 3])

	indices = []
	values = []
	for i, tweet_type in enumerate(features[3]):
		for j, t in enumerate(['Retweet', 'Quote', 'Reply', 'TopLevel']):
			if t == tweet_type:
				indices.append([i, j])
				values.append(1)
				break
	tweet_types = to_sparse(indices, values, [batch, 4])

	indices = [[i, all_language[language]] for i, language in enumerate(features[4])]
	values = [1] * batch
	languages = to_sparse(indices, values, [batch, len(all_language)])

	fer1 = np.array([[int(x)] for x in features[5]])
	fer1 = np.trunc(np.log(fer1 + 1) / LM_fer * follow_intervals).astype(np.int32)
	fer1 = to_sparse(list(zip(range(batch), fer1)), [1] * batch, [batch, follow_intervals])

	fng1 = np.array([[int(x)] for x in features[6]])
	fng1 = np.trunc(np.log(fng1 + 1) / LM_fng * follow_intervals).astype(np.int32)
	fng1 = to_sparse(list(zip(range(batch), fng1)), [1] * batch, [batch, follow_intervals])

	fer2 = np.array([[int(x)] for x in features[8]])
	fer2 = np.trunc(np.log(fer2 + 1) / LM_fer * follow_intervals).astype(np.int32)
	fer2 = to_sparse(list(zip(range(batch), fer2)), [1] * batch, [batch, follow_intervals])

	fng2 = np.array([[int(x)] for x in features[9]])
	fng2 = np.trunc(np.log(fng2 + 1) / LM_fng * follow_intervals).astype(np.int32)
	fng2 = to_sparse(list(zip(range(batch), fng2)), [1] * batch, [batch, follow_intervals])

	engaged_verified = torch.eye(2, dtype=torch.float32)[
		[int(x == 'true') for x in features[7]]]
	engaging_verified = torch.eye(2, dtype=torch.float32)[
		[int(x == 'true') for x in features[10]]]
	follow = torch.eye(2, dtype=torch.float32)[
		[int(x == 'true') for x in features[11]]]

	return [sentence_embedding,
			hashtags,
			medias,
			tweet_types,
			languages,
			fer1,
			fng1,
			engaged_verified,
			fer2,
			fng2,
			engaging_verified,
			follow], \
		   torch.tensor([bool(entries[i][-label_to_pred]) for i in range(batch)])

def raw2npy(file):
	'''
	process the raw data and store as a npy file.
	:param file: the file name without prefix directory
	'''
	data = []
	with open(os.path.join(data_dir, file)) as f:
		lines = f.readlines()
		lines = [line.split('\x01') for line in lines]
		stride = 100
		for i in trange(0, len(lines), stride):
			data += process(lines[i:i + stride])

#	np.save(os.path.join(data_dir, os.path.splitext(file)[0]), data)

if __name__ == '__main__':
#data_count(os.path.join(data_dir, 'training.tsv'), os.path.join(data_dir, 'val.tsv'))
	raw2npy('toy_training.tsv')
	raw2npy('toy_val.tsv')
#raw2npy('reduced_training.tsv')
#raw2npy('reduced_val.tsv')
	exit(0)
	with open(os.path.join(data_dir, "toy_training.tsv"), encoding="utf-8") as f:
		lines = f.readlines(100000)
		entries = [line.split('\x01') for line in lines]
		data = process(entries, None)
		print(data[0][0])
# np.save('data.npy', data)
# data = np.load('data.npy', allow_pickle=True)
#print(data)
