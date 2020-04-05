import os
from itertools import islice

import numpy as np
from pytorch_pretrained_bert import BertModel
from tqdm import trange

from config import *

all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                 "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
                "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified",
                "enaging_user_account_creation", "engagee_follows_engager"]
features_to_idx = dict(zip(all_features, range(len(all_features))))
labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23};


def data_count():
	language = dict()
	hashtag = dict()
	M_fer = 0
	M_fng = 0
	N = 0
	max_lines = 1000000
	with open(os.path.join(data_path, "training.tsv"), encoding="utf-8") as f:
		while True:
			lines = list(islice(f, max_lines))
			if not lines:
				break
			N += len(lines)
			for line in lines:
				features = line.split('\x01')
				language[features[features_to_idx['language']]] = language.get(features[features_to_idx['language']], len(language))
				for tag in features[features_to_idx['hashtags']].split():
					hashtag[tag] = hashtag.get(tag, len(hashtag))
				M_fer = max(M_fer, int(features[features_to_idx['engaged_with_user_follower_count']]))
				M_fer = max(M_fer, int(features[features_to_idx['engaging_user_follower_count']]))
				M_fng = max(M_fng, int(features[features_to_idx['engaged_with_user_following_count']]))
				M_fng = max(M_fng, int(features[features_to_idx['engaging_user_following_count']]))
			print(N)
	print(len(language))
	print(len(hashtag))
	np.savez('statistic.npz', N=N, language=language, M_fer=M_fer, M_fng=M_fng)

bert = BertModel.from_pretrained('./bert-base-multilingual-cased')
statistic = np.load('statistic.npz', allow_pickle=True)
all_language = statistic['language'][()]
LM_fer = np.log(statistic['M_fer'] + 1)
LM_fng = np.log(statistic['M_fng'] + 1)

def process(entries):
	entries = np.array(entries)
	tokens = entries[:, features_to_idx['text_tokens']]
	tokens = [[int(token) for token in line.split()] for line in tokens]
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
	sentence_embedding = (torch.sum(layers[11] * weight_mask.unsqueeze(2), 1)).cpu().numpy()

	medias = []
	for media in entries[:, features_to_idx['present_media']]:
		medias.append(np.array([0., 0., 0.]))
		for i, m in enumerate(['Photo', 'Video', 'Gif']):
			if m == media:
				medias[-1][i] = 1

	tweet_types = []
	for tweet_type in entries[:, features_to_idx['tweet_type']]:
		for i, t in enumerate(['Retweet', 'Quote', 'Reply', 'TopLevel']):
			if t == tweet_type:
				tweet_types.append(np.eye(4)[i])
				break

	languages = np.zeros((len(entries), len(all_language)))
	for i, language in enumerate(entries[:, features_to_idx['language']]):
		languages[i][all_language[language]] = 1

	fer1 = np.array([[int(x)] for x in entries[:, features_to_idx['engaged_with_user_follower_count']]])
	normed_fer1 = (np.log(fer1 + 1) / LM_fer).astype(np.float32)
	fng1 = np.array([[int(x)] for x in entries[:, features_to_idx['engaged_with_user_following_count']]])
	normed_fng1 = (np.log(fng1 + 1) / LM_fng).astype(np.float32)
	fer2 = np.array([[int(x)] for x in entries[:, features_to_idx['engaging_user_follower_count']]])
	normed_fer2 = (np.log(fer2 + 1) / LM_fer).astype(np.float)
	fng2 = np.array([[int(x)] for x in entries[:, features_to_idx['engaging_user_following_count']]])
	normed_fng2 = (np.log(fng2 + 1) / LM_fng).astype(np.float)

	engaged_verified = np.eye(2)[
		[int(x == 'true') for x in entries[:, features_to_idx['engaged_with_user_is_verified']]]]
	engaging_verified = np.eye(2)[
		[int(x == 'true') for x in entries[:, features_to_idx['engaging_user_is_verified']]]]
	follow = np.eye(2)[[int(x == 'true') for x in entries[:, features_to_idx['engagee_follows_engager']]]]

	return [[sentence_embedding[i], medias[i], tweet_types[i], languages[i], normed_fer1[i], normed_fng1[i],
			 engaged_verified[i],
			 normed_fer2[i], normed_fng2[i], engaging_verified[i], follow[i],
			 bool(entries[i][-4]), bool(entries[i][-3]), bool(entries[i][-2]), bool(entries[i][-1])]
			for i in range(len(entries))]

def raw2npy(file):
	data = []
	with open(os.path.join(data_path, file)) as f:
		lines = f.readlines()
		lines = [line.split('\x01') for line in lines]
		stride = 100
		for i in trange(0, len(lines), stride):
			data += process(lines[i:i + stride])
	np.save(os.path.join(data_path, os.path.splitext(file)[0]), data)

if __name__ == '__main__':
	raw2npy('toy_training.tsv')
	exit(0)
	with open(os.path.join(data_path, "toy_training.tsv"), encoding="utf-8") as f:
		lines = f.readlines(100000)
		entries = [line.split('\x01') for line in lines]
		data = process(entries)
		np.save('data.npy', data)
		data = np.load('data.npy', allow_pickle=True)
		print(data)
