import os
from itertools import islice

import numpy as np
import torch
from pytorch_pretrained_bert import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                 "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
                "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified",
                "enaging_user_account_creation", "engagee_follows_engager"]
features_to_idx = dict(zip(all_features, range(len(all_features))))
labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23};
data_path = '/home2/swp/data/twitter/'
#data_path = '../'

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

bert = BertModel.from_pretrained('bert-base-multilingual-cased')

def process(data):
	tokens = data[features_to_idx['text_tokens']]
	segments = [1] * len(tokens)
	tokens = torch.tensor(tokens, device=device)
	segments = torch.tensor(segments, device=device)
	bert.eval()
	with torch.no_grad():
		layers, _ = bert(tokens, segments)
	sentence_embedding = torch.mean(layers[11], 1)

	return sentence_embedding

with open(os.path.join(data_path, "training.tsv"), encoding="utf-8") as f:
	lines = f.readlines(10)
	for line in lines:
		features = line.split('\x01')
		print(process(features))

