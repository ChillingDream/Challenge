import os
from itertools import islice

import numpy as np
from pytorch_pretrained_bert import BertModel

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
	with open(os.path.join(data_path, "training.tsv"), encoding="utf-8") as file:
		while True:
			lines = list(islice(file, max_lines))
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

def process(features):
	tokens = features[features_to_idx['text_tokens']]
	tokens = [[int(token) for token in tokens.split()]]
	segments = [[1] * len(tokens)]
	tokens = torch.tensor(tokens)
	segments = torch.tensor(segments)
	bert.eval()
	with torch.no_grad():
		layers, _ = bert(tokens, segments)
	sentence_embedding = torch.mean(layers[11], 1)[0]

	media = torch.tensor([0., 0., 0.])
	if 'Thoto' in features[features_to_idx['present_media']]:
		media[0] = 1
	if 'Video' in features[features_to_idx['present_media']]:
		media[1] = 1
	if 'Gif' in features[features_to_idx['present_media']]:
		media[2] = 1

	tweet_type = torch.tensor([0., 0., 0., 0.])
	if "Retweet" == features[features_to_idx['tweet_type']]:
		tweet_type[0] = 1
	elif "Quote" == features[features_to_idx['tweet_type']]:
		tweet_type[1] = 1
	elif "Reply" == features[features_to_idx['tweet_type']]:
		tweet_type[2] = 1
	elif "Toplevel" == features[features_to_idx['tweet_type']]:
		tweet_type[3] = 1

	language = torch.zeros((len(all_language)), dtype=torch.float32)
	language[all_language[features[features_to_idx['language']]]] = 1

	normed_fer1 = torch.tensor([np.log(int(features[features_to_idx['engaged_with_user_follower_count']]) + 1) / LM_fer])
	normed_fng1 = torch.tensor([np.log(int(features[features_to_idx['engaged_with_user_following_count']]) + 1) / LM_fng])
	normed_fer2 = torch.tensor([np.log(int(features[features_to_idx['engaging_user_follower_count']]) + 1) / LM_fer])
	normed_fng2 = torch.tensor([np.log(int(features[features_to_idx['engaging_user_following_count']]) + 1) / LM_fng])

	engaged_verified = torch.tensor([0., 0.])
	if features[features_to_idx['engaged_with_user_is_verified']]:
		engaged_verified[1] = 1
	else:
		engaged_verified[0] = 1

	engaging_verified = torch.tensor([0., 0.])
	if features[features_to_idx['engaging_user_is_verified']]:
		engaging_verified[1] = 1
	else:
		engaging_verified[0] = 1

	follow = torch.tensor([0., 0.])
	if features[features_to_idx['engagee_follows_engager']]:
		follow[1] = 1
	else:
		follow[0] = 1

	return [sentence_embedding, media, tweet_type, language, normed_fer1, normed_fng1, engaged_verified,
			normed_fer2, normed_fng2, engaging_verified, follow,
			bool(features[-4]), bool(features[-3]), bool(features[-2]), bool(features[-1])]

if __name__ == '__main__':
	with open(os.path.join(data_path, "training.tsv"), encoding="utf-8") as f:
		lines = f.readlines(1000000)
		for line in lines:
			features = line.split('\x01')
			process(features)

