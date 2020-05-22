from itertools import islice

import numpy as np

all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
				"tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
				"engaged_with_user_following_count", "engaged_with_user_is_verified",
				"engaged_with_user_account_creation",
				"engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count",
				"engaging_user_is_verified",
				"engaging_user_account_creation", "engagee_follows_engager"]
features_to_idx = dict(zip(all_features, range(len(all_features))))
labels_to_idx = {"reply_timestamp":20, "retweet_timestamp":21, "retweet_with_comment_timestamp":22, "like_timestamp":23}

def data_count(path, val_path=None):
	'''
	collect the statistic of the full data described in the doc.
	'''
	language = {}
	hashtag = {}
	hashtag_count = {}
	engaged_user_count = {}
	engaging_user_count = {}
	user_language = {}
	engaging_user_media = {}
	M_fer = 0
	M_fng = 0
	N = 0
	labels_count = [0, 0, 0, 0]
	max_lines = 1000000
	with open(path, encoding="utf-8") as f:
		while True:
			lines = list(islice(f, max_lines))
			if not lines:
				break
			N += len(lines)
			for line in lines:
				features = line.strip().split('\x01')

				cur_lang = features[features_to_idx['language']]
				user1 = features[features_to_idx['engaged_with_user_id']]
				user2 = features[features_to_idx['engaging_user_id']]
				language[features[features_to_idx['language']]] = language.get(cur_lang, len(language))
				'''
				if user1 not in user_language:
					user_language[user1] = set()
				user_language[user1].add(cur_lang)
				if any(features[-4:]):
					if user2 not in user_language:
						user_language[user2] = set()
					user_language[user2].add(cur_lang)
					if user2 not in engaging_user_media:
						engaging_user_media[user2] = set()
					for media in features[features_to_idx['present_media']].split():
						engaging_user_media[user2].add(media)

					for i in range(4):
						if features[-i - 1]:
							labels_count[i] += 1
				'''
				for tag in features[features_to_idx['hashtags']].split():
					hashtag_count[tag] = hashtag_count.get(tag, 0) + 1
				M_fer = max(M_fer, int(features[features_to_idx['engaged_with_user_follower_count']]))
				M_fer = max(M_fer, int(features[features_to_idx['engaging_user_follower_count']]))
				M_fng = max(M_fng, int(features[features_to_idx['engaged_with_user_following_count']]))
				M_fng = max(M_fng, int(features[features_to_idx['engaging_user_following_count']]))
			print(N)
	print(len(language))
	print(len(hashtag_count))
	print(len(user_language))
	print(len(engaging_user_media))
	print(labels_count)
	print("")

	if val_path:
		tag_recall = 0
		engaged_engaged_recall = 0
		engaged_engaging_recall = 0
		engaging_engaged_recall = 0
		engaging_engaging_recall = 0
		with open(val_path, encoding='utf-8') as f:
			lines = f.readlines()
			for line in lines:
				features = line.strip().split('\x01')

				cur_lang = features[features_to_idx['language']]
				user1 = features[features_to_idx['engaged_with_user_id']]
				language[features[features_to_idx['language']]] = language.get(cur_lang, len(language))
				'''
				if user1 not in user_language:
					user_language[user1] = set()
				user_language[user1].add(cur_lang)
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
				'''
	print(len(user_language))
	print(len(engaging_user_media))

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

if __name__ == '__main__':
	# data_count('/home2/swp/data/twitter/training.tsv', '/home2/swp/data/twitter/val.tsv')
	data_count('data/reduced_training.tsv', 'data/reduced_val.tsv')
