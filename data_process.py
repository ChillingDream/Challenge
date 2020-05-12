from itertools import islice

from pytorch_pretrained_bert import BertModel
from torch import LongTensor
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
			   features_to_idx['engaging_user_is_verified'], features_to_idx['engagee_follows_engager'],
			   features_to_idx['engaging_user_id']]
labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23};
follow_intervals = 5
multihot_idx = [1, 2, 12, 13]
onehot_idx = [3, 4, 5, 6, 7, 8, 9, 10, 11]
field_dims = [768, 480, 3, 4, 66, follow_intervals, follow_intervals, 2, follow_intervals, follow_intervals, 2, 2, 66,
			  3]
max_length = 64

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
	max_lines = 1000000
	with open(path, encoding="utf-8") as f:
		while True:
			lines = list(islice(f, max_lines))
			if not lines:
				break
			N += len(lines)
			for line in lines:
				features = line.split('\x01')
				cur_lang = features[features_to_idx['language']]
				user1 = features[features_to_idx['engaged_with_user_id']]
				user2 = features[features_to_idx['engaging_user_id']]
				language[features[features_to_idx['language']]] = language.get(cur_lang, len(language))
				if user1 not in user_language:
					user_language[user1] = set()
				if user2 not in user_language:
					user_language[user2] = set()
				user_language[user1].add(cur_lang)
				user_language[user2].add(cur_lang)

				if user2 not in engaging_user_media:
					engaging_user_media[user2] = set()
				for media in features[features_to_idx['present_media']].split():
					engaging_user_media[user2].add(media)

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
	np.savez('statistic.npz', N=N, hashtag=hashtag, language=language, M_fer=M_fer, M_fng=M_fng,
			 user_language=user_language, engaging_user_media=engaging_user_media)

bert = BertModel.from_pretrained('./bert-base-multilingual-cased')
word_embeddings = bert.embeddings.word_embeddings.weight.data.to(device)
statistic = np.load('statistic.npz', allow_pickle=True)
all_type = dict(zip(['Retweet', 'Quote', 'Reply', 'TopLevel'], range(4)))
all_media = dict(zip(['Photo', 'Video', 'GIF'], range(3)))
all_hashtag = statistic['hashtag'][()]
all_language = statistic['language'][()]
all_user_language = statistic['user_language'][()]
all_engaging_user_media = statistic['engaging_user_media'][()]
LM_fer = np.log(statistic['M_fer'] + 1) + 1
LM_fng = np.log(statistic['M_fng'] + 1) + 1

def process(entries, token_embedding_level, use_user_info=True):
	'''
	process multiple lines including token embedding computing and average pooling, onehot encoding and numerical
	normalization.
	:param entries: the lines to be processed
	:param token_embedding_level: one of sentence, word and none. sentence means using bert forwarding; work means only
	using the word embedding; none means feed the token id into the model and make embedding inside the model.
	:return: (x,y). x is a batch of 12 tensors, each representing a feature. y is a batch of labels
	'''

	batch = len(entries)
	features = [[] for i in range(len(feature_idx))]
	for line in entries:
		for i in range(len(feature_idx)):
			features[i].append(line[feature_idx[i]])
	tokens = features[0]
	tokens = [[int(token) for token in line.split()] for line in tokens]
	if token_embedding_level == 'sentence':
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
		'''
		cur_tokens = 0
		indices = []  # the indices indicating the start of each batch, as nn.EmbeddingBag requires
		sentence_embedding = []  # the token id of all the batch as a 1-d tensor
		for line in tokens:
			indices.append(cur_tokens)
			cur_tokens += len(line)
			sentence_embedding.extend(line)
		sentence_embedding = [LongTensor(sentence_embedding).to(device), LongTensor(indices).to(device)]
		'''
		sentence_embedding = []
		for line in tokens:
			if len(line) >= max_length:
				sentence_embedding.append(line[:max_length])
			else:
				sentence_embedding.append(line + [0] * (max_length - len(line)))
		sentence_embedding = [torch.tensor(sentence_embedding).to(device)]
	else:
		raise Exception('wrong token embedding level')

	indices = []  # similar to sentence embedding
	values = []  # similar to sentence embedding
	cur = 0
	for tags in features[1]:
		indices.append(cur)
		tags = tags.split()
		for tag in tags:
			if tag in all_hashtag:
				values.append(all_hashtag[tag])
				cur += 1
	hashtags = [LongTensor(values).to(device), LongTensor(indices).to(device)]

	indices = []
	values = []
	cur = 0
	for media in features[2]:
		indices.append(cur)
		media = media.split()
		for j, m in enumerate(all_media.keys()):
			if m in media:
				values.append(j)
				cur += 1
	medias = [LongTensor(values).to(device), LongTensor(indices).to(device)]

	tweet_types = [LongTensor([all_type[type] for type in features[3]]).to(device)]
	languages = [LongTensor([all_language[language] for language in features[4]]).to(device)]

	fer1 = torch.tensor([int(x) for x in features[5]], dtype=torch.float)
	fer1 = [((fer1 + 1).log() / LM_fer * follow_intervals).long().to(device)]

	fng1 = torch.tensor([int(x) for x in features[6]], dtype=torch.float)
	fng1 = [((fng1 + 1).log() / LM_fng * follow_intervals).long().to(device)]

	fer2 = torch.tensor([int(x) for x in features[8]], dtype=torch.float)
	fer2 = [((fer2 + 1).log() / LM_fer * follow_intervals).long().to(device)]

	fng2 = torch.tensor([int(x) for x in features[9]], dtype=torch.float)
	fng2 = [((fng2 + 1).log() / LM_fng * follow_intervals).long().to(device)]

	engaged_verified = [torch.tensor([int(x == 'true') for x in features[7]]).to(device)]
	engaging_verified = [torch.tensor([int(x == 'true') for x in features[10]]).to(device)]
	follow = [torch.tensor([int(x == 'true') for x in features[11]]).to(device)]

	if use_user_info:
		indices = []
		values = []
		cur = 0
		for user in features[12]:
			indices.append(cur)
			if user in all_user_language:
				for lang in all_user_language[user]:
					values.append(all_language[lang])
					cur += 1
		user_languages = [LongTensor(values).to(device), LongTensor(indices).to(device)]

		indices = []
		values = []
		cur = 0
		for user in features[12]:
			indices.append(cur)
			if user in all_engaging_user_media:
				for media in all_engaging_user_media[user]:
					values.append(all_media[media])
					cur += 1
		engaing_user_media = [LongTensor(values).to(device), LongTensor(indices).to(device)]
	else:
		user_languages = [LongTensor([]).to(device), LongTensor([0] * batch).to(device)]
		engaing_user_media = [LongTensor([]).to(device), LongTensor([0] * batch).to(device)]

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
			follow,
			user_languages,
			engaing_user_media], \
		   torch.tensor([bool(entries[i][-label_to_pred]) for i in range(batch)]) if len(entries[0]) > len(
			   all_features) else None

def process_mp(entries, token_embedding_level, use_user_info):
	'''
	the multiprocessing version of the process
	'''

	batch = len(entries)
	features = [[] for i in range(len(feature_idx))]
	for line in entries:
		for i in range(len(feature_idx)):
			features[i].append(line[feature_idx[i]])
	tokens = features[0]
	tokens = [[int(token) for token in line.split()] for line in tokens]
	if token_embedding_level == 'sentence':
		attention_mask = []
		weight_mask = []
		for line in tokens:
			l = len(line)
			line += [0] * (max_length - l)
			attention_mask.append([1] * l + [0] * (max_length - l))
			weight_mask.append(np.array(attention_mask[-1]) / l)

		tokens = torch.tensor(tokens)
		segments = torch.ones_like(tokens)
		attention_mask = torch.tensor(attention_mask)
		weight_mask = torch.tensor(weight_mask, dtype=torch.float32)
		bert.eval()
		with torch.no_grad():
			layers, _ = bert(tokens, segments, attention_mask)
		sentence_embedding = (torch.sum(layers[11] * weight_mask.unsqueeze(2), 1)).cpu()
	elif token_embedding_level == 'word':
		sentence_embedding = torch.stack([torch.mean(word_embeddings[line], 0) for line in tokens])
	elif token_embedding_level == None:
		'''
		cur_tokens = 0
		indices = []
		sentence_embedding = []
		for line in tokens:
			indices.append(cur_tokens)
			cur_tokens += len(line)
			sentence_embedding.extend(line)
		sentence_embedding = [LongTensor(sentence_embedding), LongTensor(indices)]
		'''
		sentence_embedding = []
		for line in tokens:
			if len(line) >= max_length:
				sentence_embedding.append(line[:max_length])
			else:
				sentence_embedding.append(line + [0] * (max_length - len(line)))
		sentence_embedding = [torch.tensor(sentence_embedding)]
	else:
		raise Exception('wrong token embedding level')

	indices = []
	values = []
	cur = 0
	for i, tags in enumerate(features[1]):
		indices.append(cur)
		tags = tags.split()
		for tag in tags:
			if tag in all_hashtag:
				values.append(all_hashtag[tag])
				cur += 1
	hashtags = [LongTensor(values), LongTensor(indices)]

	indices = []
	values = []
	cur = 0
	for i, media in enumerate(features[2]):
		indices.append(cur)
		media = media.split()
		for j, m in enumerate(['Photo', 'Video', 'Gif']):
			if m in media:
				values.append(j)
				cur += 1
	medias = [LongTensor(values), LongTensor(indices)]

	tweet_types = [LongTensor([all_type[type] for type in features[3]])]
	languages = [LongTensor([all_language[language] for language in features[4]])]

	fer1 = torch.tensor([int(x) for x in features[5]], dtype=torch.float)
	fer1 = [((fer1 + 1).log() / LM_fer * follow_intervals).long()]

	fng1 = torch.tensor([int(x) for x in features[6]], dtype=torch.float)
	fng1 = [((fng1 + 1).log() / LM_fng * follow_intervals).long()]

	fer2 = torch.tensor([int(x) for x in features[8]], dtype=torch.float)
	fer2 = [((fer2 + 1).log() / LM_fer * follow_intervals).long()]

	fng2 = torch.tensor([int(x) for x in features[9]], dtype=torch.float)
	fng2 = [((fng2 + 1).log() / LM_fng * follow_intervals).long()]

	engaged_verified = [torch.tensor([int(x == 'true') for x in features[7]])]
	engaging_verified = [torch.tensor([int(x == 'true') for x in features[10]])]
	follow = [torch.tensor([int(x == 'true') for x in features[11]])]

	if use_user_info:
		indices = []
		values = []
		cur = 0
		for user in features[12]:
			indices.append(cur)
			if user in all_user_language:
				for lang in all_user_language[user]:
					values.append(all_language[lang])
					cur += 1
		user_languages = [LongTensor(values), LongTensor(indices)]

		indices = []
		values = []
		cur = 0
		for user in features[12]:
			indices.append(cur)
			if user in all_engaging_user_media:
				for media in all_engaging_user_media[user]:
					values.append(all_media[media])
					cur += 1
		engaing_user_media = [LongTensor(values), LongTensor(indices)]
	else:
		user_languages = [LongTensor([]), LongTensor([0] * batch)]
		engaing_user_media = [LongTensor([]), LongTensor([0] * batch)]
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
			follow,
			user_languages,
			engaing_user_media], \
		   torch.tensor([bool(entries[i][-label_to_pred]) for i in range(batch)])

def raw2npy(file):
	'''
	process the raw data and store as a npy file.
	:param file: the file name without prefix directory
	'''
	data = []
	with open(file) as f:
		lines = f.readlines()
		lines = [line.split('\x01') for line in lines]
		stride = 100
		for i in trange(0, len(lines), stride):
			data += process(lines[i:i + stride])

#	np.save(os.path.join(data_dir, os.path.splitext(file)[0]), data)

if __name__ == '__main__':
	data_count('data/toy_training.tsv')
