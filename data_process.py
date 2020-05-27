import time

from pytorch_pretrained_bert import BertModel
from torch import LongTensor
from tqdm import trange

from config import *
from data_count import features_to_idx

feature_idx = [features_to_idx['text_tokens'], features_to_idx['hashtags'], features_to_idx['present_media'],
			   features_to_idx['tweet_type'], features_to_idx['language'],
			   features_to_idx['engaged_with_user_follower_count'],
			   features_to_idx['engaged_with_user_following_count'], features_to_idx['engaged_with_user_is_verified'],
			   features_to_idx['engaging_user_follower_count'], features_to_idx['engaging_user_following_count'],
			   features_to_idx['engaging_user_is_verified'], features_to_idx['engagee_follows_engager'],
			   features_to_idx['engaging_user_id'], features_to_idx['engaged_with_user_account_creation'],
			   features_to_idx['tweet_timestamp']]
follow_intervals = 5
multihot_idx = [1, 2, 12, 13]
onehot_idx = [3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]
field_dims = [768, 480, 3, 4, 66, follow_intervals, follow_intervals, 2, follow_intervals, follow_intervals, 2, 2, 66,
			  3, 24]


bert = BertModel.from_pretrained('./bert-base-multilingual-cased')
word_embeddings = bert.embeddings.word_embeddings.weight.data.to(device)
statistic = np.load('statistic.npz', allow_pickle=True)
all_type = dict(zip(['Retweet', 'Quote', 'Reply', 'TopLevel'], range(4)))
all_media = dict(zip(['Photo', 'Video', 'GIF'], range(3)))
all_hashtag = statistic['hashtag'][()]
all_language = statistic['language'][()]
all_user_language = statistic['user_language'][()] if 'user_language' in statistic else {}
all_engaging_user_media = statistic['engaging_user_media'][()] if 'engaging_user_media' in statistic else {}
LM_fer = np.log(statistic['M_fer'] + 1) + 1
LM_fng = np.log(statistic['M_fng'] + 1) + 1

def process(entries, token_embedding_level, use_user_info=True):
	'''
	process multiple lines including token embedding computing and average pooling, onehot encoding and numerical
	normalization.
	:param entries: the lines to be processed
	:param token_embedding_level: one of sentence, word and none. sentence means using bert forwarding; work means only
	using the word embedding; none means feed the token id into the model and make embedding inside the model.
	:return: (x,y). x is a batch of tensors, each representing a feature. y is a batch of labels
	'''

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
		cur_tokens = 0
		indices = []
		sentence_embedding = []
		for line in tokens:
			indices.append(cur_tokens)
			cur_tokens += len(line)
			sentence_embedding.extend(line)
		sentence_embedding = [LongTensor(sentence_embedding), LongTensor(indices)]
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

	tweet_time = [LongTensor([time.localtime(int(timestamp))[3] for timestamp in features[13]])]

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
			engaing_user_media,
			tweet_time], \
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
	print(word_embeddings.size())
