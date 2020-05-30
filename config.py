import argparse
import os

import numpy as np
import torch

torch.set_num_threads(1)
np.random.seed(14159)

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument("--device", default="0", type=str)
parser.add_argument("--log_dir", default="logs", type=str)
parser.add_argument("--model", choices=['autoint', 'widedeep'])
parser.add_argument("--model_name", default="exp1", type=str)
parser.add_argument("--epoch", default=20, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--weight_decay", default=1e-6, type=float)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--batch", default=2000, type=int)
parser.add_argument("--data_name", default='reduced', type=str)
parser.add_argument("--val_size", default=2000, type=int)
parser.add_argument("--n_workers", default=4, type=int)
parser.add_argument("--label", default='like', choices=['retweet', 'reply', 'like', 'comment'])
parser.add_argument("--save_latest", action='store_true')
parser.add_argument("--make_prediction", action='store_true')
load_parser = parser.add_mutually_exclusive_group()
load_parser.add_argument("--load_latest", action='store_true')
load_parser.add_argument("--load_best", action='store_true')

arg = parser.parse_args()
if arg.device == "cpu":
	device = torch.device("cpu")
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = arg.device
	device = torch.device("cuda:0")
checkpoints_dir = 'checkpoints'
model_name = arg.model_name
log_dir = os.path.join(arg.log_dir, model_name)
if arg.data_name == 'all':
	train_file = '/home2/swp/data/twitter/training.tsv'
	test_file = 'data/val.tsv'
else:
	train_file = 'data/' + arg.data_name + '_training.tsv'
	test_file = 'data/' + arg.data_name + '_val.tsv'
epochs = arg.epoch
lr = arg.lr
weight_decay = arg.weight_decay
drop_rate = arg.dropout
batch_size = arg.batch
val_size = arg.val_size
load_checkpoint = None
label_to_pred = dict(zip(['like', 'comment', 'retweet', 'reply'], range(1, 5)))[arg.label]
save_latest = arg.save_latest
n_workers = arg.n_workers
make_prediction = arg.make_prediction
if arg.load_latest:
	load_checkpoint = 'latest'
if arg.load_best:
	load_checkpoint = 'best'

if not os.path.exists(checkpoints_dir):
	os.makedirs(checkpoints_dir)

if not os.path.exists(log_dir):
	os.makedirs(log_dir)
