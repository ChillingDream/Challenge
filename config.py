import argparse
import os

import torch

torch.set_num_threads(8)

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument("--device", default="0", type=str)
parser.add_argument("--log_dir", default="logs/", type=str)
parser.add_argument("--model_name", default="exp1", type=str)
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight_decay", default=1e-5, type=float)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--batch", default=200, type=int)
parser.add_argument("--data_name", default='toy', type=str)

arg = parser.parse_args()
device = torch.device("cpu" if arg.device == "cpu" else "cuda:" + arg.device)
# data_dir = '/home2/swp/data/twitter/'
data_dir = 'data/'
checkpoints_dir = 'checkpoints/'
model_name = arg.model_name
log_dir = os.path.join(arg.log_dir, model_name)
train_file = arg.data_name + '_training.npy'
test_file = 'toy_val.npy'
epochs = arg.epoch
lr = arg.lr
weight_decay = arg.weight_decay
drop_rate = arg.dropout
batch_size = arg.batch

if not os.path.exists(checkpoints_dir):
	os.makedirs(checkpoints_dir)

if not os.path.exists(log_dir):
	os.makedirs(log_dir)
