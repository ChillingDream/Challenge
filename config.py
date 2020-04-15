import argparse
import os

import torch

torch.set_num_threads(8)

cont_n = 768 + 4
cate_n = 3 + 4 + 66 + 2 + 2 + 2

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument("--device", default="0", type=str)

arg = parser.parse_args()
device = torch.device("cpu" if arg.device == "cpu" else "cuda:" + arg.device)
# data_dir = '/home2/swp/data/twitter/'
data_dir = 'data/'
log_dir = 'logs/'
checkpoints_dir = 'checkpoints/'
model_name = 'exp1'
train_file = 'toy_training.npy'
test_file = 'toy_val.npy'
epochs = 200

if not os.path.exists(checkpoints_dir):
	os.makedirs(checkpoints_dir)

if not os.path.exists(log_dir):
	os.makedirs(log_dir)
