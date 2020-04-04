import argparse

import torch

torch.set_num_threads(8)

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument("--device", default="0", type=str)

arg = parser.parse_args()
device = torch.device("cpu" if arg.device == "cpu" else "cuda:" + arg.device)
#data_path = '/home2/swp/data/twitter/'
data_path = 'data/'
train_file = 'toy_training.tsv'
test_file = 'toy_val.tsv'
epochs = 200
