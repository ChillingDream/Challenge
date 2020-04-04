import os

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, log_loss, mean_squared_error
from torch.utils.data import DataLoader

from FM import TorchFM
from config import *
from data_loader import TwitterDataset

def compute_prauc(pred, gt):
	prec, recall, thresh = precision_recall_curve(gt, pred)
	prauc = auc(recall, prec)
	return prauc


def calculate_ctr(gt):
	positive = len([x for x in gt if x == 1])
	ctr = positive/float(len(gt))
	return ctr


def compute_rce(pred, gt):
	cross_entropy = log_loss(gt, pred)
	data_ctr = calculate_ctr(gt)
	strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
	return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


def test(model):
	print("Loading test data...")
	test_loader = DataLoader(TwitterDataset(os.path.join(data_path, train_file)),
							 batch_size=1000, shuffle=False, num_workers=4)
	print("Testing...")
	pred = []
	gt = []
	with torch.no_grad():
		for x, y in test_loader:
			gt.extend(y.numpy())
			x, y = x.to(device), y.to(device)
			pred.extend(model(x).squeeze().cpu().numpy())
	pred = np.array(pred, dtype=np.float64)
	pred = np.clip(pred, a_min=1e-15, a_max=1-1e-15)
	mse = mean_squared_error(pred, gt)
	prauc = compute_prauc(pred, gt)
	rce = compute_rce(pred, gt)
	print("mse: ", mse)
	print("prauc: ", prauc)
	print("rce: ", rce)


if __name__ == '__main__':
	model = TorchFM(n=851, k=100)
	model.to(device)
	test(model)
