import time

import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, log_loss

from config import *
from data_count import all_features
from data_loader import TwitterDataset
from models import model

def compute_prauc(pred, gt):
	prec, recall, thresh = precision_recall_curve(gt, pred)
	prauc = auc(recall, prec)
	return prauc


def calculate_ctr(gt):
	positive = len([x for x in gt if x == 1])
	ctr = positive/float(len(gt))
	return ctr


def compute_rce(pred, gt):
	cross_entropy = log_loss(gt, pred, labels=[0, 1])
	data_ctr = calculate_ctr(gt)
	strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
	return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

def test(model, dataset=None, compute_metric=True):
	model.eval()
	if dataset:
		test_loader = dataset
	else:
		time.sleep(0.5)
		print("Loading test data...")
		time.sleep(0.5)
		test_loader = TwitterDataset(test_file, cache_size=20000000)
	if not dataset:
		time.sleep(0.5)
		print("Testing...")
		time.sleep(0.5)
	pred = []
	gt = []
	with torch.no_grad():
		for x, y in test_loader:
			if compute_metric:
				gt.extend(y.numpy())
			pred.extend(model(x).squeeze().cpu().numpy())
	if not compute_metric:
		return pred
	pred = np.array(pred, dtype=np.float64)
	pred = np.clip(pred, a_min=1e-15, a_max=1-1e-15)
	ce = log_loss(gt, pred)
	prauc = compute_prauc(pred, gt)
	rce = compute_rce(pred, gt)
	return ce, prauc, rce


if __name__ == '__main__':
	checkpoint = torch.load(os.path.join(checkpoints_dir, model_name + '_best.pt'))
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	if make_prediction:
		data = pd.read_csv(test_file, sep='\x01', header=None, names=all_features, encoding='utf-8')
		pred = test(model, compute_metric=False)
		print('prediction finished')
		pred = pd.concat([data[['tweet_id', 'engaging_user_id']], pd.DataFrame({'prediction':pred})], 1)
		pred.to_csv(arg.label + '_prediction.csv', header=False, index=False)
	else:
		ce, prauc, rce = test(model)
		print("ce: ", ce)
		print("prauc: ", prauc)
		print("rce: ", rce)
