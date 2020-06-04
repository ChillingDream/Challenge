import sys
import time

import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import trange

from config import *
from data_loader import TwitterDataset
from models import model
from test import test, log_loss, compute_rce, compute_prauc

def calc_score(prauc, rce):
	return prauc * 100 + rce

print("Loading training data...")
time.sleep(0.5)
if arg.data_name == 'all':
	train_loader = TwitterDataset(train_file, shuffle=True, cache_size=1000000, use_user_info=use_user_info,
								  n_workers=n_workers, drop_val=True)
else:
	train_loader = TwitterDataset(train_file, shuffle=True, cache_size=100000, use_user_info=use_user_info,
								  n_workers=n_workers)
time.sleep(0.5)
print("Loading validation data...")
test_loader = TwitterDataset(test_file, val_size, use_user_info=use_user_info)
print("%d entries has been loaded" % len(train_loader))

print("Preparing model...")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# sheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=100, min_lr=1e-5,
#												threshold=1e-4, threshold_mode='rel', cooldown=0)
step = 0
max_score = (0, -1e10, 0)

if load_checkpoint:
	checkpoint = torch.load(
		os.path.join(checkpoints_dir, model_name + ('_best.pt' if load_checkpoint == 'best' else '_latest.pt')))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# sheduler.load_state_dict(checkpoint['sheduler_state_dict'])
	step = checkpoint['step']
	max_score = checkpoint['max_score']
	print("Checkpoint has been loaded.")
else:
	if not sys.platform.startswith('win'):
		os.system('rm -rf ' + os.path.join(log_dir, '*'))
	else:
		os.system('rmdir /s/q' + log_dir)
writer = SummaryWriter(log_dir, flush_secs=300)

print("Training...")
time.sleep(0.5)
batches_per_epoch = (len(train_loader) - 1) // batch_size + 1
record_interval = 200
for epoch in range(epochs):
	pred = []
	gt = []
	iteration = trange(batches_per_epoch)
	for t, data in zip(iteration, train_loader):
		model.train()
		step += 1
		x, y = data
		gt.extend(y.clone().numpy())
		optimizer.zero_grad()
		logits = model(x).squeeze()
		pred.extend(logits.clone().detach().cpu().numpy())
		loss = model.loss(logits, y)
		loss.backward()
		optimizer.step()
		del x, y
		iteration.set_description("Epoch %2d train loss:%f" % (epoch + 1, loss.clone().detach().cpu()))

		if step % record_interval == record_interval - 1:
			test_ce, test_prauc, test_rce = test(model, test_loader)
			writer.add_scalars('loss/ce', {'val':test_ce}, step)
			writer.add_scalars('loss/prauc', {'val':test_prauc}, step)
			writer.add_scalars('loss/rce', {'val':test_rce}, step)
			writer.add_scalars('lr', {'lr':optimizer.state_dict()['param_groups'][0]['lr']}, step)
			# sheduler.step(test_ce)

			if calc_score(test_prauc, test_rce) > calc_score(max_score[0], max_score[1]):
				max_score = (test_prauc, test_rce, step)
				torch.save({'model_state_dict':model.state_dict(),
							'optimizer_state_dict':optimizer.state_dict(),
							# 'sheduler_state_dict':sheduler.state_dict(),
							'step':step,
							'max_score':max_score},
						   os.path.join(checkpoints_dir, model_name + '_best.pt'))

	if save_latest:
		torch.save({'model_state_dict':model.state_dict(),
					'optimizer_state_dict':optimizer.state_dict(),
					'step':step,
					#'sheduler_state_dict':sheduler.state_dict(),
					'max_score':max_score},
				   os.path.join(checkpoints_dir, model_name + '_latest.pt'))
	# for name, param in model.named_parameters():
	#	writer.add_histogram(name, param.data.clone().detach().cpu().numpy(), epoch)
	pred = np.array(pred, dtype=np.float64)
	train_ce = log_loss(gt, pred)
	train_prauc = compute_prauc(pred, gt)
	train_rce = compute_rce(pred, gt)
	writer.add_scalars('loss/ce', {'train':train_ce}, step)
	writer.add_scalars('loss/prauc', {'train':train_prauc}, step)
	writer.add_scalars('loss/rce', {'train':train_rce}, step)
	writer.flush()
train_loader.close()
writer.close()

ce, prauc, rce = test(model)
print("ce: ", ce)
print("prauc: ", prauc)
print("rce: ", rce)
print("The best performance is achieved at step %d" % max_score[2])
print("prauc: ", max_score[0])
print("rce: ", max_score[1])
