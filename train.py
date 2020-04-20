import time

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import trange

from config import *
from data_loader import TwitterDataset
from models import model
from test import test

def calc_score(prauc, rce):
	return prauc * 100 + rce

writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
print("Loading training data...")
time.sleep(0.5)
train_data = TwitterDataset(os.path.join(data_dir, train_file), model.transform)
time.sleep(0.5)
print("Loading validation data...")
test_data = TwitterDataset(os.path.join(data_dir, test_file), model.transform, val_size)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
sheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=100, min_lr=1e-5,
												threshold=1e-4, threshold_mode='rel', cooldown=0)
step = 0
max_score = (0, -1e10, 0)

if load_checkpoint:
	checkpoint = torch.load(
		os.path.join(checkpoints_dir, model_name + ('best.pt' if load_checkpoint == 'best' else 'latest.pt')))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	sheduler.load_state_dict(checkpoint['sheduler_state_dict'])
	step = checkpoint['step']
	max_score = checkpoint['max_score']
	print("Checkpoint loaded.")

print("Training...")
os.system('rm -rf ' + os.path.join(log_dir, '*'))
time.sleep(0.5)
iteration = trange(epochs)
for epoch in iteration:
	model.train()
	for data in train_loader:
		step += 1
		x, y = data
		optimizer.zero_grad()
		logits = model(x).squeeze()
		loss = model.loss(logits, y)
		loss.backward()
		optimizer.step()

		if step % 20 == 19:
			test_ce, test_prauc, test_rce = test(model, test_data)
			writer.add_scalars('loss/ce', {'val':test_ce}, step)
			writer.add_scalars('loss/prauc', {'val':test_prauc}, step)
			writer.add_scalars('loss/rce', {'val':test_rce}, step)
			writer.add_scalars('lr', {'lr':optimizer.state_dict()['param_groups'][0]['lr']}, step)
			sheduler.step(test_ce)
			if calc_score(test_prauc, test_rce) > calc_score(max_score[0], max_score[1]):
				max_score = (test_prauc, test_rce, step)
				torch.save({'model_state_dict':model.state_dict(),
							'optimizer_state_dict':optimizer.state_dict(),
							'sheduler_state_dict':sheduler.state_dict(),
							'step':step,
							'max_score':max_score},
						   os.path.join(checkpoints_dir, model_name + '_best.pt'))

	torch.save({'model_state_dict':model.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'step':step,
				'sheduler_state_dict':sheduler.state_dict(),
				'max_score':max_score},
			   os.path.join(checkpoints_dir, model_name + '_latest.pt'))
	train_ce, train_prauc, train_rce = test(model, train_data)
	writer.add_scalars('loss/ce', {'train':train_ce}, step)
	writer.add_scalars('loss/prauc', {'train':train_prauc}, step)
	writer.add_scalars('loss/rce', {'train':train_rce}, step)
	iteration.set_description("train loss:%f" % train_ce)
	writer.flush()

ce, prauc, rce = test(model)
print("ce: ", ce)
print("prauc: ", prauc)
print("rce: ", rce)
print("The best performance is achieved at step %d" % max_score[2])
print("prauc: ", max_score[0])
print("rce: ", max_score[1])
