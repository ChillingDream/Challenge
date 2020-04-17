import time

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import trange

from config import *
from data_loader import TwitterDataset
from models.wide_deep import WideDeep
from test import test

writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
print("Loading training data...")
time.sleep(0.5)
train_data = TwitterDataset(os.path.join(data_dir, train_file), WideDeep.transform)
time.sleep(0.5)
print("Loading validation data...")
test_data = TwitterDataset(os.path.join(data_dir, test_file), WideDeep.transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

model = WideDeep(emb_length=32, hidden_units=[128, 64, 32])  # recommending only change the model here
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
step = 0
if arg.load_checkpoint:
	checkpoint = torch.load(os.path.join(checkpoints_dir, model_name))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	step = checkpoint['step']
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

		if step % 20 == 0:
			test_ce, test_prauc, test_rce = test(model, test_data)
			writer.add_scalars('loss/ce', {'val':test_ce}, step)
			writer.add_scalars('loss/prauc', {'val':test_prauc}, step)
			writer.add_scalars('loss/rce', {'val':test_rce}, step)

	torch.save({'epoch':epoch,
				'model_state_dict':model.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'step':step},
			   os.path.join(checkpoints_dir, model_name))
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
