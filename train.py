import time

import numpy as np
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
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

model = WideDeep(cont_n=cont_n,
				 cate_n=cate_n,
				 emb_length=32,
				 hidden_units=[128, 64, 32])  # recommending only change the model here
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

print("Training...")
os.system('rm -rf ' + os.path.join(log_dir, '*'))
time.sleep(0.5)
iter = trange(epochs)
for epoch in iter:
	model.train()
	losses = []
	for i, data in enumerate(train_loader):
		x, y = data
		optimizer.zero_grad()
		logits = model(x).squeeze()
		loss = model.loss(logits, y)
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
	if epoch % 10 == 9:
		torch.save({'epoch':epoch,
					'model_state_dict':model.state_dict(),
					'optimizer_state_dict':optimizer.state_dict()},
				   os.path.join(checkpoints_dir, model_name))
	if epoch % 5 == 4:
		train_mse, train_prauc, train_rce = test(model, train_data)
		test_mse, test_prauc, test_rce = test(model, test_data)
		writer.add_scalars('loss/mse', {'train':train_mse, 'val':test_mse}, epoch)
		writer.add_scalars('loss/prauc', {'train':train_prauc, 'val':test_prauc}, epoch)
		writer.add_scalars('loss/rce', {'train':train_rce, 'val':test_rce}, epoch)
	else:
		train_mse = np.mean(losses)
	iter.set_description("train mse:%f" % train_mse)

writer.flush()
mse, prauc, rce = test(model)
print("mse: ", mse)
print("prauc: ", prauc)
print("rce: ", rce)
