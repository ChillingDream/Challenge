import os

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from FM import TorchFM
from config import *
from data_loader import TwitterDataset
from test import test

print("Loading training data...")
train_loader = DataLoader(TwitterDataset(os.path.join(data_path, train_file)),
						batch_size=16, shuffle=True, num_workers=4)
model = TorchFM(n=851, k=100)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("Training...")
for epoch in range(epochs):
	losses = []
	for i, data in enumerate(train_loader):
		x, y = data
		x, y = x.to(device), y.float().to(device)
		optimizer.zero_grad()
		logits = model(x).squeeze()
		loss = model.loss(logits, y)
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
	print("Epoch %d finished. mse:%f" % (epoch, np.mean(losses)))

test(model)
