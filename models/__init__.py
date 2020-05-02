from config import *
from models.autoint import AutoInt
from models.wide_deep import WideDeep

# here set configure the model
if arg.model == 'widedeep':
	model = WideDeep(emb_length=32, hidden_units=[128, 64, 32])
elif arg.model == 'autoint':
	model = AutoInt(emb_length=8, num_units=[8, 8, 8], num_heads=[2, 2, 2])
else:
	raise Exception('model required')
