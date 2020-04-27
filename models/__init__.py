from config import *
from models.autoint import AutoInt

# from models.wide_deep import WideDeep

# here set configure the model
if arg.model == 'wide_deep':
	pass
#model = WideDeep(emb_length=32, hidden_units=[128, 64, 32])
elif arg.model == 'autoint':
	model = AutoInt(emb_length=32, num_units=[32, 16], num_heads=[8, 16])
else:
	raise Exception('model required')
