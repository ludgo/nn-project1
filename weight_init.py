import numpy as np

# weight initialization
# *args added to be able to iterate over functions

def w_zeros(dims, *args):
	return np.zeros(dims)

def w_uniform(dims, scale=1.):
	# uniform = random from interval
	return np.random.RandomState(seed=42).uniform(0., scale, size=dims)

def w_gauss(dims, scale=1.):
	# gaussian = normal
	return np.random.RandomState(seed=42).normal(0., scale, size=dims)

def xavier_base(dims):
	return 1 / np.sqrt(dims[1]) # use from-layer dim

def w_xavier(dims, *args):
	return w_uniform(dims) * xavier_base(dims)

def w_relu(dims, *args):
	return w_uniform(dims) * xavier_base(dims) * np.sqrt(2)

def w_custom(dims, *args):
	return w_uniform(dims) * np.sqrt(2 / np.sum(dims)) # sum from-layer and to-layer
