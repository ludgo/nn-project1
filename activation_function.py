import numpy as np

## activation functions & derivatives

def f_linear(x):
	return x

def df_linear(x):
	return 1

def f_sigmoid(x):
	return 1 / (1 + np.exp(-x))

def df_sigmoid(x):
	return f_sigmoid(x) * (1 - f_sigmoid(x))

def f_tanh(x):
	return np.tanh(x)

def df_tanh(x):
	return 1 - np.tanh(x)**2

def f_arctan(x):
	return np.arctan(x)

def df_arctan(x):
	return 1 / (x**2 + 1)

def f_relu(x):
	return np.where(x < 0, 0, x)

def df_relu(x):
	return np.where(x < 0, 0, 1)

def f_prelu( x, alpha=.01):
	return np.where(x < 0, alpha*x, x)

def df_prelu( x, alpha=.01):
	return np.where(x < 0, alpha, 1)

def f_softplus(x):
	return np.log(1 + np.exp(x))

def df_softplus(x):
	return f_sigmoid(x)

def f_softmax(x):
	x -= np.max(x) # adds numerical stability
	exps = np.exp(x)
	return exps / sum(exps)

def df_softmax(x):
	softmax = f_softmax(x)
	return softmax * (1 - softmax)	
