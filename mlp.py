import abc

import numpy as np

from util import *
from activation_function import *
from weight_init import *


# Multi-layer perceptron
class MLP():

	def __init__(self, dim_in, dim_out, dims_hid, act_hid='sigmoid', dist='uniform', dist_scale=1.):
		
		assert(len(dims_hid) > 0)
		self.dim_in = dim_in
		self.dim_out = dim_out
		self.dims_hid = dims_hid

		self.act_hid = act_hid

		self.init_weights(dist, dist_scale)

	def init_weights(self, dist, scale):
		# fill based on distribution
		init_functions = { 'zeros': w_zeros, 'uniform': w_uniform, 'gauss': w_gauss, 'xavier': w_xavier, 'relu': w_relu, 'custom': w_custom }

		self.weights = []
		bias = 1 # standard, 1 bias cell per layer
		# input to hidden
		self.weights.append(init_functions[dist]((self.dims_hid[0], self.dim_in + bias), scale))
		# hidden to hidden
		for i in range(len(self.dims_hid) - 1):
			self.weights.append(init_functions[dist]((self.dims_hid[i+1], self.dims_hid[i] + bias), scale))
		# hidden to output
		self.weights.append(init_functions[dist]((self.dim_out, self.dims_hid[-1] + bias), scale))

	# activation function on hidden layer
	def f_hid(self, x):
		if self.act_hid == 'linear':
			return f_linear(x)
		if self.act_hid == 'sigmoid':
			return f_sigmoid(x)
		if self.act_hid == 'tanh':
			return f_tanh(x)
		if self.act_hid == 'arctan':
			return f_arctan(x)
		if self.act_hid == 'relu':
			return f_relu(x)
		if self.act_hid == 'prelu':
			return f_prelu(x)
		if self.act_hid == 'softplus':
			return f_softplus(x)
		if self.act_hid == 'softmax':
			return f_softmax(x)
		raise Exception('Unknown activation function.')

	# derivative of activation function on hidden layer
	def df_hid(self, x):
		if self.act_hid == 'linear':
			return df_linear(x)
		if self.act_hid == 'sigmoid':
			return df_sigmoid(x)
		if self.act_hid == 'tanh':
			return df_tanh(x)
		if self.act_hid == 'arctan':
			return df_arctan(x)
		if self.act_hid == 'relu':
			return df_relu(x)
		if self.act_hid == 'prelu':
			return df_prelu(x)
		if self.act_hid == 'softplus':
			return df_softplus(x)
		if self.act_hid == 'softmax':
			return df_softmax(x)
		raise Exception('Unknown activation function.')

	# activation function on output layer
	@abc.abstractmethod
	def f_out(self, x):
		pass

	# derivative of activation function on output layer
	@abc.abstractmethod
	def df_out(self, x):
		pass

	# forward propagation
	def forward(self, x):

		logits = [] # before activation function
		layers = [x] # after activation function

		n = len(self.weights)
		for i in range(n):
			# forward pass
			logits.append(self.weights[i] @ augment(layers[-1]))
			# apply activation function
			if i != n-1:
				layers.append(self.f_hid(logits[-1]))
			else:
				layers.append(self.f_out(logits[-1]))

		return layers, logits

	# forward AND backpropagation
	def backward(self, x, d, alpha):
		layers, logits = self.forward(x)

		# backward pass (last = output layer)
		backs = [(d - layers[-1]) * self.df_out(logits[-1])]
		dWs = [np.outer(backs[0], np.transpose(augment(layers[-2])))] # delta weights

		for i in range(len(self.weights) - 1, 0, -1):
			# backward pass
			backs.insert(0, (np.transpose(self.weights[i])[:-1] @ backs[0]) * self.df_hid(logits[i - 1]))
			dWs.insert(0, np.outer(backs[0], np.transpose(augment(layers[i - 1]))))

		# update weights
		for i,w in enumerate(self.weights):
			w += alpha * dWs[i]

		return layers[-1]


class MLPClassifier(MLP):
	'''
	Nomenclature:
	CE = classification error
	RE = regression error
	'''

	# cost function
	def cost(self, targets, outputs):
		# sum of squares
		return np.sum((targets - outputs)**2, axis=0)

	def f_out(self, x):
		# softmax since we suppose multi-class classification
		return f_softmax(x)

	def df_out(self, x):
		return df_softmax(x)

	def train(self, inputs, labels, alpha=.1, epochs=100, trace=False, trace_interval=10, alpha_drop_each=5, alpha_drop_rate=.5):
		(_, count) = inputs.shape
		targets = onehot_encode(labels, self.dim_out)

		if trace:
			ion()

		CEs = []
		REs = []

		for epoch in range(epochs):
			print('Ep {:3d}/{}: '.format(epoch+1, epochs), end='', flush=True)
			CE = 0
			RE = 0

			for i in np.random.RandomState(seed=42).permutation(count):
				x = inputs[:,i]
				d = targets[:,i]

				y = self.backward(x, d, alpha)

				CE += labels[i] != onehot_decode(y)
				RE += self.cost(d,y)

			CE /= count
			RE /= count

			CEs.append(CE)
			REs.append(RE)

			print('CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))

			if trace and ((epoch+1) % trace_interval == 0):
				_, predicted = self.predict(inputs)
				plot_dots(inputs, labels, predicted, block=False)
				plot_both_errors(CEs, REs, block=False)
				redraw()

			# step decay on learning rate
			if alpha_drop_each and (epoch + 1) % alpha_drop_each == 0:
				alpha *= alpha_drop_rate

		if trace:
			ioff()

		return CEs, REs

	def predict(self, inputs):
		layers, _ = self.forward(inputs) # OR np.stack([self.forward(x)[0] for x in inputs.T]) # if self.forward() cannot take a whole batch
		outputs = layers[-1]
		return outputs, onehot_decode(outputs)

	def test(self, inputs, labels):
		targets = labels
		outputs, predicted = self.predict(inputs)
		CE = np.sum(targets != predicted) / len(targets)
		RE = np.sum(self.cost(targets, outputs)) / len(targets) # mean squared error
		return CE, RE
