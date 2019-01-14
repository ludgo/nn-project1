import random
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from mlp import MLPClassifier
from util import *
from plot import plot_table


# TODO can be changed to sys.argv easily if necessary
ARG_DIMS_HID = [20]
ARG_ACTIVATION = 'sigmoid'
ARG_WEIGHT_INIT = 'uniform'
ARG_WEIGHT_SCALE = .1
ARG_ALPHA = .5
ARG_ALPHA_DROP_EACH = 5
ARG_ALPHA_DROP_RATE = .5
ARG_EPOCHS = 50
ARG_TRAINSET = os.path.join('.', 'data', '2d.trn.dat')
ARG_TESTSET = os.path.join('.', 'data', '2d.tst.dat')
ARG_NORMALIZE = True


char_to_int = {'A': 0, 'B': 1, 'C': 2}
means, devs = None, None

def load_dataset(file_path, skiprows=1, reuse_norm=False):
	data = np.loadtxt(file_path, skiprows=skiprows, dtype=str).T

	X = data[:-1].astype(float)
	y = data[-1]

	#plot_dots(X, labels=y, save_path=os.path.join('.', 'output', 'train.png')) # single set with labels

	# normalize
	if ARG_NORMALIZE:
		global means
		global devs
		if not reuse_norm:
			means, devs = np.mean(X, axis=1), np.std(X, axis=1)
		for row in range(X.shape[0]):
			X[row] = (X[row] - means[row]) / devs[row]

		#plot_dots(X, labels=y, save_path=os.path.join('.', 'output', 'train_normalized.png')) # single set with labels, normalized

	# represent letters as integers
	#char_to_int = dict((c, i) for i, c in enumerate(np.unique(y))) # make sure train & test do not differ!!
	y = pd.DataFrame(y).applymap(lambda i: char_to_int[i]).values.squeeze()

	return X, y


def nn(X_train, y_train, X_test, y_test, conf_matrix=False):

	y_unique = len(np.unique(y_train))
	model = MLPClassifier(dim_in=X_train.shape[0], dim_out=y_unique, dims_hid=ARG_DIMS_HID, act_hid=ARG_ACTIVATION, dist=ARG_WEIGHT_INIT, dist_scale=ARG_WEIGHT_SCALE)
	trainCEs, trainREs = model.train(X_train, y_train, alpha=ARG_ALPHA, epochs=ARG_EPOCHS, trace=False, alpha_drop_each=ARG_ALPHA_DROP_EACH, alpha_drop_rate=ARG_ALPHA_DROP_RATE)

	trainCE, trainRE = model.test(X_train, y_train)
	testCE, testRE = model.test(X_test, y_test)

	_, y_pred  = model.predict(X_test)

	if conf_matrix:
		# export confussion matrix
		cm = confusion_matrix(y_test, y_pred, labels=list(char_to_int.values()))
		print(cm)
		file_path = os.path.join('.', 'output', 'confusion_matrix.csv')
		with open(file_path, 'w') as f:
			for k in char_to_int.keys():
				f.write('\t{}'.format(k))
			f.write('\n')
			for k,row in zip(char_to_int.keys(), range(len(char_to_int.keys()))):
				f.write(k)
				for i in cm[row]:
					f.write('\t{}'.format(i))
				f.write('\n')
		plot_table(file_path, (6, 1))

	#plot_dots(X_train, None, None, X_test, y_test, y_pred, save_path=os.path.join('.', 'output', 'predict.png')) # predicted
	#plot_dots(X_train, None, None, X_test, y_test, None, save_path=os.path.join('.', 'output', 'test.png')) # reality
	#plot_both_errors(trainCEs, trainREs, testCE, testRE, save_path=os.path.join('.', 'output', 'errors.png'))

	return trainCE, testCE


def nn_kfold(X, y, k=10):
	assert(k > 1)
	indices = list(range(X.shape[1]))
	random.seed(42)
	random.shuffle(indices)
	indices_k = list(chunk(indices, len(indices) // k))
	CEs = []
	for indices_test in indices_k:
		indices_train = np.delete(indices, indices_test)
		X_train = X[:, indices_train]
		X_test = X[:, indices_test]
		y_train = y[indices_train]
		y_test = y[indices_test]
		CEs.append(nn(X_train, y_train, X_test, y_test))
		print('\tfold CE = {:6.2%}'.format(CEs[-1][1]))
	return np.mean(CEs, axis=0)


def cross_validation(path):
	X, y = load_dataset(path)
	estimationCE, validationCE = nn_kfold(X, y)
	print('Estimation CE = {:6.2%}'.format(estimationCE))
	print('Validation CE = {:6.2%}'.format(validationCE))


def test(path_train, path_test):
	X_train, y_train = load_dataset(path_train)
	X_test, y_test = load_dataset(path_test, reuse_norm=True)
	trainCE, testCE = nn(X_train, y_train, X_test, y_test, conf_matrix=True)
	print('Train CE = {:6.2%}'.format(trainCE))
	print('Test CE = {:6.2%}'.format(testCE))
	return trainCE, testCE


# cross-validation
if False:
	# run with default config
	cross_validation(ARG_TRAINSET)

# train and test
if False:
	# run with default config
	_ = test(ARG_TRAINSET, ARG_TESTSET)



### OPTIMAL CONFIG ###
if True:
	ARG_DIMS_HID = [18, 18]
	ARG_ACTIVATION = 'tanh'
	ARG_WEIGHT_INIT = 'custom'
	ARG_ALPHA = .5
	ARG_ALPHA_DROP_RATE = .5
	ARG_ALPHA_DROP_EACH = 15
	ARG_EPOCHS = 300
	trainCE, testCE = test(ARG_TRAINSET, ARG_TESTSET)
	file_path = os.path.join('.', 'output', 'final.csv')
	with open(file_path, 'w') as f:
		f.write('\ttrain error\ttest error\n')
		f.write('final\t{:6.2%}\t{:6.2%}\n'.format(trainCE, testCE))
	plot_table(file_path, (6, 1))
	print('\nTrain accuracy = {:6.2%}'.format(1 - trainCE))
	print('\nTest accuracy = {:6.2%}'.format(1 - testCE))












### EXPERIMENTS ###

# normalization
if False:
	ARG_DIMS_HID = [20]
	ARG_EPOCHS = 20
	K = 5
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', 'normalization.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		print('Without normalization')
		ARG_NORMALIZE = False
		estimationCE, validationCE = nn_kfold(X, y, k=K)
		f.write('none\t{:6.2%}\t{:6.2%}\n'.format(estimationCE, validationCE))
		print('With normalization')
		ARG_NORMALIZE = True
		estimationCE, validationCE = nn_kfold(X, y, k=K)
		f.write('standard normal distribution\t{:6.2%}\t{:6.2%}\n'.format(estimationCE, validationCE))
	plot_table(file_path, (6, 1))

# 1 hidden, exponents of 2
if False:
	ARG_EPOCHS = 20
	K = 5
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', '1hidden.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		for exp in range(1, 8):
			w = int(2**exp)
			ARG_DIMS_HID = [w]
			print('{} neurons'.format(w))
			estimationCE, validationCE = nn_kfold(X, y, k=K)
			f.write('{}\t{:6.2%}\t{:6.2%}\n'.format(w, estimationCE, validationCE))
	plot_table(file_path, (6, 2))

# 1 hidden, increment by 4
if False:
	K = 5
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', '1hidden_2.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		for w in range(8, 25, 4):
			ARG_DIMS_HID = [w]
			print('{} neurons'.format(w))
			estimationCE, validationCE = nn_kfold(X, y, k=K)
			f.write('{}\t{:6.2%}\t{:6.2%}\n'.format(w, estimationCE, validationCE))
	plot_table(file_path, (6, 2))

# 2 hidden, increment by 8
if False:
	K = 5
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', '2hidden.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		for w1,w2 in itertools.combinations_with_replacement(range(8, 33, 8), 2):
			ARG_DIMS_HID = [w1, w2]
			print('{} neurons'.format(ARG_DIMS_HID))
			estimationCE, validationCE = nn_kfold(X, y, k=K)
			f.write('{}\t{:6.2%}\t{:6.2%}\n'.format(ARG_DIMS_HID, estimationCE, validationCE))
	plot_table(file_path, (6, 4))

# activations, 1 hidden
if False:
	K = 5
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', 'activation.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		for a in ['sigmoid', 'tanh', 'arctan', 'relu', 'prelu', 'softplus', 'softmax']:
			ARG_ACTIVATION = a
			for w in [20, 24]:
				ARG_DIMS_HID = [w]
				print('{} {}'.format(a, w))
				estimationCE, validationCE = nn_kfold(X, y, k=K)
				f.write('{} {}\t{:6.2%}\t{:6.2%}\n'.format(a, w, estimationCE, validationCE))
	plot_table(file_path, (6, 3))

# activations, 2 hidden
if False:
	K = 5
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', 'activation_2.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		for a in ['tanh', 'relu', 'prelu']:
			ARG_ACTIVATION = a
			for w2 in range(4, 17, 4):
				ARG_DIMS_HID = [20, w2]
				print('{} {}'.format(a, ARG_DIMS_HID))
				estimationCE, validationCE = nn_kfold(X, y, k=K)
				f.write('{} {}\t{:6.2%}\t{:6.2%}\n'.format(a, ARG_DIMS_HID, estimationCE, validationCE))
	plot_table(file_path, (6, 3))


# weight init
if False:
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', 'weight_init.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		ARG_DIMS_HID = [20, 16]
		for a in ['tanh', 'relu', 'prelu']:
			ARG_ACTIVATION = a
			for wi in ['zeros', 'uniform', 'gauss', 'xavier', 'relu', 'custom']:
				ARG_WEIGHT_INIT = wi
				print('{}, {}'.format(a, wi))
				estimationCE, validationCE = nn_kfold(X, y)
				f.write('{}, {}\t{:6.2%}\t{:6.2%}\n'.format(a, wi, estimationCE, validationCE))
	plot_table(file_path, (6, 4))

# 2 hidden, tanh vs. relu
if False:
	ARG_WEIGHT_INIT = 'custom'
	ARG_EPOCHS = 100
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', '2hidden_2.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		for a in ['tanh', 'relu']:
			ARG_ACTIVATION = a
			for w1 in [18, 20, 22]:
				for w2 in [14, 16, 18]:
					ARG_DIMS_HID = [w1, w2]
					print('{} {}'.format(a, ARG_DIMS_HID))
					estimationCE, validationCE = nn_kfold(X, y)
					f.write('{} {}\t{:6.2%}\t{:6.2%}\n'.format(a, ARG_DIMS_HID, estimationCE, validationCE))
	plot_table(file_path, (6, 4))

# 2 hidden, tanh
if False:
	K = 5
	ARG_ACTIVATION = 'tanh'
	ARG_WEIGHT_INIT = 'custom'
	ARG_EPOCHS = 70
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', '2hidden_3.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		for w1 in [12, 14, 16]:
			for w2 in [14, 16, 18]:
				ARG_DIMS_HID = [w1, w2]
				print('{}'.format(ARG_DIMS_HID))
				estimationCE, validationCE = nn_kfold(X, y, k=K)
				f.write('{}\t{:6.2%}\t{:6.2%}\n'.format(ARG_DIMS_HID, estimationCE, validationCE))
	plot_table(file_path, (6, 2))

# 3 hidden, tanh
if False:
	K = 5
	ARG_ACTIVATION = 'tanh'
	ARG_WEIGHT_INIT = 'custom'
	ARG_EPOCHS = 70
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', '3hidden.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		for w2 in [18, 22]:
			for w3 in [14, 18, 22]:
				ARG_DIMS_HID = [18, w2, w3]
				print('{}'.format(ARG_DIMS_HID))
				estimationCE, validationCE = nn_kfold(X, y, k=K)
				f.write('{}\t{:6.2%}\t{:6.2%}\n'.format(ARG_DIMS_HID, estimationCE, validationCE))
	plot_table(file_path, (6, 2))

# alpha
if False:
	ARG_DIMS_HID = [18, 18]
	ARG_ACTIVATION = 'tanh'
	ARG_WEIGHT_INIT = 'custom'
	ARG_EPOCHS = 200
	X, y = load_dataset(ARG_TRAINSET)
	file_path = os.path.join('.', 'output', 'alpha.csv')
	with open(file_path, 'w') as f:
		f.write('\testimation error\tvalidation error\n')
		for ade in [5, 10, 15, 20, 25]:
			ARG_ALPHA_DROP_EACH = ade
			for adr in [.5, .7]:
				if ade > 15 and adr > .5:
					continue
				ARG_ALPHA_DROP_RATE = adr
				print('drop {}, .{}'.format(ade, int(adr*10)))
				estimationCE, validationCE = nn_kfold(X, y)
				f.write('drop {}, .{}\t{:6.2%}\t{:6.2%}\n'.format(ade, int(adr*10), estimationCE, validationCE))
	plot_table(file_path, (6, 2))

