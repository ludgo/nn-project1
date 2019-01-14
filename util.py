# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, 2017-2018

import atexit
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # todo: remove or change if not working
import matplotlib.pyplot as plt
import time


## utility

def chunk(l, n):
	'''
	Split list to chunks of specified size.
	
	:param l: list
	:param n: size of chunk
	'''
	for i in range(0, len(l), n):
		yield l[i:i+n]

def augment(X):
	if X.ndim == 1:
		return np.concatenate((X, [1]))
	else:
		pad = np.ones((1, X.shape[1]))
		return np.concatenate((X, pad), axis=0)

def onehot_decode(X):
	return np.argmax(X, axis=0)

def onehot_encode(L, c):
	if isinstance(L, int):
		L = [L]
	n = len(L)
	out = np.zeros((c, n))
	out[L, range(n)] = 1
	return np.squeeze(out)

## plotting

palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']


def limits(values, gap=0.05):
	x0 = np.min(values)
	x1 = np.max(values)
	xg = (x1 - x0) * gap
	return np.array((x0-xg, x1+xg))


def plot_errors(title, errors, test_error=None, block=True):
	plt.figure(1).canvas.mpl_connect('key_press_event', keypress)
	plt.clf()

	plt.plot(errors)

	if test_error:
		plt.plot([test_error]*len(errors))

	plt.tight_layout()
	plt.gcf().canvas.set_window_title(title)
	plt.show(block=block)


def plot_both_errors(trainCEs, trainREs, testCE=None, testRE=None, pad=None, block=True, save_path=None):
	plt.figure(2).canvas.mpl_connect('key_press_event', keypress)
	plt.clf()

	if pad is None:
		pad = max(len(trainCEs), len(trainREs))
	else:
		trainCEs = np.concatentate((trainCEs, [None]*(pad-len(trainCEs))))
		trainREs = np.concatentate((trainREs, [None]*(pad-len(trainREs))))

	plt.subplot(2,1,1)
	plt.title('Classification error [%]')
	plt.plot(100*np.array(trainCEs))

	if testCE is not None:
		plt.plot([100*testCE]*pad)

	plt.subplot(2,1,2)
	plt.title('Model loss [MSE/sample]')
	plt.plot(trainREs)

	if testRE is not None:
		plt.plot([testRE]*pad)

	plt.tight_layout()
	plt.gcf().canvas.set_window_title('Error metrics')
	if save_path:
		plt.savefig(save_path)
	plt.show(block=block)


def plot_dots(inputs, labels=None, predicted=None, test_inputs=None, test_labels=None, test_predicted=None, s=60, i_x=0, i_y=1, block=True, save_path=None):
	plt.figure(3).canvas.mpl_connect('key_press_event', keypress)
	plt.clf()

	if labels is None:
		plt.gcf().canvas.set_window_title('Data distribution')
		plt.scatter(inputs[i_x,:], inputs[i_y,:], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

	elif predicted is None:
		plt.gcf().canvas.set_window_title('Class distribution')
		for i, c in enumerate(set(labels)):
			plt.scatter(inputs[i_x,labels==c], inputs[i_y,labels==c], s=s, c=palette[i], edgecolors=[0.4]*3)

	else:
		plt.gcf().canvas.set_window_title('Predicted vs. actual')
		for i, c in enumerate(set(labels)):
			plt.scatter(inputs[i_x,labels==c], inputs[i_y,labels==c], s=2.0*s, c=palette[i], edgecolors=None, alpha=0.333)

		for i, c in enumerate(set(labels)):
			plt.scatter(inputs[i_x,predicted==c], inputs[i_y,predicted==c], s=0.5*s, c=palette[i], edgecolors=None)

	if test_inputs is not None:
		if test_labels is None:
			plt.scatter(test_inputs[i_x,:], test_inputs[i_y,:], marker='s', s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

		elif test_predicted is None:
			for i, c in enumerate(set(test_labels)):
				plt.scatter(test_inputs[i_x,test_labels==c], test_inputs[i_y,test_labels==c], marker='s', s=s, c=palette[i], edgecolors=[0.4]*3)

		else:
			for i, c in enumerate(set(test_labels)):
				plt.scatter(test_inputs[i_x,test_labels==c], test_inputs[i_y,test_labels==c], marker='s', s=2.0*s, c=palette[i], edgecolors=None, alpha=0.333)

			for i, c in enumerate(set(test_labels)):
				plt.scatter(test_inputs[i_x,test_predicted==c], test_inputs[i_y,test_predicted==c], marker='s', s=0.5*s, c=palette[i], edgecolors=None)

	plt.xlim(limits(inputs[i_x,:]))
	plt.ylim(limits(inputs[i_y,:]))
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path)
	plt.show(block=block)


## interactive drawing, very fragile....

wait = 0.0

def clear():
	plt.clf()


def ion():
	plt.ion()
	time.sleep(wait)


def ioff():
	plt.ioff()


def redraw():
	plt.gcf().canvas.draw()
	plt.waitforbuttonpress(timeout=0.001)
	time.sleep(wait)


def keypress(e):
	if e.key in {'q', 'escape'}:
		os._exit(0) # unclean exit, but exit() or sys.exit() won't work

	if e.key in {' ', 'enter'}:
		plt.close() # skip blocking figures


## non-blocking figures still block at end

def finish():
	plt.show(block=True) # block until all figures are closed


atexit.register(finish)
