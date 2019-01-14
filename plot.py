# https://stackoverflow.com/a/44751285/5148218

import pandas as pd
import matplotlib.pylab as plt
from pandas.plotting import table

def plot_table(file_path, figsize=(6, 2)):
	df = pd.read_csv(file_path, index_col=0, sep='\t')
	print(df)

	# set fig size
	fig, ax = plt.subplots(figsize=figsize)
	# no axes
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	# no frame
	ax.set_frame_on(False)
	# plot table
	tab = table(ax, df, loc='upper right', colWidths=[.3 for col in range(df.shape[1])])
	# set font manually
	tab.auto_set_font_size(False)
	tab.set_fontsize(8)
	# save the result
	image_path = file_path.replace('.csv', '.png')
	_ = plt.savefig(image_path)

#plot_table('./output/final.csv', (6, 1))
