import sys
import os
import pandas as pd
import numpy as np

def summary_generator(dataframe):
	column_names = list(dataframe.columns.values)
	print ('ncolums = '+ str(len(dataframe.axes[1])))
	print ('')
	n_cols = len(dataframe.axes[1])
	for i in range(0,n_cols):
		print(column_names[i])
		print (np.unique(dataframe.iloc[:,i]))
		print ('')
	print ('')



def folder_summary(folder):
	listing = os.listdir(folder)
	for csv_name in listing:
		file_csv = open(folder+csv_name,'r')
		dataframe = pd.read_csv(file_csv, engine='python')
		print('##### ' + csv_name)
		summary_generator(dataframe)
		print('--------------------------------------')