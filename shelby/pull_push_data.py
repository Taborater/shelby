# -*- coding: utf-8 -*-
"""Pull Push Data
(part of shelby package)

This module contains function for reading csv files into pandas DataFrame
and function for writing pandas DataFrame into csv file.

Usage example:
	``
	# reading
	df_train, df_test = shelby.pull_push_data.read_data(tr_path, te_path, index_col='Id')

	# writing
	shelby.pull_push_data.write_data(y_pred, index_col='Id', fname='predicted', sample_path='sample.csv')
	``
"""
import pandas as pd
import os

def read_data(train_path="./data/train.csv", test_path='./data/test.csv', index_col=None):
	"""Read train and test data from csv files into pandas DataFrames.

	Args:
		train_path (str): path to train.csv from current dir (default './data/train.csv').
		test_path (str): path to test.csv from current dir (default './data/test.csv').
		index_col (str): index column name (default 'None').

	Returns:
		df_train (pandas.DataFrame): train data.
		df_test (pandas.DataFrame): test data.

	"""

	# Get full path to the current dir
	file_dir = os.path.dirname(os.path.realpath('__file__'))

	# Combine train and test paths with full dir path
	train_path = os.path.join(file_dir, train_path)
	test_path = os.path.join(file_dir, test_path)

	# Read and return
	df_train = pd.read_csv(train_path, index_col=index_col)
	df_test = pd.read_csv(test_path, index_col=index_col)

	return df_train, df_test



def write_data(y_pred, index_col, file_name, dir_name = 'preds', sample_path='./data/sample_submission.csv'):
	"""Write pandas DataFrame into csv file.

	Args:
		y_pred (ndarray): array of predicted values.
		index_col (str): index column name.
		file_name (str): new file name.
		dir_name (sts): name of the directory for predictions (if not exists - will be created).
		sample_path (str): path to sample submission csv file.

	"""

	# Get full path of the current dir
	file_dir = os.path.dirname(os.path.realpath('__file__'))

	# Combine sample submission path with full dir path
	sample_path = os.path.join(file_dir, sample_path)

	# Read sample submission csv file into pandas DataFrame
	submission = pd.read_csv(sample_path, index_col=index_col)

	# Replace values in sample submission pandas DataFrame with predicted values
	submission.values[:] = y_pred.reshape(-1,1)

	# Check if directory exists - create if not
	if not os.path.exists('./' + dir_name + '/'):
		os.makedirs('./' + dir_name + '/')

	# Save predictions
	submission.to_csv('./' + dir_name + '/' + file_name + '.csv')










