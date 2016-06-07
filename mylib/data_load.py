#####################
### DATA LOADING
#####################

import csv

# Example:
#features,labels=load_csv('train.csv',',')
def load_csv_training(csv_file,delimiter):

	# Load the data set (just the training set, as the test set does not have labels):
	training_csvfile = open(csv_file, 'rb')
	training_set = csv.reader(training_csvfile, delimiter=delimiter)

	# Read column names:
	training_column_names = training_set.next()

	# [TEST] the data loading:
	#print training_column_names
	#print training_set.next()

	# Build the feature and label arrays for the entire dataset:
	features = list()
	labels = list()

	for row in training_set:

		# Create a list containing the features of this datapoint:
		new_datapoint = list()
		for feature in row[1:-1]: #fixed, it was row[:-1], now row[1:-1] because the first attribute is just an unnecessary index
			new_datapoint.append(int(feature))

		# Add the list to the list of list features
		features.append(new_datapoint)

		# Add a label to the label list:
		labels.append(int(row[-1]))

	return features,labels



# Example:
#features_test=load_csv('test.csv',',')
def load_csv_test(csv_file,delimiter):

	test_csvfile = open(csv_file, 'rb')
	test_set = csv.reader(test_csvfile, delimiter=delimiter)


	# Read column names:
	test_column_names = test_set.next()

	# Build the feature and label arrays for the entire dataset:
	features_test = list()
	for row in test_set:
		# Create a list containing the features of this datapoint:
		new_datapoint = list()
		for feature in row[1:]: #fixed, it was row[:-1], now row[1:] because the first attribute is just an unnecessary index
			new_datapoint.append(int(feature))

		# Add the list to the list of list features
		features_test.append(new_datapoint)

	return features_test