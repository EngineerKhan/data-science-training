import csv
from sklearn import datasets, naive_bayes, linear_model
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities, matutils
from auxiliar_functions import wikidoc2vec, get_info
from sklearn.externals import joblib


# load tfidf model and dictionary
tfidf = models.TfidfModel.load('./model.tfidf')
dictionary = corpora.Dictionary.load('./dictionary.dict')


# train the stochastic gradient descent classifier iteratively:
clf = linear_model.SGDClassifier()

# read 8 datapoints from disk to memory, and apply the gradient method. Repeat till all the training set has been read.
with open('dataset_training.csv', 'rb') as training_set:
	training_reader = csv.reader(training_set, delimiter=',')

	iterative_X = []
	iterative_y = []

	n_data_points = 8
	counter = n_data_points

	for row in training_reader:
		counter=counter-1
		iterative_X.append(row[0:-2])
		iterative_y.append(int(row[-1]))

		if counter==0:
			clf.partial_fit(iterative_X, iterative_y, classes=[0,1])
			counter=n_data_points
			iterative_X = []
			iterative_y = []


	if counter % n_data_points != 0:
		clf.partial_fit(iterative_X, iterative_y)



# persist learner:
joblib.dump(clf, './SGD_model.pkl') 


# load learner: 
clf = joblib.load('./SGD_model.pkl') 


# test model
with open('dataset_test.csv', 'rb') as test_set:
	test_reader = csv.reader(test_set, delimiter=',')

	iterative_X = []
	iterative_y = []

	true_positives = 0
	true_negatives = 0
	total = 0
	
	for row in test_reader:
		if clf.predict([float(i) for i in row[0:-2]])[0] == int(row[-1]):
			
			if row[-1] == '1':
				true_positives = true_positives+1
			else:
				true_negatives = true_negatives+1
		total=total+1

	print 'true positives: ', true_positives
	print 'true negatives: ', true_negatives
	print 'total: ', total

	print float(true_positives + true_negatives)/total