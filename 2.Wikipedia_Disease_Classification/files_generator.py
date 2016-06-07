 #-*- coding: utf-8 -*-

# LIBRARIES
import os
import os.path
from bs4 import BeautifulSoup
import re
from gensim import corpora, models, similarities, matutils
from collections import defaultdict
from pprint import pprint
from auxiliar_functions import html2vec
import csv
from random import shuffle, random
import subprocess


# READING POSITIVE DATA
positive_documents = []
for filename in os.listdir("./training/positive-big/"):
	positive = open("./training/positive-big/"+filename,"r+")
	positive_vector = html2vec(positive)
	positive_documents.append(positive_vector)



# READING NEGATIVE DATA
negative_documents = []
for filename in os.listdir("./training/negative-big/"):
	negative = open("./training/negative-big/"+filename,"r+")
	negative_vector = html2vec(negative)
	negative_documents.append(negative_vector)



# Create dic
dictionary = corpora.Dictionary(positive_documents+negative_documents)
dictionary.save('./dictionary.dict') # store the dictionary, for future reference


# create full corpus
corpus_all = [dictionary.doc2bow(text) for text in positive_documents+negative_documents]
#corpora.MmCorpus.serialize('./corpus_all.mm', corpus_all) # store to disk, for later use

# create positive corpus
corpus_positives = [dictionary.doc2bow(text) for text in positive_documents]
#corpora.MmCorpus.serialize('./corpus_positives.mm', corpus_positives) # store to disk, for later use
	
# create negative corpus
corpus_negatives = [dictionary.doc2bow(text) for text in negative_documents]
#corpora.MmCorpus.serialize('./corpus_negatives.mm', corpus_negatives) # store to disk, for later use

# Create tf-idf version of corpuses
tfidf = models.TfidfModel(corpus_all)
tfidf.save('./model.tfidf')
corpus_positives_tf_idf = tfidf[corpus_positives]
corpus_negatives_tf_idf = tfidf[corpus_negatives]


# write tf-idf representations of documents into a csv file
dataset = open('dataset.csv', 'wb')
dataset_writer = csv.writer(dataset, delimiter=',')

for document in corpus_positives_tf_idf:
	
	document_list = [0]*(len(dictionary)+1)

	for element in document:
		document_list[element[0]]=element[1]
	document_list[-1]=1

	dataset_writer.writerows([document_list])


for document in corpus_negatives_tf_idf:
	
	document_list = [0]*(len(dictionary)+1)

	for element in document:
		document_list[element[0]]=element[1]
	document_list[-1]=0

	dataset_writer.writerows([document_list])

dataset.close()


# call external file to shuffle rows of the csv file
FNULL = open(os.devnull, 'w')    # use this if you want to suppress output to stdout from the subprocess
subprocess.call("./shuf-t.exe dataset.csv -o dataset_shuffled.csv")

# Read the shuffled dataset and split into training(~80%) and test set (~20%)
len_dataset = len(corpus_positives_tf_idf) + len(corpus_negatives_tf_idf)
dataset = open('dataset_shuffled.csv', 'rb')
dataset_reader = csv.reader(dataset, delimiter=',')

dataset_training = open('dataset_training.csv', 'wb')
dataset_training_writer = csv.writer(dataset_training, delimiter=',')

dataset_test = open('dataset_test.csv', 'wb')
dataset_test_writer = csv.writer(dataset_test, delimiter=',')

#
for i in range(0, len_dataset):

	row=dataset_reader.next()
	row[-1]=row[-1].strip()

	if (random()>0.2): # 80% chances to write rows into the training set file
		dataset_training_writer.writerows([row])
	else:
		dataset_test_writer.writerows([row])


dataset_training.close()
dataset_test.close()