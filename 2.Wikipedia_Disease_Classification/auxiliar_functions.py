# LIBRARIES
import os
import os.path
from bs4 import BeautifulSoup
import re
from gensim import corpora, models, similarities, matutils
from collections import defaultdict
from pprint import pprint
import csv
import urllib2
import random



def wikidoc2vec_scikit(url):
	
	# TO - DO



# given an url, a tf-idf model and a dictionary, return the extended tf-idf representation of the vector
def wikidoc2vec(url, tfidf, dictionary):

	# decode an ascii string to produce unicode string, ignoring non-ascii characters
	wikidoc_html = urllib2.urlopen(url).read().decode('ascii', 'ignore')

	# tokenize text
	paragraphs_body = html2vec(wikidoc_html)

	# get tfidf representation of the document
	bow_body = dictionary.doc2bow(paragraphs_body)
	tfidf_body = tfidf[bow_body]

	# convert to extended vector
	vector_body = [0]*(len(dictionary)-1)
	for element in tfidf_body:
		vector_body[element[0]]=element[1]

	return vector_body


# given an html document, return its text tokenized
def html2vec(wikidoc_html):

	# PARSING DATA
	soup = BeautifulSoup(wikidoc_html, 'lxml')

	# get title of the article
	title = soup.h1.getText()

	# Get the body of the article
	paragraphs = soup.find_all('p')
	paragraphs_body = ''

	# Text pre-processing, for each paragraph:
	for paragraph in paragraphs:
		paragraph = re.sub('<[^<]+?>', '', str(paragraph)) # remove all html tags
		paragraph = re.sub('\[\d\]', '', str(paragraph)) # remove all [i], where i is a wikipedia reference number
		paragraph = re.sub('[^a-zA-Z0-9_\s]', '', str(paragraph)) # remove all non-alphanumeric characters but whitespaces
		paragraph = re.sub('[0-9]', '', str(paragraph)) # remove all numbers
		paragraphs_body = paragraphs_body + ' ' + paragraph # concatenate paragraph to the resulting string

	# Text pre-processing, for the final pre-processed text of the document
	paragraphs_body = paragraphs_body.strip().lower() # remove initial and final whitespaces, set lowercase
	paragraphs_body = re.sub('\s+', ' ', str(paragraphs_body)) # remove extra whitespaces
	
	# remove common words and tokenize
	stoplist = set('for a of the and to in'.split())
	paragraphs_body = [word for word in paragraphs_body.lower().split() if word not in stoplist]


	# remove words that appear only once
	frequency = defaultdict(int)
	for token in paragraphs_body:
		frequency[token] += 1
	paragraphs_body = [token for token in paragraphs_body if frequency[token] > 1]


	return paragraphs_body



def get_info(url):

	wikidoc_html = urllib2.urlopen(url).read().decode('ascii', 'ignore')

	# PARSING DATA
	soup = BeautifulSoup(wikidoc_html, 'lxml')

	# get title of the article
	title = soup.h1.getText()

	print ('title: ', title)