try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
from bs4 import BeautifulSoup
import re
import numpy
from collections import defaultdict
##### new packages ######
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

def preprocessor(paragraph_input):
	paragraph = re.sub('<[^<]+?>', '', str(paragraph_input)) # remove all html tags
	paragraph = re.sub('\[\d\]', '', str(paragraph)) # remove all [i], where i is a wikipedia reference number
	paragraph = re.sub('[^a-zA-Z0-9_\s]', '', str(paragraph)) # remove all non-alphanumeric characters but whitespaces
	paragraph = re.sub('[0-9]', '', str(paragraph)) # remove all numbers
	return paragraph

def tokenizer(paragraph_input):
	return paragraph_input.split()


def wikidoc2vec_scikit(url):
	
	# decode an ascii string to produce unicode string, ignoring non-ascii characters
	wikidoc_html = urllib2.urlopen(url).read().decode('ascii', 'ignore')
	soup = BeautifulSoup(wikidoc_html, 'lxml')
	
	# Get the body of the article
	paragraphs = soup.find_all('p')

	#for paragraph in paragraphs:
	#	print (paragraph)
	#	print('####################')


	# get tfidf representation of the document with scikit-learn:

	tfidf_vectorizer = TfidfVectorizer(
		input='content', 
		encoding='utf-8', 
		decode_error='replace', 
		strip_accents='unicode', 
		lowercase=True, 
		preprocessor=preprocessor, 
		tokenizer=tokenizer, 
		analyzer='word', 
		stop_words=list('for a of the and to in'.split()), 
		token_pattern='(?u)\b\w\w+\b', 
		ngram_range=(1, 1), 
		max_df=1.0, 
		min_df=1, 
		max_features=None, 
		vocabulary=None, 
		binary=False, 
		dtype=numpy.int64, 
		norm='l2', 
		use_idf=True, 
		smooth_idf=True, 
		sublinear_tf=False
		)


	#print(type(paragraphs))
	return tfidf_vectorizer.fit_transform(iter(paragraphs))
	#preproc = tfidf_vectorizer.build_analyzer()
	#print(preproc("hola que tal   estás"))



X_train = wikidoc2vec_scikit('https://en.wikipedia.org/wiki/Shrub')
#print(X_train)

# Now, we can use X_train as input for classifiers. y_train would be the label for each document:
#SGDClassifier.fit(X_train, y_train)