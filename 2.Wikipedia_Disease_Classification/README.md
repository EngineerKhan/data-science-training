
------------
Description:
------------

This is part of a selection process I was involved in for a startup company based in San Francisco. I won't give details about it nor the dataset I worked with. I just uploaded this to get some feedback from anyone interested in discussing better approaches for this kind of problems.

This project consists on the classification of wikipedia pages: the result is = 1 when the page is written about a disease, 0 otherwise.
You can find the full description of the tools and the approach taken below. The dataset consists on already downloaded wikipedia pages in html format, which requires parsing and preprocessing with BeautifulSoup, and TF-IDF vector transformations with Gensim. 

The biggest problem found was the memory limitations, given that I chose to use Python and scikit-learn as the main tools. Loading an entire dataset in numpy arrays was not possible, so I had to use an iterative approach where I read just a few rows of the data set, apply an operation, and repeat till the full dataset was read. That brought me to use an iterative learner, the stochastic gradient descent.


*UPDATE: I recently learnt how to use the tfidf vectorizer from scikit-learn. Given we needed an iterative approach for this problem, I wonder if it would be useful for it. I still consider gensim was required for this problem, as it allows to store the vocabulary and the tfidf model in disk. Anyways, you can find the scikit-learn implementation of this vectorizer in the file test.py. It has several input parameters. For more information about how I chose these, have a look at the documentation:

http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html




-------------------
Running the script:
-------------------

Note: To run this software, python 2.7, BeautifulSoup, gensim and scikit-learn must be available in the system.

To install BeautifulSoup: pip install beautifulsoup4
To instal the lxml parser for BeautifulSoup: pip install lxml
For gensim: pip install --upgrade gensim
For scikit-learn: pip install -U scikit-learn

1. Download the python script and the data in the same folder.

2. Run the script in the terminal as follows:

	python classificator.py

This will show the model performance in terms of accuracy


------------------------------
Summary of the approach taken:
------------------------------


---------
1. Tools: 
---------

Given this is a document classification problem based in html data, for which a quick solution is required, the chosen language and tools are the following:

- Python: given the simple but still powerful machine learning and natural language processing tools it offers, plus the rapid protoyping capabilities inherent to this language, Python is the main tool to be used.

- BeautifulSoup: powerful tool which easily allows to read and parse html documents.

- Gensim: complete natural language processing tool used to create TF-IDF vectors, among other features.

- Scikit-learn: widely used machine learning library for Python.



---------------
2. Development:
---------------

- The tasks of this software consist of reading and creating a dataset of TF-IDF vectors, each corresponding to a document. BeatifulSoup is used for parsing, and Gensim for the TF-IDF creation.

- The initial tests were performed with a small version of the dataset. Big Data related problems arose when the big data set was used:
	
	the size of the datasets generated is bigger than memory, so all operations, including training a test set generation, model training and testing are done by reading small amounts of data each time. This has implications when choosing the model: only a classifier which can be trained iteratively can be used. Scikit-learn offers several models based on this idea, and the most suited for floating point attributes like tf-idf scores is the Stochastic Gradient Descent (SGD) Classifier. It is trained by reading 8 rows per iteration and applying the partial_fit() method, which applies the gradient method.

- To ensure a non-biased training, the rows of the generated dataset are randomly shuffled using an C++ based external software previously used in a Kaggle competition: 

	https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/11018/shuffling-lines-of-big-data-files
	https://github.com/trufanov-nok/shuf-t/tree/master/binaries

- the function get_info() in auxiliar_functions.py returns the name of a disease of a given wikipedia url.



-------------
- 3. Results:
-------------

By running the example, the obtained results are the following:

true positives:  690
true negatives:  1964
total:  2713

accuracy: 0.978252856616 


- Further improvements could follow this basic version, such as:

	- Feature selection
	
	- cross-validation: to accurately predict the efficency of the classificator.

	- better classifiers

	- ensemble methods (averaging methods like random forest)


