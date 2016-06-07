# data-science-training
My Kaggle/CodaLab projects.

Packages:
- Scikit-learn
- Pandas
- Seaborn
- Spark - MLlib

Languages and tools:
- IPhyton Notebook - PyCharm - IntelliJ IDEA
- Scala - IntelliJ IDEA



# 1.Forest_Classification
1.Forest_Classification/ForestClassification.ipynb contains the interactive notebook. For those who don't use IPython, 1.Forest_Classification/ForestClassification.py is the corresponding Python script.



# 2.Wikipedia_Dissease_Classification
It contains an iterative classificator, the Stochastic Gradient Descent. This is part of a selection process I was involved in for a startup company based in San Francisco. I won't give details about it nor the dataset I worked with. I just uploaded this to get some feedback from anyone interested in discussing better approaches for this kind of problems.

*UPDATE: I recently learnt how to use the tfidf vectorizer from scikit-learn. Given we needed an iterative approach for this problem, I wonder if it would be useful for it. I still consider gensim was required for this problem, as it allows to store the vocabulary and the tfidf model in disk. Anyways, you can find the scikit-learn implementation of this vectorizer in the file test.py. It has several input parameters. For more information about how I chose these, have a look at the documentation:

http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html



# 3.Telstra_Network_Disruptions
This project consists on the solution to a Kaggle competition: Telstra Network Disruptions - Predict service faults on Australia's largest telecommunications network. I found this competition quite late, when there were just a few days left before the final submission. I decided to start it anyways, as I considered it very useful to learn how to apply some data tyding operations, given the dataset was split between several tables, in which some variables were spread along the rows instead the columns. Python does not offer a clear set of data tyding operations like those found in R's tidyr package. I haven't spent more time on this competition, as I would like to start a new one whose deadline has not been reached yet. That is why you won't find the data science side of this project. For a data science focused project, check my 4th folder, the Santander Customer Satisfaction prediction.

* UPDATE 1: After I finished my solution and uploaded it to Kaggle, I had a look at two other solutions I found:

http://www.analyticsvidhya.com/blog/2016/03/complete-solution-top-11-telstra-network-disruptions-kaggle-competition/

http://gereleth.github.io/Telstra-Network-Disruptions-Writeup/

They both mention a magic feature hidden in the data set. Turns out that ids and locations are sorted by time, so a new feature (a time index), can be added to the dataset to significantly improve the results. Following the second link, I found a very useful ipython notebook explaining how to find this magic feature:

https://github.com/gereleth/kaggle-telstra/blob/master/Discovering%20the%20magic%20feature.ipynb



# 4.Santander_Customer_Satisfaction

Competition finished. I used scikit-learn, pandas and seaborn.
Now I want to write the equivalent code with PySpark and Scala-Spark. It will be a good practice to learn Spark
with these two languages.