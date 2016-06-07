
# coding: utf-8

# In[1]:

############## IMPORTS ###################
import sys
sys.path.insert(0, '../mylib/')
from data_load import load_csv_training
from data_load import load_csv_test
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from numpy import *
import pandas as pd
import seaborn as sns


# In[2]:

############## FILES ###################
output=open('output','wb')


# In[3]:

#####################
### DATA LOADING
#####################

#features,labels=load_csv_training('train.csv',',')

training_set=pd.read_csv('train.csv',index_col=0)

# [TEST] the data loading by writing arrays in file:

# counter=0
# for new_datapoint in features:
# 	for feature in new_datapoint:
# 		output.write(str(feature)+" ")
# 	output.write(' ')	
# 	output.write(str(labels[counter]))
# 	output.write("\n")
# 	counter=counter+1


# From now onwards: 
	# features is a list containing lists of datapoint features
	# labels is a list of labels
# This structures can be used as inputs for the .fit() function of the learners


# In[4]:

#################################
### DATA VISUALIZATION (SEABORN)
#################################

get_ipython().magic(u'matplotlib tk')

# Create a plot with cover type vs any other variable
#sns.regplot("Elevation", "Cover_Type", training_set);

#print training_set.ix[:,0:3]

# Create a scatter plot (regplot()) 
#sns.pairplot(training_set.ix[:,50:55], size=5, diag_kind='kde') # it fails when using a larger amount of features
#g = sns.PairGrid(training_set)
#g.map(plt.scatter)


# Create a correlation plot (corrplot())

# Show an histogram for th features
#training_set.hist()


# In[5]:

###################################################
# extract features and labels from training_set (): 
###################################################
training_set_array = array(training_set)
features = training_set_array[0:,0:-1]
labels = ravel(training_set_array[0:,-1:]) # Can we actually do this? It may use a lot of memory for big data sets


# In[6]:

######################
### FEATURE SELECTION
######################

#print len(features)

# Variance threshold
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#features = sel.fit_transform(features)

# Univariate feature selection
#features = SelectKBest(f_classif, k=2).fit_transform(features, labels)

# Use Linear Discriminant Analysis (LDA)
#lda = LDA() # lda = LDA(n_components=5) 1..5 (Number of components (< n_classes - 1) for dimensionality reduction)
#features= lda.fit(features, labels).transform(features)

# Use Principal Component Analysis (PCA)




# In[7]:

############################
### COMMENT FOR SUBMISSION:
###############################################
### LEARNING AND TESTING WITH CROSS-VALIDATION
###############################################
###############################################
print ""

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,labels, test_size=0.4, random_state=0)

# [TEST] the splits
# for list_a in a:
# 	for elem in list_a:
# 		output.write(str(elem))
# 		output.write(" ")
# 	output.write("\n\n\n\n\n\n")


# In[ ]:

# Logistic Regression
estimator = linear_model.LogisticRegression()
clf = estimator.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "Score for Logistic Regression, default parameters (liblinear, l2, max_it=100,multi_class='ovr'): ", score

estimator = linear_model.LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr')
clf = estimator.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "Score for Logistic Regression, penalty='l1', solver='liblinear', multi_class='ovr': ", score

estimator = linear_model.LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial')
clf = estimator.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "Score for Logistic Regression, penalty='l2', solver='lbfgs', multi_class='multinomial': ", score

estimator = linear_model.LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial')
clf = estimator.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "Score for Logistic Regression, penalty='l2', solver='newton-cg', multi_class='multinomial': ", score


# In[12]:

estimator = linear_model.LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial', max_iter=200)
clf = estimator.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "Score for Logistic Regression, penalty='l2', solver='newton-cg', multi_class='multinomial', max_iter=200: ", score

#print "Using recursive feature elimination (RFE):"

for i in []:
    # adding feature selection to this estimator, i features:
    selector = RFE(estimator, i, step=1)
    selector = selector.fit(features_train, labels_train)
    score = selector.score(features_test, labels_test)
    print "Score for Logistic Regression with the " + str(i) + " most relevant features (RFE): ", score
    print selector.get_support(indices=True)
    
# adding feature selection to this estimator:
#selector = RFECV(estimator, step=1, cv=5)
#selector = selector.fit(features_train, labels_train)
#score = selector.score(features_test, labels_test)
#print "Score for Logistic Regression with the  most relevant features (RFECV): ", score
#print selector.get_support(indices=True)

print ""


# In[28]:

# Support Vector Classification
estimator = svm.LinearSVC() # tune the parameters to outperform logistic regression
clf = estimator.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "Score for Support Vector Classification: ", score

print "Using recursive feature elimination (RFE):"

for i in []:
    # adding feature selection to this estimator, i features:
    selector = RFE(estimator, i, step=1)
    selector = selector.fit(features_train, labels_train)
    score = selector.score(features_test, labels_test)
    print "Score for Support Vector Classification with the " + str(i) + " most relevant features (RFE): ", score
    print selector.get_support(indices=True)

# adding feature selection to this estimator:
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(features_train, labels_train)
score = selector.score(features_test, labels_test)
print "Score for Support Vector Classification with the  most relevant features (RFECV): ", score
print selector.get_support(indices=True)

print ""


# In[8]:

# Naive Bayes
estimator = naive_bayes.GaussianNB() # tune the parameters to outperform logistic regression
clf = estimator.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "Score for Naive Bayes: ", score

print "Using recursive feature elimination (RFE):"

for i in []:
    # adding feature selection to this estimator, i features:
    selector = RFE(estimator, i, step=1)
    selector = selector.fit(features_train, labels_train)
    score = selector.score(features_test, labels_test)
    print "Score for Naive Bayes with the " + str(i) + " most relevant features (RFE): ", score

# adding feature selection to this estimator:
#selector = RFECV(estimator, step=1, cv=5)
#selector = selector.fit(features_train, labels_train)
#score = selector.score(features_test, labels_test)
#print "Score for Naive Bayes with the  most relevant features (RFECV): ", score

print ""


# In[10]:

# Nearest Neighbors
estimator = neighbors.KNeighborsClassifier() # tune the parameters to outperform logistic regression
clf = estimator.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "Score for Nearest Neighbors: ", score

print "Using recursive feature elimination (RFE):"

for i in []:
    # adding feature selection to this estimator, i features:
    selector = RFE(estimator, i, step=1)
    selector = selector.fit(features_train, labels_train)
    score = selector.score(features_test, labels_test)
    print "Score for Nearest Neighbors with the " + str(i) + " most relevant features (RFE): ", score

# adding feature selection to this estimator:
#selector = RFECV(estimator, step=1, cv=5)
#selector = selector.fit(features_train, labels_train)
#score = selector.score(features_test, labels_test)
#print "Score for Nearest Neighbors with the  most relevant features (RFECV): ", score

print ""


# In[11]:

# Decision Tree
estimator = tree.DecisionTreeClassifier() # tune the parameters to outperform logistic regression
clf = estimator.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "Score for Decision Tree: ", score

print "Using recursive feature elimination (RFE):"

for i in [5,10,15,20,25,30,35]:
    # adding feature selection to this estimator, i features:
    selector = RFE(estimator, i, step=1)
    selector = selector.fit(features_train, labels_train)
    score = selector.score(features_test, labels_test)
    print "Score for Decision Tree with the " + str(i) + " most relevant features (RFE): ", score

# adding feature selection to this estimator:
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(features_train, labels_train)
score = selector.score(features_test, labels_test)
print "Score for Decision Tree with the  most relevant features (RFECV): ", score

print ""


# In[12]:

######################
### ENSEMBLE METHODS
######################

# AVERAGING
###########

# Random Forest
estimator = RandomForestClassifier(n_estimators=10)
clf = estimator.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "Score for Random Forest: ", score

print "Using recursive feature elimination (RFE):"

for i in [5,10,15,20,25,30,35]:
    # adding feature selection to this estimator, i features:
    selector = RFE(estimator, i, step=1)
    selector = selector.fit(features_train, labels_train)
    score = selector.score(features_test, labels_test)
    print "Score for Random Forest with the " + str(i) + " most relevant features (RFE): ", score

# adding feature selection to this estimator:
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(features_train, labels_train)
score = selector.score(features_test, labels_test)
print "Score for Random Forest with the  most relevant features (RFECV): ", score

print ""


# In[13]:

# BOOSTING
##########

# AdaBoost
clf = AdaBoostClassifier(n_estimators=10) # default = DecisionTreeClassifier. (base_estimator=tree.DecisionTreeClassifier, )
scores = cross_val_score(clf, features, labels)
print "Score for AdaBoost: ", scores.mean()


# In[8]:

#############################
## UNCOMMENT FOR SUBMISSION: 
##############################################################
### LEARNING FROM DATA WITH THE ENTIRE DATASET ###############
##############################################################


# learner = linear_model.LogisticRegression() # uncomment when preparing
# #learner = svm.LinearSVC() # for submission
# #learner = RandomForestClassifier(n_estimators=10)
# learner.fit(features,labels) 

# # [TEST] if learning works with Logistic Regression
# # predicted_class_LogReg = learnerLogReg.predict([2596,51,3,258,0,510,221,232,148,6279,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
# # print predicted_class_LogReg

# # Test if learning works with Support Vector Machines
# # predicted_class_SVC = learner.predict([2596,51,3,258,0,510,221,232,148,6279,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
# # print predicted_class_SVC


# In[9]:

#############################
## UNCOMMENT FOR SUBMISSION: 
# #######################################
# ### CREATE A CSV FILE FOR THE RESULTS
# #######################################


# features_test = load_csv_test('test.csv',',')

# # SCORE the model
# # predict and write in the csv

# output.write("Id,Cover_Type\n")

# id_data = 15121
# for datapoint in features_test:

# 	# predict the cover type
# 	predicted_label = learner.predict(datapoint)

# 	# write id and cover type
# 	output.write(str(id_data)+","+str(predicted_label[0])+"\n")

# 	id_data += 1


# # [TEST] print into a file the result of predictions:
# counter=0
# # .. TO-DO ..


# In[10]:

###### FILES ######################################
# #close output
output.close()
##################################################

