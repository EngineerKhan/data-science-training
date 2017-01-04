
# coding: utf-8

# In[2]:

# PACKAGES
import os
os.chdir('./4.Santander_Customer_Satisfaction')
import sys
sys.path.insert(0, '../mylib/')
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from show_data import print_full
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import robust_scale


# In[3]:

# MAGIC COMMANDS
get_ipython().magic(u'matplotlib inline')


# In[4]:

# LOAD DATA FILES
training_df = pd.read_csv('train.csv')
scoring_df = pd.read_csv('test.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')


# In[ ]:

#######################
# EXPLORATORY ANALYSIS
#######################


# In[ ]:

training_df.shape


# In[ ]:

# Get statistics about the features
desc = training_df.describe()
#desc.to_csv('describe.csv')
desc


# In[ ]:

# Get name and type of features
#for x in training_df.columns:
# print (x, training_df[x].dtype)
print ('number of features: ', len(training_df.columns))


# In[ ]:

# Deal with missing values: not missing values
for x in training_df.columns:
    if training_df[x].isnull().any():
        print x


# In[ ]:

# check if there is any repeated ID, which would imply to tidy the data set:
id_counts_df = training_df['ID'].value_counts().sort_index() # count the number of occurrences of each ID
max(id_counts_df) # if the max value is 1, then there are no repeated IDs = 1 row for each observation.


# In[ ]:

# Percentage of target == 1: ~ 4% --> very unbalanced!
float(len(training_df[training_df['TARGET']==1]))/len(training_df['TARGET']) * 100


# In[ ]:

#######################
# PREPROCESSING
#######################


# In[ ]:

# Remove IDs
training_df.drop(['ID'], axis=1, inplace=True)


# In[ ]:

# Remove constant features
# "Constant features can lead to errors in some models and obviously provide no information in the training set that can be learned from."
remove = []
count_constants = 0
for col in training_df.columns:
    if training_df[col].values.std() == 0: # pandas.series std() is not correct, use numpy std() instead (.values.std() instead of std())
        #print col
        remove.append(col)
        count_constants += 1
training_df.drop(remove, axis=1, inplace=True)
#test_df.drop(remove, axis=1, inplace=True)
print ('number of constant features removed: ', count_constants)


# In[ ]:

# Remove duplicated columns
remove = []
c = training_df.columns
count_duplicated = 0
for i in range(len(c)-1):
    v = training_df[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,training_df[c[j]].values):
            #print c[j]
            remove.append(c[j])
            count_duplicated += 1

training_df.drop(remove, axis=1, inplace=True)
#test_df.drop(remove, axis=1, inplace=True)
print ('number of duplicated features removed: ', count_duplicated)


# In[ ]:

positives = training_df[training_df['TARGET']==1]
negatives = training_df[training_df['TARGET']==0]


# In[ ]:

# BALANCE THE DATASET: 3008 with TARGET==1, 3008 with TARGET==0
training_positive = positives[0:1504]#.reset_index(drop=True)
training_negative = negatives[0:1504]


# In[ ]:

training_df = pd.concat([training_positive, training_negative])
training_df = training_df.reindex(np.random.permutation(training_df.index))


# In[ ]:

test_positive = positives[1504:3008]
test_negative = negatives[1504:]


# In[ ]:

test_df = pd.concat([test_positive, test_negative])
test_df = test_df.reindex(np.random.permutation(test_df.index))


# In[ ]:

####################
# FEATURE SELECTION
####################


# In[ ]:

# Remove features with low variance:
from sklearn.feature_selection import VarianceThreshold
def VarianceThreshold_selector(data, th):
    #Select Model
    selector = VarianceThreshold(th) #Defaults to 0.0, e.g. only remove features with the same value in all samples
    #Fit the Model
    selector.fit(data)
    features = selector.get_support(indices = True) #returns an array of integers corresponding to nonremoved features
    features = [column for column in data[features]] #Array of all nonremoved features' names
    #Format and Return
    selector = pd.DataFrame(selector.transform(data))
    selector.columns = features
    return selector


# In[ ]:

training_df.reset_index(drop=True, inplace=True)


# In[ ]:

# keep the ID by index
#index_id = training_df['ID']


# In[ ]:

training_df_vth = pd.concat([VarianceThreshold_selector(training_df.iloc[:,:-1], 0.9), training_df.iloc[:,-1]], axis=1)


# In[ ]:

training_df_vth.columns


# In[ ]:

# Univariate feature selection


# In[ ]:

# Split between features and target variable
X, y = training_df_vth.iloc[:,:-1], training_df_vth.iloc[:, -1]


# In[ ]:

# Remove features which are not correlated to the target variable by using recursive feature elimination (by checking subsets)
# with cross-validation. Wrapped method with logistic Regression (with C=1, low regularization strengh)
#from sklearn.feature_selection import RFECV
#from sklearn.linear_model import LogisticRegression


# In[ ]:

# My computer takes ages to compute this
#estimator = LogisticRegression()
#selector = RFECV(estimator, step=1, cv=3, n_jobs=4)
#selector = selector.fit(X, y)
#selector.support_
#selector.ranking_


# In[ ]:

# Remove features which are not correlated to the target variable by using univariate feature selection, keep the top 5%
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
selector = SelectPercentile(score_func = f_classif, percentile = 35)


# In[ ]:

X.shape


# In[ ]:

X_reduced = selector.fit_transform(X, y)


# In[ ]:

X_reduced.shape


# In[ ]:

#########################
# EXPLORATORY ANALYSIS 2
#########################


# In[ ]:

# Chosen features:
keep = X.iloc[:,selector.get_support()].columns


# In[ ]:

#X.iloc[:,selector.get_support()].hist(color='k', alpha=0.5, bins=10)


# In[ ]:

##########################################################################################################
##########################################################################################################


# In[ ]:

#################
# SIMPLEST MODEL:
#################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# FEATURES: X.iloc[:,selector.get_support()],
# TARGET: y


# In[ ]:

# SIMPLE TRAINING
#X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,selector.get_support()], y, test_size=0.20)
X_train = X.iloc[:,selector.get_support()]
#clf = LogisticRegression(C=1)
#clf = GaussianNB()
#clf = SVC()
#clf = DecisionTreeClassifier()
#clf = RandomForestClassifier()
#clf = AdaBoostClassifier(LogisticRegression(C=1), n_estimators=100)
clf = GradientBoostingClassifier()
#clf = AdaBoostClassifier(n_estimators=100)
#clf.fit(X_train, y)


# In[ ]:

clf.get_params().keys()


# In[ ]:

parameters = {'max_depth':[1, 2, 3, 4, 5]}
est = GradientBoostingClassifier()
clf = GridSearchCV(est, parameters)
clf.fit(X_train, y)
sorted(clf.cv_results_.keys())


# In[ ]:

# SIMPLE TESTING

roc_auc_score(test_df['TARGET'], clf.predict(test_df[keep]))

# Logistic Regression:
# 0.7091457517325841 con 18% de relevant features (25)

# Gaussian Naive Bayes
# 0.69264412016390964 con 18% de relevant features (25)

# Support Vector Classifier
# 0.71902264097217505 con 5% de relevant features (8)

# Decision Tree
# 0.67486172131908473 (0.97539893617021267 en training set) con 18% de relevant features (25).
# Parece tener overfitting (mucha varianza),
# podría reducirse utilizando un método de averagging entre varios árboles - > RandomForest
# -> no lo soluciona, probemos con menos features -> Ahora si
# Con 8% de relevant features (12) mejora bastante, ya no tiene tanto overfitting

# Random Forest
# 0.73459806312401899 con 8% de relevant features (12)
# Probemos Boosting con un modelo que tenga high bias (Logistic Regression con pocas features): no mejora nada, no reduce bias
# ~ Logistic Regression

# AdaBoost con Trees
# 0.75120072861956222 con 18% de relevant features (25)

# Gradient Boosting
# 0.7572604396889383 con 18% de relevant features (25)
# 0.7610946893012418 con el 35% de relevant features (51)


# In[ ]:

roc_auc_score(y, clf.predict(X_train))


# In[ ]:

# Cross-validation TEST with auc score


# In[ ]:

scores = cross_val_score(clf, X_train, y, cv=5, scoring='roc_auc')


# In[ ]:

scores


# In[ ]:

np.mean(scores)


# In[ ]:

#sample_submission_df


# In[ ]:

scoring_y = clf.predict(scoring_df[keep])


# In[ ]:

result = pd.concat([scoring_df['ID'], pd.DataFrame(scoring_y, index=scoring_df.index)], axis=1)


# In[ ]:

result = result.rename(columns={0: "TARGET"})


# In[ ]:

result.to_csv('leaderboard_submission.csv', index=False)


# In[ ]:

#################
# LEARNING CURVES
#################

rg = range(100, 10000, 10)
training_scores = np.zeros((2, len(rg)))
test_scores = np.zeros((2, len(rg)))
nrows, ncols = X_train.shape
clf = LogisticRegression()


# In[ ]:

counter = 0

for j in rg:
    
    clf.fit(X_train.iloc[1:j,:], y_train[1:j])

    # TRAINING score
    training_scores[0, counter] = roc_auc_score(y_train, clf.predict(X_train))
    training_scores[1, counter] = j

    # TESTING score
    test_scores[0, counter] = roc_auc_score(y_test, clf.predict(X_test))
    test_scores[1, counter] = j
    counter += 1


# In[ ]:

# PLOT THESE LEARNING CURVES:
#plt.plot(training_scores[1,:], training_scores[0,:],'r', test_scores[1,:], test_scores[0,:],

