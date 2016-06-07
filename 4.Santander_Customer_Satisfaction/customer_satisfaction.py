
# CHANGE WORKING DIRECTORY AND ADD MY LIBRARIES
import os
os.chdir('./4.Santander_Customer_Satisfaction')
import sys
sys.path.insert(0, '../mylib/')

# PACKAGES
import pandas as pd
import numpy as np
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

# MAGIC COMMANDS
%matplotlib inline

# LOAD DATA FILES
training_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')

# Remove constant features
# "Constant features can lead to errors in some models and obviously provide no information in the training set that can be learned from."
remove = []

for col in training_df.columns:
    if training_df[col].values.std() == 0: # pandas.series std() is not correct, use numpy std() instead (.values.std() instead of std())
        remove.append(col)

training_df.drop(remove, axis=1, inplace=True)
test_df.drop(remove, axis=1, inplace=True)

# Remove duplicated columns
remove = []
c = training_df.columns

for i in range(len(c)-1):
    v = training_df[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,training_df[c[j]].values):
            remove.append(c[j])
            
training_df.drop(remove, axis=1, inplace=True)
test_df.drop(remove, axis=1, inplace=True)

# EXPLORATORY ANALYSIS:
# check if there is any repeated ID, which would imply to tidy the data set:
id_counts_df = training_df['ID'].value_counts().sort_index()  # count the number of occurrences of each ID
max(id_counts_df)  # if the max value is 1, then there are no repeated IDs = 1 row for each observation.

#training_df.describe()  # describe training set
column_names = list(training_df)  # show all column names. Shorter alternative to my_dataframe.columns.values.tolist()
#column_names
describe_file = open('describe_file.txt', 'w+')
describe_file.write(training_df.describe().to_json()) 
describe_file.close()

# Explore each variable individually:

# An important factor would be the demographics. Younger generations tend to be less conformist, so they are very
# likely to be unhappy clients if there is something wrong. There is no variable named "age", so it would be important 
# to find it. Describe could give us this information, for example, by looking at the mean. var15 has a mean of 33.21. 
# It sounds like the average age of clients, so lets explore this variable and other variables with similar mean values.

# var15:

# By looking at the describe table, var15 min = 5, var15 max = 105.
# Not very convincing... yet. Do 5 year old kids already have bank accounts? is it usual to find 105 years old people?
# We should check which % of values are between 5-18 and 85-105. 
# If this % consists on the long tails of the distribution, then this might be the age variable. 
#sns.set(rc={"figure.figsize": (16, 8)})
#plt = sns.distplot(training_df['var15'], 
#             hist_kws={"linewidth": 1},  # histogram
#             rug_kws={"color": "g"},  # Plot datapoints in an array as sticks on an axis.
#             kde_kws={"color": "b", "lw": 2, "label": "mean"}  # Fit and plot a univariate or bivariate kernel density estimate.
#             )

# * I cant figure out how to change the number of data points shown in the x axis: instead of 0, 20, 40... etc I would like it to look like: 
# 0, 10, 20, 30, 40... etc

# Well, it seems like a big amount of the population is near 25 years old. It is the right variable. I think Santander offers good deals for 
# young professionals and recent graduates when it comes to bank accounts. Given the name of the variables, I think this data set corresponds 
# to the spanish branch of the bank. However, they might have ideas in common with other branches, such as Santander UK. I was student there, 
# and I remember they were one the banks which made it easier for students to open a bank account (few paperwork involved). 
# However, the used to charge 5 pounds per month for maintenance. If this trend has been up for years, it could explain why there are 
# so many young customers, and less older ones. I actually left the bank a month ago because of these monthly fees, so this could be an 
# explanation for this distribution. The combination young customer + fees might be quite significative for customers to be unsatisfied. 
# I will keep this in mind.



# There are several boolean variables in the dataset. Lets figure out what they mean. First, I will get all their names and the proportion 
# of 0s and 1s
booleans = [col for col in training_df.columns if min(training_df[col]) == 0 if max(training_df[col]) == 1]
#list(training_df[booleans])

for i in range(0,len(booleans)):
    print(str(booleans[i]) + ': ',sum(training_df[booleans[i]].values)/len(training_df[booleans[i]].values))

# I realized the TARGET variable is very unbalanced (value 1 = 0.0395685345962%). I should apply something in order to fix this problem. 
# That might improve the final results -> Support Vector Classifier with linear kernel

# DATA TRANSFORMATION

# remove the ID field from DataFrames, but save first
training_IDs = training_df['ID']
test_IDs = test_df['ID']

training_df = training_df.drop('ID',axis=1)
test_df = test_df.drop('ID',axis=1)

# Transform age column to: likely to change = 1, unlikely to change = 0. Lets establish the threshold above 40 years old.
# http://discuss.analyticsvidhya.com/t/difference-between-map-apply-and-applymap-in-pandas/2365
#training_df['var15'] = training_df['var15'].map(lambda x: 1 if x < 40 else 0)
#test_df['var15'] = test_df['var15'].map(lambda x: 1 if x < 40 else 0)

# Lets explore more variables:
# if we have delta_imp_amort_var18_1y3, delta_imp_amort_var34_1y3, delta_imp_aport_var13_1y3, delta_imp_aport_var17_1y3, delta_imp_aport_var33_1y3,
# lets explore just the first and the second one, as they should be similar.

# var3: most values around zero. I don't find this variable meaningful nor useful
# delta_imp_amort_var18_1y3: " most values around zero
# delta_num_aport_var13_1y3: " most values around zero
# imp_amort_var18_ult1: " most values around zero
# imp_amort_var34_ult1: " most values around zero
# imp_aport_var13_hace3: " most values around zero
# imp_compra_var44_hace3: " most values around zero

# Let's try one of the variables chosen by the feature selector: ind_var5. Cool, the distribution seems to be definately more useful. It is a boolean variable.
# How does it correlate to the TARGET variable?

# Now, let's explore a variable chosen by the feature selector which is not in the list of boolean variables: num_var4: It has values = [0, 1, 2, 3, 4, 5, 6, 7]
# How does it correlate to the TARGET variable?

# Description:
#training_df['num_var4'].describe()

# Distribution:
#sns.set(rc={"figure.figsize": (16, 8)})
#plt = sns.distplot(training_df['num_var4'], hist_kws={"linewidth": 1}, rug_kws={"color": "g"}, kde_kws={"color": "b", "lw": 2, "label": "mean"})

# Correlation with TARGET variable: lmplot with or without regression line (fit_reg=True or False). A flat regression line (slope = 0) means no correlation:
#sns.lmplot("num_var4", "TARGET", data=training_df, fit_reg=True)

#ordered_columns = sorted(training_df.columns.tolist())
#for col in ordered_columns:
#    print(col)

# DataFrames to numpy arrays
training_array = training_df.values 
X = training_array[:, :-1] 
y = training_array[:, -1] 

test_array = test_df.values 
X_test = test_array[:, ]

# Feature selection
feature_selector = SelectKBest(f_classif, k=5).fit(X, y) 
support = feature_selector.get_support(indices=True)

i = 0 
for index in support:
    print(str(i) + " - " + str(index) + ": " + str(training_df.columns[index]))
    i += 1

#X_transformed = feature_selector.transform(X)
#X_test_transformed = feature_selector.transform(X_test)

# Check the matching:
# print(training_df.iloc[73517,280])
# print(X_transformed[73517,19])

# Back to dataframe format

#y = np.transpose(y) # this does not work, as the array is 1-Dimensional: http://stackoverflow.com/questions/5954603/transposing-a-numpy-array
#concatenated = np.concatenate((X_transformed, np.transpose([y])), axis=1)

#transformed_columns = training_df.columns[support].values.tolist()
#transformed_columns.append('TARGET')

#transformed_training_df = pd.DataFrame(concatenated, columns=transformed_columns)

# Remove more columns
#final_columns = training_df[['var15']+booleans[:-1]].columns

# Get the union of the boolean variables and the chosen variables by the feature selector
#final_columns = training_df[['var15']+booleans[:-1]].columns | training_df.columns[support]
extra_list = ['var15','saldo_var30','var36']
# From the feature analysis of the next code block, I test the SVC with similar variables, without improvements:
#extra_list = ['var15','saldo_var30','var36','ind_var8','ind_var12','ind_var13','ind_var24','ind_var30']
final_columns = training_df[extra_list + booleans[:5]].columns | training_df.columns[support]
final_columns = final_columns.tolist()

# From the feature analysis of the next code block, I removed those variables which does not seem very significative, given they got most
# values around one of the two possible values (boolean variables)
#final_columns = [x for x in final_columns if x not in ['ind_var1', 'ind_var1_0', 'ind_var5_0', 'ind_var6_0']]

#final_columns.extend(['TARGET']) != final_columns+['TARGET'] # The former doesnt modify final_columns, the latter does.

print(len(final_columns))
for e in final_columns:
    print(e) 
final_training_df = training_df[final_columns+['TARGET']]
final_test_df = test_df[final_columns]

# Lets study these variables:
# ind_var1: 
#   Distribution: it seems to have many zeros and few ones, so this variable might not be very significative.
#   Correlation with TARGET: not much
# ind_var_0: same as the previous one
# ind_var_30:
#   Distribution: this one seems to be more balanced, ~75% = 1, 25% = 0
#   Correlation with TARGET: not much
# ind_var_5:
#   Distribution: this one seems to be more balanced
#   Correlation with TARGET: not much
# ind_var5_0:
#   Distribution: very biased to 1
#   Correlation with TARGET: not much
# ind_var6_0:
#   Distribution: very biased to 0
#   Correlation with TARGET: not much
# num_meses_var5_ult3:
#   Distribution: 0,1,2,3 not biased
#   Correlation with TARGET: not much
# num_var30:
#   Distribution: [ 0,  3,  6,  9, 12, 15, 18, 33, 21] -> 0, 3 and 6 are the peaked values
#   Correlation with TARGET: interesting
# num_var42:
#   Distribution: [ 0,  3,  6,  9, 12, 15, 18] -> 0, 3 and 5 are the peaked values
#   Correlation with TARGET: interesting
# saldo_var30:
#   Distribution: [-4942.260000, 3458077.320000]
#   Correlation with TARGET: very interesting
# var15:
#   Distribution: [5, 105] -> mean at 33.21
#   Correlation with TARGET: interesting
# var36:
#   Distribution: [99,  3,  2,  1,  0] -> mean at 40.44, peaked at 1 and 99
#   Correlation with TARGET: not much


# remove: 'ind_var1','ind_var_0','ind_var_5_0','ind_var_6_0','','',''

current_var = 'var36'

# Description:
training_df[current_var].unique()
training_df[current_var].describe()

#sns.set(rc={"figure.figsize": (16, 8)})
# Distribution:
#plt = sns.distplot(training_df[current_var], hist_kws={"linewidth": 1}, rug_kws={"color": "g"}, kde_kws={"color": "b", "lw": 2, "label": "mean"})

# Correlation with TARGET variable: lmplot with or without regression line (fit_reg=True or False). A flat regression line (slope = 0) means no correlation:
#sns.lmplot(current_var, 'TARGET', data=training_df, fit_reg=True)

# Scatterplot with seaborn
#sns_plot = sns.pairplot(transformed_training_df, hue='TARGET', size=2.5)
#sns_plot.savefig("scatterplot10.png")
#plt.show()

# DataFrames to numpy arrays
X_transformed = final_training_df.ix[:, final_training_df.columns != 'TARGET'].values
y = final_training_df['TARGET'].values

X_test_transformed = final_test_df.ix[:, final_test_df.columns != 'TARGET'].values

# Balancing the training set:

# Get 100 examples with TARGET = 1
positives_df = final_training_df[final_training_df['TARGET'] == 1].head(100)

# GET 100 examples with TARGET = 0
negatives_df = final_training_df[final_training_df['TARGET'] == 0].head(100)

# merge the two subsets
balanced_training_df = pd.concat([positives_df,negatives_df])

# shuffle the rows
balanced_training_df = balanced_training_df.sample(frac=1).reset_index(drop=True)

# Transform to numpy arrays
X_transformed = balanced_training_df.ix[:, balanced_training_df.columns != 'TARGET'].values
y = balanced_training_df['TARGET'].values

X_test_transformed = final_test_df.ix[:, final_test_df.columns != 'TARGET'].values

# CLASSIFIER & TRAINING: support vector machine, class imbalance handling
#clf = svm.SVC(kernel='linear', C=1.0)
clf = svm.SVC(kernel='linear', class_weight='balanced')
clf.fit(X_transformed, y)

# TEST: cross-validation
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_transformed, y, test_size=0.4, random_state=0)
roc_auc_score(y_test_cv, clf.predict(X_test_cv).astype(int))

# CLASSIFIER & TRAINING: bagging with kneighbors
#clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5) 
clf = BaggingClassifier(GaussianNB(), max_samples=0.5, max_features=0.5) 
clf.fit(X_transformed, y)

# CLASSIFIER & TRAINING: adaboost with gaussian naive bayes
clf = AdaBoostClassifier(GaussianNB(),
                         algorithm="SAMME",
                         n_estimators=200)

clf.fit(X_transformed, y)

# CLASSIFIER & TRAINING: random forest
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_transformed, y)

# CLASSIFIER & TRAINING: naive bayes, gaussian
clf = GaussianNB()
clf.fit(X_transformed, y)  #svm.SVC(kernel='linear', C=1).fit(X, y)

# Using a subset:
#X_first_half, X_second_half = np.split(X_transformed, 2, axis=0)
#y_first_half, y_second_half = np.split(y, 2, axis=0)

X_part = X_transformed[300:450 , :]
y_part = y[300:450]

sum(y_part)/len(y_part)
# 300 -> 450 is a good subset (auc = 0.73442800039992329), with 5.33% of TARGET = 1 examples

# CLASSIFIER & TRAINING: support vector machine, class imbalance handling
#clf = svm.SVC(kernel='linear', C=1.0)
clf = svm.SVC(kernel='linear', class_weight='balanced')
clf.fit(X_part, y_part)

# TEST: cross-validation
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_transformed, y, test_size=0.4, random_state=0)
roc_auc_score(y_test_cv, clf.predict(X_test_cv).astype(int))

# PREDICTION
prediction = clf.predict(X_test_transformed).astype(int)

# RESULTS: preparation
result_df = pd.concat([pd.DataFrame(test_IDs), pd.DataFrame(prediction).astype(int)], axis=1)
result_df.columns = ['ID', 'TARGET']
#result_df

# RESULTS: to csv
result_df.to_csv('result_SVC_rbf_balanced_3000.csv', index=False, dtype=int)

#result_df.to_csv('result_AdaBoostGNB.csv', index=False, dtype=int)

#result_df.to_csv('result_BaggingGaussianNaiveBayes.csv', index=False, dtype=int)

#result_df.to_csv('result_BaggingKNearestNeighbors.csv', index=False, dtype=int)

#result_df.to_csv('result_GaussianNaiveBayes.csv', index=False, dtype=int)

#result_df.to_csv('result_RandomForest.csv', index=False, dtype=int)
