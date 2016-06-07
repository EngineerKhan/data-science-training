
from aux_functions import summary_generator, folder_summary
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


folder = './data/'


# 0.1. Open folder or csv files
# ------------------------------

# training set
train_df = pd.read_csv(open(folder+'train.csv','r'), engine='python')
#print ('training set nrows = ' + str(len(train_df.axes[0])))

# test set
test_df = pd.read_csv(open(folder+'test.csv','r'), engine='python')
#print ('test set nrows = ' + str(len(test_df.axes[0])))

# event_type set
event_type_df = pd.read_csv(open(folder+'event_type.csv','r'), engine='python')
#print ('event_type set nrows =' + str(len(event_type_df.axes[0])))

# log_feature set
log_feature_df = pd.read_csv(open(folder+'log_feature.csv','r'), engine='python')
#print ('log_feature set nrows =' + str(len(event_type_df.axes[0])))

# resource_type set
resource_type_df = pd.read_csv(open(folder+'resource_type.csv','r'), engine='python')
#print ('resource_type set nrows =' + str(len(event_type_df.axes[0])))

# severity_type set
severity_type_df = pd.read_csv(open(folder+'severity_type.csv','r'), engine='python')
#print ('severity_type set nrows =' + str(len(event_type_df.axes[0])))




# 0.2. Show summary 
# -----------------
#folder_summary(folder)
#summary_generator(dataframe)


# 1.1.1 clean the columns of the training dataframe
# ------------------------------------------------
train_df['location'] = train_df['location'].map(lambda x: int(x.split(' ')[1]))
#train_df = pd.get_dummies(train_df, columns=['fault_severity']) # get dummy variables for event type
#print(train_df)

# 1.1.2 clean the columns of the test dataframe
# --------------------------------------------
test_df['location'] = test_df['location'].map(lambda x: int(x.split(' ')[1]))
#test_df = pd.get_dummies(test_df, columns=['fault_severity']) # get dummy variables for event type
#print(test_df)


# 1.2. clean the columns of the event_type dataframe and tidy
# -----------------------------------------------------------
event_type_df['event_type'] = event_type_df['event_type'].map(lambda x: int(x.split(' ')[1]))
event_type_df = pd.get_dummies(event_type_df, columns=['event_type']) # get dummy variables for event type
event_type_df = event_type_df.groupby(event_type_df.id,as_index=False).sum() # compact the rows with the same id
#event_type_df.iloc[:,1:] = event_type_df.iloc[:,1:].astype(bool) # convert the rows to boolean types # keep it as float, 
# to count the occurrences of event types. This conversion can be made with the final dataset
#print (event_type_df)


# 1.3 clean the columns of the log_feature dataframe (I wonder if this dataframe is useful)
# -----------------------------------------------------------------------------------------
#log_feature_df['log_feature'] = log_feature_df['log_feature'].map(lambda x: int(x.split(' ')[1]))
#print(log_feature_df)
#print(sorted(log_feature_df.log_feature.unique()))


# 1.4 clean the columns of the resource_type dataframe  and tidy
# --------------------------------------------------------------
resource_type_df['resource_type'] = resource_type_df['resource_type'].map(lambda x: int(x.split(' ')[1]))
resource_type_df = pd.get_dummies(resource_type_df, columns=['resource_type'])
resource_type_df = resource_type_df.groupby(resource_type_df.id,as_index=False).sum()
#print(resource_type_df)


# 1.5 clean the columns of the severity_type dataframe
# ----------------------------------------------------
severity_type_df['severity_type'] = severity_type_df['severity_type'].map(lambda x: int(x.split(' ')[1]))
#print(severity_type_df)



#########
# MERGE #
#########

# 2.1. Left join training/test set <-> event_type set
merged_df = pd.merge(left=train_df, right=event_type_df, how='left', left_on='id', right_on='id')
merged_test_df = pd.merge(left=test_df, right=event_type_df, how='left', left_on='id', right_on='id')
#print(merged_df)
#print ('merged set nrows =' + str(len(merged_df.axes[0])))


# 2.2. Left join training/test set <-> log_feature set
#merged_df = pd.merge(left=merged_df, right=log_feature_df, how='left', left_on='id', right_on='id')
#merged_test_df = pd.merge(left=merged_test_df, right=log_feature_df, how='left', left_on='id', right_on='id')
#print(merged_df)
#print ('merged set nrows =' + str(len(merged_df.axes[0])))


# 2.3. Left join training/test set <-> resource_type set
merged_df = pd.merge(left=merged_df, right=resource_type_df, how='left', left_on='id', right_on='id')
merged_test_df = pd.merge(left=merged_test_df, right=resource_type_df, how='left', left_on='id', right_on='id')
#print(merged_df)
#print ('merged set nrows =' + str(len(merged_df.axes[0])))


# 2.4. Left join training set <-> severity_type set
merged_df = pd.merge(left=merged_df, right=severity_type_df, how='left', left_on='id', right_on='id')
merged_test_df = pd.merge(left=merged_test_df, right=severity_type_df, how='left', left_on='id', right_on='id')
#print(merged_df)
#print ('merged set nrows =' + str(len(merged_df.axes[0])))

# Keep location and id sorted
merged_df = merged_df.sort_values(by=['location','id'], ascending=True)

# Create the time variable
merged_df['time'] = merged_df.groupby('location').cumcount()


#########
# TRAIN #
#########
print(merged_df[['id','location','time']].to_string(index=False))
#exit()
# remove location for training. These two should only be used to map feature vectors
del merged_df['location']
del merged_df['id']
#merged_df = merged_df.iloc[:,2:]
#merged_df.to_csv('./merged_df', index=False) # write to csv
#print(merged_df)
# Convert dataframe to numpy array:
training_array = merged_df.values#.astype(int)

X = training_array[:,1:]
y = training_array[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

print('here 1')

#clf = GaussianNB().fit(X_train, y_train)
clf = svm.SVC(kernel='linear', C=1).fit(X, y)
print(clf.score(X_test, y_test))

print('here 2')
exit()

########
# TEST #
########

merged_test_df = merged_test_df.sort_values(by=['location','id'], ascending=True)
merged_test_df['time'] = merged_test_df.groupby('location').cumcount()

# remove id and location for test. These two should only be used to map feature vectors
del merged_test_df['location']
del merged_test_df['id']
#merged_test_df_noloc = merged_test_df.iloc[:,2:]
#merged_test_df.to_csv('./merged_test_df', index=False) # write to csv
#print(merged_test_df)

print('here 3')

# Convert dataframe to numpy array:
X_test = merged_test_df.values

print('here 4')

y_test = clf.predict(X_test).astype(int)

print('here 5')

# build submission dataframe and file:
submission_df = pd.DataFrame(data=y_test, columns=['predict'], dtype=int)
submission_df = pd.get_dummies(submission_df, columns=['predict']).astype(int) # IMPORTANT! keeping so mant floats in memory is not good, I would like get_dummies to convert to int directly (I would add this functionality)
submission_df = pd.concat([merged_test_df.iloc[:,0:1], submission_df], axis=1)

submission_df.to_csv('submission_df.csv', index=False)