#################################################
### TESTING THE MODEL IN LOCAL MODE (EVALUATION)
### INVOLVES LEARNING WITH HALF THE DATA SET
### AND TESTING WITH THE OTHER HALF
#################################################

# Divide the given data set into training and test data (comment when preparing for submission:

features_training = features[:len(features)/2]
features_test = features[len(features)/2:]

labels_training = labels[:len(labels)/2]
labels_test = labels[len(labels)/2:]


# [TEST] the division of features
# counter=0
# for datapoint in features_training:
# 	for feature in datapoint:
# 		output.write(str(feature)+" ")
# 	output.write(' ')	
# 	output.write(str(labels[counter]))
# 	output.write("\n")
# 	counter=counter+1

# output.write("\n\n\n\n\n\n\n")

# # let the counter start from where it stopped in the previous loop
# for datapoint in features_test:
# 	for feature in datapoint:
# 		output.write(str(feature)+" ")
# 	output.write(' ')	
# 	output.write(str(labels[counter]))
# 	output.write("\n")
# 	counter=counter+1


# [TEST] the division of labels
# counter=0
# for label in labels_training:
# 	output.write(str(label))
# 	output.write("\n")
# 	counter+=1

# output.write("\n\n\n\n\n\n\n")
# counter=0
# for label in labels_test:	
# 	output.write(str(label))
# 	output.write("\n")
# 	counter+=1


#learner = svm.LinearSVC() # comment when preparing
learner = linear_model.LogisticRegression()
learner.fit(features_training,labels_training) # for submission

# Test if learning works with Support Vector Machines
#predicted_class_SVC = learner.predict(features_test[0])
#print features_test[0], predicted_class_SVC


# SCORE the model
# create a list for predicted labels:
predicted_labels = list()
for datapoint in features_test:
	predicted_labels.append(learner.predict(datapoint))


# [TEST] print into a file the result of predictions:
counter=0
for label in labels_test:	
	output.write(str(label))
	output.write(" -- ")
	output.write(str(predicted_labels[counter]))
	output.write("\n")
	counter+=1



# Count how many predictions are correct:
counter=0
i=0
for label in labels_test:
	if label==predicted_labels[i]:
		counter+=1
	i+=1


# Percentage:
percentage=counter/float(len(labels_test))
print percentage

# close output file 
output.close()