# coding=utf-8
#####################################################
# Tyler Hedegard
# 7/1/16
# Thinkful Data Science
# Decision Tree
#####################################################

# Imports ###########################################
import pandas as pd
import numpy as np
import re
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# Functions #########################################
def rem_regex(data, column, chars):
    data[column] = data[column].map(lambda x: re.sub(chars, '', x, count=0, flags=0))

# Extract raw data ##################################
# Training data
header = pd.read_csv('UCI HAR Dataset/features.txt', sep=' ', header=None, engine='python')
data = pd.read_table('UCI HAR Dataset/train/X_train.txt', header=None, sep=r"\s*", engine='python')
label = pd.read_table('UCI HAR Dataset/train/y_train.txt', header=None)
# Test data
test_data = pd.read_table('UCI HAR Dataset/test/X_test.txt', header=None, sep=r"\s*", engine='python')
test_label = pd.read_table('UCI HAR Dataset/test/y_test.txt', header=None)

# Clean data ########################################
# Identify and fix the inclusion of ( ) in column names.
# Identify and fix the inclusion of ‘-’ in column names.
# Identify and fix extra ) in some column names.
# Identify and fix inclusion of multiple ‘,’ in column names.
rem_regex(header, 1, '[(),-]')
# Identify and fix column names containing “BodyBody”
# Drop 'Body' and 'Mag' from column names.
rem_regex(header, 1, 'Body')
rem_regex(header, 1, 'Mag')
# Map 'mean' and 'std' to 'Mean' and 'STD'
header[1] = header[1].map(lambda x: re.sub('mean', 'Mean', x, count=0, flags=0))
header[1] = header[1].map(lambda x: re.sub('std', 'STD', x, count=0, flags=0))

# Identify and remove duplicate column names.
sys.setrecursionlimit(1500) # Needed to handle 'RuntimeError: maximum recursion depth exceeded in cmp'
data.columns = header[1].transpose()
test_data.columns = header[1].transpose()
data = data.T.drop_duplicates().T
test_data = test_data.T.drop_duplicates().T

# Make 'activity' a categorical variable.
label_meaning = pd.read_table('UCI HAR Dataset/activity_labels.txt', header=None, sep="\s", engine='python')
label_meaning.columns = ('Category','Activity')
# data['label'] = label

# Plot a histogram of Body Acceleration Magnitude (i.e. histogram of all 6 activities)
# to see how each variable does as a predictor of static versus dynamic activities.
# data['tAccMean'].hist(by=label[0])

# Split the data into training, test, and validation sets.
# training set = data
# test set = test_data
# Create validation sets
kf = KFold(len(data), n_folds=10)
print("-----Using cross validation-------------------------------------------")
for train, test in kf:
    X_train, X_test = data.T[train].T, data.T[test].T
    y_train, y_test = label[0].T[train].T, label[0].T[test].T
    # Fit a random forest classifier with 500 estimators to your training set.
    clf = RandomForestClassifier(n_estimators=500)
    clf.fit(X_train, y_train)
    importance = clf.feature_importances_
    rank = importance.argsort()[::-1]
    # Rank the features by their importance scores. What are the top 10 important features?
    top_ten = header[1][rank[:10]]
    print("What are the top 10 important features?")
    print(top_ten)
    # What is the 10th feature's importance score?
    print("What is the 10th feature's importance score?")
    print("#10: " + top_ten[rank[9]] + ' : ' + str(importance[9]))
    # What is your model's mean accuracy score on the validation and test sets?
    print("Test score: {}".format(clf.score(X_test, y_test)))
    # What is your model's precision and recall score on the test set?
    predict = clf.predict(X_test)
    precision = precision_score(y_test, predict, average='weighted')
    print("Precision: {}".format(precision))
    recall = recall_score(y_test, predict, average='weighted')
    print("Recall: {}".format(recall))

print("-----Not using cross validation-------------------------------------------")
clf = RandomForestClassifier(n_estimators=500)
clf.fit(data, label[0])
importance = clf.feature_importances_
rank = importance.argsort()[::-1]
top_ten = header[1][rank[:10]]
print("What are the top 10 important features?")
print(top_ten)
print("What is the 10th feature's importance score?")
print("#10: " + top_ten[rank[9]] + ' : ' + str(importance[9]))
print("Test score: {}".format(clf.score(test_data, test_label)))
predict = clf.predict(test_data)
precision = precision_score(test_label, predict, average='weighted')
print("Precision: {}".format(precision))
recall = recall_score(test_label, predict, average='weighted')
print("Recall: {}".format(recall))
