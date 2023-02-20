# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:02:28 2023

@author: Dhrumit Patel
"""
"""
Name: Dhrumit Patel
Student ID: 301171388
"""
import numpy as np
from sklearn.datasets import fetch_openml
MNIST_Dhrumit = fetch_openml('mnist_784', version=1)

# Storing the random numbers to avoid generating numbers differently
np.random.seed(88);

MNIST_Dhrumit.keys()

X_Dhrumit = MNIST_Dhrumit['data']
y_Dhrumit = MNIST_Dhrumit['target']

print(type(X_Dhrumit))
print(type(y_Dhrumit))

print(X_Dhrumit.shape)
print(y_Dhrumit.shape)

import pandas as pd
# Changing dataset to numpy array
X_Dhrumit = pd.DataFrame(X_Dhrumit).to_numpy()
y_Dhrumit = pd.DataFrame(y_Dhrumit).to_numpy()

some_digit1, some_digit2, some_digit3 = X_Dhrumit[7], X_Dhrumit[5], X_Dhrumit[0]

import matplotlib.pyplot as plt
plt.imshow(some_digit1.reshape(28, 28), cmap="binary")
plt.axis("off")
plt.show()
plt.imshow(some_digit2.reshape(28, 28), cmap="binary")
plt.axis("off")
plt.show()
plt.imshow(some_digit3.reshape(28, 28), cmap="binary")
plt.axis("off")
plt.show()

# Pre-processing the data

# Changing the type of y to uint8
y_Dhrumit = y_Dhrumit.astype(np.uint8)

# Any digit between 0 and 3 inclusive should be assigned a target value of 0
y_Dhrumit_1 = np.where(y_Dhrumit <= 3, 0, y_Dhrumit)
# Any digit between 4 and 6 inclusive should be assigned a target value of 1
y_Dhrumit_2 = np.where(y_Dhrumit >= 4, 1, y_Dhrumit_1)
# Any digit between 7 and 9 inclusive should be assigned a target value of 9
y_Dhrumit_3 = np.where(y_Dhrumit >= 7, 9, y_Dhrumit_2)
y_Dhrumit_3class = y_Dhrumit_3

# Actual Values
print("Actual Values:-",y_Dhrumit_3[7])
print("Actual Values:-",y_Dhrumit_3[5])
print("Actual Values:-",y_Dhrumit_3[0])

# Counting the frequencies of each class
unique, counts = np.unique(y_Dhrumit_3class, return_counts=True)

# Printing the frequencies
print("Frequencies of target classes:")
for i in range(len(unique)):
    print("Class", unique[i], ":", counts[i])

# Plot the bar chart
plt.bar(unique, counts, align='center')
plt.xticks(unique)
plt.xlabel("Target Classes")
plt.ylabel("Frequency")
plt.title("Frequency of Target Classes")
plt.show()

# Splitting the data
X_train, X_test, y_train, y_test = X_Dhrumit[:50000], X_Dhrumit[50000:], y_Dhrumit_3[:50000], y_Dhrumit_3[50000:]

"""Logistic Regression"""

# Training a Logistic Regression classifier
from sklearn.linear_model import LogisticRegression

# Logistic Regression classifier with lbfgs solver
LR_clf_Dhrumit_lbfgs = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1200, tol=0.1)
LR_clf_Dhrumit_lbfgs.fit(X_train, y_train)

# Train a Logistic Regression classifier with Saga solver
LR_clf_Dhrumit_saga = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1200, tol=0.1)
LR_clf_Dhrumit_saga.fit(X_train, y_train)

# Cross Validation Score
from sklearn.model_selection import cross_val_score
clf_lbfgs = LR_clf_Dhrumit_lbfgs
clf_saga = LR_clf_Dhrumit_saga
X = X_test
y = y_test

# Using 3-fold cross validation using lbfgs
scores_lbfgs = cross_val_score(clf_lbfgs, X, y, cv=3)
print("The score using 3-fold cross validation (lbfgs):- ",scores_lbfgs)
# Using 3-fold cross validation using saga
scores_saga = cross_val_score(clf_saga, X, y, cv=3)
print("The score using 3-fold cross validation (saga):- ",scores_saga)

# Accuracy Score
from sklearn.metrics import accuracy_score

# Predict the labels for the test data using the trained classifier
y_pred_lbfgs = LR_clf_Dhrumit_lbfgs.predict(X_test)
y_pred_saga = LR_clf_Dhrumit_saga.predict(X_test)

# Calculate the accuracy score using the test data and the predicted labels
accuracy_lbfgs = accuracy_score(y_test, y_pred_lbfgs)
accuracy_saga = accuracy_score(y_test, y_pred_saga)
# Print the accuracy score
print("Accuracy score using lbfgs solver: ", accuracy_lbfgs)
print("Accuracy score using saga solver: ", accuracy_saga)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Calculate and prining the confusion matrix, precision, and recall
cm_lbfgs = confusion_matrix(y_test, y_pred_lbfgs)
cm_saga = confusion_matrix(y_test, y_pred_saga)
print("Confusion Matrix (lbfgs): \n", cm_lbfgs)
print("Confusion Matrix (saga): \n", cm_saga)

precision_lbfgs = precision_score(y_test, y_pred_lbfgs, average='weighted')
precision_saga = precision_score(y_test, y_pred_saga, average='weighted')
print("Precision (lbfgs): ", precision_lbfgs)
print("Precision (saga): ", precision_saga)

recall_lbfgs = recall_score(y_test, y_pred_lbfgs, average='weighted')
recall_saga = recall_score(y_test, y_pred_saga, average='weighted')
print("Recall (lbfgs): ", recall_lbfgs)
print("Recall (saga): ", recall_lbfgs)

# Prediction using LBFGS Solver
some_digit1_pred = LR_clf_Dhrumit_lbfgs.predict(some_digit1.reshape(1, 784))
some_digit2_pred = LR_clf_Dhrumit_lbfgs.predict(some_digit2.reshape(1, 784))
some_digit3_pred = LR_clf_Dhrumit_lbfgs.predict(some_digit3.reshape(1, 784))

print("Prediction for some_digit1(lbfgs): ", some_digit1_pred)
print("Prediction for some_digit2(lbfgs): ", some_digit2_pred)
print("Prediction for some_digit3(lbfgs): ", some_digit3_pred)

# Prediction using SAGA Solver
some_digit1_pred = LR_clf_Dhrumit_saga.predict(some_digit1.reshape(1, 784))
some_digit2_pred = LR_clf_Dhrumit_saga.predict(some_digit2.reshape(1, 784))
some_digit3_pred = LR_clf_Dhrumit_saga.predict(some_digit3.reshape(1, 784))

print("Prediction for some_digit1(saga): ", some_digit1_pred)
print("Prediction for some_digit2(saga): ", some_digit2_pred)
print("Prediction for some_digit3(saga): ", some_digit3_pred)

"""Naive Bayes"""

# Training naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
NB_clf_Dhrumit = MultinomialNB()
NB_clf_Dhrumit.fit(X_train, y_train)

# Using 3-fold cross validation
from sklearn.model_selection import cross_val_score
nb_scores = cross_val_score(NB_clf_Dhrumit, X_train, y_train, cv=3, scoring="accuracy")
print("Naive Bayes 3-fold cross validation:", nb_scores)

y_pred_NB = NB_clf_Dhrumit.predict(X_test)

# Accuracy
accuracy_NB = accuracy_score(y_test, y_pred_NB)
print("Naive Bayes Classifier Test Accuracy:", accuracy_NB)

# Generating the confusion matrix
cm_NB = confusion_matrix(y_test, y_pred_NB)
print("Confusion Matrix: \n", cm_NB)

# Using the classifier to predict the three variables that created above
some_digit1_pred = NB_clf_Dhrumit.predict(some_digit1.reshape(1, 784))
some_digit2_pred = NB_clf_Dhrumit.predict(some_digit2.reshape(1, 784))
some_digit3_pred = NB_clf_Dhrumit.predict(some_digit3.reshape(1, 784))

print("Prediction for some_digit1: ", some_digit1_pred)
print("Prediction for some_digit2: ", some_digit2_pred)
print("Prediction for some_digit3: ", some_digit3_pred)
