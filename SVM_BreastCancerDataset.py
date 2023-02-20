# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:02:20 2023

@author: Dhrumit Patel
"""

"""Assignment 2 - exercise 1"""

"""Load and check data"""
import pandas as pd
data_dhrumit = pd.read_csv("C:/Users/Dhrumit Patel/College/3402 - Semester 4/COMP 247 - Supervised Learning/Assignments/Assignment_2/breast_cancer.csv")
# Check the names and types of columns
data_dhrumit.info()
data_dhrumit.columns
data_dhrumit.dtypes
# Check the missing values
data_dhrumit.isnull()
data_dhrumit.isnull().sum()
# Check the statistics of the numeric fields (mean, min, max, median, count..etc.)
data_dhrumit.describe()
data_dhrumit.mean()
data_dhrumit.min()
data_dhrumit.max()
data_dhrumit.median()
data_dhrumit.count()

"""Pre-process and visualize the data"""
# Replacing ? with np.nan() method
import numpy as np
data_dhrumit['bare'] = data_dhrumit['bare'].replace('?', np.nan)
# Changing the type to float
data_dhrumit['bare'] = data_dhrumit['bare'].astype(float)
# Checking
data_dhrumit['bare'].dtype

# Finding the median of each column
median = data_dhrumit.median()
median
# Filling any missing data with the median of the column
data_dhrumit = data_dhrumit.fillna(median)
data_dhrumit
# Dropping ID column
data_dhrumit = data_dhrumit.drop('ID',axis=1)
# Checking
data_dhrumit.columns

# Plotting the graphs
data_dhrumit.hist(bins=50, figsize=(20,15))

import seaborn as sns
import matplotlib.pyplot as plt

# plot a histogram of the 'shape' column
sns.histplot(data_dhrumit['shape'], kde=False)
plt.xlabel('Shape')
plt.ylabel('Count')
plt.title('Distribution of Shape')
plt.show()

# plot a scatter plot of the 'size' and 'shape' columns
sns.scatterplot(data_dhrumit, x='size', y='thickness')
plt.xlabel('Size')
plt.ylabel('Thickness')
plt.title('Size vs Thickness')
plt.show()

# plot a bar chart of the 'class' column
sns.countplot(data_dhrumit, x='class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Class')
plt.show()


# Seperating the features from the class
X = data_dhrumit.drop('class',axis=1)
y = data_dhrumit['class'] # class is the target column

# Train and Test Split of 80% and 20% of data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=88)

# Building classification model
# SVM model
from sklearn.svm import SVC
# Training an SVM classifier using the training data and
# setting the kernel to linear and set the regularization parameter to C= 0.1.
clf_linear_dhrumit = SVC(kernel='linear', C=0.1)

# Training the model
clf_linear_dhrumit.fit(X_train,y_train)

# Printing Accuracy Score
from sklearn.metrics import accuracy_score

# Accuracy score for training data
y_train_pred_lin = clf_linear_dhrumit.predict(X_train)
acc_score_train_lin = accuracy_score(y_train, y_train_pred_lin)
print("Accuracy Score on training set using linear kernel:- ",acc_score_train_lin)

# Accuracy score for testing data
y_test_pred_lin = clf_linear_dhrumit.predict(X_test)
acc_score_test_lin = accuracy_score(y_test, y_test_pred_lin)
print("Accuracy Score on testing set using linear kernel:- ",acc_score_test_lin)

# Generating accuracy matrix
from sklearn.metrics import confusion_matrix
# Predicting against X_test
y_pred_lin = clf_linear_dhrumit.predict(X_test)
cm_dhrumit_lin = confusion_matrix(y_test, y_pred_lin)
print("Confusion Matrix (linear kernel):- \n",cm_dhrumit_lin)  
# Calculating Precision and Recall of the matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precision_lin = precision_score(y_test, y_pred_lin, average='weighted')
precision_lin
recall_lin = recall_score(y_test, y_pred_lin, average='weighted')
recall_lin


# Now using RBF kernel
clf_rbf_dhrumit = SVC(kernel='rbf')

# Training the model
clf_rbf_dhrumit.fit(X_train,y_train)

# Printing Accuracy Score
from sklearn.metrics import accuracy_score

# Accuracy score for training data
y_train_pred_rbf = clf_rbf_dhrumit.predict(X_train)
acc_score_train_rbf = accuracy_score(y_train, y_train_pred_rbf)
print("Accuracy score on training set using rbf kernel:- ",acc_score_train_rbf)

# Accuracy score for testing data
y_test_pred_rbf = clf_rbf_dhrumit.predict(X_test)
acc_score_test_rbf = accuracy_score(y_test, y_test_pred_rbf)
print("Accuracy score on testing set using rbf kernel:- ",acc_score_test_rbf)
# Predicting against X_test
y_pred_rbf = clf_rbf_dhrumit.predict(X_test)
cm_dhrumit_rbf = confusion_matrix(y_test, y_pred_rbf)
print("Confusion Matrix (RBF kernel):- \n",cm_dhrumit_rbf)
# Calculating Precision and Recall of the matrix
precision_rbf = precision_score(y_test, y_pred_rbf, average='weighted')
precision_rbf
recall_rbf = recall_score(y_test, y_pred_rbf, average='weighted')
recall_rbf

# Now using Poly kernel
clf_poly_dhrumit = SVC(kernel='poly')

# Training the model
clf_poly_dhrumit.fit(X_train,y_train)

# Printing Accuracy Score
# Accuracy score for training data
y_train_pred_poly = clf_poly_dhrumit.predict(X_train)
acc_score_train_poly = accuracy_score(y_train, y_train_pred_poly)
print("Accuracy score on training test using poly kernel:- ",acc_score_train_poly)

# Accuracy score for testing data
y_test_pred_poly = clf_poly_dhrumit.predict(X_test)
acc_score_test_poly = accuracy_score(y_test, y_test_pred_poly)
print("Accuracy score on testing set using poly kernel:- ",acc_score_test_poly)
# Predicting against X_test
y_pred_poly = clf_poly_dhrumit.predict(X_test)
cm_dhrumit_poly = confusion_matrix(y_test, y_pred_poly)
print("Confusion Matrix (Poly kernel):- \n",cm_dhrumit_poly)
# Calculating Precision and Recall of the matrix
precision_poly = precision_score(y_test, y_pred_poly, average='weighted')
precision_poly
recall_poly = recall_score(y_test, y_pred_poly, average='weighted')
recall_poly

# Now using Sigmoid kernel
clf_sigmoid_dhrumit = SVC(kernel='sigmoid')

# Training the model
clf_sigmoid_dhrumit.fit(X_train,y_train)

# Printing Accuracy Score
# Accuracy score for training data
y_train_pred_sigmoid = clf_sigmoid_dhrumit.predict(X_train)
acc_score_train_sigmoid = accuracy_score(y_train, y_train_pred_sigmoid)
print("Accuracy score on training set using sigmoid kernel:- ",acc_score_train_sigmoid)

# Accuracy score for testing data
y_test_pred_sigmoid = clf_sigmoid_dhrumit.predict(X_test)
acc_score_test_sigmoid = accuracy_score(y_test, y_test_pred_sigmoid)
print("Accuracy score on testing set using sigmoid kernel:- ",acc_score_test_sigmoid)
# Predicting against X_test
y_pred_sigmoid = clf_sigmoid_dhrumit.predict(X_test)
cm_dhrumit_sigmoid = confusion_matrix(y_test, y_pred_sigmoid)
print("Confusion Matrix (Sigmoid kernel):- \n",cm_dhrumit_sigmoid)
# Calculating Precision and Recall of the matrix
precision_sigmoid = precision_score(y_test, y_pred_sigmoid, average='weighted')
precision_sigmoid
recall_sigmoid = recall_score(y_test, y_pred_sigmoid, average='weighted')
recall_sigmoid

"""-----------------------------------------------------------------------------------------------------------"""

"""Assigment 2 - exercise 2"""

# Loading data using pandas dataframe
data_dhrumit_df2 = pd.read_csv("C:/Users/Dhrumit Patel/College/3402 - Semester 4/COMP 247 - Supervised Learning/Assignments/Assignment_2/breast_cancer.csv")
# Replacing ? with np.nan() method
data_dhrumit_df2['bare'] = data_dhrumit_df2['bare'].replace('?', np.nan)
# Changing the type to float
data_dhrumit_df2['bare'] = data_dhrumit_df2['bare'].astype(float)
# Checking
data_dhrumit_df2['bare'].dtype
# Dropping ID column
data_dhrumit_df2 = data_dhrumit_df2.drop('ID',axis=1)
# Checking
data_dhrumit_df2.columns
# Seperating the features from the class
X = data_dhrumit_df2.drop('class',axis=1)
y = data_dhrumit_df2['class'] # class is the target column
# Train and Test Split of 80% and 20% of data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=88)

# Defining 2 transformers
# 1. SimpleImputer to fill the missing values with the median
from sklearn.impute import SimpleImputer
miss_val_median = SimpleImputer(strategy='median')
# 2. StandardScalar to scale the data
from sklearn.preprocessing import StandardScaler
scaled_data = StandardScaler()
# Combining both transformers into one pipeline
from sklearn.pipeline import Pipeline
num_pipe_dhrumit = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scalar', StandardScaler())
])
num_pipe_dhrumit
# Creating a new Pipeline with two steps the first is the num_pipe_dhrumit
# and the second is an SVM classifier with random state = 88 and naming the pipeline pipe_svm_dhrumit.  
pipe_svm_dhrumit = Pipeline([
    ('transfomer', num_pipe_dhrumit),
    ('classifier', SVC(kernel='linear',C=0.1,random_state=88))
])
pipe_svm_dhrumit
# Grid Search Parameter
param_grid = {
    'classifier__kernel': ['linear','rbf','poly'],
    'classifier__C': [0.01,0.1, 1, 10, 100],
    'classifier__gamma':[0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    'classifier__degree':[2,3]
}
param_grid
# Creating a grid search object and naming it grid_search_dhrumit with the mentioned parameters in the document
from sklearn.model_selection import GridSearchCV
grid_search_dhrumit = GridSearchCV(estimator=pipe_svm_dhrumit, param_grid=param_grid, scoring='accuracy',refit=True,verbose=3)
grid_search_dhrumit
# Fitting the training data to the gird search object
grid_search_dhrumit.fit(X_train,y_train)
# Printing out the best parameters 
print("Best Parameters:- ",grid_search_dhrumit.best_params_)
# Printout the best estimator 
print("Best Estimator:- ",grid_search_dhrumit.best_estimator_)


# Fitting test data into grid search object
grid_search_dhrumit.predict(X_test)
# Accuracy Score
print("Accuracy Score:- ",grid_search_dhrumit.score(X_test,y_test))
# Storing best estimator of model in best_model_dhrumit
best_model_dhrumit = grid_search_dhrumit.best_estimator_

import joblib
# Saving the model using joblib
joblib.dump(best_model_dhrumit,"best_model_dhrumit.pkl")
# Saving the pipeline
joblib.dump(pipe_svm_dhrumit, "full_pipeline_dhrumit.pkl")
