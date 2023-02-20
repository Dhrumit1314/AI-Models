# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:04:50 2023

@author: Dhrumit Patel
"""

# Converting tar file from url into csv file
import urllib

urllib.request.urlretrieve("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz", "housing.tgz")

import tarfile
housing_tgz = tarfile.open('housing.tgz')
housing_tgz.extractall()
housing_tgz.close()

# Loading data in pandas dataframe
import pandas as pd
housing = pd.read_csv("housing.csv")

housing.shape
housing.head(5)

housing.info()
housing.describe()

# Checking whether our datasets have null values or not
housing.isnull().values.any()
# Calculating null values in dataasets
housing.isnull().sum()

# Unique values in the column ocean_proximity
housing["ocean_proximity"].unique()
# Counting them
housing["ocean_proximity"].value_counts()

# Note:- median_house_value is target column

# Visualizing

import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize=(20,15))
plt.show()

# Making categories for median income
import numpy as np
housing["income_cat"] = pd.cut(housing["median_income"], bins = [0,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])
housing["income_cat"].hist()
                    # OR
np.ceil(housing['median_income']/1.5)
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
# Less than 5 then keep same or else if value >5 then it will be replace by 5
housing['income_cat'].where(housing['income_cat']<5,5,inplace=True)
housing['income_cat']
housing['income_cat'].hist()

# Random sampling / Spiltting Shuffling Split
housing['income_cat'].value_counts()
# Calculating percentage by dividing by the length of the house
housing['income_cat'].value_counts() / len(housing)

# Random Sampling
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# Calculating percentage by dividing by the length of the test_set
test_set['income_cat'].value_counts() / len(test_set)

# Stratified Shuffling Split
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Creating index value for trainig and testing // Based on income_cat we are splitting
for train_index, test_index in split.split(housing,housing['income_cat'] ):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]

# Evaluating model using this below set
strat_test_set.head(5)

strat_test_set['income_cat'].value_counts() / len(strat_test_set)

# Visualizing the samplings error
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props

# Dropping income_cat column from training and testing sets
for items in (strat_train_set, strat_test_set):
    items.drop("income_cat",axis=1, inplace=True)

# Creating copy of the strat_train_set into housing variable
housing = strat_train_set.copy()
housing.shape
# Checking if column is dropped or not
housing.info()

# Finding the correlation
corr_matrix = housing.corr()
# Comparing matrix with the median_house_value(target) and sorting them in descending order
corr_matrix['median_house_value'].sort_values(ascending=False) # We can see median_income is more related to median_house_value
# Visualizing the above thing in scatter plot
from pandas.plotting import scatter_matrix
attributes = ['median_house_value', 'median_income', 'total_bedrooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,12)) # median_income is linear against median_house_value

# Adding new attributes and making matrix to check if any other feature makes difference or not
# Calculated Attributes
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['population_per_household'] = housing['population']/housing['households']
housing['bedroom_per_room'] = housing['total_bedrooms']/housing['total_rooms']
# Finding correlation and comapring them with target value
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False) # Again median_income is more linearly correlated to median_house_value

# Preparing the dataset - Seperating the data and target(label)
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing.info()

# Data Cleaning
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
# Option 1- Dropping Bedrooms
sample1 = housing.drop('total_bedrooms', axis=1)
sample1.info()
# Option 2 - using dropna method //dropping rows which are NaN
sample2 = housing.dropna(subset=['total_bedrooms'])
sample2.info()
# Option 3 - Computing median and using the imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
# Removing ocean_proximity beacuse it has string values
housing_num = housing.drop('ocean_proximity',axis=1)
imputer.fit(housing_num)
# Printing the median of each and every feature (except: ocean_proximity column)
housing_num.median().values
            # OR
imputer.statistics_

X = imputer.transform(housing_num)
X
# Converting numpy to dataframe
house_tr = pd.DataFrame(X, columns=housing_num.columns)
house_tr
house_tr.info() # No more null values

# Text and Categorical Attribute
housing_cat = housing['ocean_proximity']
housing_cat

# One hot encoding
house_cat_encoded, housing_categories = housing_cat.factorize()
housing_categories
house_cat_encoded[:15] # First 15 values

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(house_cat_encoded.reshape(1,-1))
housing_cat_1hot.toarray() #It will be sparse matrix so conveting it to array to see values

from sklearn.base import BaseEstimator, TransformerMixin
# hard code the column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6

# Creating class
class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:,rooms_ix]/X[:,household_ix]
        population_per_household = X[:,population_ix]/X[:,household_ix]
        
        if self.add_bedrooms_per_room:
            bedroom_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedroom_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
    
attr_adder = CombinedAttributeAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs
housing.info()

# Feature Scaling
housing_scaled = housing.drop("ocean_proximity",axis=1)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
# fit_transform - train and update the model
housing_scaled1 = scalar.fit_transform(housing_scaled)
housing_scaled1

# Building Pipelines
# All the three are called Transformer
# 1.Simple imputer to fix null values
# 2. Adding new attributes - CombinedAttributeAdder
# 3. Feature Scaling - StandardScalar

housing_num = housing.drop('ocean_proximity', axis=1)

# Creating pipleine
from sklearn.pipeline import Pipeline
# Creating numerical pipeline
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributeAdder()),
    ('std_scalar',StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr
# Converting to pandas dataframe for visualizing values
# pd.DataFrame(housing_num_tr)

# Column Transformer - which can handle both numerical and categorical pipeline
from sklearn.compose import ColumnTransformer
# Getting the list of numerical columns and categorical columns
num_attribs = list(housing_num)
num_attribs
cat_attribs = ['ocean_proximity']
cat_attribs

full_pipeline = ColumnTransformer([
    ("num" , num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
# pd.DataFrame(housing_prepared)
# Pipeline created

# Building the model
# Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

# Note :- houing_prepared is training data, housing_labels is testing
lin_reg.fit(housing_prepared, housing_labels) # (Xtrain, Ytrain)
housing_predictions = lin_reg.predict(housing_prepared)

# Computing RMSE , RMSE = (1/m (yactual - ypred)^2)^1/2
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(housing_labels, housing_predictions) # (yactual, ypred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse # if house price is 100k then it will show 100k + 68627 = $168627 (so 68627 is error which is bad)

# SVR (Support Vector Regression) model
from sklearn.svm import SVR
svr_model = SVR(kernel="linear")
svr_model.fit(housing_prepared, housing_labels)

housing_predictions = svr_model.predict(housing_prepared)
SVR_mse = mean_squared_error(housing_labels, housing_predictions)
SVR_rmse = np.sqrt(SVR_mse)
SVR_rmse # error is 111k which is worse

# Decision Tree - Based on feature it spilts dataset. 
# Random Forest - Multiple decision trees
# n estimator is to set how many decison trees we have to add
# max feature is to set the feature/condition 

# Random Forest
# ensemble algorithm
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
# Building model
forest_reg.fit(housing_prepared, housing_labels)
# Predicting
housing_predictions = forest_reg.predict(housing_prepared)
# Calculating MSE and RMSE
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse # 18k which is much better than above 2

# Minimizing error with hyper parameter tuning
# Building parameter grid for random forest

param_grid = [
    {
     'n_estimators' : [3,10,30],
     'max_features' : [2,4,6,8]
    },
    {
     'bootstrap' : [False], # by default it is true
     'n_estimators' : [3,10,30],
     'max_features' : [2,4,6,8]
    }
]

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
best_model = grid_search.best_estimator_
best_model.fit(housing_prepared, housing_labels)
housing_predictions = best_model.predict(housing_prepared)
# Calculating MSE and RMSE
best_model_mse = mean_squared_error(housing_labels, housing_predictions)
best_model_rmse = np.sqrt(best_model_mse)
best_model_rmse # 19k which is bad than random forest one


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Parameter Distribution
param_dist = {
    'n_estimators' : randint(low=1, high=200),
    'max_features' : randint(low=1, high=8)
}

# Building new model
forest_reg_new = RandomForestRegressor(random_state=42)
forest_reg_rand_search = RandomizedSearchCV(forest_reg_new, param_distributions=param_dist, cv=5,scoring='neg_mean_squared_error')
forest_reg_rand_search.fit(housing_prepared, housing_labels)
forest_reg_rand_search.best_params_

final_model = forest_reg_rand_search.best_estimator_
final_model.fit(housing_prepared, housing_labels)

housing_predictions = final_model.predict(housing_prepared)
# Calculating MSE and RMSE
final_model_mse = mean_squared_error(housing_labels, housing_predictions)
final_model_rmse = np.sqrt(final_model_mse)
final_model_rmse

# Testing
X_test = test_set.drop("median_house_value",axis=1)
y_test = test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test,final_predictions) # (yactual, ypred)
final_rmse = np.sqrt(final_mse)
final_rmse

# Pickle File 
# Saving the model
import joblib
joblib.dump(final_model, "final_model.pkl")
# Reloading the model
final_model_reloaded = joblib.load("final_model.pkl")
final_model_reloaded

"""
Note:- 
.fit --> Only trains the data ..if you use this, you need to call .transform again
.transform --> It is used to Manipulate Data by updating it.
.fit_transform --> It trains model as well as transforms. It is used to Manipualte/train model
"""