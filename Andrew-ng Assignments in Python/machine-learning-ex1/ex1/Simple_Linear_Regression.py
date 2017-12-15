#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:01:54 2017

@author: siddhantchadha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data pre-processing
dataset = pd.read_csv('ex1data1.txt')
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, [1]].values

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

coefficients = regressor.coef_
intercept = regressor.predict([[0]])

# Predicting the results
y_pred = regressor.predict(X_test)

# Visualise the results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
