# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Pre-processing
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,[1]].values

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Making prediction
y_pred = regressor.predict(X_test)

# Visualising the results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Simple Linear Regression')
plt.xlabel('Years of Experience (years)')
plt.ylabel('Salary ($)')
plt.show()



