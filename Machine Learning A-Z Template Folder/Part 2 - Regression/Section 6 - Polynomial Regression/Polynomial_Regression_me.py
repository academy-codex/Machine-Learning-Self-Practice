# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, [2]].values

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Predicting the results
y_pred = lin_reg.predict(X_poly)

# Predicting a single result
predicted_salary = lin_reg.predict(poly_reg.fit_transform(6.5))

"""
# Visualising the results
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
"""