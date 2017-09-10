# Simple Linear Regression Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values

# Splitting the dataset into training and testing dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

X_train = X_train.reshape(len(X_train),1)
Y_train = Y_train.reshape(len(Y_train),1)
X_test = X_test.reshape(len(X_test),1)

#  Fitting Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(X_train,Y_train)

# Predicting the test results
Y_pred = regresor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regresor.predict(X_train), color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary');
plt.title('Salary vs Experience (Training set)')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regresor.predict(X_train), color='blue')
plt.xlabel('Years of Experience');
plt.ylabel('Salary');
plt.show() 