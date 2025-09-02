# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and split data into training and test sets; select required features and targets.

2.Standardize both input and output using StandardScaler.

3.Train model using SGDRegressor wrapped in MultiOutputRegressor.

4.Predict and inverse-transform output; compute and display mean squared error.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: KATHI HASINI
RegisterNumber:
*/

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
X=data.data[:,:3]
Y=np.column_stack((data.target,data.data[:,6]))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train =scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
sgd=SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Square Error:",mse)
print("\nPredictions:\n",Y_pred[:5])
```

## Output:
![WhatsApp Image 2025-09-02 at 15 38 09_db9f6120](https://github.com/user-attachments/assets/f42d9720-bd7b-40b5-b4d1-ad4f6deb724d)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
