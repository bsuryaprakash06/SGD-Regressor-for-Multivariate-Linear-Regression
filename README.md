# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
**Step 1 — Data Preparation**  
- Load the California Housing dataset.  
- Select the first three features (`X[:, :3]`) as inputs.  
- Create a multi-output target matrix by combining the housing target and the 7th feature (`Y = [target, X[:, 6]]`).  
- Split the dataset into training and testing subsets (80/20 split).

**Step 2 — Feature Scaling**  
- Apply `StandardScaler` to normalize both the input (`X`) and output (`Y`) data.  
- Fit the scalers on the training data, then transform both training and testing sets (avoid data leakage).

**Step 3 — Model Training**  
- Initialize an `SGDRegressor` with appropriate hyperparameters (`max_iter=1000`, `tol=1e-3`).  
- Wrap it using `MultiOutputRegressor` to enable multi-target regression.  
- Train the model on the scaled training data.

**Step 4 — Prediction & Evaluation**  
- Predict outputs for the test data.  
- Apply inverse scaling to return predictions to the original scale.  
- Evaluate model performance using **Mean Squared Error (MSE)**.  
- Display sample predictions for verification.

## Program:
```python
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: B Surya Prakash
RegisterNumber:  212224230281
*/

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data.data[:, :3]
Y = np.column_stack((data.target, data.data[:, 6]))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
model = MultiOutputRegressor(sgd)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

mse = mean_squared_error(Y_test, Y_pred)
print("Name: B Surya Prakash")
print("Reg No: 212224230281\n")
print("Mean Squared Error:", mse)
print("Predictions:\n", Y_pred[:5])

```

## Output:
<img width="999" height="181" alt="image" src="https://github.com/user-attachments/assets/505c6f64-4fd4-4b89-abc4-4219cb998d89" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
