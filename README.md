# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib, and scipy.
2.Visualize the data and define the sigmoid function, cost function, and gradient descent.
3.Plot the decision boundary.
4. Calculate the y-prediction.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: YUVAN RAJ R
RegisterNumber:  212225230315
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("Placement_Data.csv")
data.head()

# Copy and preprocess
data1 = data.copy()
data1 = data1.drop(['sl_no', 'salary'], axis=1)

# Encode categorical variables
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Split features and target
x = data1.iloc[:, :-1]
y = data1["status"]

# Initialize theta
theta = np.random.randn(x.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function
def loss(theta, x, y):
    h = sigmoid(x.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Gradient descent
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(x.dot(theta))
        gradient = x.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

# Train model
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

# Prediction function
def predict(theta, x):
    h = sigmoid(x.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# Predictions and accuracy
y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print("Predicted:\n", y_pred)
print("Actual:\n", y.values)

# Test with new data
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print("Predicted Result:", y_prednew)
*/
```

## Output:
<img width="1056" height="325" alt="image" src="https://github.com/user-attachments/assets/aacd9a18-0c87-4161-9095-b3959b24b9d3" />

<img width="882" height="367" alt="image" src="https://github.com/user-attachments/assets/2d97eea9-8d99-430b-809f-67adab470fc1" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

