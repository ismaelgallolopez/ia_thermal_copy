# Importing pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('deep learning/datasets/student_data.csv')

W = [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
]
W2 = [1.0, 1.0, 1.0, 1.0]

epochs = 100

X = data.iloc[:, -3:].to_numpy()
y = data.iloc[:,0].to_numpy()

ones = np.ones((X.shape[0], 1))
X = np.column_stack((X, ones))

def softmax(y_hat):
    # Shift the inputs to a stable range
    y_hat_shifted = y_hat - np.max(y_hat)
    exp_values = np.exp(y_hat_shifted)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

def prediction(X, W, W2):
    y_hat1 = []
    y_hat2 = []
    y_hat3 = []
    for i in range(len(X)):
        y_hat1.append(np.dot(X[i],W[0]))
        y_hat2.append(np.dot(X[i],W[1]))
        y_hat3.append(np.dot(X[i],W[2]))
    
    y_hat1 = np.array(y_hat1)
    y_hat2 = np.array(y_hat2)
    y_hat3 = np.array(y_hat3)

    ones = np.ones((X.shape[0], 1))
    y_hat = softmax(y_hat1)*W2[0]+softmax(y_hat2)*W2[1]+softmax(y_hat3)*W2[2]+ones*W2[3]
    return softmax(y_hat)
# lo que yo entendí

""" def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(z_shifted)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

def prediction(X, W, W2):
    # Assuming W is a matrix of shape (input_features, hidden_units)
    # and W2 is a matrix of shape (hidden_units, output_classes)
    y_hat = np.dot(X, W)  # This is a simplified version; might need adjustments based on your actual model architecture
    y_hat = np.dot(softmax(y_hat), W2)
    return softmax(y_hat) """
pred = prediction(X, W, W2)


print(pred)








