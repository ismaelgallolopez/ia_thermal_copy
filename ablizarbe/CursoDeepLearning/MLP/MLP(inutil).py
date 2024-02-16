import numpy as np

def sig(x):
    return 1/(1+np.exp(-x))

def prediction(X, W, b):
    return sig(np.dot(X,W)+b)

def error(y, y_hat, X):
    err = 0
    for i in range(len(X)):
        err += -(y-y_hat)*X[i]
    return err - (y-y_hat)

     
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

