# Importing pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def error_term_formula(x, y, output2,weights2, input, input2):
#    for binary cross entropy loss
    return (y - output2)*sigmoid_prime(input2)*weights2*np.dot(sigmoid_prime(input),x)




# Reading the csv file into a pandas DataFrame
data = pd.read_csv('deep learning/datasets/student_data.csv')

one_hot_data = pd.get_dummies(data, columns=['rank'], prefix='rank', drop_first=True)

processed_data = one_hot_data[:]

processed_data  ['gre'] = (processed_data['gre']/800)-0.5
processed_data  ['gpa'] = (processed_data['gpa']/4)-0.5

sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))

features = train_data.drop('admit', axis=1).astype(float)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1).astype(float)
targets_test = test_data['admit']


# Neural Network hyperparameters
epochs = 5000
learnrate = 0.00006
n_neurons = 5
n_records, n_features = features_test.shape

# Training function
def train_nn(features, targets, epochs, learnrate,n_neurons):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    
    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=(n_features,n_neurons))
    weights2 = np.random.normal(scale=1 / n_features**.5, size=(n_features))

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        del_w2 = np.zeros(weights2.shape)
        for x, y in zip(features.values, targets):

            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable
            input = np.dot(x,weights)
            output = sigmoid(input)
            input2 = np.dot(output, weights2)
            output2 = sigmoid(input2)

            # The error, the target minus the network output


            # error = error_formula(y, output)

            # The error term
            #   Notice we calulate f'(h) here instead of defining a separate
            #   sigmoidmoid_prime function. This just makes it faster because we
            #   can re-use the result of the sigmoidmoid function stored in
            #   the output variable
            
            error_term = error_term_formula(x, y, output2, weights2, input, input2)
            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term
            del_w2 += (y - output2)*sigmoid_prime(input2)*output

        weights += learnrate*del_w #/ n_records
        weights2 += learnrate*del_w2




        # Update the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
        weights += learnrate*del_w #/ n_records
        weights2 += learnrate*del_w2

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features_test,weights))
            out2 = sigmoid(np.dot(out,weights2))
            loss = np.mean((out2 - targets_test) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Test loss: ", loss, "  WARNING - Loss Increasing")
                break
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights, weights2
    
weights, weights2 = train_nn(features, targets, epochs, learnrate, n_neurons)


# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test,weights))
test_out2 = sigmoid(np.dot(test_out,weights2))
predictions = test_out2 > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

