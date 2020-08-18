import numpy as np
# Setting a seed value allows the random numbers to be repeatable.
# This is useful for scenarios in which testing may be done
np.random.seed(0)

# (Assuming a linear boundary)For a point with coordinates (p,q), label
# y and a prediction y_prediction = step_function(WX + b)(prediction())
# where W is a vector of the weights, X is a vector of the inputs and b
# is the bias. If the point is correctly classified, do nothing. If the 
# point is classified as positive but has a negative label, subtract
# (learn_rate*p), (learn_rate*q) & the learn_rate from W1, W2 & b
# respectively. If the point is classified negative but has a positive
# label, add (learn_rate*p),(learn_rate*q) & learn_rate to W1, W2, & b. 
# The perceptron algorithm is run repeatedly (up to a certain point) on
# the training dataset, improving the placement of the boundary line


def prediction(X, W, b)
    """Generates a prediction on whether a point should be classified +ve/-ve

    Args:
    X: array. Inputs used for training
    W: array. Weights for the inputs
    b: double. Bias

    Returns:
    0 or 1: 0 indicates the point is classified as negative, 1 shows positive
    """
    # This is a discrete function
    if (np.matmul(X,W)+b)[0]) >= 0:
        return 1
    return 0


def perceptron_step(X, y, W, b, learn_rate):
    """Generates new weights and bias (generating better placed boundaries)

    Args: 
    X: array. Inputs used for training
    y: array. The correct classifications of our inputs (labels)
    W: array. Weights for the inputs
    b: double. Bias
    learn_rate: double. Impacts how drastically the boundary line moves 
    towards a point

    Returns:
    W, b: W is an array of new weights and b is the new bias
    """
    for i in range(len(X)):
        y_prediction = prediction(X[i],W,b)
        if y[i]-y_prediction == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_prediction == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b


def train(X, y, learn_rate = .01, epochs = 100):
    """Trains the model by obtaining a new boundary line with each epoch

    Args:
    X: array. Inputs
    y: array. Labels
    learn_rate: double. Impacts how drastically the boundary line moves towards a
    point
    epochs: int. Number of times the algorithm passes through the entire dataset
    """
    # Generate random weights and bias for initial use. They are to get
    # better with training 
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] # + max(of X1 values)
    for i in range(epochs):
        # Apply perceptron_step in each epoch
        W, b = perceptron_step(X, y, W, b, learn_rate)
