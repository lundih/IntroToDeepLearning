import numpy as np
# An error function allows us to tell how far a point is from the
# boundary line. With discrete functions, we can only tell whether a 
# point is misclassified or not. But with a continuous function we can 
# also tell by what magnitude the point is misclassified (using the 
# error function). This makes it more efficient when optimising the 
# boundary line.


def equation_result(X, W, b)
    """Gives the result of the inputs being applied to a linear boundary equation

    Args:
    X: array. Inputs used for training 
    W: array. Weights for the inputs
    b: double. Bias

	Returns:
    value after inputs are applied to a linear equation
    """
    return (np.matmul(X,W)+b)[0]


# Example of discrete (as used in the perceptron algorithm example)
def prediction_discrete(X, W, b)
    """Generates a prediction on whether a point should be classified +ve/-ve

    Args:
    X: array. Inputs used for training
    W: array. Weights for the inputs
    b: double. Bias

    Returns:
    0 or 1: 0 indicates the point is classified as negative, 1 shows positive
    """
    # This is a step function
    if equation_result(X, W, b) >= 0:
        return 1
    return 0


# Example of continuous (Sigmoid function)
# sigmoid(X) = 1/(1+e^(-X)
def prediction_sigmoid(X, W, b)
    """Gives the probability of a point being positive or negative

    Args:
    X: array. Inputs used for training
    W: array. Weights for the inputs
    b: double. Bias

    Returns:
    value between 0 and 1: values close to one are large positive numbers,
    values close to 0 are large negative numbers, and values close to .5
    are numbers that are close to zero
    """
    return 1/(1 + np.exp(-equation_result(X, W, b))


# Example of continuous (Softmax function)
# The Sigmoid function works when working with binary options. But when
# working with more than one class that something can be classified 
# into, we need another function, the Softmax function. 
# softmax(X) = e^(score_of_class)/sum_of_((e^score_of_class)_for_all_classes)
def softmax(list_of_scores)
    """Takes a list of scores for each of classes and returns values when 
    the Softmax function is applied to the list 

    Args:
    list_of_scores: list. Scores per class. These are used to determine the
    probability that what we are examining belongs to which class

    Returns:
    List of probabilities of each class
    """ 
    return np.divide(np.exp(list_of_scores), np.exp(list_of_scores).sum())
