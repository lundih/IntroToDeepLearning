import numpy as np

# Cross entropy lets us know how likely it is for an arrangement of 
# events to occur based on their probabilities.
# If we have only 2 classes:
# Cross entropy is given by -sum_of(Yln(P) + (1-Y)ln(1-P) 
# where Y is an array of one hot encoded scores of classes and P is the
# probabilities of the points being classified correctly.

"""Returns cross entropy for 2 classes

Args:
Y: array. Values of the one hot encoded scores of classes
P: array. Probabilities of the events in Y happening

Returns:`
Float that represents cross entropy
"""
def calculate_cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum((Y*np.log(P)) + ((1-Y)*np.log(1-P)))

# For multi-class cases:
# Cross entropy is given by 
# -sum_of(from i=1 to i=n for j=1 to J=m in Y_ij ln(P_ij)) where m is 
# the number of classes.