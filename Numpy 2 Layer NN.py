import numpy as np

# sigmoid function provides
# non linearity by converting
# numbers to a probability
# which can be interpreted as
# neuron's activation strength
def sigmoid(x,deriv=False):
    # in addition, this function
    # is easily differentiated
    # which is neccessary for
    # back propagation
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset with a (3,4)
# shape. This can be interpreted
# as 4 training examples, each
# with 3 input values
X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])

# ouput dataset
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# the one seems arbitrary but the command
# standardizes any random calls to a uniform
# range. This means everytime the program is
# run, syn0 will initialize with the exact
# same random numbers as it did in previous
# runtimes
np.random.seed(1)

# initialize weight matrix randomly
syn0 = 2*np.random.random((3,1)) - 1

# initialize the training loop with 10000 
# steps the number of steps here is an 
# abitrary hyperparameter that should be 
# optimized for the dataset
for itet in range(10000):
    # forward propagation (input and hidden layers)
    # in this case the entire dataset is processed
    # in each loop. This purely for simplifying 
    # the demonstration
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))

    # error calculation
    l1_error = y - l1

    # multiply the error by the sigmoid
    # derivative for values l1
    l1_delta = l1_error * sigmoid(l1,deriv=True)

    # update the weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output after Training:")
print(l1)