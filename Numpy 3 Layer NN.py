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

# If you analyze the X matrix you 
# will find no direct correlation 
# between a single column and the 
# output matrix y. Instead, you may 
# notice that the combination of 
# columns 1 and 2 form a XOR gate
# with the output. This considered a
# non linear correlation
y = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
# the one seems arbitrary but the command
# standardizes any random calls to a uniform
# range. This means everytime the program is
# run, syn0 will initialize with the exact
# same random numbers as it did in previous
# runtimes
np.random.seed(1)

# initialize 2 weight matrix randomly. This
# neural network contains two hidden layers
# and therefore requires another weight matrix
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

# initialize the training loop with 60000 
# steps the number of steps here is an 
# abitrary hyperparameter that should be 
# optimized for the dataset
for itet in range(60000):
    # forward propagation (input and hidden layers)
    # in this case the entire dataset is processed
    # in each loop. This purely for simplifying 
    # the demonstration
    l0 = X
    z1 = np.dot(l0,syn0)
    l1 = sigmoid(z1)
    z2 = np.dot(l1,syn1)
    l2 = sigmoid(z2)

    # The error calculation is initially performed on
    # the last layer as we backwards propogate through
    # the network
    l2_error = y - l2

    # if a network is truly learning, the error should
    # always be decreasing
    if (itet% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
    
    # multiply the error by the sigmoid
    # derivative for values l2
    l2_delta = l2_error * sigmoid(l2,deriv=True)

    # now we continue the propogation through to the next
    # layer of the network
    l1_error = np.dot(l2_delta, syn1.T)
    
    # multiply the error by the sigmoid
    # derivative for values l2
    l1_delta = l1_error * sigmoid(l1,deriv=True)

    # update the weights
    syn1 += np.dot(l1.T, l2_delta)
    syn0 += np.dot(l0.T, l1_delta)

print("Output after Training:")
print(l1)