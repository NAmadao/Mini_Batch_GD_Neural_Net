"""
This is the code for a 2 layer Neural Network and is used on a breast cancer
dataset. The dataset is space separated and the first column contains the
labels.

The labels are -1(FALSE) and 1(TRUE).

run program on terminal as
python mini_batch_gd.py train test hiddennodes(optional) batchsize(optional)
e.g
python mini_batch_gd.py bc.train.0.txt bc.test.0.txt 3 20

train - trainfile
test - testfile
hiddennodes - number of hidden nodes
batchsize - minibatch size

Author: Prithwish Ganguly
Create Date : 8 March 2019
Last Update : 15 March 2019
"""
# We use the 3 dictionaries. Their names and contents are:
# Parameters: W_1, B_1, W_2, B_2
# Cache: Z_1, A_1, Z_2, A_2
# Grads: dZ_1, dA_1, dZ_2, dA_2

# Importing Packages
import sys
import random
import time
import numpy as np

def read_file(DAT, NAME):
    """
    Loads features and labels from files. First column of file is labels.
    Converts -1 labels to 0.

    Args:
        (str) DAT - type of file, train or test
        (str) NAME - name of file
    Returns:
        X - features
        Y - labels
    """
    F = open(NAME)
    DATA = np.loadtxt(F)
    X = DATA[:, 1:]
    Y = DATA[:, 0]

    Y[Y == -1] = 0  # Converting -1 labels to 0 for classification

    print('\n', DAT, "=\n", X)
    print(DAT, "shape=\n", X.shape)

    print(DAT, "=\n", Y)
    print(DAT, "labels shape=\n", Y.shape)

    return X, Y


def sigmoid(X, derivative=False):
    """
    Applies sigmoid activation or derivative of sigmoid activation.

    Args:
        (float) X - np array
        derivative - returns sigmoid derivative if True else sigmoid activation
    Returns:
        Sigmoid activation output or sigmoid derivative
    """
    return X*(1-X) if derivative else 1/(1+np.exp(-X))


def forward_prop(TRAIN, PARAMETERS, showoutput=False):
    """
    Compute forward propagation of 2 layer neural net using parameters and
    features.

    Args:
        (float) TRAIN - input features
        (dict) PARAMATERS - Contains layer weights and bias
    Returns:
        CACHE - dictionary of layer linear output and activations
    """
    W_1 = PARAMETERS.get('W_1')
    B_1 = PARAMETERS.get('B_1')
    W_2 = PARAMETERS.get('W_2')
    B_2 = PARAMETERS.get('B_2')

    Z_1 = np.dot(W_1, TRAIN) + B_1
    A_1 = sigmoid(Z_1)

    Z_2 = np.dot(W_2, A_1) + B_2
    A_2 = sigmoid(Z_2)

    if showoutput is True:
        print("Hidden layer output =\n", A_1)
        print("Output layer =\n", A_2)

    CACHE = {'Z_1': Z_1,
             'A_1': A_1,
             'Z_2': Z_2,
             'A_2': A_2}

    return CACHE


def back_prop(TRAIN, TRAIN_LABELS, CACHE, PARAMETERS):
    """
    Do backpropagation on 2 layer neural network.

    Args:
        TRAIN - input features
        TRAIN_LABELS - input labels
        CACHE - dictionary of layer linear output and activations
        PARAMATERS - dictionary that contains the layer weights and bias units
    Returns:
        GRADS - disctionary that contains the gradients of weight and bias
    """

    Z_1 = CACHE.get('Z_1')
    A_1 = CACHE.get('A_1')
    Z_2 = CACHE.get('Z_2')
    A_2 = CACHE.get('A_2')

    W_2 = PARAMETERS.get('W_2')

    m = TRAIN_LABELS.shape[1]

    dZ_2 = A_2 - TRAIN_LABELS
    dW_2 = (1/m) * np.dot(dZ_2, A_1.T)
    dB_2 = (1/m) * np.sum(dZ_2, axis=1, keepdims=True)

    dZ_1 = np.dot(W_2.T, dZ_2) * sigmoid(A_1, derivative=True)
    dW_1 = (1/m) * np.dot(dZ_1, TRAIN.T)
    dB_1 = (1/m) * np.sum(dZ_1, axis=1, keepdims=True)

    GRADS = {'dW_1': dW_1,
             'dB_1': dB_1,
             'dW_2': dW_2,
             'dB_2': dB_2}

    return GRADS


def compute_cost(CACHE, TRAIN_LABELS):
    """
    Compute the cost of the network

    Args:
        CACHE - dictionary of layer linear output and activations
        TRAIN_LABELS - input labels
    Returns:
        COST - cost of the network (logloss)
    """

    A_2 = CACHE.get('A_2')
    m = TRAIN_LABELS.shape[1]

    P1 = np.multiply(np.log(A_2),TRAIN_LABELS)
    P2 = np.multiply(np.log((1-A_2)),(1-TRAIN_LABELS))
    LOGPROBS = P1 + P2
    COST = (-1/m) * np.sum(LOGPROBS)

    # e.g. convert [[17]] to 17
    COST = (1/m)*np.squeeze(COST)

    return COST


def update_params(PARAMETERS, GRADS, LEARNING_RATE=0.01):
    """
    Update the layer parameters using the gradients

    Args:
        PARAMETERS - dictionary that contains the layer weights and bias units
        GRADS - disctionary that contains the gradients of weight and bias
        LEARNING_RATE - learning rate of gradient descent
    Returns:
        PARAMETERS - dictionary that contains the updated layer weights and
        bias units
    """
    W_1 = PARAMETERS.get('W_1')
    B_1 = PARAMETERS.get('B_1')
    W_2 = PARAMETERS.get('W_2')
    B_2 = PARAMETERS.get('B_2')

    dW_1 = GRADS.get('dW_1')
    dB_1 = GRADS.get('dB_1')
    dW_2 = GRADS.get('dW_2')
    dB_2 = GRADS.get('dB_2')

    W_1 = W_1 - LEARNING_RATE*dW_1
    B_1 = B_1 - LEARNING_RATE*dB_1
    W_2 = W_2 - LEARNING_RATE*dW_2
    B_2 = B_2 - LEARNING_RATE*dB_2

    PARAMETERS = {'W_1': W_1,
                  'B_1': B_1,
                  'W_2': W_2,
                  'B_2': B_2}

    return PARAMETERS


def randomize(TRAIN, TRAIN_LABELS):
    """
    Randomly shuffle the rows of the dataset

    Args:
        TRAIN - input features
        TRAIN_LABELS - input labels
    Returns:
        TRAIN - randomly shuffled rows of input features
        TRAIN_LABELS - corresponding input labels
    """
    """
    input: train - train features
            TRAIN_LABELS - train labels
    output: train - train features
            TRAIN_LABELS - train labels

    Randomly shuffles the entire dataset and outputs corresponding features and
    labels
    """
    Z = list(zip(TRAIN, TRAIN_LABELS))
    random.shuffle(Z)
    X, Y = zip(*Z)

    TRAIN = np.array(X)
    TRAIN_LABELS = np.array([Y])

    return TRAIN, TRAIN_LABELS


def normalize(DATA):
    """
    Normalize the data to scale features to the same range

    Args:
        DATA - dataset of examples and features
    Returns:
        DATA - normalized dataset
    """
    X = DATA.T
    for i in range(len(X)):
        X[i] = (X[i] - np.mean(X[i])) * (1/(np.std(X[i]) + 1e-5))

    return DATA


# Parameter and hyperparameter declarations
STOP = 0.01
EPOCHS = 1000
ETA = 1
PREV_COST = np.inf
ITERATIONS = 0
COST = 0

# If batchsize is not given as input default to batchsize = 10
if len(sys.argv) > 3:
    BATCH_SIZE = int(sys.argv[3])
else:
    BATCH_SIZE = 10
# batch_size = 1 -> stochastic gradient Descent
# batch_size = train.shape[0] batch gradient descent

# If number of hidden nodes not provided as input default to hidden_nodes = 3
if len(sys.argv) > 4:
    HIDDEN_NODES = int(sys.argv[4])
else:
    HIDDEN_NODES = 3

# Read Data from input file names
TRAIN, TRAIN_LABELS = read_file('train', sys.argv[1])
TEST, TEST_LABELS = read_file('test', sys.argv[2])


TRAIN = normalize(TRAIN)
TEST = normalize(TEST)

# Initialize all weights
FEATURES = TRAIN.shape[1]

CLASSES = 2

W_1 = np.random.rand(HIDDEN_NODES, FEATURES)
B_1 = np.zeros((HIDDEN_NODES, 1))

W_2 = np.random.rand(CLASSES - 1, HIDDEN_NODES)
B_2 = np.zeros((CLASSES - 1, 1))

print("W_1=\n", W_1)
print("B_1=\n", B_1)
print("W_2=\n", W_2)
print("B_2=\n", B_2)

PARAMETERS = {'W_1': W_1,
              'B_1': B_1,
              'W_2': W_2,
              'B_2': B_2}


# Start Batch Gradient Descent
START = time.time()
while(abs(PREV_COST - COST) > STOP and ITERATIONS < EPOCHS):

    if ITERATIONS > 0:
        PREV_COST = COST

    TRAIN_RAND, TRAIN_LABELS_RAND = randomize(TRAIN, TRAIN_LABELS)

    for i in range(0, TRAIN.shape[0], BATCH_SIZE):
        TRAIN_BATCH = TRAIN_RAND[i:i+BATCH_SIZE].T
        TRAIN_LABELS_BATCH = np.array([TRAIN_LABELS_RAND[0][i:i+BATCH_SIZE]])

        CACHE = forward_prop(TRAIN_BATCH, PARAMETERS, showoutput=False)
        GRADS = back_prop(TRAIN_BATCH, TRAIN_LABELS_BATCH, CACHE, PARAMETERS)
        PARAMETERS = update_params(PARAMETERS, GRADS, LEARNING_RATE=ETA)

    COST = compute_cost(CACHE, TRAIN_LABELS_BATCH)
    print("Cost after epoch %i: %f" % (ITERATIONS, COST))

    ITERATIONS += 1
END = time.time()

# Do final predictions
TRAIN_CACHE = forward_prop(TRAIN.T, PARAMETERS, showoutput=False)
TRAIN_PRED = np.round(TRAIN_CACHE.get("A_2"))

TEST_CACHE = forward_prop(TEST.T, PARAMETERS, showoutput=False)
TEST_PRED = np.round(TEST_CACHE.get("A_2"))

TRAIN_ERROR = np.mean(TRAIN_PRED != TRAIN_LABELS) * 100
TEST_ERROR = np.mean(TEST_PRED != TEST_LABELS) * 100

print("Train predictions:\t", TRAIN_PRED)
print("Train error:\t", TRAIN_ERROR, '%')

print("Test predictions\t", TEST_PRED)
print("Test error\t", TEST_ERROR, '%')

print("Gradient Descent time taken =\t", END - START, 'seconds')
