import sklearn
import sklearn.datasets
import numpy as np
import idx2numpy
import argparse
import math

percent_for_train = 0.8

def unison_shuffled_copies(a, b, seed=42):
    assert len(a) == len(b)

    # use seed to be determinist
    np.random.seed(seed)
    p = np.random.permutation(len(a))
    return a[p], b[p]


X, Y = sklearn.datasets.load_digits(n_class=10, return_X_y=True)

X = np.reshape(X.astype('uint8'), (-1, 8, 8))
Y = Y.astype('uint8')

unique, counts = np.unique(Y, return_counts=True)
labels_to_count = dict(zip(unique, counts))

train_labels_to_count = {digit: math.floor(count*percent_for_train) for digit, count in labels_to_count.items()}
test_labels_to_count = {digit: labels_to_count[digit] - count for digit, count in train_labels_to_count.items()}

inds = Y.argsort()
sortedX = X[inds]
sortedY = Y[inds]

split_inds = np.cumsum([count for digit, count in labels_to_count.items()])

sortedXs = np.split(sortedX, split_inds)[:-1]
sortedYs = np.split(sortedY, split_inds)[:-1]

Xtrains = [sortedXi[0:train_labels_to_count[digit]] for digit, sortedXi in enumerate(sortedXs)]
Ytrains = [sortedYi[0:train_labels_to_count[digit]] for digit, sortedYi in enumerate(sortedYs)]

Xtests = [sortedXi[train_labels_to_count[digit]:labels_to_count[digit]] for digit, sortedXi in enumerate(sortedXs)]
Ytests = [sortedYi[train_labels_to_count[digit]:labels_to_count[digit]] for digit, sortedYi in enumerate(sortedYs)]

Xtrain = np.concatenate(Xtrains, axis=0)
Ytrain = np.concatenate(Ytrains, axis=0)

Xtest = np.concatenate(Xtests, axis=0)
Ytest = np.concatenate(Ytests, axis=0)

Xtrain_randomized, Ytrain_randomized = unison_shuffled_copies(Xtrain, Ytrain)
Xtest_randomized, Ytest_randomized = unison_shuffled_copies(Xtest, Ytest)

idx2numpy.convert_to_file('train-images-idx3-ubyte', Xtrain_randomized)
idx2numpy.convert_to_file('train-labels-idx1-ubyte', Ytrain_randomized)

idx2numpy.convert_to_file('t10k-images-idx3-ubyte', Xtest_randomized)
idx2numpy.convert_to_file('t10k-labels-idx1-ubyte', Ytest_randomized)
