import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Bidirectional
from keras.layers import LSTM, Dropout, GRU, Convolution1D,  MaxPooling1D, Flatten
from keras import optimizers
from itertools import cycle
import itertools
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import scipy.signal
import cnnrnn_source as src

import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne.nonlinearities import rectify, softmax, sigmoid, tanh


def plotroc(test_y, pred_y, n_classes, filename):
    """
    Compute ROC curve and ROC area for each class
    """
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(test_y[:, i], pred_y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot all ROC curves
    plt.figure()
    lw = 1
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'''.
                 format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LSTM Performance')
    plt.legend(loc="lower right")
    plt.savefig(filename + '.png')
    return


def read_data(file_path):
    data = pd.read_excel(file_path, header=0)
    return data


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size - 1
        start += (window_size)


def extract_segments(data, window_size):
    segments = None
    labels = np.empty((0))
    for (start, end) in windows(data, window_size):
        if (len(data.ix[start:end]) == (window_size)):
            signal = data.ix[start:end][range(1, 16)]
            if segments is None:
                segments = signal
            else:
                segments = np.vstack([segments, signal])
            labels = np.append(labels, data.ix[start:end]["goal"][start])
    return segments, labels


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


if __name__ == '__main__':

    """Hyperparameters"""
    win_size = 89
    num_var = 15
    split_ratio = 0.8


    """Load data:
    segment: Each time series of 89 samples is named as segment.
    label: Each segment is associated with a label
    """
    data = read_data("multi-variate-large-set.xlsx")
    segments, labels = extract_segments(data, win_size)
    #labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    reshaped_segments = segments.reshape(
        [len(segments) / (win_size), (win_size), num_var])
    segments_upscaled=np.zeros((reshaped_segments.shape[0],1,128,858))

    for i in range(reshaped_segments.shape[0]):
        segment = reshaped_segments[i, :, :]
        seg_up = scipy.signal.resample(segment, 858, axis=0)
        seg_up = scipy.signal.resample(seg_up, 128, axis=1)
        seg_up=np.transpose(seg_up)
        segments_upscaled[i,0,:,:]=seg_up

    """Create Train and Test Split based on split ratio"""

    train_test_split = np.random.rand(len(segments_upscaled)) < split_ratio
    train_x = segments_upscaled[train_test_split]
    train_y = labels[train_test_split]
    test_x = segments_upscaled[~train_test_split]
    test_y = labels[~train_test_split]



    batch_size = 32
    input_var=T.tensor4('X')
    target_var = T.ivector('y')
    answer_var = T.ivector('y')
    num_units=500
    batch_norm=1
    l2=0
    dropout=0

    print "==> building network"
    example = np.random.uniform(size=(batch_size, 1, 128, 858), low=0.0, high=1.0).astype(
        np.float32)  #########
    answer = np.random.randint(low=0, high=176, size=(batch_size,))  #########

    network = layers.InputLayer(shape=(None, 1, 128, 858), input_var=input_var)
    print layers.get_output(network).eval({input_var: example}).shape

    # CONV-RELU-POOL 1
    network = layers.Conv2DLayer(incoming=network, num_filters=16, filter_size=(7, 7),
                                 stride=1, nonlinearity=rectify)
    print layers.get_output(network).eval({input_var: example}).shape
    network = layers.MaxPool2DLayer(incoming=network, pool_size=(3, 3), stride=(2, 1), pad=2)
    print layers.get_output(network).eval({input_var: example}).shape
    if (batch_norm):
        network = layers.BatchNormLayer(incoming=network)

    # CONV-RELU-POOL 2
    network = layers.Conv2DLayer(incoming=network, num_filters=32, filter_size=(5, 5),
                                 stride=1, nonlinearity=rectify)
    print layers.get_output(network).eval({input_var: example}).shape
    network = layers.MaxPool2DLayer(incoming=network, pool_size=(3, 3), stride=(2, 1), pad=2)
    print layers.get_output(network).eval({input_var: example}).shape
    if (batch_norm):
        network = layers.BatchNormLayer(incoming=network)

    # CONV-RELU-POOL 3
    network = layers.Conv2DLayer(incoming=network, num_filters=32, filter_size=(3, 3),
                                 stride=1, nonlinearity=rectify)
    print layers.get_output(network).eval({input_var: example}).shape
    network = layers.MaxPool2DLayer(incoming=network, pool_size=(3, 3), stride=(2, 1), pad=2)
    print layers.get_output(network).eval({input_var: example}).shape
    if (batch_norm):
        network = layers.BatchNormLayer(incoming=network)

    # CONV-RELU-POOL 4
    network = layers.Conv2DLayer(incoming=network, num_filters=32, filter_size=(3, 3),
                                 stride=1, nonlinearity=rectify)
    print layers.get_output(network).eval({input_var: example}).shape
    network = layers.MaxPool2DLayer(incoming=network, pool_size=(3, 3), stride=(2, 1), pad=2)
    print layers.get_output(network).eval({input_var: example}).shape
    if (batch_norm):
        network = layers.BatchNormLayer(incoming=network)

    params = layers.get_all_params(network, trainable=True)

    output = layers.get_output(network)
    output = output.transpose((0, 3, 1, 2))
    output = output.flatten(ndim=3)

    # NOTE: these constants are shapes of last pool layer, it can be symbolic
    # explicit values are better for optimizations
    num_channels = 32
    filter_W = 852
    filter_H = 8

    # InputLayer
    network = layers.InputLayer(shape=(None, filter_W, num_channels * filter_H), input_var=output)
    print layers.get_output(network).eval({input_var: example}).shape

    # GRULayer
    network = layers.GRULayer(incoming=network, num_units=num_units, only_return_final=True)
    print layers.get_output(network).eval({input_var: example}).shape
    if (batch_norm):
        network = layers.BatchNormLayer(incoming=network)
    if (dropout > 0):
        network = layers.dropout(network, dropout)

    # Last layer: classification
    network = layers.DenseLayer(incoming=network, num_units=3, nonlinearity=softmax)
    print layers.get_output(network).eval({input_var: example}).shape

    params += layers.get_all_params(network, trainable=True)
    prediction = layers.get_output(network)

    # print "==> param shapes", [x.eval().shape for x in params]

    loss_ce = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
    if (l2 > 0):
        loss_l2 = l2 * lasagne.regularization.apply_penalty(params,
                                                                      lasagne.regularization.l2)
    else:
        loss_l2 = 0
    loss = loss_ce + loss_l2

    # updates = lasagne.updates.adadelta(loss, params)
    # updates = lasagne.updates.momentum(loss, params, learning_rate=0.003) # good one
    updates = lasagne.updates.momentum(loss, params, learning_rate=0.001)

    # compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")

    num_epochs=100
    import time

    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(train_x, train_y, 32, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(test_x, test_y, 32, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(test_x, test_y, 32, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

