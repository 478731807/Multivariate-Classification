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

def plot_confusion_matrix(test_y, pred_y, class_names, filename):
    """
    This function prints and plots the confusion matrix.
    """
    cmap = plt.cm.Blues
    # Compute confusion matrix
    cm = confusion_matrix(
        np.argmax(test_y, axis=1), np.argmax(pred_y, axis=1))
    np.set_printoptions(precision=2)
    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("LSTM Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename + ".png")
    # Plot normalized confusion matrix


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


def LSTM_model(num_var,win_size):
    model = Sequential()

    model.add(LSTM(128, input_dim=num_var, input_length=win_size, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                          epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def BiLSTM_model():
    model = Sequential()
    #model.add(Bidirectional(LSTM(128,return_sequences=True), input_dim=num_var, input_length=win_size))
    #model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(89, 14)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                          epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

# def CNN_model():
#     model = Sequential()
#     return model

def CNN_model(window_size=89, filter_length=1, nb_input_series=14, nb_outputs=3, nb_filter=89):
    """:Return: a Keras Model for predicting the next value in a timeseries given a fixed-size lookback window of previous values.
    The model can handle multiple input timeseries (`nb_input_series`) and multiple prediction targets (`nb_outputs`).
    :param int window_size: The number of previous timeseries values to use as input features.  Also called lag or lookback.
    :param int nb_input_series: The number of input timeseries; 1 for a single timeseries.
      The `X` input to ``fit()`` should be an array of shape ``(n_instances, window_size, nb_input_series)``; each instance is
      a 2D array of shape ``(window_size, nb_input_series)``.  For example, for `window_size` = 3 and `nb_input_series` = 1 (a
      single timeseries), one instance could be ``[[0], [1], [2]]``. See ``make_timeseries_instances()``.
    :param int nb_outputs: The output dimension, often equal to the number of inputs.
      For each input instance (array with shape ``(window_size, nb_input_series)``), the output is a vector of size `nb_outputs`,
      usually the value(s) predicted to come after the last value in that input instance, i.e., the next value
      in the sequence. The `y` input to ``fit()`` should be an array of shape ``(n_instances, nb_outputs)``.
    :param int filter_length: the size (along the `window_size` dimension) of the sliding window that gets convolved with
      each position along each instance. The difference between 1D and 2D convolution is that a 1D filter's "height" is fixed
      to the number of input timeseries (its "width" being `filter_length`), and it can only slide along the window
      dimension.  This is useful as generally the input timeseries have no spatial/ordinal relationship, so it's not
      meaningful to look for patterns that are invariant with respect to subsets of the timeseries.
    :param int nb_filter: The number of different filters to learn (roughly, input patterns to recognize).
    """
    model = Sequential((
        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(window_size, nb_input_series)),
        MaxPooling1D(),     # Downsample the output of convolution by 2X.
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(nb_outputs, activation='linear'),     # For binary classification, change the activation to 'sigmoid'
    ))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # To perform (binary) classification instead:
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model


if __name__ == '__main__':

    """Hyperparameters"""
    win_size = 89
    num_var = 15
    split_ratio = 0.8

    """Load data:
    segment: Each time series of 89 samples is named as segment.
    label: Each segment is associated with a label
    """
    #data = read_data("Multi-variate-Time-series-Data.xlsx")
    data = read_data("multi-variate-large-set.xlsx")
    segments, labels = extract_segments(data, win_size)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
    segments_scaled = preprocessing.scale(segments)
    scaler = preprocessing.StandardScaler().fit(segments_scaled)
    reshaped_segments = segments_scaled.reshape(
        [len(segments) / (win_size), (win_size), num_var])

    """Create Train and Test Split based on split ratio"""

    train_test_split = np.random.rand(len(reshaped_segments)) < split_ratio
    train_x = reshaped_segments[train_test_split]
    train_y = labels[train_test_split]
    test_x = reshaped_segments[~train_test_split]
    test_y = labels[~train_test_split]

    # create  the Network

    #model=CNN_model()
    model = LSTM_model(num_var,win_size)
    #model = BiLSTM_model(num_var,win_size)


    # Fit the network


    model.fit(train_x, train_y, nb_epoch=1000, batch_size=149,
              verbose=2, validation_split=0.1)

    # Predict Test Data and Plot ROC
    pred_y = model.predict(test_x, batch_size=64, verbose=2)
    plotroc(test_y, pred_y, 3, 'test_roc')
    class_names = ['Class 0', 'Class 1', 'Class 3']
    plot_confusion_matrix(test_y, pred_y, class_names, 'test_conf')

    pred_y = model.predict(train_x, batch_size=64, verbose=2)
    plotroc(train_y, pred_y, 3, 'train_roc')
    plot_confusion_matrix(train_y, pred_y, class_names, 'train_conf')
