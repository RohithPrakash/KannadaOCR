import os
import pickle
import numpy as np
import tensorflow as tf

from time import time
from datetime import datetime
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def loadPickle(XTrainPath, yTrainPath, XTestPath=None, yTestPath=None):
    """Loads the train and test data pickle files

    Args:

        XTrainPath (string): Path of the train-feature pickle file

        yTrainPath (string): Path of the train-label pickle file

        XTestPath (string, optional): Path of the test-feature pickle file. Defaults to None.

        yTestPath (string, optional): Path of the test-label pickle file. Defaults to None.

    Returns:

        tuple: Numpy arrays of the loaded pickle files

    """
    XTrain = pickle.load(open(XTrainPath, 'rb'))
    yTrain = np.array(pickle.load(open(yTrainPath, 'rb')), dtype=np.int32)
    XTrain = XTrain / 255.0
    XTrain = np.reshape(XTrain, (XTrain.shape[0], 200, 200, 1))
    yTrain = np_utils.to_categorical(yTrain)

    XTest, yTest = [], []
    if XTestPath != None:
        XTest = pickle.load(open(XTestPath, 'rb'))
        XTest = XTest / 255.0
        XTest = np.reshape(XTest, (XTest.shape[0], 200, 200, 1))

    if yTestPath != None:
        yTest = np.array(pickle.load(open(yTestPath, 'rb')), dtype=np.int32)
        yTest = np_utils.to_categorical(yTest)

    if XTest == None and yTest == None:
        return(XTrain, yTrain)
    else:
        return(XTrain, yTrain, XTest, yTest)


def generateModel(XTrain, yTrain, epochs, XTest=None, yTest=None, trainSplit=None, outputPath=None):
    """Trains and saves a convolutional neural network to recognize characters

    Args:

        XTrain (numpy array): Numpy array of training features

        yTrain (numpy array): Numpy array of training labels

        epochs (int): Number of times to train, why am i explaining this

        XTest (numpy array, optional): Numpy array of testing features, use when not using 'trainSplit'. Defaults to None.

        yTest (numpy array, optional): Numpy array of testing labels, use when not using 'trainSplit'. Defaults to None.

        trainSplit (float, optional): Ratio to split the training data into test data, use when not using explicit test data. Defaults to None.

        outputPath (string, optional): Absolute or relative path to save the trained model. Defaults to None.

    """

    denseLayers = [1]
    layerSizes = [64]
    convLayers = [3]
    for denseLayer in denseLayers:
        for layerSize in layerSizes:
            for convLayer in convLayers:

                time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                name = "{}-conv-{}-nodes-{}-dense-{}-epochs-{}".format(
                    convLayer, layerSize, int(512), epochs, time)
                log = os.path.join("logs", name)
                tensorboard = TensorBoard(log_dir=log)

                model = Sequential()

                model.add(Conv2D(layerSize, (5, 5), input_shape=(
                    200, 200, 1), activation='tanh'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

                for i in range(convLayer - 1):
                    model.add(Conv2D(layerSize, (5, 5), activation='tanh'))
                    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5)))

                model.add(Flatten())

                for i in range(denseLayer):
                    model.add(Dense(512, activation='tanh'))
                    model.add(Dropout(0.25))

                model.add(Dense(15, activation='softmax'))

                model.compile(loss='categorical_crossentropy',
                              optimizer='adam', metrics=['accuracy'])

                if XTest != None or yTest != None:
                    model.fit(XTrain, yTrain, validation_data=(
                        XTest, yTest), epochs=epochs, batch_size=40, callbacks=[tensorboard])
                else:
                    model.fit(XTrain, yTrain, validation_split=trainSplit,
                              epochs=epochs, batch_size=40, callbacks=[tensorboard])

                modelName = "{}.h5".format(name)

                if not os.path.exists(outputPath):
                    print('Creating output directory...')
                    os.makedirs(outputPath)

                if not outputPath[-1] == '/' or outputPath[-1] == '\\':
                    outputPath = outputPath + '/'

                model.save(outputPath+modelName)
                print('Model generated and saved')
