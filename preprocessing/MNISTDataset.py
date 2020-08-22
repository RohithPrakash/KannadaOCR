import os
import cv2
import random
import pickle
import numpy as np

from time import time


def createMNISTDataset(path):
    """Creates MNIST like dataset from images provided

    Args:

        path (string): Relative or Absolute path of the dataset.

    Returns:

        list: List containing numpy array representations of images
    """

    start = time()

    data = []
    characters = os.listdir(path)
    for character in characters:
        if character == '.ipynb_checkpoints':
            continue
        imagePath = os.path.join(path, character)
        characterIndex = characters.index(character)
        for image in os.listdir(imagePath):
            if character == '.ipynb_checkpoints':
                continue
            try:
                imageArray = cv2.imread(os.path.join(
                    imagePath, image), cv2.IMREAD_GRAYSCALE)
                data.append([imageArray, characterIndex])
            except Exception as e:
                pass
    random.shuffle(data)

    print('Operation completed in ', time() - start, 'seconds')
    return data


def exportAsPickle(data, type, storeAt):
    """Exports dataset into pickle files (features and labels seperately)

    Args:

        data (list): List containing numpy array representations of images

        type (string): Type of data, 'train' or 'test'

        storeAt (string): Path to store the generated pickle file

    """

    start = time()

    X = []
    y = []

    for features, label in data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, 200, 200, 1)

    file = 'features_' + type + '.pickle'
    path = os.path.join(storeAt, file)

    pickleOut = open(path, 'wb')
    pickle.dump(X, pickleOut)
    pickleOut.close()

    file = 'labels_' + type + '.pickle'
    path = os.path.join(storeAt, file)

    pickleOut = open(path, 'wb')
    pickle.dump(y, pickleOut)
    pickleOut.close()

    print('Export completed in ', time() - start, 'seconds')
