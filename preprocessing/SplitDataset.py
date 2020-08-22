import os
import shutil
import random
from time import time


def copyFiles(datasetPath, folders, train, test, output):
    """Copies files from actual dataset to train and test folders

    Args:

        datasetPath (string): Relative or Absolute path of the dataset

        folders (list): List having names of folders of all characters in dataset

        train (list): List having names of images to be copied to train folder

        test (list): List having names of images to be copied to test folder

        output (string): Path to store the split training and testing data

    """
    testPath = os.path.join(output, "test")
    trainPath = os.path.join(output, "train")

    if not os.path.exists(testPath):
        os.makedirs(testPath)
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)

    for index, folder in enumerate(folders):

        if folder == '.ipynb_checkpoints':
            continue

        testSubPath = os.path.join(testPath, folder)
        if not os.path.exists(testSubPath):
            os.makedirs(testSubPath)

        trainSubPath = os.path.join(trainPath, folder)
        if not os.path.exists(trainSubPath):
            os.makedirs(trainSubPath)

        for image in train[index]:
            if image == '.ipynb_checkpoints':
                continue
            imagePath = os.path.join(datasetPath, folder, image)
            copyPath = os.path.join(trainSubPath, image)
            shutil.copy(imagePath, copyPath)

        for image in test[index]:
            if image == '.ipynb_checkpoints':
                continue
            imagePath = os.path.join(datasetPath, folder, image)
            copyPath = os.path.join(testSubPath, image)
            shutil.copy(imagePath, copyPath)


def splitDataset(datasetPath, trainRatio, output):
    """Splits dataset into training and testing data

    Args:

        datasetPath (string): Relative or Absolute path of the dataset

        trainRatio (float): Ratio of training data required from the dataset

        output (string): Path to store the split training and testing data

    """

    start = time()

    folders = os.listdir(datasetPath)
    files = []
    for folder in folders:
        files.append(os.listdir(datasetPath + folder))

    test = []
    train = []
    length = len(files[1])
    trainRatio = int(trainRatio * length)

    for i in range(len(files)):
        train = train + [random.choices(files[i], k=trainRatio)]
        test = test + [list(set(files[i]) - set(train[i]))]

    copyFiles(datasetPath, folders, train, test, output)

    print("Splitting completed in ", time() - start, " seconds")


#splitDataset('../temp2/Img/', 0.8, '../temp2/')
