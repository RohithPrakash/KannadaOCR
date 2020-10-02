import os
import sys


def performOperation(datasetPath, datasetSplitPath, trainSplitRatio=0.8):
    """Performs preprocessing, augmentation, dataset splitting (into train and text data) and exports the split dataset into pickle files

    Args:

        datasetPath (string): Absolute or relative path of the dataset

        datasetSplitPath (string): Path to store the split training and testing data

        trainSplitRatio (float, optional): Ratio to split the dataset into training and testing data. Defaults to 0.8.

    """

    from SplitDataset import splitDataset
    from ImagePreprocessing import parallelPreProcessing
    from ImageAugmentation import parallelAugmentation
    from MNISTDataset import createMNISTDataset, exportAsPickle

    parallelPreProcessing(datasetPath)
    parallelAugmentation(datasetPath)
    splitDataset(datasetPath, trainSplitRatio, datasetSplitPath)

    trainPath = os.path.join(datasetSplitPath, 'train')
    dataset = createMNISTDataset(trainPath)
    exportAsPickle(dataset, 'train', '../dataset')

    testPath = os.path.join(datasetSplitPath, 'test')
    dataset = createMNISTDataset(testPath)
    exportAsPickle(dataset, 'test', '../dataset')


if __name__ == "__main__":

    if len(sys.argv) == 4:
        performOperation(sys.argv[1], sys.argv[2], sys.argv[3])

    elif len(sys.argv) == 3:
        performOperation(sys.argv[1], sys.argv[2])

    else:
        sys.exit("Incorrect number of arguments passed.")
