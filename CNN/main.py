from CNN import loadPickle, generateModel


if __name__ == "__main__":
    XTrainPath = input("Enter path of the train-feature pickle file: ")
    yTrainPath = input("Enter path of the train-label pickle file: ")
    XTestPath = input("Enter path of the test-feature pickle file, optional: ")
    yTestPath = input("Enter path of the test-label pickle file, optional: ")
    XTrain, yTrain, XTest, yTest = None

    if len(XTestPath) == 0 or len(yTestPath) == 0:
        XTrain, yTrain = loadPickle(XTrainPath, yTrainPath)
    else:
        XTrain, yTrain, XTest, yTest = loadPickle(
            XTrainPath, yTrainPath, XTestPath, yTestPath)

    trainSplit, outputPath = None
    epochs = input("Enter number of epochs to train: ")

    if len(XTestPath) == 0 or len(yTestPath) == 0:
        trainSplit = input("Enter ratio to split train data for validation: ")

    outputPath = input("Enter path to save trained model, optional: ")

    generateModel(XTrain, yTrain, epochs, XTest, yTest, trainSplit, outputPath)
