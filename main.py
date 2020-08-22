import sys


def recognize(imagePath, storeSegmentedImagesAt="recognizer/segmentedImage",
              modelPath="model/3-conv-64-nodes-512-dense-20-epochs-2020_05_08_16_19_14.h5",
              outputPath=None):
    """Recognizes text from input image and stores text to output.txt file

    Args:

        imagePath (string): Path of input image

        storeSegmentedImagesAt (string, optional); Path to store images of segmented character. Defaults to "recognizer/segmentedImage"

        modelPath (string, optional): Path to the model to be used for character recognition. Defaults to prebuilt model

        output (string, optional) : Path to store recognized text in a '.txt' file. Defaults to None

    Returns:

        String: Recognized text
    """

    import os
    import cv2
    import numpy as np
    from preprocessing.Segmentation import segmentImage
    from unicode.Unicode import getCharacter
    from tensorflow.keras.models import load_model
    from preprocessing.ImagePreprocessing import preProcessImages

    numberOfLines, numberOfWords, numberOfLetters = segmentImage(
        imagePath, storeSegmentedImagesAt)
    imageName = os.path.basename(imagePath)
    characters = os.listdir(os.path.join(storeSegmentedImagesAt, imageName))
    for character in characters:
        if character == ".ipynb_checkpoints":
            continue
        preProcessImages(storeSegmentedImagesAt, imageName, character)

    predictions = [" "] * numberOfLines
    for i in range(len(numberOfWords)):
        predictions[i] = [" "] * len(numberOfLetters[i])
        for j in range(len(numberOfLetters[i])):
            predictions[i][j] = [" "] * numberOfLetters[i][j]

    predictedString = ""
    model = load_model(modelPath)
    characters = os.listdir(os.path.join(storeSegmentedImagesAt, imageName))
    for character in characters:
        if character == ".ipynb_checkpoints":
            continue
        image = cv2.imread(os.path.join(
            storeSegmentedImagesAt, imageName, character), cv2.IMREAD_GRAYSCALE)
        image = np.reshape(image, (-1, 200, 200, 1))
        predictionProbabs = list(model.predict(image)[0])
        predictedIndex = predictionProbabs.index(max(predictionProbabs))
        predictedCharacter = getCharacter(predictedIndex)
        position = character.split("-", 3)
        position.pop()
        position = list(map(int, position))
        predictions[position[0] - 1][position[1] -
                                     1][position[2] - 1] = predictedCharacter

    print("Note: Requires 'Noto Sans Kannada' font or equivalent to view output.")
    print("Individually found characters are: ")
    print(predictions)

    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            for k in range(len(predictions[i][j])):
                predictedString = predictedString + predictions[i][j][k]
            predictedString = predictedString + " "
        predictedString = predictedString + "\n"

    print("The recognized text from the image is: ")
    print(predictedString)

    if not os.path.exists(outputPath):
        print('Creating output directory...')
        os.makedirs(outputPath)

    outputFile = open(os.path.join(outputPath, 'output.txt'), 'w')
    outputFile.write(prediction)
    outputFile.close()
    print('Predicted text saved to output.txt')

    return predictedString


if __name__ == "__main__":
    if len(sys.argv) == 2:
        prediction = recognize(sys.argv[1])
    elif len(sys.argv) == 3:
        prediction = recognize(sys.argv[1], storeSegmentedImagesAt=sys.argv[2])
    elif len(sys.argv) == 4:
        prediction = recognize(
            sys.argv[1], storeSegmentedImagesAt=sys.argv[2], modelPath=sys.argv[3])
    elif len(sys.argv) == 5:
        prediction = recognize(
            sys.argv[1], storeSegmentedImagesAt=sys.argv[2], modelPath=sys.argv[3], outputPath=sys.argv[4])
    else:
        sys.exit("Wrong number of arguments passed.")
