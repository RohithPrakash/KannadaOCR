import os
import cv2
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt

from time import time
from pathlib import Path


def showImages(sourceImage):
    """Displays specified image. Note: To be used only in notebooks

    Args:

        sourceImage (numpy array): Image to be displayed
    """
    plt.imshow(sourceImage)
    plt.show()


def lineArray(array):
    listXUpper = []
    listXLower = []
    for y in range(5, len(array) - 5):
        s_a, s_p = startLine(y, array)
        e_a, e_p = endline(y, array)
        if s_a >= 7 and s_p >= 5:
            listXUpper.append(y)
        if e_a >= 5 and e_p >= 7:
            listXLower.append(y)
    return listXUpper, listXLower


def startLine(y, array):
    countAhead = 0
    countPrevious = 0
    for i in array[y:y+10]:
        if i > 3:
            countAhead += 1
    for i in array[y-10:y]:
        if i == 0:
            countPrevious += 1
    return countAhead, countPrevious


def endline(y, array):
    countAhead = 0
    countPrevious = 0
    for i in array[y:y+10]:
        if i == 0:
            countAhead += 1
    for i in array[y-10:y]:
        if i > 3:
            countPrevious += 1
    return countAhead, countPrevious


def endLineWord(y, array, a):
    countAhead = 0
    countPrevious = 0
    for i in array[y:y+2*a]:
        if i < 2:
            countAhead += 1
    for i in array[y-a:y]:
        if i > 2:
            countPrevious += 1
    return countPrevious, countAhead


def endLineArray(array, a):
    listEndLines = []
    for y in range(len(array)):
        e_p, e_a = endLineWord(y, array, a)
        if e_a >= int(1.5*a) and e_p >= int(0.7*a):
            listEndLines.append(y)
    return listEndLines


def refineEndWord(array):
    refineList = []
    for y in range(len(array)-1):
        if array[y]+1 < array[y+1]:
            refineList.append(array[y])
    refineList.append(array[-1])
    return refineList


def refineArray(arrayUpper, arrayLower):
    upperLines = []
    lowerLines = []
    for y in range(len(arrayUpper)-1):
        if arrayUpper[y] + 5 < arrayUpper[y+1]:
            upperLines.append(arrayUpper[y]-10)
    for y in range(len(arrayLower)-1):
        if arrayLower[y] + 5 < arrayLower[y+1]:
            lowerLines.append(arrayLower[y]+10)
    upperLines.append(arrayUpper[-1]-10)
    lowerLines.append(arrayLower[-1]+10)
    return upperLines, lowerLines


def letterWidth(contours):
    """Finds the mean width of letters in the image

    Args:

        contours (list): List having contours in a binary image

    Returns:

        float: Mean width of letters
    """

    letterWidthSum = 0
    count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            _, _, w, _ = cv2.boundingRect(contour)
            letterWidthSum += w
            count += 1
    return letterWidthSum/count


def endWordDetect(lines, i, binaryImage, meanLetterWidth, width, finalThreshold):
    """Detects the last word in the line

    Args:

        lines (list): Numpy array representations of lines in the actual image

        i (int): Index of line in focus

        binaryImage (numpy array): Binary / black and white image of actual image

        meanLetterWidth (float): Mean width of letters

        width (float): Width of actual image

        finalThreshold (list): Closed morphological transformation of actual image

    Returns:

        list: List having last word of line in focus
    """

    countY = np.zeros(shape=width)
    for x in range(width):
        for y in range(lines[i][0], lines[i][1]):
            if binaryImage[y][x] == 255:
                countY[x] += 1
    endLines = endLineArray(countY, int(meanLetterWidth))
    endlines = refineEndWord(endLines)
    for x in endlines:
        finalThreshold[lines[i][0]:lines[i][1], x] = 255
    return endlines


def createFolderToStoreSegmentedImage(segmentedImageStorePath, imageName):
    """Creates folder to store segmented images

    Args:

        segmentedImageStorePath (string): Path to store the segmented letters' images

        imageName (string): Name of the image being segmented, including file type
    """

    segmentedImageDir = os.path.join(segmentedImageStorePath, imageName)
    if os.path.exists(segmentedImageDir) and os.path.isdir(segmentedImageDir):
        shutil.rmtree(segmentedImageDir)
    Path(segmentedImageDir).mkdir(parents=True, exist_ok=True)


def letterSegment(linesImage, xLines, i, imageName, segmentedImageStorePath, newDimension=(200, 200)):
    """Segments and saves characters from words, with optional segmented image resizing

    Args:

        linesImage (list): List holding images of individual lines 

        xLines (list): List having last word of each line

        i (int): Index of the line in focus

        imageName (string): Name of the image being segmented, including the file type

        segmentedImageStorePath (string): Path to store the segmented letters' images

        newDimension (tuple, optional): Dimensions to resize the segmented letter to. Format = (width, height) in pixels. Defaults to (200, 200). Note: Must be the same or greater dimension to the dimension of images used in model generation 

    Returns:

        list: Number of letters per word for line in focus
    """

    copyImage = linesImage[i].copy()
    xLinesCopy = xLines[i].copy()
    letterImage = []
    letterK = []

    contours, _ = cv2.findContours(
        copyImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            letterK.append((x, y, w, h))

    letter = sorted(letterK, key=lambda student: student[0])

    numberOfLetters = []
    word = 1
    letterIndex = 0
    for e in range(len(letter)):
        if(letter[e][0] < xLinesCopy[0]):
            letterIndex += 1
            letterImageTemp = linesImage[i][letter[e][1]-5:letter[e][1] +
                                            letter[e][3]+5, letter[e][0]-5:letter[e][0]+letter[e][2]+5]
            letterImage = cv2.resize(
                letterImageTemp, dsize=newDimension, interpolation=cv2.INTER_LANCZOS4)

            segmentedImageDir = os.path.join(
                segmentedImageStorePath, imageName)
            Path(segmentedImageDir).mkdir(parents=True, exist_ok=True)

            segmentedImageName = str(i+1)+'-'+str(word) + \
                '-'+str(letterIndex)+'.png'
            segmentedImagePath = os.path.join(
                segmentedImageDir, segmentedImageName)
            cv2.imwrite(segmentedImagePath, 255-letterImage)
        else:
            numberOfLetters.append(letterIndex)
            xLinesCopy.pop(0)
            word += 1
            letterIndex = 1
            letterImageTemp = linesImage[i][letter[e][1]-5:letter[e][1] +
                                            letter[e][3]+5, letter[e][0]-5:letter[e][0]+letter[e][2]+5]
            letterImage = cv2.resize(
                letterImageTemp, dsize=newDimension, interpolation=cv2.INTER_LANCZOS4)

            segmentedImageDir = os.path.join(
                segmentedImageStorePath, imageName)
            Path(segmentedImageDir).mkdir(parents=True, exist_ok=True)

            segmentedImageName = str(i+1)+'-'+str(word) + \
                '-'+str(letterIndex)+'.png'
            segmentedImagePath = os.path.join(
                segmentedImageDir, segmentedImageName)
            cv2.imwrite(segmentedImagePath, 255-letterImage)
    numberOfLetters.append(letterIndex)
    return numberOfLetters


def segmentImage(imagePath, segmentDestination):
    """Segments individual characters in provided image

    Args:

        imagePath (string): Path of image to be segmented

        segmentDestination (string): Path to store segmented characters

    Returns:

        numberOfLines (int) : Number of lines present in image

        numberOfWords (list) : List having number of words per line 

        numberOfLetters (list) : List having number of letters per word, per line
    """

    start = time()

    imageName = os.path.basename(imagePath)
    sourceImage = cv2.imread(imagePath, 1)
    copy = sourceImage.copy()
    height = sourceImage.shape[0]
    width = sourceImage.shape[1]

    sourceImage1 = sourceImage
    sourceImage2 = sourceImage
    sourceImage = cv2.resize(copy, dsize=(
        1320, int(1320*height/width)), interpolation=cv2.INTER_LANCZOS4)

    height = sourceImage.shape[0]
    width = sourceImage.shape[1]

    grayImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)

    binaryImage = cv2.adaptiveThreshold(
        grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 20)
    binaryImage1 = binaryImage.copy()
    binaryImage2 = binaryImage.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)

    finalThreshold = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)
    countourRetrival = finalThreshold.copy()

    countX = np.zeros(shape=(height))
    for y in range(height):
        for x in range(width):
            if binaryImage[y][x] == 255:
                countX[y] = countX[y]+1

    upperLines, lowerLines = lineArray(countX)

    upperLines, lowerLines = refineArray(upperLines, lowerLines)

    if len(upperLines) == len(lowerLines):
        lines = []
        for y in upperLines:
            finalThreshold[y][:] = 255
        for y in lowerLines:
            finalThreshold[y][:] = 255
        for y in range(len(upperLines)):
            lines.append((upperLines[y], lowerLines[y]))
    else:
        print("Segmentation unsuccessful. Ended in " +
              str(time() - start) + " seconds")
        sys.exit(
            "Too much noise in image, unable to process. Please try with another image.")

    lines = np.array(lines)
    numberOfLines = len(lines)
    linesImage = []

    for i in range(numberOfLines):
        linesImage.append(binaryImage2[lines[i][0]:lines[i][1], :])

    contours, _ = cv2.findContours(
        countourRetrival, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContour = np.zeros(
        (finalThreshold.shape[0], finalThreshold.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(sourceImage, contours, -1, (0, 255, 0), 1)

    meanLetterWidth = letterWidth(contours)

    xLines = []

    for i in range(len(linesImage)):
        xLines.append(endWordDetect(lines, i, binaryImage,
                                    meanLetterWidth, width, finalThreshold))

    numberOfWords = []
    for line in xLines:
        numberOfWords.append(len(line))

    for i in range(len(xLines)):
        xLines[i].append(width)

    createFolderToStoreSegmentedImage(segmentDestination, imageName)

    numberOfLetters = []
    for i in range(len(lines)):
        numberOfLetters.append(letterSegment(
            linesImage, xLines, i, imageName, segmentDestination))
    characterImage = binaryImage1.copy()

    contours, _ = cv2.findContours(
        characterImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(sourceImage, (x, y), (x+w, y+h), (0, 255, 0), 2)
    showImages(sourceImage)
    print("Segmentation completed in " + str(time() - start) + " seconds")

    return numberOfLines, numberOfWords, numberOfLetters
