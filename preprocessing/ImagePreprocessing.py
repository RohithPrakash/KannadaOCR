import os
import cv2
import warnings
import numpy as np

from time import time
from joblib import Parallel, delayed
from skimage.util import invert
from skimage import io, filters
from skimage.morphology import medial_axis

warnings.filterwarnings('ignore')


def resizeImage(image, newDimension=(200, 200)):
    """Resizes image to provided dimension

    Args:

        image (numpy array): Array representation of an image

        newDimension (tuple, optional): Width and height of resized image. Format = (width, height) in pixels. Defaults to (200, 200).

    Returns:

        numpy array: Resized image
    """

    resized = cv2.resize(image, newDimension, interpolation=cv2.INTER_LANCZOS4)
    return resized


def rgbToGray(image):
    """Converts color image to grayscale image

    Args:

        image (numpy array): Array representation of an image

    Returns:

        numpy array: Grayscale image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def grayToBinary(image):
    """Converts grayscale image to black and white image

    Args:

        image (numpy array): Array representation of an image

    Returns:

        numpy array: Binary image
    """
    (thresh, binary) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary


def denoiseImage(image):
    """Denoise image

    Args:

        image (numpy array): Array representation of an image

    Returns:

        numpy array: Denoised image
    """
    return cv2.fastNlMeansDenoising(image)


def imageSmoothing(image):
    """Smoothens pixelated image

    Args:

        image (numpy array): Array representation of an image

    Returns:

        numpy array: Smoothened image
    """
    return cv2.medianBlur(image, 5)


def imageThinning(datasetPath, subFolder, file):
    """Thins and saves binary image

    Args:

        datasetPath (string): Relative or Absolute path of the dataset.

        subFolder (string): Folder name of specific character

        file (string): Name of image

    Purpose:

        Saves the thinned image with tag "-thinned" to the actual file name in the actual images' location
    """

    path = os.path.join(datasetPath, subFolder, file)

    original = invert(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    binary = original > filters.threshold_otsu(original)
    skeleton, distance = medial_axis(binary, return_distance=True)
    distanceOnSkeleton = distance * skeleton
    invertedImage = invert(distanceOnSkeleton)

    index = file.find('.png')
    newFile = file[:index] + "-thinned" + file[index:]
    newPath = os.path.join(datasetPath, subFolder, newFile)
    io.imsave(newPath, invertedImage)


def preProcessImages(datasetPath, subFolder, file):
    """Preprocesses individual images

    Args:
        datasetPath (string): Relative or Absolute path of the dataset.

        subFolder (string): Folder name of specific character

        file (string): Name of image

    Purpose:

        Preprocesses image specified in the path "datasetPath + subFolder + file"
    """

    if(file == ".ipynb_checkpoints"):
        return

    start = time()

    path = os.path.join(datasetPath, subFolder, file)

    image = cv2.imread(path)
    image = resizeImage(image)
    image = rgbToGray(image)
    image = grayToBinary(image)
    image = denoiseImage(image)
    image = imageSmoothing(image)

    os.remove(path)
    cv2.imwrite(path, image)
    imageThinning(datasetPath, subFolder, file)
    os.remove(path)

    print("Preprocessing completed in ", time() - start, " seconds")


def parallelPreProcessing(datasetPath, threads=4):
    """Performs parallel preprocessing of multiple images

    Args:

        datasetPath (string): Relative or Absolute path of the dataset.

        threads (int, optional): Number of images to be preprocessed in parallel. Defaults to 4. Note: This value should not be greater than the number of physical or logical (if processor virtualization present) cores present.

    Purpose:

        Enables preprocessing multiple images in parallel

    """
    start = time()

    folders = np.asarray(os.listdir(datasetPath))
    files = []
    for folder in folders:
        if folder == ".ipynb_checkpoints" or folder.find(".ipynb"):
            continue
        files.append(np.asarray(os.listdir(datasetPath + folder)))
    for i in range(len(folders)):
        Parallel(n_jobs=threads, prefer="threads")(
            delayed(preProcessImages)(datasetPath, folders[i], files[i][j]) for j in range(len(files[i])))

    print("Process completed in ", time() - start, " seconds")
