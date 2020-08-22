import os
import cv2
import numpy as np

from time import time
from joblib import Parallel, delayed
from skimage import io, transform, img_as_uint


def rotateClockwise(image, path, angle=20):
    """Rotates image to clockwise direction

    Args:

        image (numpy array): Array representation of an image

        path (string): Path of the image

        angle (int, optional): angle of clockwise rotation. Defaults to 20.

    Purpose:

        Saves the Rotated image with tag "-clock" to the actual file name in the actual images' location
    """

    if(angle > 0):
        angle = -angle

    image = transform.rotate(image, angle, resize=True, cval=1)
    image = transform.resize(image, (200, 200))
    image = img_as_uint(image)

    index = path.find('.png')
    newPath = path[:index] + "-clock" + path[index:]
    io.imsave(newPath, image)


def rotateAntiClockwise(image, path, angle=20):
    """Rotates image to anti-clockwise direction

    Args:

        image (numpy array): Array representation of an image

        path (string): Path of the image

        angle (int, optional): angle of anti-clockwise rotation. Defaults to 20.

    Purpose:

        Saves the Rotated image with tag "-anticlock" to the actual file name in the actual images' location
    """

    image = transform.rotate(image, angle, resize=True, cval=1)
    image = transform.resize(image, (200, 200))
    image = img_as_uint(image)

    index = path.find('.png')
    newPath = path[:index] + "-anticlock" + path[index:]
    io.imsave(newPath, image)


def imageAugmentation(datasetPath, subFolder, file):
    """Augments individual images

    Args:

        datasetPath (string): Relative or Absolute path of the dataset.

        subFolder (string): Folder name of specific character

        file (string): Name of image

    Purpose:

        Augments image specified in the path "datasetPath + subFolder + file"
    """

    if(file == ".ipynb_checkpoints"):
        return

    start = time()
    path = os.path.join(datasetPath, subFolder, file)
    image = io.imread(path)
    rotateClockwise(image, path)
    rotateAntiClockwise(image, path)

    print("Augmentation completed in ", time() - start, " seconds")


def parallelAugmentation(datasetPath, threads=4):
    """Performs parallel augmentation of multiple images

    Args:

        datasetPath (string): Relative or Absolute path of the dataset.

        threads (int, optional): Number of images to be augmented in parallel. Defaults to 4. Note: This value should not be greater than the number of physical or logical (if processor virtualization present) cores present.

    Purpose:

        Enables augmenting multiple images in parallel
    """

    start = time()

    folders = np.asarray(os.listdir(datasetPath))
    files = []
    for folder in folders:
        files.append(np.asarray(os.listdir(datasetPath + folder)))
    for i in range(len(folders)):
        if folders[i] == ".ipynb_checkpoints":
            break
        Parallel(n_jobs=threads, prefer="threads")(
            delayed(imageAugmentation)(datasetPath, folders[i], files[i][j]) for j in range(len(files[i])))

    print("Process completed in ", time() - start, " seconds")
