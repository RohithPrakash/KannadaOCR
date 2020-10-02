# Optical Character Recognizer

An Optical Character Recognizer for handwritten Kannada Scripts. It is currently a POC and is partially capable of recognizing the first 15 characters of the Kannada script.

The [dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) used was compiled by [T de Campos](http://personal.ee.surrey.ac.uk/Personal/T.Decampos/) at Microsoft Research India.

## Requirements

    At the time of building this application, Tensorflow did not support Python 3.8. The application is not tested on Python 3.8 as of now.

* [Python 3.7.9](https://www.python.org/downloads/release/python-379/ "Download Python 3.7.9")

You could also use a virtual environment like [conda](https://docs.conda.io/ "Conda docs").

### Dependencies

* joblib
* keras
* matplotlib
* numpy
* opencv-python
* scikit-image
* tensorboard
* tensorflow

Install dependencies from requirements.txt file

```
pip install -r requirements.txt
```
## How To Run

In terminal, run

```
python main.py <path of the image to be recognized>
```

Recognized text is displayed on console and also exported to `output.txt`.
