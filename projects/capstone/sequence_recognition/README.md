# Project 5: Street view sequence recognition

The goal of this project is to provide a machine learning model that can interpret and recognize sequences of digits in real-world street images.
![Image recognition example](https://github.com/thalles753/machine-learning/blob/master/projects/capstone/sequence_recognition/images/fig1.PNG?raw=true)

## Data Exploration and Visualization
The SVHN dataset, is used to train and evaluate the model proposed here. It consists of approximately 600,000 training digits and roughly 26,000 digits for testing. For more information refer to: http://ufldl.stanford.edu/housenumbers/.

![Image of dataset distribution](https://github.com/thalles753/machine-learning/blob/master/projects/capstone/sequence_recognition/images/fig2.PNG?raw=true)

## Model
The proposed convolutional neural network receives as input 54 x 54 pixel image(s) and outputs, for each image, a collection of N random variables S1,…,SN, where N is 5, representing the possible sequence digits on each image. Each of these output variables has a set of possible outcomes therefore, each Si has 11 possible results each, ranging from 0 to 10 where 0 represent the digit 0, 1 represents the digits 1 and so fourth with 10 representing no digit. For instance, when analyzing the image with the sequence 316 (first image on the third row of Figure 2), the desired output of the model will be the sequence [3, 1, 6, 10, 10].

![Deep learning model archtecture](https://github.com/thalles753/machine-learning/blob/master/projects/capstone/sequence_recognition/images/fig3.PNG?raw=true)

## Results

Our model was able to get 94.2% accuracy in the sequence transcription section and 98.37% in the character-level accuracy. For more information, see the Results section of the report.

These are the Confusion matrices for the first two digits placeholder of a sequence that might go up to five digits.

![confusion matrices](https://github.com/thalles753/machine-learning/blob/master/projects/capstone/sequence_recognition/images/fig4.PNG?raw=true)

## Reproduce

To reproduce the work please make sure you have python 2.7 along with the following libraries:

- Tensorflow
- Anaconda (Optional)
- numpy
- matlibplot
- scipy
- planar
- PIL
- h5py
- sys
- tarfile
- os

This package contains the following files:

- DataProcessing.ipynb
- Model.ipynb
- Capstone Project Report.pdf
- README.md
