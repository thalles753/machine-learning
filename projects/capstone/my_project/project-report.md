#Capstone Project
####Machine Learning Engineer 
####Nanodegree


## Definition

###Peoject Overview

The goal of this project is to provide a web app that is capable of interpreting sequencies of numbers in an image. With the massive increase of camera smartphones and the latter advances on the field of image processing, the necessity of having a tool capable of automatically identifying objects on these pictures was never so high as it is now. Based on that, this project aims to provide an interactive tool that will make anyone capable of having digits on their images automatically identified in real time. In other words, for each picture with a sequence of digits on it, the system descrived in this document will automatically output an indication of what numbers are in the picture. 

The core of this project is about creating a mechanism that can identify a combination of digits on an image in the various different conditions. The system should be able to identify the number(s) on the picture regardless of the pictures' conditions such as ilumination (daily or nightly photograph) and quality i.e. ability to identify the sequence on either high or small quality pictures. 

###Problem Statement

The problem of image recognition has been around for a long time in the field of computer science. In order to understand the complexity of this job, one might think about how many combinations of pictures of one particular object there are and therefore, the system should be aware of to complete the task successfully. Clearly, we note that the strategy of comparing one particular photograph with many others so that we can identify what objects there are on that picture is not practical neither possible. We need a system able to not only recognize a set of digits on pictures that we might have, but also able to perform well on images that we did not take yet. Strictly speaking, the system has to have the hability of recognizing digits on images that we have taken and digits on images that we have not yet taken. 

To do that, this system cannnot only be aware of the underlying structure of any particular image, rather, it has to learn the general structure that is present everytime one specifically digit appears regardless of its shape or light conditions. Furthermore, it has to learn what a digit is, it has to learn its commom structure and design so that it will look for that structure on any image that that we take in the future. Based on these facts, the goal of this report is to describe a computational model that will learn the basic patterns of digits and apply this knoledge for every new picture we take.

To address this challenge, we propose a Deep Convolutional Neural Network (DCNN) archtecture that will take as input a set of images obtained from house numbers in Google Street View. The Street View House Numbers (SVHN) Dataset comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). The dataset contains three separete sets of data, a 73257 image traning dataset, a 26032 testing dataset and a 531131 additional extra digits dataset, with somewhat less difficult samples, to use as extra training data. Originally the dataset contains 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10, in our setup, we converted the labels 10 to become 0 so images with 0s will be identified as 0 instead of 10.

The main methods used in the project to properly evaluate the model descrived are accuracy and error rate. For some problems in which the distribution of examples among the class labels is not will balanced, the accuracy can be a misleading measure. Let's take as an example the SVHN training dataset with 73257 image used in to train this model. If we take a look at the class labels distribution of this dataset, the first thing one might notice is the very high number of samples with class label 1 over the others. In fact, the histogram shows that there are nearly 14000 images of class label 1 (one) on the dataset whilst there are only roughly 5000 image of class label 0. This descrepance on the data distribution could result in a very strange behavious using the accuracy measure. For instance, let's now consider a model that regardless of the input image, it would always says that the digit on the image is the digit 1 (one). Since there are 13861 images of class label 1 the model would get all of the 13861 samples correct resulting a score of roughly 18.8%. Note that this is not a terrible accuracy but it is a very bad model.

Additionally to our visual ituition, it would be very good to have a mathematical way of representing the degree of unballacing in this dataset. To do that, we can define the Degree of Class Imbalance (DCI) as described in [], as the equation:

\begin{equation*}
DCI = \frac{1}{N} \left( \frac{1}{(K-1)} \sum_{k} \left( |c_k| - \frac{N}{K} \right)^2  \right)^{\frac{1}{2} }
\end{equation