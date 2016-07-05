#Capstone Project
####Machine Learning Engineer 
####Nanodegree


## Definition

###Peoject Overview

The goal of this project is to provide a web app that is capable of interpreting sequencies of numbers in an image. With the massive increase of camera smartphones and the latter advances on the field of image processing, the necessity of having a tool capable of automatically identifying objects on these pictures was never so high as it is now. Based on that, this project aims to provide an interactive tool that will make anyone capable of having digits on their images automatically identified in real time. In other words,  for each picture taken in real time, this system will automatically output an indication of what number is in the picture. 

The core of this project is about creating a mechanism that can identify a combination of digits on an image in the various different conditions. The system should be able to identify the number(s) on the picture regardless of the pictures' conditions such as ilumination (daily or nightly photograph) and quality i.e. ability to identify the sequence on either high quality or very small pictures. 

###Problem Statement

The problem of image recognition has been around for a long time in the field of computer science. In order to understand the complexity of this job, one might think about how many combinations of pictures of one particular object there are and therefore, the system should be aware of to complete the task successfully. Clearly, we note that the strategy of comparing one particular photograph with many others so that we can identify what objects there are on that picture is not practical neither possible. We need a system able to not only recognize a set of digits on pictures that we might have but also able to perform well on images that we did not take yet. Strictly speaking, the system has to have the ability of recognizing images that we have taken and images that we do not yet have taken. 

To do that, this system cannnot only be aware of the underlying structure of any particular image, rather, it has to learn the general structure that is present everytime this specifically digit appears. Furthermore, it has to learn what a digit is, it has to learn its commom structure and design so that it will look for that structure on any image that that we take in the future. Based on these facts, the goal of this report is to describe a system that will learn the basic patterns of digits and apply this knoledge for every new picture we take.

To address this challenge, we propose a Deep Convolutional Neural Network (DCNN) archtecture that will take as input a set of images obtained from house numbers in Google Street View. The Street View House Numbers (SVHN) Dataset comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). The dataset contains three separete sets of data, a 73257 image traning dataset, a 26032 testing dataset and a 531131 additional extra dataset, with somewhat less difficult samples, to use as extra training data. Originally the dataset contains 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10, in our setup, we converted the labels 10 to become 0 so images with 0s will have will be identified as 0 instead of 10.