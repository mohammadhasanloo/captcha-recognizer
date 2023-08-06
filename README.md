# Captcha Recognition Neural Network

## Description

This repository contains a neural network model for recognizing captchas. It includes a collection of captchas in the [samples] folder, and the neural network model is used to predict the characters present in new captchas. For more detailed information, please refer to the [Report.html] file.

## Installation

To run this project, follow the steps below:

1. Clone the repository to your local machine.

2. Open the command prompt (cmd) in the folder of the project.

3. Run the following command to execute the Captcha Recognition script:

```
python Captcha_Recognization.py
```

During the running process, you will see a converted sample figure after preprocessing. After that, some information will be printed, and the data will be fitted (estimated 2-3 minutes). Finally, a file named [result.csv] will be created, which shows the predicted values and true values in CSV format for the test data.

## AI Medic Internship - Second Project

This project implements a neural network for captcha recognition using Python and TensorFlow. The model is designed using object-oriented programming to improve modularity and maintainability.

## Import Necessary Libraries

The necessary libraries are imported to perform image processing, neural network modeling, and visualization. The libraries include cv2, os, numpy, pandas, tensorflow, and matplotlib.

## Load Images from Folder and Convert Images to Grayscale Form

The images are loaded from the specified folder, and preprocessing operations are applied to the data. The function `delete_noisy_line()` is used to delete noise lines from the images. The images are then converted to grayscale and stored along with their labels and unique characters.

## Preprocess Data

The loaded images and labels are preprocessed to convert them to the proper shape for training the neural network. The data is divided into X (input images) and y (target labels). The characters in the labels are converted to one-hot encoded vectors to be used as target values for the model.

## Design an Object-Oriented Network

An object-oriented neural network model is designed using the TensorFlow Keras API. The model consists of five Conv2D layers, each followed by a MaxPool2D layer. It is then followed by a Flatten, Dropout, and BatchNormalization layers. The model is designed to predict each character of the captcha separately. Each character prediction is represented by a separate sequential part of the model.

## Fiting Data

The model is compiled using the Adam optimizer and categorical cross-entropy loss function. The data is split into training and testing sets (80% for training and 20% for testing). The model is trained for 30 epochs using a batch size of 32.

## Evaluation

The model is evaluated on the test data, and the accuracy of predicting each character is printed. The accuracy achieved on each character is more than 94%.

## See Predictions

The model's predictions and true values are obtained for the test data. The predicted values are converted to a dataframe, and a CSV file named "result.csv" is created to show the predictions.

## Note

Please make sure to have all the necessary dependencies installed before running the script.
