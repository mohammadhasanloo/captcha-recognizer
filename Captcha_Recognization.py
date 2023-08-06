# %% [markdown]
# # Mohammad GharehHasanloo
# ## AI Medic Internship
# ## Second Project

# %% [markdown]
# ### Import Necessary Libraries

# %%
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential
import tensorflow.keras.layers as tfkl
from random import shuffle

from pathlib import Path
PNG_FOLDER_NAME = str(Path(__file__).parent.absolute()) + '\\samples\\'

DEFAULT_CAPTCHA_LENGTH = 5

# %% [markdown]
# ### Load Images from Folder and Convert Images to Grayscale Form
# load data and apply necessary preprocessing operations to the data

# %%


def delete_noisy_line(image: list):
    """ Some processes to delete line on the letters

    Args:
        image (list): input image that has been loaded recently

    Returns:
        image (list): converted image to approximately delete noisy line
    """
    image = cv2.blur(image, (3, 3))
    ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)

    image = cv2.dilate(image, np.ones((3, 1), np.uint8))
    image = cv2.erode(image, np.ones((2, 2), np.uint8))

    return image


def load_images_from_folder(folder: str, shuffled_files: list):
    """ This function load images from the determined folder

    Args:
        folder (str): the name of folder
        shuffled_files (list): list of files in determined folder those are shuffled

    Returns:
        images (list): the variable to save images
        splited_labels (list): return the characters of what images show
        unique_chars (dict): the unique characters that have been found in labels
    """
    images = []
    labels = []
    unique_chars = {}

    for filename in shuffled_files:
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        img = delete_noisy_line(img)

        # Add image with its label to `images` and `labels`
        if img is not None:
            images.append(img)
            labels.append(filename.split('.')[0])

            # Find unique characters with their number
            for char in filename.split('.')[0]:
                if char in unique_chars:
                    unique_chars[char] += 1
                else:
                    unique_chars[char] = 1

    return images, labels, unique_chars


def preprocess_data(images: np.ndarray, labels: np.ndarray, unique_chars: dict, shuffled_files: list):
    """ This function apply preprocess images and labels to convert them to proper shape 

    Args:
        images (np.ndarray): The images those are saved in the type of array
        labels (np.ndarray):  the labels in shape: (sample's count,)
        unique_chars (dict): unique chars that are found in captchas
        shuffled_files (list): list of files in determined folder those are shuffled

    Returns:
        X (np.ndarray): converted images in shape: (sample's count, height, width, 1)
        y (np.ndarray): converted labels in shape: (length of each example, sample's count, unique chars)
    """
    captcha_length = len(labels[0])
    unique_chars_count = len(unique_chars)

    X = np.zeros(images[...,  np.newaxis].shape)
    y = np.zeros((captcha_length, len(labels), unique_chars_count))

    for i, img in enumerate(shuffled_files):
        # Divide cells of images by 255 to learn more quickly in neural network
        captcha_img = images[i]
        captcha_img = captcha_img/255.0

        # Reshape `images`
        captcha_img = np.reshape(
            captcha_img, images[0, ...,  np.newaxis].shape)
        X[i] = captcha_img

        # Reshape `labels`
        curr_name = np.zeros((captcha_length, unique_chars_count))
        for j, char in enumerate(labels[i]):
            curr_name[j, list(unique_chars.keys()).index(char)] = 1
        y[:, i] = curr_name

    return X, y


# %%
# Shuffle folders' name to unorder the files
shuffled_files = os.listdir(PNG_FOLDER_NAME)
shuffle(shuffled_files)

images, labels, unique_chars = load_images_from_folder(
    PNG_FOLDER_NAME, shuffled_files)
images, labels = np.array(images), np.array(labels)
images, labels = preprocess_data(images, labels, unique_chars, shuffled_files)

unique_chars_count = len(unique_chars)

print(f"images' shape is {images.shape}")
print(f"labels' shape is {labels.shape}")
print(f"unique characters count equals: {unique_chars_count}")

# %% [markdown]
# #### Plot a Sample

# %%
index = 32
plt.imshow(images[index])
plt.show()
print(f"array of label number {index} equals:\n {labels[:, index]}")

# %% [markdown]
# ### Design an Object Oriented Network
# I design a network that has five `Conv2D` that each one is followed by a `MaxPool2D`.<br/>
# Then, it's followed by a `Flatten`, `Dropout` and `BatchNormalization`..<br/>
# At last, we have five `Sequential` part those are parallel and each one is used for predicting one letter of the captcha..<br/>
# Each Sequential part has one `Dense` with 64 neurons, `Dropout`, `BatchNormalization` and `Dense` the number of output(`num_classes`)

# %%


class MyModel(Model):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()

        self.layer1 = Sequential([
            tfkl.Conv2D(16, activation='relu', kernel_size=5,
                        strides=1, padding="same"),
            tfkl.MaxPool2D(pool_size=(2, 2), padding='same'),
        ])
        self.layer2 = Sequential([
            tfkl.Conv2D(32, activation='relu', kernel_size=3,
                        strides=1, padding="same"),
            tfkl.MaxPool2D(pool_size=(2, 2), padding='same'),
        ])
        self.layer3 = Sequential([
            tfkl.Conv2D(64, activation='relu', kernel_size=3,
                        strides=1, padding="same"),
            tfkl.MaxPool2D(pool_size=(2, 2), padding='same'),
        ])
        self.layer4 = Sequential([
            tfkl.Conv2D(128, activation='relu', kernel_size=3,
                        strides=1, padding="same"),
            tfkl.MaxPool2D(pool_size=(2, 2), padding='same'),
        ])
        self.layer5 = Sequential([
            tfkl.Conv2D(256, activation='relu', kernel_size=3,
                        strides=1, padding="same"),
            tfkl.MaxPool2D(pool_size=(2, 2), padding='same'),
        ])
        self.layer6 = Sequential([
            tfkl.Flatten(),
            tfkl.Dropout(0.2),
            tfkl.BatchNormalization(),
        ])
        self.layer7_1 = Sequential([
            tfkl.Dense(64, activation='relu'),
            tfkl.Dropout(0.2),
            tfkl.BatchNormalization(),
            tfkl.Dense(num_classes, activation='softmax', name='char1'),
        ])
        self.layer7_2 = Sequential([
            tfkl.Dense(64, activation='relu'),
            tfkl.Dropout(0.2),
            tfkl.BatchNormalization(),
            tfkl.Dense(num_classes, activation='softmax', name='char2'),
        ])
        self.layer7_3 = Sequential([
            tfkl.Dense(64, activation='relu'),
            tfkl.Dropout(0.2),
            tfkl.BatchNormalization(),
            tfkl.Dense(num_classes, activation='softmax', name='char3'),
        ])
        self.layer7_4 = Sequential([
            tfkl.Dense(64, activation='relu'),
            tfkl.Dropout(0.2),
            tfkl.BatchNormalization(),
            tfkl.Dense(num_classes, activation='softmax', name='char4'),
        ])
        self.layer7_5 = Sequential([
            tfkl.Dense(64, activation='relu'),
            tfkl.Dropout(0.2),
            tfkl.BatchNormalization(),
            tfkl.Dense(num_classes, activation='softmax', name='char5'),
        ])

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        output1, output2, output3, output4, output5 = self.layer7_1(
            x), self.layer7_2(x), self.layer7_3(x), self.layer7_4(x), self.layer7_5(x)
        output = [output1, output2, output3, output4, output5]
        return output


# %%
input = tfkl.Input(images[0].shape)
output = MyModel(unique_chars_count)(input)

model = Model(inputs=input, outputs=output)

# %%
model.summary()

# %%
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')

# %% [markdown]
# #### Separate Data to Train and Test
# Because we shuffle files based on their names, it doesn't need to shuffle

# %%
# Assign 80% data to train
train_size = 0.8
numTrainingSamples = int(len(images) * train_size)
numTestingSamples = int(len(images) - numTrainingSamples)

X_train = images[:numTrainingSamples]
y_train = labels[:, :numTrainingSamples]
X_test = images[numTrainingSamples:]
y_test = labels[:, numTrainingSamples:]

# %% [markdown]
# #### Fiting Data

# %%
history = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], epochs=30,
                    batch_size=32, validation_data=(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]]))

# %% [markdown]
# ### Evaluation
# I evaluate by characters one by one but it doesn't work in character number 1 as good as other characters --> I solve this problem by shuffling<br/>
# UPDATE: It works on all characters and return accuracy more than 94% for each character

# %%
modelEvaluation = model.evaluate(
    X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])

# %%
print(f'Character1 Prediction Accuracy : {modelEvaluation[6]*100} %')
print(f'Character2 Prediction Accuracy : {modelEvaluation[7]*100} %')
print(f'Character3 Prediction Accuracy : {modelEvaluation[8]*100} %')
print(f'Character4 Prediction Accuracy : {modelEvaluation[9]*100} %')
print(f'Character5 Prediction Accuracy : {modelEvaluation[10]*100} %')

# %% [markdown]
# #### Plot Diagram to Show Loss in Epochs
# the loss in train and test has droped but we can see a little overfitting but that's not considerable

# %%


def draw_loss_diagram_of_history(curr_history):
    plt.plot(curr_history.history['loss'])
    plt.plot(curr_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# %%
draw_loss_diagram_of_history(history)

# %% [markdown]
# ### See Predictions
# return `true_word` and `predicted_word` from `X_test` and `y_test`

# %%
characters = list(unique_chars.keys())
predicted_word = []
true_word = []

for i, sample in enumerate(X_test):
    sample = np.reshape(X_test[i], images[0][np.newaxis, :].shape)
    sample = model.predict(sample)
    sample = np.reshape(sample, (DEFAULT_CAPTCHA_LENGTH, unique_chars_count))
    predicted = ''.join([characters[np.argmax(i)] for i in sample])
    predicted_word.append(predicted)

for i in range(0, numTestingSamples):
    temp = ''.join([characters[i] for i in (np.argmax(y_test[:, i], axis=1))])
    true_word.append(temp)

# %% [markdown]
# #### Convert to csv
# Convert true values and predicted values to dataframe and then make a csv file to show predictions

# %%
predicted_word = np.array(predicted_word)
true_word = np.array(true_word)
df = pd.DataFrame({'Predicted_value': predicted_word, 'True_value': list(
    true_word)}, columns=['Predicted_value', 'True_value'])
print(df)

# %%
df.to_csv('result.csv', index=False)
