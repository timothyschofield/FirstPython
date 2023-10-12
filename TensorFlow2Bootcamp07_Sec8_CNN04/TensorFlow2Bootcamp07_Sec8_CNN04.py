"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

30 September 2023

Spent whole day looking for cell_images.zip

Section 8: Convolutional Neural Networks - CNNs Coding

"""
print("========== 69. Downloading Data Set for Real Image Lectures ==========")
# pip install pandas
# pip install numpy
import pandas as pd
import numpy as np

# pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

# pip install tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

import os

data_dir = "cell_images" # The folder cell_images is in current directory
print(os.listdir(data_dir))
# ['test', 'train']
#"cell_images\\train"
#"cell_images\\test\\"

print("========== 70. CNN on Real Image Files - Part One - Reading in the Data ==========")
"""
1 October 2023
TensorFlow has built in tools for generating batches of image data from 
    directories of files
"""
from matplotlib.image import imread

test_path = data_dir + "\\test\\"
print(os.listdir(test_path))
# ['parasitized', 'uninfected']

train_path = data_dir + "\\train\\"
print(os.listdir(train_path))
# ['parasitized', 'uninfected']

# Name of first image in parasitized:
para_cell_name = os.listdir(train_path + "parasitized")[0]
print("Name of first image", para_cell_name)

para_cell_full_path = train_path + "parasitized\\" + para_cell_name
para_cell_data = imread(para_cell_full_path) # Converts a png file to an array
# print(para_cell_data)
"""
[[[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]
  ...
"""

print(para_cell_data.shape) # (148, 142, 3)
# plt.imshow(para_cell_data)
# plt.show()

# NOTE: the image files vary in their dimensions
# Lets look at the dimensions of the files
dim1 = []
dim2 = []

for image_filename in os.listdir(test_path + "uninfected"):

    img = imread(test_path + "uninfected\\" + image_filename)
    d1, d2, num_channels = img.shape
    dim1.append(d1)
    dim2.append(d2)

dim1 = np.array(dim1)
dim2 = np.array(dim2)
df = pd.DataFrame({"dim1":dim1, "dim2":dim2})

#sns.jointplot(df) # New!: Draws a plot of two variables with bivariate and univariate graphs
#plt.show()
#sns.scatterplot(df)
#plt.show()

# But CNNs can only train on a single size - choose the mean
print("mean dimensions", np.mean(dim1), np.mean(dim2)) # 130.92538461538462 130.75
image_shape = (130, 130, 3)

print("========== 71. CNN on Real Image Files - Part Two - Data Processing ==========")
# There is too much data to read in all at once
print(28*28)        # 784
print(32*32*3)      # 3072
print(130*130*3)    # 50700
# The model should be robust enough to deal with images it hasn't seen before
# One way we can do this is by performing transformations,
# like rotation, scaling and resizing

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# print(help(ImageDataGenerator))
# First thing it tell us is that
# tf.keras.preprocessing.image.ImageDataGenerator is deprecated - Oh well

# Our narrator comes out with a strange point:
# That the total number of images in our dataset is quite small < 30,000 compared
# the the MNIST dataset which was 60,000.
# So what we want to be able to do is "expand" the amount of images, without having
# to get more data - we can't just grab people and take more blood samples.
# So what we can do (he says) is take our current images and
# randomly rotate, scale, change width and height, skew etc.
# Generate batches of tensor image data
image_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode="nearest")

# These transformed images are called "Augmented" images or "Data Augmentation"

# It appears that the pixel values are already normalized to 0.0->1.0, so no rescale
# fill_mode="nearest" - when you have (say) shrunk an image, and end up with some
# blank pixels on the canvas, what do you fill them with?

# "Fundamental algorithms for scientific computing in Python"
# scipy, not mentioned in lesson
import scipy

# Let's try image_gen out on a single image
# random_transform, based of the restrictions we passed into the constructor
para_cell_data_transformed = image_gen.random_transform(para_cell_data)
#plt.imshow(para_cell_data_transformed)
#plt.show()
# ----------------------------------------------------------------
# So, how do we setup our directories, to flow batches from a directory?
# The directories have to be set up in a certain way - which they are
# train_path
#   parasitized (Class 1)
#   uninfected  (Class 2)
#   ...
#   folderN     (Class n)

image_gen.flow_from_directory(train_path)
# Prints out: "Found 24958 images belonging to 2 classes." 12,479 in each folder

image_gen.flow_from_directory(test_path)
# Prints out: "Found 2600 images belonging to 2 classes." 1,300 in each folder

print("========== 72. CNN on Real Image Files - Part Three - Creating the Model ==========")
# 02 October 2023

# The larger the images sizes, and the more complex the task,
# the more convolutional layers you should have, see:
# https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters
model = Sequential()

# Specified up top, image_shape = (130, 130, 3)
model.add(Conv2D(filters=32,
             kernel_size=(3,3),
             input_shape=image_shape,
             activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,
             kernel_size=(3,3),
             input_shape=image_shape,
             activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,
             kernel_size=(3,3),
             input_shape=image_shape,
             activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten()) # 3or2D to 1D

model.add(Dense(128, activation="relu")) # 128 neurons to accept the flattened input

# Dropout helps reduce overfitting by randomly turning off during training
# (here) 50% of the neurons
model.add(Dropout(0.5))

# Last layer is binary, so sigmoid
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# Prints out the structure of the model and
# the number of parameters per layer - 1,605,760 fo the first dense layer
# print(model.summary())
"""
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 128, 128, 32)      896  <<<< where does this number come from?
                                                                 
 max_pooling2d (MaxPooling2  (None, 64, 64, 32)        0         
 D)
...
"""
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", patience=2)

batch_size = 16 # images at a time
# image_shape = (130, 130, 3) - just need first two hence image_shape[:2]
# image_gen (instance of ImageDataGenerator), generates batches of tensor image data from batches
# of image files in the directory
train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                color_mode="rgb",
                                                batch_size=batch_size,
                                                class_mode="binary",
                                                shuffle=True)
# shuffle=False - don't understand
test_image_gen = image_gen.flow_from_directory(test_path,
                                                target_size=image_shape[:2],
                                                color_mode="rgb",
                                                batch_size=batch_size,
                                                class_mode="binary",
                                                shuffle=False)
# print(train_image_gen.class_indices)
# {'parasitized': 0, 'uninfected': 1}

# Apparently all "_generator" methods can no be replaced
# fit_generator -> fit
# predict_generator -> predict
# evaluate_generator -> evaluate
#results = model.fit_generator(train_image_gen,
#                              epochs=20,
#                              validation_data=test_image_gen,
#                              callbacks=[early_stop])

"""
6 min 26 sec per epoch

Where does the 1560 come from?
Epoch 1/20
  18/1560 [..............................] - ETA: 6:53 - loss: 13.9208 - accuracy: 0.4896
  ...
Takes too long - so we actualy have a pre-trained model called malaria_detector.h5

"""
from tensorflow.keras.models import load_model

model = load_model("malaria_detector.h5")
# print(model.summary())
# He acknowledges that no training history has been stored with the model
# i.e. model.history.history is null
"""
# pickle is an object serializer
# Save histry as a Dictionary in file_pi (file_pickle)
with open("/trainHistoryDict", "wb") as file_pi:
    pickle.dump(history.history, file_pi)
"""
# However, we can still evaluate the model results
# print(model.evaluate_generator(test_image_gen)) # depricated
print(model.metrics_names)
# ['loss', 'accuracy']

print("========== 73. CNN on Real Image Files - Part Four - Evaluating the Model ==========")

# Apparently all "_generator()" methods can no be replaced by the shorter version
# fit_generator() -> fit()
# predict_generator() -> predict()
# evaluate_generator() -> evaluate()

pred = model.predict(test_image_gen) # don't realy know if this is working
#print(pred)
"""
[[0.]
 [0.]
 [0.]
 ...
 [1.]
 [1.]
 [0.]]
"""
predictions = pred > 0.5
#print(predictions)
"""
[[False]
 [False]
 [False]
 ...
 [ True]
 [ True]
 [ True]]
"""
# print("len pred", len(pred))
# len pred 2600

# pip install scikit-learn
from sklearn.metrics import classification_report, confusion_matrix

# print(test_image_gen.classes)
# [0 0 0 ... 1 1 1]

print(classification_report(test_image_gen.classes, predictions))
"""
              precision    recall  f1-score   support

           0       0.81      0.98      0.88      1300
           1       0.97      0.76      0.86      1300

    accuracy                           0.87      2600
   macro avg       0.89      0.87      0.87      2600
weighted avg       0.89      0.87      0.87      2600
"""

print(confusion_matrix(test_image_gen.classes, predictions))
"""
[[1276   24]
 [ 299 1001]]
"""
#-------------------------------------------------------------
# Single image prediction
from tensorflow.keras.preprocessing import image

# image.load_img allows you to change the shape of an image, so lets
# make it our mean image shape
my_image = image.load_img(para_cell_full_path, target_size=image_shape)
# plt.imshow(my_image)
# plt.show()

my_image_array = image.img_to_array(my_image)
print(my_image_array.shape)
# (130, 130, 3)
# But the model expects batches of images, so we want the
# shape to be a batch of 1, i.e. (1, 130, 130, 3)
my_image_array = np.expand_dims(my_image_array, axis=0)
print(my_image_array.shape)
# (1, 130, 130, 3)

print(model.predict(my_image_array))
# [[0.]]
print(train_image_gen.class_indices)
# {'parasitized': 0, 'uninfected': 1}
# So the model predicts it to be a parasitized cell - CORRECT!

# END - off to CNN Exercise









