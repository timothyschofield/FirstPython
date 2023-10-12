"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

26 September 2023

Section 8: Convolutional Neural Networks - CNNs Coding

"""
print("========== 63. MNIST Dataset Overview ==========")
"""
- MNIST is thee classic data set using in deep learning 0 - 9
- It's the "Hello World!" of deep learning
- Let's cover some basic facts about it - how we organize these datasets into 4D arrays
- 60,000 training images, 10,000 test images
- We can think of the entire group of 60,000 images as a 4D array
- 60,000 of 1 channel 28 x 28 pixels
        (60000, 28, 28, 1)
        (Samples, x, y, channels)
- For the labels, we will use One-Hot encoding
- This means that instead of having labels "One", "Two", etc. we'll have 
     a single array for each image.
- The origonal labels of the images are given as a list of numbers, [5,0,4,...6,3,8]
- We will convert them to one-hot encoding
- So 
0 becomes [1,0,0,0,0,0,0,0,0,0]
1 becomes [0,1,0,0,0,0,0,0,0,0]
2 becomes [0,0,1,0,0,0,0,0,0,0]
...
8 becomes [0,0,0,0,0,0,0,0,1,0]
9 becomes [0,0,0,0,0,0,0,0,0,1]

The reason we do this is that it works really well, with a single 
layer output with 10 neurons and each of them is triggered to fire off a sigmaoid to 0 or 1
As a result the labels for the training array end up being a 60,000 x 10 bit array
"""
print("========== 64. CNN on MNIST - Part One - The Data ==========")
# pip install pandas
# pip install numpy
import pandas as pd
import numpy as np

# pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

# pip install tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train)
print(x_train.shape)
# (60000, 28, 28) # and 0->255 greyscale
"""
[[[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
...
"""
single_image = x_train[0]
print(single_image.shape)
# (28, 28)
#plt.imshow(single_image)
#plt.show()

print(y_train)
# [5 0 4 ... 5 6 8]
print(y_train.shape)
# (60000,)
"""
If we were to pass in to the network [5 0 4 ... 5 6 8] as our training data
the network would assume it was continuous and try to predict 5.5, 0.3 etc.
Really the 10 values 0 to 9 we have are categories - this is a classification problem.
So we need to one-hot encode it
"""

y_example = to_categorical(y_train, num_classes=10)
print(y_example.shape) #(60000, 10)
print(y_example)
"""
starts out
[4, 0, 2, etc.

after to_categorical
[[0. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 ...
"""

y_cat_train = to_categorical(y_train, num_classes=10)
y_cat_test = to_categorical(y_test, num_classes=10)

# Now we need to normalize the training data
# each pixel pixel 0->255
# We could use MinMaxScaler at this point

# He gives an explanation of why we don't need to use it here
# which I didn't quite understand...anyway

x_train = x_train/255
x_test = x_test/255
# each pixel value now 0.0->1.0

#  Now we need to shape the data
print(x_train.shape)
# (60000, 28, 28)
# This is correct of a CNN, but we need to add one more dimension
# to let the network know we are dealing with a single channel
                        # batch_size, width, height, num_channels
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print("========== 65. CNN on MNIST - Part Two - Creating and Training the Model ==========")
"""
- Here we talk about the aspects of a model you can edit vs. aspects that, 
    based on the constraints of the model, should always be fixed more or less.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
"""
- Typical for a CNN, the first layer that it encounters is a convolutional layer
    The more complex the dataset you are trying to classify, the more filters you shold have
    The number of filters is commonly chosen as a power of 2: filters=32 for a simple problem
- kernel_size=(4,4)
- strides=(1,1), in this case, the images are only 28 x 28, so leave stride small
- padding, only two possible values, "valid" or "same"
    "valid" - no padding, we assume that the input image, 
    gets fully covered by the filter and stride specified.
    "same" - The system will automaticaly figure out what padding to apply 
    so the image gets fully covered by the filter and stride specified. 
    For stride 1, this will ensure the output image size is the same as the input image.
"""

model.add(Conv2D(
    filters=32,
    kernel_size=(4,4),
    strides=(1,1),
    padding="valid",
    input_shape=(28,28,1),
    activation="relu"))

# pool_size=(2,2) half our kernal_size
model.add(MaxPool2D(pool_size=(2,2)))

# What we could do, is keep on adding
# convolutional layers and pooling layers - but lets keep it simple

# We need to flatten the 28 x 28 images to 728 * 1
# We are going from the 2D world of images to the 1D world of categories
model.add(Flatten())

# It's a good idea to have one final Dense layer before output
# 128, is a reduction, but is of the same order as 728
model.add(Dense(128, activation="relu"))

# Output layer is softmax because it a multi-class problem
model.add(Dense(10,  activation="softmax"))

# https://keras.io/api/metrics/
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

"""
So, we come back to the question:
    What aspects of a model can we edit and what aspects of the model must stay the same?

These are hyperparameters

What is fixed, and is determined by your input shape?

FIXED: There is a CORRECT value for these - and this is it!
input_shape=(28, 28, 1)
model.add(Flatten())     <<<<< you are going to HAVE to do this at some point
model.add(Dense(10, activation="softmax")
    
CAN PLAY AROUND WITH:
1) Add as many convolutional and pooling layers as you want
2) Within the convolutional layers, you can play with the number of filters, kernel_size,
    pool_size and padding
3) After you have Flattened it out, you can play around with 
    the number of Dense layers (but one is typical) as well as 
    the number of neurons within those layers
"""

# Now lets train the model
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", patience=1)

history = model.fit(x_train, y_cat_train,
          epochs=10,
          validation_data=(x_test, y_cat_test),
          callbacks=[early_stop])

# Prints out the structure of the model
# print(model.summary())
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 25, 25, 32)        544 
 ...
=================================================================
Total params: 591786 (2.26 MB)
Trainable params: 591786 (2.26 MB)
Non-trainable params: 0 (0.00 Byte)
"""
# This is a bit slow
# So I want to save a previously trained model and use that to speed things up
# but history is not saved with the model - so problem.
# Might save history as JSON seperatly, but too far off piece
# from tensorflow.keras.models import load_model
# model = load_model("CNN_sec8_MNIST_model.keras")
# model.save("CNN_sec8_MNIST_model.keras")

# from pprint import pprint
# pprint(vars(history))
"""
Returned from model.fit()
i.e. history
{'_chief_worker_only': None,
 '_supports_tf_logs': False,
 'epoch': [0, 1, 2],
 'history': {'accuracy': [0.958383321762085,
                          0.9860000014305115,
                          0.9904000163078308],
             'loss': [0.13789065182209015,
                      0.04603048413991928,
                      0.02995491400361061],
             'val_accuracy': [0.9835000038146973,
                              0.9858999848365784,
                              0.9835000038146973],
             'val_loss': [0.05151214078068733,
                          0.04219073802232742,
                          0.05437849089503288]},
 'model': <keras.src.engine.sequential.Sequential object at 0x0000017AC6E7C340>,
 'params': {'epochs': 10, 'steps': 1875, 'verbose': 1},
 'validation_data': None}
"""

print("========== 66. CNN on MNIST - Part Three - Model Evaluation ==========")
# 29 September 2023
# metrics, a more general term than "loss", this covers loss and accuracy etc.
metrics = pd.DataFrame(model.history.history)

# Note: That because we set  metrics=["accuracy"] in model.compile()
# we not only get loss and val_loss in the history, we get
# accuracy and val_accuracy also recorded (see history above)

# print(metrics)
"""
       loss  accuracy  val_loss  val_accuracy
0  0.137540  0.958667  0.055955        0.9835
1  0.048311  0.985200  0.052293        0.9823
2  0.031289  0.990317  0.042092        0.9865
3  0.021857  0.993117  0.037348        0.9876
4  0.014780  0.995433  0.038335        0.9884
"""
# It is meaningless to compair loss to accuracy
# But loss and val_loss
# and accuracy and val_accuracy make sense
# metrics[["loss", "val_loss"]].plot()
# plt.show()

# print(model.metrics_names)
# ["loss", "accuracy"]

# pip install scikit-learn
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix

"""
Remember: 
y_cat_test looks like
[[0. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 ...
 
y_test looks like
[4, 0, 2, etc.
"""
# predictions = model.predict_classes(x_test) # depricated
# Surely there is an easier way
predictions = model.predict(x_test)             # floating point one_hot - needs rounding
one_hot = np.round(predictions).astype(int)     # one_hot
predictions = np.argmax(one_hot, axis=1)        # [7 2 1 ... 4 5 6] - back labels

print("--------------classification_report------------------")
print(classification_report(y_test, predictions))
"""
              precision    recall  f1-score   support

           0       0.96      0.98      0.97       980
           1       0.99      1.00      0.99      1135
           2       0.98      0.98      0.98      1032
           3       0.97      0.99      0.98      1010
           4       0.99      0.99      0.99       982
           ...
"""
print("-------------confusion_matrix-----------------")
print(confusion_matrix(y_test, predictions))
"""
[[ 974    0    1    1    0    1    1    1    1    0]
 [   0 1133    0    1    0    0    1    0    0    0]
 [   4    1 1014    3    0    0    2    5    3    0]
 [   2    0    1 1003    0    3    0    0    1    0]
 [   2    0    0    0  962    0    2    0    0   16]
 [   0    0    0    8    0  882    1    0    0    1]
 [   4    2    0    0    2    4  944    0    2    0]
 [   3    5    5    0    0    0    0 1010    1    4]
 [   8    0    1    4    0    4    0    0  952    5]
 [   3    3    0    0    3    4    0    1    1  994]]
"""
# Great - how to predict the category of a single number image
my_number = x_test[0] # shape (28, 28, 1)

                                        # num_images, x, y, channels
predictions = model.predict(my_number.reshape(1, 28, 28, 1))
one_hot = np.round(predictions).astype(int)
predictions = np.argmax(one_hot, axis=1)
print("my_number is a", predictions) # 7 - Correct!

# END















































