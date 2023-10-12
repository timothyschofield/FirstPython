"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

29 September 2023

Section 8: Convolutional Neural Networks - CNNs Coding

"""
"""
Colour images 32 x 32

10 different objects

airplane 0
car 1
bird 2
cat 3
deer 4
dog 5
fog 6
horse 7
ship 8
truck 9

We will be reusing a lot of the code from the previous MNIST lesson
and concentrating on the add itions nessessary due to having 3 colour channels.
"""
print("========== 67. CNN on CIFAR-10 - Part One - The Data ==========")
# pip install pandas
# pip install numpy
import pandas as pd
import numpy as np

# pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

# pip install tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape) # (50000, 32, 32, 3)
print(y_train.shape) # (50000, 1)
print(x_test.shape) # (10000, 32, 32, 3)
print(y_test.shape) # (10000, 1)

print(x_train[0]) # (32 32 3)
"""
one line of colour data
[[[ 59  62  63]
  [ 43  46  45]
  [ 50  48  43]
  ...
  [158 132 108]
  [152 125 102]
  [148 124 103]]
  ...
  another 31 of these
"""
#plt.imshow(x_train[0]) # A frog
#plt.show()

# Need to convert colour cannels from 0->255 to 0.0->1.0
x_train = x_train/255
x_test = x_test/255

print(y_train)
"""
[[6]
 [9]
 [9]
 ...
 [9]
 [1]
 [1]]
"""
y_cat_train = to_categorical(y_train, num_classes=10)
y_cat_test = to_categorical(y_test, num_classes=10)
print(y_cat_train)
"""
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]
 ...
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# The images for MNIST were 28*28*1 = 784
# Here we have 32*32*3 = 3072
# So we will probably need more convolutional and pooling layers
model = Sequential()

model.add(Conv2D(
    filters=32,
    kernel_size=(4,4),
    strides=(1,1),
    padding="valid",
    input_shape=(32, 32, 3),
    activation="relu"))

# pool_size=(2,2) half our kernal_size
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(
    filters=32,
    kernel_size=(4,4),
    strides=(1,1),
    padding="valid",
    input_shape=(32, 32, 3),
    activation="relu"))

model.add(MaxPool2D(pool_size=(2,2)))

# We need to flatten the 32*32*3 images to 3072*1
# We are going from the 2D world of images to the 1D world of categories
model.add(Flatten())

# Increase neurons to 256 form 128 in MNIST
model.add(Dense(256, activation="relu"))

model.add(Dense(10,  activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=2)

history = model.fit(x_train, y_cat_train,
          epochs=15,
          validation_data=(x_test, y_cat_test),
          callbacks=[early_stop])
# Terminates after 7 epochs

print("========== 68. CNN on CIFAR-10 - Part Two - Evaluating the Model ==========")

metrics = pd.DataFrame(model.history.history)
print(metrics.columns)
# ["loss","accuracy","val_loss","val_accuracy"]

# metrics[["accuracy","val_accuracy"]].plot()
# plt.show()

# metrics[["loss","val_loss"]].plot()
# plt.show()

# pip install scikit-learn
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix

predictions = model.predict(x_test)             # floating point one_hot - needs rounding
one_hot = np.round(predictions).astype(int)     # one_hot
predictions = np.argmax(one_hot, axis=1)        # [7 2 1 ... 4 5 6] - back to labels

print("--------------classification_report------------------")
# print(classification_report(y_test, predictions))
"""
              precision    recall  f1-score   support

           0       0.34      0.83      0.49      1000
           1       0.85      0.77      0.81      1000   <<< 81% cars, good! - very distinct
           2       0.70      0.47      0.56      1000
           3       0.59      0.38      0.46      1000   <<< 46% cats problem - can look like dogs
           4       0.66      0.64      0.65      1000
           5       0.64      0.53      0.58      1000
           6       0.73      0.78      0.76      1000
           7       0.84      0.66      0.74      1000
           8       0.82      0.71      0.77      1000
           9       0.83      0.73      0.78      1000   <<< 78% truck pretty good - quite distinct

    accuracy                           0.65     10000   <<< 65% accuracy overall, remember, random = 10%, OKish
   macro avg       0.70      0.65      0.66     10000
weighted avg       0.70      0.65      0.66     10000
"""
print("-------------confusion_matrix-----------------")
# print(confusion_matrix(y_test, predictions))
"""
[[828  14  29   9  25   9  14   8  52  12]
 [104 769   2   3   1   5  18   4  36  58]
 [286   4 465  29  72  43  69  16  11   5]
 [254  12  31 375  59 145  72  22  12  18]
 [163   3  36  40 641  27  45  33   8   4]
 [208   4  31 115  43 530  36  25   2   6]
 [ 97   1  33  24  40  12 780   7   5   1]
 [150   2  20  24  81  40  11 655   4  13]
 [186  26   7   7   7   7   7   3 714  36]
 [130  74   7   7   4   6  11   8  22 731]]
"""
my_number = x_test[0]
                                        # num_images, x, y, channels
predictions = model.predict(my_number.reshape(1, 32, 32, 3))
one_hot = np.round(predictions).astype(int)
predictions = np.argmax(one_hot, axis=1)
print("my_number is predicted to be a", predictions) # [3] cat
print("correct is", y_test[0]) # [3] cat

# END



