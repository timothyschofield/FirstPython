"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

03 October 2023

Spent whole day looking for cell_images.zip

Section 8: Convolutional Neural Networks - CNNs Coding Project

"""
print("========== 74. CNN Exercise Overview ==========")
print("========== 75. CNN Exercise Solutions ==========")
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

# pip install scikit-learn
from sklearn.metrics import classification_report, confusion_matrix

import os

"""

"""
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)
# x_train (60000, 28, 28)
# y_train (60000,)
# x_test (10000, 28, 28)
# y_test (10000,)

# print(x_train[0])
# [[  0   0   0   0   0   0 ...
#plt.imshow(x_train[0])
#plt.show()

x_train = x_train/255
x_test = x_test/255

# Now reshape the x train and test data to reflect the one channel
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Now reshape the y train and test data to one-hot
print(y_train)
# [9 0 0 ... 3 0 5]

y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

# Now build the model
model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(4, 4),
                 input_shape=(28, 28, 1),
                 activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())  # 2D to 1D

model.add(Dense(128, activation="relu"))

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# print(model.summary())
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 25, 25, 32)        544       

 max_pooling2d (MaxPooling2  (None, 12, 12, 32)        0         
 D)                                                              

 flatten (Flatten)           (None, 4608)              0         

 dense (Dense)               (None, 128)               589952    

 dense_1 (Dense)             (None, 10)                1290      

=================================================================
Total params: 591786 (2.26 MB)
Trainable params: 591786 (2.26 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""

history = model.fit(x_train, y_cat_train, validation_data=[x_test, y_cat_test], epochs=3)

# Now, how did the model do?
# We can see, pretty well,
# after 3 epochs: loss: 0.2313 - accuracy: 0.9140 - val_loss: 0.2740 - val_accuracy: 0.9028

print(model.metrics_names)
# ["loss", "accuracy"]

metrics = pd.DataFrame(model.history.history)
print(metrics)
"""
       loss  accuracy  val_loss  val_accuracy
0  0.419479  0.850200  0.330585        0.8809
1  0.288706  0.895633  0.289766        0.8931
2  0.243420  0.910300  0.279636        0.9002
"""
# metrics[["loss", "val_loss"]].plot()
# plt.show()

# predict_classes is deprecated
predictions = model.predict(x_test)
predictions = np.round(predictions).astype(int)
# print("predictions", predictions)
"""
predictions [[0 0 0 ... 0 0 1]
 [0 0 1 ... 0 0 0]
 [0 1 0 ... 0 0 0]
"""
# print(classification_report(y_cat_test, predictions))
"""
              precision    recall  f1-score   support

           0       0.94      0.67      0.78      1000
           1       0.99      0.98      0.98      1000
           2       0.91      0.75      0.82      1000
           3       0.93      0.88      0.90      1000
           4       0.86      0.81      0.84      1000
           5       0.98      0.98      0.98      1000
           6       0.62      0.84      0.72      1000
           7       0.97      0.95      0.96      1000
           8       0.99      0.97      0.98      1000
           9       0.96      0.97      0.96      1000

   micro avg       0.90      0.88      0.89     10000
   macro avg       0.91      0.88      0.89     10000
weighted avg       0.91      0.88      0.89     10000
 samples avg       0.88      0.88      0.88     10000
"""

# confusion_matrix take labels like [7 2 1 ... 4 5 6], not one-hot
# print(confusion_matrix(y_cat_test, predictions))

# END and end of section - onto RNNs

















