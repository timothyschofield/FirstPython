"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

18 September 2023

Section 7: Basic Artificial Neural Networks - ANNs

Remember to have Num Lock on
Alt + 228 = Σ
Alt + 0178 = ²
Alt + 251 = √
remember |x| = abs(x)
remember ||x|| = mean(x)

δl = w(l+1)ᵀ * δ(l+1) ⊙ σ(Zl)

"""
print("========= 39. TensorFlow vs. Keras Explained =========")
"""
TensorFlow
Is an open-source deep learning library developed by Google, 
with TF 2.0 being released in late 2019. TensorFlow has a large
ecosystem of related components, including libraries like Tensorboard,
Deployment and Production APIs, and support for various programming languages.

Keras
Is a high-level Python library that can use a variety of deep learning
libraries underneath such as: TensorFlow, 
CNTK (The Microsoft Cognitive Toolkit), or Theano.

On release of TF 2.0 Keras was adopted as the official API for TF.
Keras can now be imported from TF - there is no need to do a 
seperate installation. Keras is part of TF and is its offical API.
"""
print("=== 40. Keras Syntax Basics - Part One - Preparing the Data ===")
"""

"""
# pip install pandas
# pip install numpy
# pip install seaborn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#path to root of FirstPython
root_path = "C:\\Tim\\VSCodeProjects\\FirstPython"

df = pd.read_csv(root_path +"\\TensorFlow2Bootcamp06_Sec7_ANN02\\fake_reg.csv")

print(df.head())
"""
        price     feature1     feature2
0  461.527929   999.787558   999.766096
1  548.130011   998.861615  1001.042403
2  410.297162  1000.070267   998.844015
3  540.382220   999.952251  1000.440940
4  546.024553  1000.446011  1000.338531
"""
#sns.pairplot(df)
#plt.show()

# pip install scikit-learn
from sklearn.model_selection import train_test_split

# Uppercase X because it is a tensor
X = df[["feature1", "feature2"]].values # The "values" because we require a np array

# Lowercase y because it is a vector
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)

# We will scale our feature set to something from 0 to 1
from sklearn.preprocessing import MinMaxScaler # "Scaler" = one who scales
# help(MinMaxScaler)
# We need to scale the features, but we don't need to scale
# the label/output/y because it is not being passed through the network

# Make an instance of MinMaxScaler()
scaler = MinMaxScaler()

# fit calculates the standard deviation, min and max of the data
# so it can do scaling (the transform) later on
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print("=== 41. Keras Syntax Basics - Part Two - Creating and Training the Model ===")

# This is how keras is packaged inside tensorflow
# pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1)) # a single price

"""
ASSIDE - some tips on chooseing optimizers and loss functions
What kind of problem are you trying to solve:

- For a multi-class classification problem
    model.compile(  optimizer="rmsprop",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])

- For a binary classification problem
    model.compile(  optimizer="rmsprop",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])

- For a mean squared error regression problem
    model.compile(  optimizer="rmsprop",
                    loss="mse")
"""
# We want price
model.compile(optimizer="rmsprop", loss="mse")
model.fit(x=X_train, y=y_train, epochs=250, verbose=1)

# loss_df = pd.DataFrame(model.history.history)
# loss_df.plot()
# plt.show()
print("=== 42. Keras Syntax Basics - Part Three - Model Evaluation ===")
# A bunch of different methods of evaluating the model
this_error = model.evaluate(X_test, y_test, verbose=0)

# We has decided on Mean Square Error (MSE) when we compiled the model
print("this_error=",this_error) # 25.885665893554688 - What does it mean?

# Let's see what the model actualy predicts (the y_test) given the X_test
test_predictions = model.predict(X_test)
print(test_predictions)
"""
[[405.69412]
 [623.6921 ]
 [592.26685]
 [572.5605 ]
 [367.30008]
 [579.418  ]
 ...
"""
# Lets comepare them to the actualy, required values, in y_test
# First turn them into a pandas series
test_predictions = pd.Series(test_predictions.reshape(300,))
# print(test_predictions)
"""
0      405.694122
1      623.692078
2      592.266846
3      572.560486
4      367.300079
...
"""
# Now we can make a combined DataFrame of the test_predictions and
# required/True values, y_test - so we can compare them

pred_df = pd.DataFrame(y_test)

pred_df = pd.concat([pred_df, test_predictions],axis=1)
pred_df.columns = ["y_test required/True", "test_predictions"]
"""
     y_test required/True  test_predictions
0              402.296319        405.274719
1              624.156198        623.682739
2              582.455066        592.247803
3              578.588606        572.342224
...
"""
# Let's create a scatter plot of these two columns
# sns.scatterplot(x="y_test required/True", y="test_predictions",data= pred_df)
# plt.show()

# Now for some other error metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

mae = mean_absolute_error(pred_df["y_test required/True"], pred_df["test_predictions"])
print("mae=", mae) # around about $4 out, not bad on prices around $500

print(df.describe())

mse = mean_squared_error(pred_df["y_test required/True"], pred_df["test_predictions"])
print("mse=", mse)
rmse = mse**0.5 # square root
print("rmse=", rmse)

# ===== Predicting on Brand New Data =====

new_gem = [[998, 1000]] # What should I price this at?
new_gem = scaler.transform(new_gem)
print("new_gem price", model.predict(new_gem)) # $421

# How to save a model and load a model?
from tensorflow.keras.models import load_model

# HDF5/H5 format is legacy
# keras format is where its at
model.save("my_gem_model.keras")

later_model = load_model("my_gem_model.keras")
print("later_model new_gem price", later_model.predict(new_gem)) # $421 again
#===============================================================================
























































































