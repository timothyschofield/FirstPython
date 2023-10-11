"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

24 September 2023

Section 7: Basic Artificial Neural Networks - ANNs

TensorBoard is a webapp which is a visualization tool for neural networks

After creating your model by running model.fit(... in the code,
Start TensorBoard in the console by:

tensorboard --logdir logs\\fit
TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)

"""
print("========== 58. TensorBoard ==========")
# TensorBoard is from Google

# pip install pandas
# pip install numpy
# pip install seaborn
import pandas as pd
import numpy as np
import seaborn as sns

# pip install tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#path to root of FirstPython
root_path = "C:\\Tim\\VSCodeProjects\\FirstPython"

df = pd.read_csv(root_path + "\\TensorFlow2Bootcamp06_Sec7_ANN05_TensorBoard\\cancer_classification.csv")

X = df.drop("benign_0__mal_1", axis=1).values
y = df["benign_0__mal_1"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

# === Creating the TensorBoard callback ===
# When you create the TensorBoard callback,
# there are many things you can tell TensorBoard to track - lots of arguments

from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d--%H%M")

log_directory = "logs\\fit" + "\\" + timestamp

board = TensorBoard(log_dir=log_directory, histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq="epoch",
                    profile_batch=2,
                    embeddings_freq=1)

model = Sequential()
model.add(Dense(units=30, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(units=15, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit(x=X_train,
          y=y_train,
          epochs=600,
          validation_data=(X_test, y_test),
          verbose=1,
          callbacks=[early_stop, board])

# After creating your model by running model.fit(... in the code,
# Start TensorBoard in the console by:

# tensorboard --logdir logs\fit
# TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)

# Play with the tabs at the top of the screen SCALERS, IMAGES, GRAPHS etc
# The HISTOGRAMS and DISTIBUTIONS come in more useful with CNNs










