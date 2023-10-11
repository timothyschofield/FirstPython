"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

21 September 2023

Section 7: Basic Artificial Neural Networks - ANNs

"""
# pip install pandas
# pip install numpy
# pip install seaborn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler # "Scaler" = one who scales

# pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("========== 47. Keras Classification Code Along - EDA and Preprocessing ==========")
"""
- How to deal with Classification using TensorFlow
- Are tumors malignant or benign
- Fow to identify and deal with Overfitting useing early stopping callback techniques
    and adding in "dropout" layers

Dropout can be added to "turn off" neurons during training
to prevent overfitting.

Dropout Layers
Each Dropout layer will "drop" a user-defined percentage
of neuron units in the previous layer every batch.
Which means that certain neuron's weights and biases don't change
"""

"""
###################### So, some exploritory visualizations ######################
"""
#path to root of FirstPython
root_path = "C:\\Tim\\VSCodeProjects\\FirstPython"

# tumors malignant = 1 or benign = 0
df = pd.read_csv(root_path + "\\TensorFlow2Bootcamp06_Sec7_ANN04\\cancer_classification.csv")
print(df.describe())
"""
       mean radius  mean texture  ...  worst fractal dimension  benign_0__mal_1
count   569.000000    569.000000  ...               569.000000       569.000000
mean     14.127292     19.289649  ...                 0.083946         0.627417
std       3.524049      4.301036  ...                 0.018061         0.483918
min       6.981000      9.710000  ...                 0.055040         0.000000
25%      11.700000     16.170000  ...                 0.071460         0.000000
50%      13.370000     18.840000  ...                 0.080040         1.000000
75%      15.780000     21.800000  ...                 0.092080         1.000000
max      28.110000     39.280000  ...                 0.207500         1.000000
"""

# The "label" we are trying to predict is the column
# called benign_0__mal_1 == malignant (1) or benign (0)
# Let us see if we have a "well balenced problem"
#sns.countplot(x="benign_0__mal_1", data=df)
#plt.show()

# Checkout the correlation between the features themselfs
print(df.corr().head().to_string())
"""
                 mean radius  mean texture  mean perimeter  mean area  mean smoothness  mean compactness  mean concavity  mean concave points  mean symmetry  mean fractal dimension  radius error  texture error  perimeter error  area error  smoothness error  compactness error  concavity error  concave points error  symmetry error  fractal dimension error  worst radius  worst texture  worst perimeter  worst area  worst smoothness  worst compactness  worst concavity  worst concave points  worst symmetry  worst fractal dimension  benign_0__mal_1
mean radius         1.000000      0.323782        0.997855   0.987357         0.170581          0.506124        0.676764             0.822529       0.147741               -0.311631      0.679090      -0.097317         0.674172    0.735864         -0.222600           0.206000         0.194204              0.376169       -0.104321                -0.042641      0.969539       0.297008         0.965137    0.941082          0.119616           0.413463         0.526911              0.744214        0.163953                 0.007066        -0.730029
mean texture        0.323782      1.000000        0.329533   0.321086        -0.023389          0.236702        0.302418             0.293464       0.071401               -0.076437      0.275869       0.386358         0.281673    0.259845          0.006614           0.191975         0.143293              0.163851        0.009127                 0.054458      0.352573       0.912045         0.358040    0.343546          0.077503           0.277830         0.301025              0.295316        0.105008                 0.119205        -0.415185
mean perimeter      0.997855      0.329533        1.000000   0.986507         0.207278          0.556936        0.716136             0.850977       0.183027               -0.261477      0.691765      -0.086761         0.693135    0.744983         -0.202694           0.250744         0.228082              0.407217       -0.081629                -0.005523      0.969476       0.303038         0.970387    0.941550          0.150549           0.455774         0.563879              0.771241        0.189115                 0.051019        -0.742636
mean area           0.987357      0.321086        0.986507   1.000000         0.177028          0.498502        0.685983             0.823269       0.151293               -0.283110      0.732562      -0.066280         0.726628    0.800086         -0.166777           0.212583         0.207660              0.372320       -0.072497                -0.019887      0.962746       0.287489         0.959120    0.959213          0.123523           0.390410         0.512606              0.722017        0.143570                 0.003738        -0.708984
mean smoothness     0.170581     -0.023389        0.207278   0.177028         1.000000          0.659123        0.521984             0.553695       0.557775                0.584792      0.301467       0.068406         0.296092    0.246552          0.332375           0.318943         0.248396              0.380676        0.200774                 0.283607      0.213120       0.036072         0.238853    0.206718          0.805324           0.472468         0.434926              0.503053        0.394309                 0.499316        -0.358560
...
"""
# And we can just view this agianst the label we are trying to predict thus:
print(df.corr()["benign_0__mal_1"])
"""
mean radius               -0.730029
mean texture              -0.415185
mean perimeter            -0.742636
mean area                 -0.708984
mean smoothness           -0.358560
...
worst symmetry            -0.416294
worst fractal dimension   -0.323872
benign_0__mal_1            1.000000
Name: benign_0__mal_1, dtype: float64
"""

# And then sort_values, thus:
print(df.corr()["benign_0__mal_1"].sort_values())
"""
worst concave points      -0.793566 <<<< Negative correlations
worst perimeter           -0.782914
mean concave points       -0.776614
worst radius              -0.776454
mean perimeter            -0.742636
worst area                -0.733825
mean radius               -0.730029
mean area                 -0.708984
...
mean fractal dimension     0.012838
smoothness error           0.067016
benign_0__mal_1            1.000000
Name: benign_0__mal_1, dtype: float64
"""

# Sometimes it help to plot the above.
#df.corr()["benign_0__mal_1"].sort_values().plot(kind="bar")
#plt.show()

# Obviously benign_0__mal_1 correlates with itself, so lets drop that.
# So grab everything but the last column
#df.corr()["benign_0__mal_1"][:-1].sort_values().plot(kind="bar")
#plt.show()

# A not very useful heatmap
#plt.figure(figsize=(10,10)) # inches
#sns.heatmap(df.corr())
#plt.show()

"""
Trian test split and scaling the data
"""
# Drop our target label
X = df.drop("benign_0__mal_1", axis=1).values # Just values, so its a np array
y = df["benign_0__mal_1"].values # Just values, so its a np array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("=== 48. Keras Classification Dealing with Overfitting and Evaluation ===")

print(X_train.shape) # (398, 30)

model = Sequential() # model 1, demonstrating 600 epochs overfitting
model.add(Dense(30, activation="relu"))
model.add(Dense(15, activation="relu"))

# BINARY CLASSIFICATION therefore SIGMOID
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

# 600 epochs is way too much - just so we can show overfitting on a graph
# With 600 epochs, the validation loss scoots right up - so clearly over trained
# model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test))

# So we have availiable the training loss (error) and the validation loss
#losses = pd.DataFrame(model.history.history)
#losses.plot()
#plt.show()

# Let's uses early stop callbacks to stop the traing before it gets out of hand
# First off - we are told to redefine our model
# OK...

model = Sequential() # model 2, demonstrating EarlyStopping
model.add(Dense(30, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam")

from tensorflow.keras.callbacks import EarlyStopping
# We have to choose a metric for early stopping - in our case validation loss
# also other parameters
# min_delta - changes < min_delta count as no improvement
# patience - number of epochs with no improvent before stopping
# mode: min, max - are you trying to minimize the thing you are monitoring or maximize it?
# If our metric were acuracy, we would be wanting to maximise it
# Our metric is validation loss - which we are trying to minimize

#early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
#model.fit(x=X_train, y=y_train, epochs=600,
#          validation_data=(X_test, y_test),
#          callbacks=[early_stop])

# The training stops < 100 epochs
# Note: Its OK if the validation loss is flat,
# its when it starts to rise we want to stop training
#losses = pd.DataFrame(model.history.history)
#losses.plot()
#plt.show()

from tensorflow.keras.layers import Dropout

# Dropout layers - turn off a number of neurons randomly
model = Sequential() # model 3, demonstrating Dropout layers
model.add(Dense(30, activation="relu"))

# batch or epoch??? - not sure
# 0.5 = half the neurons in the previous layer,
# are randomly turned off during each batch of training.
# i.e. their weights and biases are not going to be updated in that batch
# So different neurons are turned on/off per batch of training
model.add(Dropout(0.5)) # Usualy a value between 0.2 and 0.5

model.add(Dense(15, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam")

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
model.fit(x=X_train, y=y_train, epochs=600,
          validation_data=(X_test, y_test),
          callbacks=[early_stop])

#losses = pd.DataFrame(model.history.history)
#losses.plot()
#plt.show()

# It stopped training after 125 epochs - a bit longer that with just early stopping.
# This is good, because it means that the model is still doing meaningful
# training onto those latter epochs - because of the dropout layers.

"""
Let us now do a full evaluation on the classification power of our trained model
"""

# In the good old days, we would just use "model.predict" but with keras:
# predictions = model.predict_classes(X_test) # Deprecated
# predictions = np.argmax(model.predict(X_test),axis=1) # don't work
predictions = model.predict(X_test)
predictions = np.round(predictions).astype(int)
#print(predictions)
"""
[[1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [0]
 ...
"""
from sklearn.metrics import classification_report, confusion_matrix
#print(classification_report(y_test,predictions)) # Very good
"""
              precision    recall  f1-score   support

           0       0.95      0.95      0.95        66
           1       0.97      0.97      0.97       105

    accuracy                           0.96       171
   macro avg       0.96      0.96      0.96       171
weighted avg       0.96      0.96      0.96       171
"""

#print(confusion_matrix(y_test,predictions))
"""
[[ 64   2]
 [  3 102]]
"""





















