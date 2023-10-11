"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

21 September 2023

Section 7: Basic Artificial Neural Networks - ANNs

This is the Project at the end of Section 7

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
from sklearn.metrics import classification_report, confusion_matrix

# pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

print("========== 49. TensorFlow 2.0 Keras Project Options ==========")
print("========== 50. TensorFlow 2.0 Keras Project Description ==========")
"""
22 September 2023

We are going to be using a subset of the LendingClub DataSet obtained from Kaggle:
https://www.kaggle.com/wordsforthewise/lending-club
BUT - don't download the origonal file
We use some specialy addapted ones in the project.
The Lending Club lend out money
Sometimes people do not pay it back and it has to be written off - this is called a "charge-off"

We are going to try to predict, based on historical data, 
whether or not a potential client is going to default.

The "loan_status" column contains our label/target.

"""
print("========== 51. Keras Project Solutions - Explorotory Data Analysis ==========")

#path to root of FirstPython
root_path = "C:\\Tim\\VSCodeProjects\\FirstPython"

# The second argument defines the column index_col to be the index column - why?
# This is simply a description of what each feature means
data_info = pd.read_csv(root_path + "\\TensorFlow2Bootcamp06_Sec7_ANN05_Project\\lending_club_info.csv", index_col="LoanStatNew")

def feat_info(col_name):
    print(data_info.loc[col_name]["Description"])


df = pd.read_csv(root_path + "\\TensorFlow2Bootcamp06_Sec7_ANN05_Project\\lending_club_loan_two.csv")
#print(df.info())
"""
RangeIndex: 396030 entries, 0 to 396029
Data columns (total 27 columns):
 #   Column                Non-Null Count   Dtype  
---  ------                --------------   -----  
 0   loan_amnt             396030 non-null  float64
 1   term                  396030 non-null  object 
 2   int_rate              396030 non-null  float64
 3   installment           396030 non-null  float64
 4   grade                 396030 non-null  object 
 5   sub_grade             396030 non-null  object 
 ...
"""

# First explore the balencing of the labels
# sns.countplot(x="loan_status", data=df)
# plt.show()

# By far the majority of people pay - it is an inbalenced problem
# So we will do well in Accuracy but
# precision and recall are going to be the true metrics we will have to evaluate our model from
# and we should not expect to do so well on these
"""
===== Accuracy ===== 
Very common and the eysiest to understand
In classification problems
accuracy = numcorrect predictions/total number of predictions
0.0 -> 1.0
- Accuracy is good when there are roughly 
    the same amount of cat images as we do dog images
- But image an "unbalenced class" situation - say 99 dog images
    and only 1 cat image. If our model was a line that always predicted dog,
    it would get 99% acuaracy

- If you happen to have an unbalenced class, 
    that is where the other metrics come into play...

===== Recall ===== 
- The ability of a model to find all the relevant
    cases within a dataset

- Think of confusion matrix here
recall = num of true positives/(num of true positives + num of false negatives)
===== Precision ===== 
- The ability of a clssification model to identify
    only the relevant data points.
    
- Think of confusion matrix here
precision = num of true positives/(num of true positives + num of false positives)
"""
# Lets look at the amount loaned
# distplot is deprecated in v0.11.0 - Guide to updating your code
# https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
#sns.distplot(df["loan_amnt"], kde=False, bins=40)
#plt.show()
# Notice the spikes at even doller loans - 10K, 20K etc.

# Lets look at the correlation between features that are continuous values
print(df.corr(numeric_only=True))
"""
                      loan_amnt  int_rate  ...  mort_acc  pub_rec_bankruptcies
loan_amnt              1.000000  0.168921  ...  0.222315             -0.106539
int_rate               0.168921  1.000000  ... -0.082583              0.057450
installment            0.953929  0.162758  ...  0.193694             -0.098628
annual_inc             0.336887 -0.056771  ...  0.236320             -0.050162
dti                    0.016636  0.079038  ... -0.025439             -0.014558
open_acc               0.198556  0.011649  ...  0.109205             -0.027732
pub_rec               -0.077779  0.060986  ...  0.011552              0.699408
revol_bal              0.328320 -0.011280  ...  0.194925             -0.124532
revol_util             0.099911  0.293659  ...  0.007514             -0.086751
total_acc              0.223886 -0.036404  ...  0.381072              0.042035
mort_acc               0.222315 -0.082583  ...  1.000000              0.027239
pub_rec_bankruptcies  -0.106539  0.057450  ...  0.027239              1.000000
"""

#plt.figure(figsize=(12,7))
#sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="viridis")
#plt.show()

# We always want to check that we are not accidentaly "leaking" information
# from our features into our label. We want to make sure that there is not
# a single feature that is the perfect predictor of the label because that
# basicaly indicates that it is not realy a feature, its just some duplicate information
# that is very similar to the label.
# With this in mind,
# we can see the feature "installment" and "loan_amnt" have a 0.95 correlation
print("\n")
feat_info("installment")
feat_info("loan_amnt")
print("\n")
# If you load someone $1,000,000, then following some formula, your
# monthly payments will be quite hight, and the same formula will be used
# for loans of $10,000
# So there is some direct, formulaic, relation between installment and loan_amnt

#sns.scatterplot(x="installment",  y="loan_amnt", data=df)
#plt.show()

# Q: So is there a relationship between really expensive loans and
# not being able to pay them off?
# A: Not really
#sns.boxplot(x="loan_status", y="loan_amnt", data=df)
#plt.show()

# Remember: "describe()" gives you some useful statistics like, mean, min, max etc.
# With no second argument, this just lists mean, std, min, etc. for All features
# print(df.groupby("loan_status").describe().to_string())

# With a second argument, "loan_amnt", you just get that one feature described
print(df.groupby("loan_status")["loan_amnt"].describe().to_string())
"""
                count          mean          std     min     25%      50%      75%      max
loan_status                                                                                
Charged Off   77673.0  15126.300967  8505.090557  1000.0  8525.0  14000.0  20000.0  40000.0
Fully Paid   318357.0  13866.878771  8302.319699   500.0  7500.0  12000.0  19225.0  40000.0
"""

# Explore the grade and sub-grade columns (presumably, something to do with reliability)
print(df["grade"].unique())
print(df["sub_grade"].unique())
"""
['B' 'A' 'C' 'E' 'D' 'F' 'G']
['B4' 'B5' 'B3' 'A2' 'C5' 'C3' 'A1' 'B2' 'C1' 'A5' 'E4' 'A4' 'A3' 'D1'
 'C2' 'B1' 'D3' 'D5' 'D2' 'E1' 'E2' 'E5' 'F4' 'E3' 'D4' 'G1' 'F5' 'G2'
 'C4' 'F1' 'F3' 'G5' 'G4' 'F2' 'G3']
"""

# Lets look at grade vs. Paid/Charged-off
#grade_order = sorted(df["grade"].unique()) # However the x-axis is out of order, so sort
#sns.countplot(x="grade", data=df, hue="loan_status", order=grade_order)
#plt.show()

# Lets just do a countplot of subgraded
#plt.figure(figsize=(12, 4))
#subgrade_order = sorted(df["sub_grade"].unique())
#sns.countplot(x="sub_grade", data=df, order=subgrade_order, hue="loan_status")
#plt.show()

# From the above, it looks like F and G don't get paid off very often
# So let's isolate them to have a closer look
#f_and_g = df[(df["grade"] == "G") | (df["grade"] == "F")]
#plt.figure(figsize=(12, 4))
#subgrade_order = sorted(f_and_g["sub_grade"].unique())
#sns.countplot(x="sub_grade", data=f_and_g, order=subgrade_order, hue="loan_status")
#plt.show()

# Task: create a new column called loan_repaid,
# which is 1 loan_status = Fully Paid and 0 if loan_status = Charged Off
# We will be using loan_repaid
df["loan_repaid"] = df["loan_status"].map({"Fully Paid":1, "Charged Off":0})
print(df[["loan_repaid", "loan_status"]])
"""
        loan_repaid  loan_status
0                 1   Fully Paid
1                 1   Fully Paid
2                 1   Fully Paid
3                 1   Fully Paid
4                 0  Charged Off
"""

# Task: Which neumeric features have the highest correlation with the actual label
corr_loan_repaid_sorted = df.corr(numeric_only=True)["loan_repaid"].sort_values().drop("loan_repaid")
print(corr_loan_repaid_sorted)

#corr_loan_repaid_sorted.plot(kind="bar")
#plt.show()

print("========== 52. Keras Project Solutions - Dealing with Missing Data - Part 1 ==========")
"""
This is very nuanced - oh my!
1. Keep the missing data
2. Drop the missing data
3. Fill in the missing data
"""
print(df.head().to_string())

print("length of dataframe", len(df))

# Total count of missing values a column
print(df.isnull().sum())
"""
loan_amnt                   0
...
sub_grade                   0
emp_title               22927 <<< Employment title
emp_length              18301 <<< Employment length
home_ownership              0
...
purpose                     0
title                    1756
dti                         0
...
revol_bal                   0
revol_util                276
total_acc                   0
initial_list_status         0
application_type            0
mort_acc                37795 <<< mortgage account
pub_rec_bankruptcies      535
address                     0
loan_repaid                 0
"""

# Get the above in terms % of total df
print(100 * df.isnull().sum()/len(df))
"""
loan_amnt               0.000000
...
emp_title               5.789208 <<< Employment title
emp_length              4.621115 <<< Employment length
home_ownership          0.000000
...
purpose                 0.000000
title                   0.443401
dti                     0.000000
...
revol_bal               0.000000
revol_util              0.069692
total_acc               0.000000
initial_list_status     0.000000
application_type        0.000000
mort_acc                9.543469 <<< Mortgage account
pub_rec_bankruptcies    0.135091
address                 0.000000
loan_repaid             0.000000
dtype: float64
"""
# Lets examine Employment title and Employment length
# emp_title emp_length - to see if we can drop them

# What about emp_title? How many unique employment titles are there?
print(df["emp_title"].nunique())
# 173105 - half the size of the df!!!

print(df["emp_title"].value_counts())
"""
emp_title
Teacher                    4389
Manager                    4250
Registered Nurse           1856
RN                         1846
Supervisor                 1830
                           ... 
Postman                       1
McCarthy & Holthus, LLC       1
jp flooring                   1
Histology Technologist        1
Gracon Services, Inc          1
"""
# This is way too many to create a dummy variable feature from them.
# Remember dummy variables are where you create bit fields to give the strings neumeric value.
# We can't use this - So let's drop the emp_title column
# Also remember you can only run this once
df = df.drop("emp_title", axis=1)

# Task: lets create a countplot of emp_length feature - also sort them
print("\n--------------")

# Well, we were supposed to sort them - he cludged it by manualy editing the list at the end
emp_length_order = sorted(df["emp_length"].dropna().unique())
print(emp_length_order)

# Just checking types
print("df type", type(df))                              # type DataFrame
print("df[emp_length] type", type(df["emp_length"]))    # type Series
print("emp_length_order type", type(emp_length_order))  # type list

#plt.figure(figsize=(12,4))
#sns.countplot(x="emp_length", data=df, order=emp_length_order, hue="loan_status")
#plt.show()

# But, what we really want, is the ration of Fully Paid/Charged Off per emp_length category
emp_charged_off = df[df["loan_status"]=="Charged Off"].groupby("emp_length").count()["loan_status"]
emp_fully_paid = df[df["loan_status"]=="Fully Paid"].groupby("emp_length").count()["loan_status"]

print("\ntype=",type(emp_charged_off)) # type Series

# Dividing one Series by another
percent_paid_off = emp_charged_off/(emp_fully_paid + emp_charged_off)
print(percent_paid_off) # Percentage
print("One series divided by another", type(percent_paid_off) ) # Series

"""
emp_length
1 year       0.199135
10+ years    0.184186
2 years      0.193262
3 years      0.195231
4 years      0.19238
...
"""
#percent_paid_off.plot(kind="bar")
#plt.show()
# They all look pretty similar,
# i.e. length of employment does not have much effect on Pay Off rate
# So lets drop that column
df = df.drop("emp_length", axis=1)

print("\n========== 53. Keras Project Solutions - Dealing with Missing Data - Part 2 ==========")
# What features do we still have missing data?
# print(df.isnull().sum())
"""
loan_amnt                   0
...
purpose                     0
title                    1756 <<<< still got
dti                         0
...
revol_bal                   0
revol_util                276 <<<< still got
total_acc                   0
...
application_type            0
mort_acc                37795 <<<< still got - this is 10% of all rows
pub_rec_bankruptcies      535 <<<< still got
address                     0
"""
# Task: Compare the title column to the purpose column
#print(df["purpose"].head())
"""
0              vacation
1    debt_consolidation
2           credit_card
3           credit_card
4           credit_card
...
Name: purpose, dtype: object
"""
# print(df["title"].head())
"""
0                   Vacation
1         Debt consolidation
2    Credit card refinancing
3    Credit card refinancing
4      Credit Card Refinance
...
Name: title, dtype: object
"""
# They look the same, so drop the title column
df = df.drop("title", axis=1)

# Note: This is the hardest part of the project!
# Its filling in missing data in mort_acc, based on the values of another column

# Task: Find what mort_acc means
# mort_acc means - Number of mortgage accounts - if they have remorgaged or not?
#print(df["mort_acc"].value_counts())
"""
mort_acc
0.0     139777  <<<< No mortgage account
1.0      60416
2.0      49948
3.0      38049
4.0      27887
5.0      18194
...
26.0         2
28.0         1
30.0         1
34.0         1 <<<< Pretty extreem
"""
# We have mort_acc   37795 <<<< still got - this is 10% of all rows
# Can we fill it in?
# So... we can look for other columns,
# which have full information and correlate well with mort_acc to fill in mort_acc
print("\n")
print(df.corr(numeric_only=True)["mort_acc"].sort_values())
"""
int_rate               -0.082583
dti                    -0.025439
revol_util              0.007514
pub_rec                 0.011552
pub_rec_bankruptcies    0.027239
loan_repaid             0.073111
open_acc                0.109205
installment             0.193694
revol_bal               0.194925
loan_amnt               0.222315
annual_inc              0.236320
total_acc               0.381072
mort_acc                1.000000
"""
# So total_acc has the strongest correlation - better than nothing anyway
# total_acc: The total number of credit lines currently in the borrower's credit file
# The idea is to group the DataFrame by total_acc and then calculate the
# mean value of mort_acc per total_acc

total_acc_avg = df.groupby("total_acc")["mort_acc"].mean()
#print(total_acc_avg)

# Note:
# My way is df.groupby("total_acc")["mort_acc"].mean()
# His way is df.groupby("total_acc").mean()["mort_acc"] - and you get the same result

"""
total_acc
2.0      0.000000
3.0      0.052023
4.0      0.066743
5.0      0.103289
6.0      0.151293
           ...   
124.0    1.000000
129.0    1.000000
135.0    3.000000
150.0    2.000000
151.0    0.000000
"""
# The second column (remember) is the average number of mort_accs
# in each total_acc category.
# So we are going to go through the df and, where there is
# no mort_acc value, look-up its total_acc category, and fill in
# average number of mort_accs for that total_acc category
# ...I tryed this - and I think my heart was in the right place

def fill_mort_acc(total_acc, mort_acc):

    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df["mort_acc"] = df.apply(lambda x: fill_mort_acc(x["total_acc"], x["mort_acc"]), axis=1)
# print(df.isnull().sum())
"""
loan_amnt                 0
term                      0
...
pub_rec                   0
revol_bal                 0
revol_util              276
total_acc                 0
initial_list_status       0
application_type          0
mort_acc                  0 <<<< mort_acc now has no missing data
pub_rec_bankruptcies    535
address                   0
loan_repaid               0
"""
# There are two other rows with missing data:
# revol_util 276 missing values, and pub_rec_bankruptcies 535 missing values
# This is a small amounts of data when compared to the whole size of the data frame,
# so lets just drop these missing values (not the columns/featurs themselfs)

# dropna = remove missing values (from the whole df in this example)
df = df.dropna()
# print(df.isnull().sum())
"""
loan_amnt               0
term                    0
...
revol_bal               0
revol_util              0 OK
total_acc               0
initial_list_status     0
application_type        0
mort_acc                0 OK
pub_rec_bankruptcies    0 OK
address                 0
loan_repaid             0

No missing data.
"""

print("\n====== 54. Keras Project Solutions - Catagorical Data (non-neumeric) ======")
# 24 September 2023
# How to deal with Categorical Variable and Dummy Variables
# Categorical and string data, do we convert to dummy variables?
# Task: List all the columns that are non-neumetic
# print(df.select_dtypes(["object"]).columns)
"""
Let's go through this and deal with them one by one

Index(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',
       'issue_d', 'loan_status', 'purpose', 'earliest_cr_line',
       'initial_list_status', 'application_type', 'address'],
      dtype='object')
"""
# Task1: Convert "term" feature into a neumeric data type
print("unique",df["term"].unique())                 # nd array (because its just a list)
print("value_counts", df["term"].value_counts())    # panda Series (because its a table)

# Well its binary, either 36 or 60 months.
# We could throw the neumeric relation of 36 and 60 away and just do a 1 and 0 or
# other binary coding - but there is information there, so lets keep the numbers
# and convert them from string to int 36 or 60
df["term"] = df["term"].apply(lambda term: int(term[:3]))
# print(df["term"])

# Task2: We already know about 'grade' and 'sub_grade' from the last lesson,
# and grade was basicaly duplicate infomation
df = df.drop("grade",axis=1)
# OK, lets convert the sub_grade column to dummy variables (in a new df)
# and then concatenate to the origonal df. Remember to drop the origonal sub_grade column

# drop_first means "drop first level"
# Whether to get k-1 dummies out of k categorical levels by removing the first level.
dummies = pd.get_dummies(df["sub_grade"], drop_first=True)
df_with_dropped = df.drop("sub_grade", axis=1)
df = pd.concat([df_with_dropped, dummies], axis=1)

# Task3: Convert the following variables into dummy variables
"""
# These columns are all just of a few string values, and convert nicely to bit codes
"verification_status"
"application_type"
"initial_list_status"
"purpose"
"""
dummies = pd.get_dummies(df[["verification_status",
                             "application_type",
                             "initial_list_status",
                             "purpose"]], drop_first=True)

df = df.drop(["verification_status",
                         "application_type",
                         "initial_list_status",
                         "purpose"], axis=1)

df = pd.concat([df, dummies], axis=1)

# Task4: Now home_ownership
"""
home_ownership
MORTGAGE    198022
RENT        159395
OWN          37660
OTHER          110
NONE            29
ANY              3
"""
# There are so few NONE and ANY,... lets put them in the OTHER category
# Almost - but not quite
# df["home_ownership"] = df["home_ownership"].map({"NONE":"OTHER", "ANY":"OTHER"})
df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")
#print(df["home_ownership"].value_counts()) # To check
"""
home_ownership
MORTGAGE    198022
RENT        159395
OWN          37660
OTHER          142
"""
# Now turn it into dummy variables
dummies = pd.get_dummies(df["home_ownership"], drop_first=True)
df_with_dropped = df.drop("home_ownership", axis=1)
df = pd.concat([df_with_dropped, dummies], axis=1)

# Task5: Create a new zip_code column from the address coulumn
# Assume the zip code is always the last 5 characters in an address - v unsafe
df["zip_code"] = df["address"].apply(lambda address:address[-5:])
df = df.drop("address", axis=1)

#print(df["zip_code"])
"""
0         22690
1         05113
2         05113
...
"""
#print(df["zip_code"].value_counts()) # Only 10 unique zip codes
"""
zip_code
70466    56880
22690    56413
30723    56402
...
"""
# Only 10 unique zip codes, so lets create 9 unique columns (why not 4 binary?)
dummies = pd.get_dummies(df["zip_code"], drop_first=True)
df_with_dropped = df.drop("zip_code", axis=1)
df = pd.concat([df_with_dropped, dummies], axis=1)

# Task6: "issue_d" - the month in which the load was funded
# But we are trying to determine, from someones known features, whether or
# not they are going to pay back the load - so in the real world we would be
# running the model, before issue_d has a value.
# So drop this feature/column
df = df.drop("issue_d", axis=1)

# Task7: earliest_cr_line - earliest credit line "The month the borrower's
# earliest reported credit line was open" - This has the full date.
# Lets just extract the year and create a new column called earliest_cr_year
#print(df["earliest_cr_line"])
"""
0         Jun-1990
1         Jul-2004
2         Aug-2007
...
"""
# So lets make same assumbtion as the zip code above and extract the last 4 characters
# It might be safer, to first convert earliest_cr_line to a date format, and then extract the year
df["earliest_cr_year"] = df["earliest_cr_line"].apply(lambda date: int(date[-4:]) )
df = df.drop("earliest_cr_line", axis=1)

print("========== 55. Keras Project Solutions - Data PreProcessing ==========")
# Remember we made a bit column called loan_repaid from the column loan_status
# drop the loan_status column - lets use loan_repaid
df = df.drop("loan_status", axis=1)

df = df.sample(frac=0.1, random_state=101) # Full thing is too big, take 10%

X = df.drop("loan_repaid",axis=1).values # just a np.array without loan_repaid
y = df["loan_repaid"].values # just loan_repaid np.array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("========== 56. Keras Project Solutions - Creating and Training a Model ==========")
# 78 -> 39 -> 19 -> 1
# Note: The number of neurons, sort of drop by half, at each layer

model = Sequential()

model.add(Dense(78, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(39, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(19, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer= "adam")

model.fit(X_train, y_train, epochs=25, batch_size=256,validation_data=(X_test, y_test) )

print("========== 57. Keras Project Solutions - Model Evaluation ==========")

from tensorflow.keras.models import load_model
model.save("Project_sec7_model.keras")

losses = pd.DataFrame(model.history.history)
#losses.plot()
#plt.show()

# classification_report, confusion_matrix
# predictions = model.predict_classes(X_test) # depricated
predictions = model.predict(X_test)
predictions = np.round(predictions).astype(int)
#print(predictions)
print( classification_report(y_test, predictions) )
"""
              precision    recall  f1-score   support

           0       0.92      0.45      0.60      2382
           1       0.88      0.99      0.93      9475

    accuracy                           0.88     11857
   macro avg       0.90      0.72      0.77     11857
weighted avg       0.89      0.88      0.86     11857
"""

# Last task: Given the customer below - would you give them a loan
# First grab a random customer row from df
import random
random.seed(101)
random_index = random.randint(0, len(df))

# Grab the random row, without wheather it was paid on not
new_customer = df.drop("loan_repaid", axis=1).iloc[random_index]
# print(new_customer)

new_customer = new_customer.values.reshape(1, 78)
new_customer = scaler.transform(new_customer)

# prediction = model.predict_classes(new_customer) # depricated
prediction = model.predict(new_customer)
prediction = np.round(prediction).astype(int)
print("prediction", prediction)
# prediction [[1]] # Person is predicted to get a loan

# Task - check that this person actualy got a loan
print(df["loan_repaid"].iloc[random_index])
# 1  # yes this person got a loan

# END OF PROJECT


























