"""
Tensorflow 2 and Keras Deep Learning Bootcamp
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

18 September 2023

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

#path to root of FirstPython
root_path = "C:\\Tim\\VSCodeProjects\\FirstPython"

print("========== 43. Keras Regression Code Along - Exploritory Data Analysis ==========")
# Exploritory Data Analyisis
# Feature Engineering
df = pd.read_csv(root_path + "\\TensorFlow2Bootcamp06_Sec7_ANN03\\kc_house_data.csv")

# Let's see if there is any missing data
print(df.isnull().sum()) # All zeros, so no missing data
"""
id               0
date             0
price            0
bedrooms         0
bathrooms        0
sqft_living      0
...
"""
# Let's get a quick statistical analysis of the dataset
print(df.describe())

# Let's look the the distribution of prices
# It looks like most houses are less than about $1.5M.
# There are a few outliers at $5M etc., so we might drop those in our analysis.
# sns.displot(df["price"])
# plt.show()

# 19 September 2023

# Count plot of bedrooms
# dosn't work properly
# sns.countplot(df["bedrooms"])
# plt.show()

# Remember the "label" in this example is price
print(df.corr(numeric_only=True)["price"].sort_values(ascending=False))

# sns.scatterplot(x="price", y="sqft_living", data=df)
# plt.show()

#sns.boxplot(x="bedrooms", y="price", data=df)
#plt.show()

# The data includes lat and long information
#plt.figure(figsize=(12,8))
#sns.scatterplot(x="long", y="lat", data=df, hue="price")
#plt.show()

# Lets try to drop some outliers - the top 1%
# len(df) * 0.01  # = 215.97

"""
# So grab all rows AFTER the top 1% of houses
non_top_1_perc = df.sort_values("price", ascending=False).iloc[216:]
plt.figure(figsize=(12,8))
sns.scatterplot(x="long", y="lat", data=non_top_1_perc,
                palette="RdYlGn", hue="price", edgecolor=None, alpha=0.2)
plt.show()
"""
# We actualy have a waterfront feature
# sns.boxplot(x="waterfront",y="price", data=df)
# plt.show()

print("========== 44. Keras Regression Code Along - Exploritory Data Analysis - Continued ==========")
"""
So, let's do a bit of feature engineering and feature dropping
"""
print(df.head().to_string())
"""
           id        date     price  bedrooms  bathrooms  sqft_living  sqft_lot  floors  waterfront  view  condition  grade  sqft_above  sqft_basement  yr_built  yr_renovated  zipcode      lat     long  sqft_living15  sqft_lot15
0  7129300520  10/13/2014  221900.0         3       1.00         1180      5650     1.0           0     0          3      7        1180              0      1955             0    98178  47.5112 -122.257           1340        5650
1  6414100192  12/09/2014  538000.0         3       2.25         2570      7242     2.0           0     0          3      7        2170            400      1951          1991    98125  47.7210 -122.319           1690        7639
2  5631500400   2/25/2015  180000.0         2       1.00          770     10000     1.0           0     0          3      6         770              0      1933             0    98028  47.7379 -122.233           2720        8062
3  2487200875  12/09/2014  604000.0         4       3.00         1960      5000     1.0           0     0          5      7        1050            910      1965             0    98136  47.5208 -122.393           1360        5000
4  1954400510   2/18/2015  510000.0         3       2.00         1680      8080     1.0           0     0          3      8        1680              0      1987             0    98074  47.6168 -122.045           1800        7503
"""
# Something we can drop immediatly is the id - it dosn't mean anything
df = df.drop("id",  axis=1)

# The date column, at the moment, is just a string - so it is difficult to do process.
# Let's convert it to a datetime object
df["date"] = pd.to_datetime(df["date"])
print("\n")
print(df.head().to_string()) # Note, date is now in a datetime format

# We can now easily create a new "year" column
df["year"] = df["date"].apply(lambda date: date.year)
df["month"] = df["date"].apply(lambda date: date.month)
print("\n")
print(df.head().to_string())

"""
lambda date: date.year
is equivalent to

def year_extraction(date):
    return date.year
"""
# So we see that the year and month that where "hidden"
# inside the string date are now rendered useful
# Let's see if our new month data is useful
# sns.boxplot(x="month", y="price", data=df)
# plt.show()

# Or we could see the mean price grouped-by month
# print(df.groupby("month").mean()["price"])
"""
month
1     525963.251534
2     508520.051323
3     544057.683200
4     562215.615074
5     550849.746893
6     557534.318182
7     544892.161013
8     536655.212481
9     529723.517787
10    539439.447228
11    522359.903478
12    524799.902041
Name: price, dtype: float64
"""
# Now we can get rid of the redundant date column
df = df.drop("date", axis=1)
print(df.columns)
# The zipcode is interesting
# Because they are neumetical (98178, 98125, etc.) the model will assume they are some sort
# of continuous neumerical value - this isn't the case - but is it viable/meaningful to start
# using zipcode as some sort of categorical value? You need to explore its meaning.

# How many unique zipcodes to we have - 70 unique zipcode,
print(df["zipcode"].value_counts())

# So this is probaly too much to treate as a category, with dummy variables etc.
# So let's drop it, because we can't use it, and the model
# would speand too much effort typing to use something useless.
# There may be more subtle ways of using zipcodes
# A lot of this is domain knowledge - ask an expert

df = df.drop("zipcode", axis=1)

# Another column that looks problomatic is yr_renovated
print(df["yr_renovated"].value_counts())
"""
0       20683    <<<<< So most properties were never renovated - loads of zeros
2014       91
2013       37
2003       36
2005       35
        ...  
1951        1
1959        1
1948        1
1954        1
1944        1
"""
# Now in the column "year", 0 is not a year.
# It is an indication that the house was NOT renavated.
# So it may make more sense to categorize this column as renavated vs. not-renovated
# But let's not do this.

# A similar situating with loads of zeros is sqft_basement - no basement
# These make sense as a continuouse variable, so we can keep as is.
print(df["sqft_basement"].value_counts())
"""
sqft_basement
0      13110
600      221
700      218
500      214
800      206
...
"""
print("=== 45. Keras Regression Code Along - Data Preprocessing and Creating a Model ===")
"""
So we are going to train the model 
"""
# First thing to do is seperate the label from the features
# The "values" so it returns the numpy array
X = df.drop("price", axis=1).values     # Now the features
y = df["price"].values                  # Now the label

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# The MinMaxScaler method "fit", calculates the standard deviation,
# min and max of the data so it can do scaling (the transform) later on.

# We can do this in two stages, thus:
# scaler.fit(X_train) <<<<< not to be confused with training a model.fit()
# X_train = scaler.transform(X_train)
# Or use the more convienient method "fit_transform", which does both stage at once.
X_train = scaler.fit_transform(X_train)

# We only transform the test set - no fit, because we do not want to assume
# any prior knowledge about the training set!?
X_test = scaler.transform(X_test)

# print(X_train.shape) # (15117, 19)

model = Sequential()

model.add(Dense(19, activation="relu")) # Rectified Linear Unit
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

# validation_data - After each epoch of training on the train data, we will quickly
# run the model (so far) on our test data, to see how well we are doing on that.
# Note: The results of our testing on the test data is NOT USED for updating the
# weights in the model - only the results of training on the training data do that.
# Because the test set is not used in the weight training, we will be able to see
# if the model is becomeing overfitted - I almost get this!
# You get two losses per epoch:
# loss (training): 28737253376.0000 - val_loss (test): 26383943680.0000
model.fit(x=X_train, y=y_train,
          validation_data=(X_test, y_test),
          batch_size=128, epochs=100)
# The smaller the batch_size the longer the training will take but the less likley you
# are to overfit, because each batch is effectivly an indipendent training set.

print("=== 46. Keras Regression Code Along - Model Evaluation and Predictions ===")

# print(model.history.history)
"""
{'loss': [430240825344.0, 
          428868861952.0, 
          408797577216.0, 
          315282161664.0, 
          159248547840.0, 
          101095677952.0, 
          97691025408.0, 
          95958048768.0, 
          94194671616.0, 
          92392325120.0, 
          90554187776.0, 
          88621203456.0, 86663577600.0, 84597858304.0, 82477621248.0, 80206520320.0, 77902987264.0, 75543388160.0, 73010741248.0, 70515220480.0, 67956113408.0, 65471782912.0, 62997499904.0, 60668628992.0, 58604896256.0, 56722268160.0, 55094386688.0, 53754130432.0, 52613931008.0, 51599777792.0, 50816811008.0, 50065854464.0, 49394061312.0, 48785432576.0, 48283369472.0, 47752953856.0, 47265751040.0, 46762295296.0, 46351917056.0, 45979906048.0, 45595697152.0, 45231706112.0, 44904247296.0, 44640485376.0, 44239372288.0, 43933261824.0, 43671175168.0, 43277643776.0, 42936655872.0, 42656567296.0, 42351132672.0, 42022989824.0, 41748189184.0, 41449861120.0, 41154404352.0, 40873197568.0, 40598732800.0, 40356044800.0, 40088076288.0, 39862222848.0, 39664001024.0, 39392923648.0, 39148191744.0, 38944268288.0, 38705745920.0, 38506389504.0, 38250131456.0, 38042972160.0, 37893341184.0, 37628084224.0, 37496578048.0, 37282926592.0, 37145157632.0, 36977221632.0, 36823269376.0, 36737429504.0, 36637814784.0, 36477341696.0, 36382916608.0, 36211548160.0, 36126584832.0, 36003315712.0, 35868860416.0, 35786391552.0, 35657641984.0, 35578322944.0, 35477364736.0, 35378728960.0, 35269206016.0, 35186753536.0, 35097939968.0, 35015901184.0, 34947854336.0, 34905149440.0, 34842476544.0, 34755256320.0, 34664341504.0, 34609160192.0, 34516262912.0, 34537717760.0, 34431246336.0, 34339733504.0, 34280034304.0, 34234941440.0, 34171932672.0, 34100211712.0, 34012338176.0, 33990881280.0, 33932128256.0, 33860657152.0, 33816383488.0, 33733027840.0, 33698230272.0, 33631985664.0, 33632983040.0, 33560342528.0, 33527412736.0, 33450907648.0, 33455157248.0, 33397069824.0, 33385611264.0, 33314576384.0, 33264474112.0, 33281761280.0, 33203286016.0, 33140992000.0, 33159477248.0, 33113438208.0, 33054795776.0, 33065467904.0, 33021962240.0, 32954912768.0, 32992856064.0, 32910598144.0, 32874258432.0, 32890343424.0, 32793397248.0, 32824946688.0, 32766713856.0, 32698310656.0, 32696068096.0, 32632076288.0, 32628713472.0, 32607578112.0, 32555001856.0, 32521250816.0, 32518166528.0, 32529190912.0, 32469835776.0, 32443295744.0, 32404168704.0, 32378439680.0, 32367861760.0, 32346112000.0, 32300752896.0, 32286418944.0, 32272871424.0, 32246292480.0, 32192251904.0, 32187908096.0, 32147179520.0, 32134688768.0, 32117331968.0, 32081727488.0, 32083130368.0, 32046262272.0, 31994460160.0, 31994984448.0, 31966330880.0, 31943899136.0, 31931713536.0, 31927314432.0, 31873079296.0, 31882805248.0, 31837814784.0, 31849940992.0, 31761790976.0, 31819976704.0, 31832795136.0, 31725051904.0, 31756169216.0, 31704547328.0, 31703863296.0, 31684935680.0, 31656300544.0, 31619180544.0, 31626829824.0, 31608274944.0, 31595259904.0, 31572811776.0, 31552917504.0, 31508932608.0, 31521267712.0, 31495555072.0, 31468648448.0, 31454388224.0, 31441444864.0, 31434508288.0, 31426746368.0, 31408596992.0, 31351496704.0, 31368593408.0, 31351826432.0, 31358552064.0, 31363315712.0, 31286767616.0, 31274143744.0, 31300433920.0, 31286882304.0, 31234750464.0, 31223459840.0, 31231272960.0, 31203926016.0, 31206277120.0, 31153035264.0, 31122432000.0, 31129624576.0, 31144560640.0, 31070613504.0, 31072284672.0, 31055495168.0, 31055128576.0, 31047956480.0, 31067314176.0, 31060172800.0, 31079352320.0, 30988404736.0, 30976301056.0, 30948734976.0, 30926133248.0, 30930534400.0, 30906036224.0, 30950881280.0, 30872774656.0, 30896218112.0, 30861918208.0, 30829406208.0, 30864349184.0, 30844891136.0, 30799437824.0, 30816184320.0, 30801389568.0, 30800404480.0, 30777917440.0, 30757097472.0, 30707806208.0, 30740160512.0, 30745620480.0, 30773440512.0, 30716753920.0, 30766094336.0, 30652962816.0, 30647353344.0, 30657619968.0, 30625593344.0, 30627102720.0, 30611279872.0, 30609633280.0, 30572412928.0, 30596864000.0, 30548598784.0, 30536175616.0, 30537287680.0, 30482741248.0, 30551242752.0, 30502146048.0, 30493870080.0, 30469937152.0, 30443632640.0, 30485381120.0, 30439925760.0, 30424180736.0, 30397659136.0, 30397315072.0, 30388748288.0, 30371897344.0, 30371129344.0, 30350850048.0, 30394564608.0, 30304743424.0, 30328973312.0, 30323046400.0, 30280601600.0, 30280179712.0, 30296561664.0, 30272985088.0, 30234759168.0, 30345994240.0, 30212161536.0, 30209495040.0, 30208239616.0, 30196813824.0, 30201315328.0, 30169344000.0, 30132862976.0, 30184056832.0, 30131087360.0, 30137716736.0, 30199457792.0, 30153244672.0, 30107922432.0, 30079719424.0, 30066657280.0, 30088671232.0, 30052882432.0, 30032271360.0, 30067785728.0, 30046011392.0, 30044942336.0, 29992122368.0, 30015715328.0, 30026719232.0, 30011910144.0, 30019241984.0, 29966409728.0, 30016937984.0, 29947428864.0, 29959059456.0, 29947777024.0, 29912393728.0, 29891827712.0, 29888415744.0, 29882597376.0, 29873305600.0, 29888874496.0, 29875568640.0, 29852141568.0, 29863569408.0, 29846251520.0, 29841631232.0, 29782906880.0, 29805312000.0, 29810636800.0, 29806331904.0, 29776424960.0, 29765142528.0, 29746024448.0, 29748862976.0, 29752074240.0, 29736130560.0, 29732227072.0, 29813596160.0, 29715128320.0, 29704194048.0, 29677590528.0, 29651132416.0, 29669795840.0, 29652463616.0, 29648422912.0, 29611573248.0, 29615372288.0, 29645727744.0, 29656078336.0, 29602545664.0, 29598861312.0, 29567516672.0, 29588498432.0, 29583616000.0, 29672632320.0, 29551886336.0, 29556150272.0, 29511243776.0, 29526884352.0, 29514600448.0, 29530839040.0, 29488252928.0, 29492393984.0, 29470078976.0, 29439641600.0, 29451937792.0, 29529720832.0, 29439938560.0, 29494325248.0, 29452978176.0, 29400334336.0, 29413746688.0, 29395924992.0, 29399558144.0, 29379383296.0, 29365710848.0, 29331415040.0, 29383827456.0, 29317156864.0, 29342738432.0, 29339904000.0, 29367310336.0, 29290115072.0, 29280104448.0, 29306540032.0, 29285767168.0, 29233225728.0, 29249974272.0, 29248823296.0, 29218498560.0, 29238294528.0, 29225832448.0, 29201862656.0, 29201600512.0, 29223227392.0, 29214666752.0], 
'val_loss': [418912075776.0, 
             413960208384.0, 
             368466591744.0, 
             225106018304.0, 104603860992.0, 95329640448.0, 93672644608.0, 92035260416.0, 90306510848.0, 88621162496.0, 86796926976.0, 85039505408.0, 83025682432.0, 81050345472.0, 79026782208.0, 76703907840.0, 74549772288.0, 72140644352.0, 69729525760.0, 67214557184.0, 64933847040.0, 62461022208.0, 60121346048.0, 57916833792.0, 56095608832.0, 54326259712.0, 53029703680.0, 51717533696.0, 50725871616.0, 49846124544.0, 49133813760.0, 48443269120.0, 47857053696.0, 47290830848.0, 46768377856.0, 46279712768.0, 45832773632.0, 45394546688.0, 44997279744.0, 44643647488.0, 44276244480.0, 43978022912.0, 43656867840.0, 43310252032.0, 43022184448.0, 42669436928.0, 42358099968.0, 42157608960.0, 41750724608.0, 41459789824.0, 41116188672.0, 40877539328.0, 40483577856.0, 40181366784.0, 39880777728.0, 39596359680.0, 39297822720.0, 39054155776.0, 38794674176.0, 38625136640.0, 38302920704.0, 38064476160.0, 37848129536.0, 37603430400.0, 37379948544.0, 37145817088.0, 36912779264.0, 36721704960.0, 36524249088.0, 36379283456.0, 36127485952.0, 36038746112.0, 35808358400.0, 35684552704.0, 35605893120.0, 35396419584.0, 35312955392.0, 35161706496.0, 35034877952.0, 34905964544.0, 34768945152.0, 34659983360.0, 34564501504.0, 34436476928.0, 34339033088.0, 34241630208.0, 34146445312.0, 34049931264.0, 33957877760.0, 33871945728.0, 33776545792.0, 33715404800.0, 33620912128.0, 33547296768.0, 33496522752.0, 33439232000.0, 33324845056.0, 33270136832.0, 33266593792.0, 33108377600.0, 33044148224.0, 33022058496.0, 33034047488.0, 32843264000.0, 32764041216.0, 32697526272.0, 32690358272.0, 32580478976.0, 32540514304.0, 32455047168.0, 32379930624.0, 32321550336.0, 32264128512.0, 32239097856.0, 32216940544.0, 32292577280.0, 32073213952.0, 32007686144.0, 32023932928.0, 31979390976.0, 31858808832.0, 31925620736.0, 31775342592.0, 31721422848.0, 31684323328.0, 31654027264.0, 31645349888.0, 31566321664.0, 31523749888.0, 31559317504.0, 31451371520.0, 31412011008.0, 31527589888.0, 31547475968.0, 31404634112.0, 31304003584.0, 31256014848.0, 31240282112.0, 31191324672.0, 31254521856.0, 31168325632.0, 31135438848.0, 31107258368.0, 31011741696.0, 30972948480.0, 31042631680.0, 30961061888.0, 30874552320.0, 30850385920.0, 30822889472.0, 30847363072.0, 30797561856.0, 30740602880.0, 30701705216.0, 30677843968.0, 30676148224.0, 30587705344.0, 30562543616.0, 30581151744.0, 30521384960.0, 30488055808.0, 30444832768.0, 30451800064.0, 30447822848.0, 30372487168.0, 30328348672.0, 30330181632.0, 30279563264.0, 30271418368.0, 30279501824.0, 30273267712.0, 30211516416.0, 30179008512.0, 30130470912.0, 30192138240.0, 30094182400.0, 30396180480.0, 30082977792.0, 30029768704.0, 29984282624.0, 30030086144.0, 29960919040.0, 29916803072.0, 30056491008.0, 29915793408.0, 29888858112.0, 29856034816.0, 29870655488.0, 29783490560.0, 29754038272.0, 29743544320.0, 29722624000.0, 29702445056.0, 29677111296.0, 29654624256.0, 29648496640.0, 29609637888.0, 29606291456.0, 29531287552.0, 29614053376.0, 29532313600.0, 29500108800.0, 29555130368.0, 29490212864.0, 29456701440.0, 29436127232.0, 29418115072.0, 29401999360.0, 29357641728.0, 29320681472.0, 29353177088.0, 29301356544.0, 29522178048.0, 29298968576.0, 29245261824.0, 29219024896.0, 29197127680.0, 29169111040.0, 29259327488.0, 29134952448.0, 29176066048.0, 29116596224.0, 29140027392.0, 29093746688.0, 29095102464.0, 29071595520.0, 29040687104.0, 29039833088.0, 29002819584.0, 28976723968.0, 28972908544.0, 28959510528.0, 28962803712.0, 28911484928.0, 28916971520.0, 28895277056.0, 29127776256.0, 28868663296.0, 28843634688.0, 28853067776.0, 28841336832.0, 28809170944.0, 28791435264.0, 28755988480.0, 28752736256.0, 28734324736.0, 28720988160.0, 28700717056.0, 28674574336.0, 28652322816.0, 28665511936.0, 28663605248.0, 28699607040.0, 28626210816.0, 28617449472.0, 28603934720.0, 28590256128.0, 28564436992.0, 28570404864.0, 28519708672.0, 28508811264.0, 28475975680.0, 28477972480.0, 28551931904.0, 28449662976.0, 28471246848.0, 28452648960.0, 28436805632.0, 28403656704.0, 28385581056.0, 28445700096.0, 28419549184.0, 28406093824.0, 28334069760.0, 28302106624.0, 28290275328.0, 28264734720.0, 28274155520.0, 28345911296.0, 28252616704.0, 28280688640.0, 28216260608.0, 28195608576.0, 28159735808.0, 28165042176.0, 28234997760.0, 28206774272.0, 28119554048.0, 28132374528.0, 28088293376.0, 28062754816.0, 28060971008.0, 28052807680.0, 28042938368.0, 28038699008.0, 28001210368.0, 28458270720.0, 28008486912.0, 28012789760.0, 27988185088.0, 27965290496.0, 27974035456.0, 27918780416.0, 27907401728.0, 27878920192.0, 27919118336.0, 27896025088.0, 27914217472.0, 27854137344.0, 27819966464.0, 27819016192.0, 27802875904.0, 27794468864.0, 27795712000.0, 27791144960.0, 27795003392.0, 27780708352.0, 27761029120.0, 27826225152.0, 27750426624.0, 27726907392.0, 27711698944.0, 27714557952.0, 27807350784.0, 27670775808.0, 27659902976.0, 27632568320.0, 27664775168.0, 27589457920.0, 27593842688.0, 27598331904.0, 27578910720.0, 27565967360.0, 27541743616.0, 27671625728.0, 27519541248.0, 27528134656.0, 27574059008.0, 27534114816.0, 27541719040.0, 27482869760.0, 27482312704.0, 27651330048.0, 27481106432.0, 27457767424.0, 27477612544.0, 27464994816.0, 27441920000.0, 27400212480.0, 27390570496.0, 27447644160.0, 27410245632.0, 27352412160.0, 27332001792.0, 27338512384.0, 27307388928.0, 27337781248.0, 27371456512.0, 27289759744.0, 27276359680.0, 27304300544.0, 27234408448.0, 27246098432.0, 27213602816.0, 27219505152.0, 27204685824.0, 27158091776.0, 27188377600.0, 27193729024.0, 27224823808.0, 27207991296.0, 27313936384.0, 27152848896.0, 27118882816.0, 27232655360.0, 27110875136.0, 27159914496.0, 27093647360.0, 27070345216.0, 27051880448.0, 27108151296.0, 27125403648.0, 27128494080.0, 27049293824.0, 27089106944.0, 27031326720.0, 26978183168.0, 26938308608.0, 26951174144.0, 26927030272.0, 26999343104.0, 26923530240.0, 27011350528.0, 26915266560.0, 26950885376.0, 26880491520.0, 26927824896.0, 26892926976.0, 26928951296.0, 26866518016.0]}
"""

losses = pd.DataFrame(model.history.history)
print(losses)
"""
             loss      val_loss
0    4.302274e+11  4.188551e+11
1    4.284640e+11  4.127177e+11
2    4.044566e+11  3.589031e+11
3    2.978502e+11  2.022008e+11
4    1.423363e+11  9.929247e+10
..            ...           ...
395  2.960330e+10  2.739949e+10
396  2.961048e+10  2.740892e+10
397  2.957119e+10  2.732788e+10
398  2.956181e+10  2.741252e+10
399  2.961652e+10  2.732261e+10
"""

# So... am I overfitting my training data to my model?
# No
#losses.plot()
#plt.show()
"""
This is exactley the sort of signal we want to see, where there is
decrease in both traing and validation data, then there is no increae in 
the validation set from the training set.

If you saw the validation line on the graph starting to increase then
this would be an indication of overfitting.
i.e. as a result of over-training, you are JUST and ONLY fitting 
to the training data. The validation data, no longer fits this over-exact fit and 
so its error increases.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

predictions = model.predict(X_test)

# Very big because it is of the order of the square of house prices
print("mse", mean_squared_error(y_test, predictions))

# More sensible, of the order of house prices
print("mae", mean_absolute_error(y_test, predictions)) # Off by around 25% - not so good

print(explained_variance_score(y_test, predictions))
# 0.7751854446013382 - best is 1.0

#plt.scatter(y_test, predictions) # x, y
#plt.plot(y_test, y_test,"red") # Straight line, best fit
#plt.show()
# You can see from the plot, that we are getting punished by a few
# very expensive properties, which we are undervaluing (they come under the red line)
# It might be an idea to conside properties above $3M as outliers and drop them

# So... how to we use our trained model, to predict on an previously unseen house
# Lets just select everything but the price from the first row in df
single_house = df.drop("price", axis=1).iloc[0]
print(single_house)
"""
bedrooms            3.0000
bathrooms           1.0000
sqft_living      1180.0000
sqft_lot         5650.0000
...
"""
# Next we need to scale the data - just get the values from df and reshape it
reshaped_values = single_house.values.reshape(-1, 19) # -1 means?

single_house = scaler.transform(reshaped_values)
print(single_house)
"""
[[0.2        0.08       0.08376422 0.00310751 0.         0.
  0.         0.5        0.4        0.10785619 0.         0.47826087
  0.         0.57149751 0.21760797 0.16193426 0.00582059 0.
  0.81818182]]
"""

single_house_prediction = model.predict(single_house)
print(single_house_prediction)
# [[282926.2]] # dollars

print(df.head(1))
# 221900.0 - not so good, but in the range
# END


