# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.simplefilter(action = 'ignore',category = FutureWarning)


import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam


train_d = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_d = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train_d.head(5)


train_d.describe()


train_d.shape,test_d.shape


train_d.isnull().sum()


# Values Count of Target Column
train_d['accident_risk'].value_counts()


print("Information of train dataset")
print("-"* 60)
train_d.info()
print("\n")
print("information of test set")
print("-"*60)

test_d.info()


print("printing  unique values of all columns")

for col in train_d:
    print("-"*60)
    unq = train_d[col].unique()
    if len(unq)>15 :
        print(col,":", unq[:5],"....", "so much unique values")
    else:
        print(col,":", unq)


# checking missing values 
train_d.isnull().sum()


num_col = train_d.select_dtypes(include = ['int64','float64']).columns
random_search = ['royalblue', 'seagreen', 'orange', 'crimson', 'purple', 'gold', 'teal', 'tomato']

plt.figure(figsize = (14,12))
for i,col in enumerate(num_col,start = 1):
    choose_color = random.choice(random_search)
    plt.subplot(3,3,i)
    sns.histplot(x = col,data = train_d,kde = True,color = choose_color,bins = 20 )

plt.tight_layout()
plt.show()


# countplot of target variable 
plt.figure(figsize = (14,12))
object_col = train_d.select_dtypes('object').columns
colors = ('Set1','Set2','Set3')
for i,col in enumerate(object_col,1):
    random_color = random.choice(colors)
    plt.subplot(3,3,i)
    sns.countplot(x = col ,data= train_d,palette =random_color )
    plt.xticks(rotation = 90)
    plt.title(f"Countplot of {col}")

plt.tight_layout()
plt.show()


train_d = train_d.drop('id',axis = 1)
test_d = test_d.drop('id',axis = 1)


# applying label encoder 
ln = LabelEncoder()

for col in object_col:
    train_d[col] = ln.fit_transform(train_d[col])
    test_d[col] = ln.transform(test_d[col])


bool_col = train_d.select_dtypes('bool').columns
for col in bool_col:
    train_d[col] = train_d[col].astype(int)
    test_d[col] = test_d[col].astype(int)


scaler = StandardScaler()
num_float = test_d.select_dtypes(include = ['int64','float64']).columns
train_d[num_float] = scaler.fit_transform(train_d[num_float])
test_d[num_float] = scaler.transform(test_d[num_float])


plt.figure(figsize = (10,8))
corr = train_d.corr()

sns.heatmap(corr.round(2),annot = True,cmap = 'coolwarm',linewidth = 0.5)
plt.title("Correlation Map ")
plt.show()


x = train_d.drop('accident_risk',axis = 1)
y = train_d['accident_risk']
test = test_d.copy()


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


classifier = Sequential()


# adding input layer
classifier.add(Dense(units =12, activation = 'relu'))


# now adding hidden layer 
classifier.add(Dense(units = 8,activation = 'relu'))

# adding one more layer 
classifier.add(Dense(units = 4,activation = 'relu'))



# now output layer 
classifier.add(Dense(units = 1,activation = 'linear'))



optimizer = Adam(learning_rate = 0.01)


classifier.compile(optimizer = optimizer,loss = 'binary_crossentropy',metrics = ['mse'])



early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    min_delta = 0.001,
    patience = 3,
    verbose = 0,
    mode = 'auto',
    baseline = None,
    restore_best_weights = True
)


classifier.fit(x_train,y_train,validation_split = 0.33,batch_size = 20,epochs = 30,callbacks = early_stopping)


# making predictions 
preds = classifier.predict(test_d)


submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


submission['accident_risk'] = preds


submission.head()


# saving the submission file 
submission.to_csv('submission.csv',index = False)
print("Submission file is saved successfully✅")




