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


pip install tensorflow-ranking



import tensorflow_ranking as tfr



from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense,Dropout

from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


df_train 


df_train.info()


df_train.describe()  ## descriptive analysis


df_train['Fertilizer Name'].unique()


### Temperature 
plt.xticks(rotation=75)
sns.set_theme(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Fertilizer Name", y="Temparature", data=df_train)


### Humidity  
plt.xticks(rotation=75)
sns.set_theme(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Fertilizer Name", y="Humidity", data=df_train)


### Humidity  
plt.xticks(rotation=75)
sns.set_theme(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Fertilizer Name", y="Moisture", data=df_train)


### Potassium   
plt.xticks(rotation=75)
sns.set_theme(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Fertilizer Name", y="Potassium", data=df_train)


### Nitrogen   
plt.xticks(rotation=75)
sns.set_theme(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Fertilizer Name", y="Nitrogen", data=df_train)


#Phosphorous    
plt.xticks(rotation=75)
sns.set_theme(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Fertilizer Name", y="Phosphorous", data=df_train)


df_train.columns


univariate_analysis_soil_type = pd.crosstab(df_train['Soil Type'],df_train['Fertilizer Name'])




ax= univariate_analysis_soil_type.plot(kind='bar', stacked=True)
for container in ax.containers:
    print(container)
    ax.bar_label(container)


univariate_analysis_crop_type = pd.crosstab(df_train['Crop Type'],df_train['Fertilizer Name'])
univariate_analysis_crop_type

ax= univariate_analysis_crop_type.plot(kind='bar', stacked=True)
for container in ax.containers:
    print(container)
    ax.bar_label(container)


df_train


df_train=pd.concat([df_train,pd.get_dummies(df_train['Soil Type'],prefix='Soil Type',dtype =int)],axis=1)




df_train


df_train=df_train.drop(['Soil Type','Soil Type_Sandy'],axis=1)


df_train


df_train=pd.concat([df_train,pd.get_dummies(df_train['Crop Type'],prefix='Crop Type',dtype =int)],axis=1)
df_train = df_train.drop(['Crop Type','Crop Type_Wheat'],axis =1)


df_train


df_train['Fertilizer Name'].value_counts()


min_count = df_train['Fertilizer Name'].value_counts().min()

# Group by label and take the first 'min_count' rows from each group
balanced_df = df_train.groupby('Fertilizer Name').head(min_count)




balanced_df


balanced_df['Fertilizer Name'].value_counts()


## lets one hot  encode the target class  
target = df_train['Fertilizer Name']

target =  pd.get_dummies(df_train['Fertilizer Name'],prefix='FN',dtype =int)


target


pd.from_dummies(target)


X= df_train.drop(['Fertilizer Name','id'],axis =1).values
y = target.values
X



y


### scaling
scaler = preprocessing.MinMaxScaler((0,1))
X = scaler.fit_transform(X)






X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.20)


def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=chi2, k=40)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs





X.shape ,y.shape


X_train.shape ,y_train.shape , X_val.shape, y_val.shape




def baseline_model():
  model = Sequential()
  model.add(Dense(128,input_dim=20,activation='sigmoid'))
  model.add(Dense(128,activation='sigmoid'))
  model.add(Dense(64,activation='sigmoid'))
  model.add(Dense(64,activation='sigmoid'))
  model.add(Dropout(rate = 0.3))
  model.add(Dense(32,activation='sigmoid'))
  model.add(Dense(32,activation='sigmoid'))
  model.add(Dense(7,activation='softmax'))
	
  # Compile model  
  model.compile( optimizer='adam',loss='categorical_crossentropy',metrics = [tfr.keras.metrics.MeanAveragePrecisionMetric('5')])
  return model



model=baseline_model()
print(model.summary())
model_checkpoint_callback = ModelCheckpoint(filepath='/kaggle/working/',
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss',
    patience=10,         
    verbose=1,          
    mode='min',         
    restore_best_weights=True )
model.fit(X,y,epochs=100,batch_size=128,validation_data=(X_val, y_val),callbacks = [model_checkpoint_callback] )
#validation_data=(x_val, y_val),


id_ = df_test['id']
df_test.drop(['id'],inplace=True,axis =1)


df_test.head(5)


#prepare test data 
df_test=pd.concat([df_test,pd.get_dummies(df_test['Crop Type'],prefix='Crop Type',dtype =int)],axis=1)
df_test = df_test.drop(['Crop Type','Crop Type_Wheat'],axis =1)
df_test=pd.concat([df_test,pd.get_dummies(df_test['Soil Type'],prefix='Soil Type',dtype =int)],axis=1)
df_test=df_test.drop(['Soil Type','Soil Type_Sandy'],axis=1)

X_test = scaler.transform(df_test.values)


y_pred = model.predict(X_test)


from keras.utils import to_categorical
labels = to_categorical(np.argmax(y_pred, 1), dtype = "int64")


labels


df = pd.DataFrame(labels,columns = target.columns)


df_ = pd.from_dummies(df)


df_['id'] = id_


df_.columns = ['Fertilizer Name','id']


df_[['id','Fertilizer Name']].to_csv('output.csv')


model.save_weights("model_weights.h5")
model_architecture = model.to_json()

with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_architecture)



model.save("model.keras")

