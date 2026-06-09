import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import  mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from xgboost import XGBRegressor
from catboost import CatBoostClassifier, Pool




df=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')


df.info()


df.describe()


df.head()


df.isnull().sum()


df.dropna(subset=['num_sold'], inplace=True)


df.num_sold.isnull().sum()


 df['num_sold'] = df['num_sold'].apply(lambda x: 3000 if x > 3000 else x)


# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Extract year, month, and day into separate columns
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

to_drop =['date','id']



df.drop(to_drop,axis=1, inplace=True)


df


sns.barplot(x=df['year'], y=df['num_sold'])



sns.barplot(x=df['month'], y=df['num_sold'])


sns.barplot(x=df['day'], y=df['num_sold'])


y=df.pop('num_sold')


train_X, val_X, train_y, val_y = train_test_split(df, y, random_state = 0)


object_cols=['country','store','product']


encoder =OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(encoder.fit_transform(train_X[object_cols]))
OH_cols_val = pd.DataFrame(encoder.transform(val_X[object_cols]))


# One-hot encoding removed index; put it back
OH_cols_train.index = train_X.index
OH_cols_val.index =  val_X.index
# Remove categorical columns (will replace with one-hot encoding)
num_X_train = train_X.drop(object_cols, axis=1)
num_X_val = val_X.drop(object_cols, axis=1)
# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_val = pd.concat([num_X_val, OH_cols_val], axis=1)
# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_val.columns = OH_X_val.columns.astype(str)



kmeans= KMeans(n_clusters= 6, max_iter=300, n_init=5)


x_train_cluster=pd.DataFrame(OH_X_train)
x_train_cluster.columns = OH_X_train.columns
x_val_cluster=pd.DataFrame(OH_X_val)
x_val_cluster.columns = OH_X_val.columns


x_train_cluster["Cluster"] = kmeans.fit_predict(x_train_cluster)
x_train_cluster["Cluster"] = x_train_cluster["Cluster"].astype("int")
x_val_cluster["Cluster"] = kmeans.fit_predict(x_val_cluster)
x_val_cluster["Cluster"] = x_val_cluster["Cluster"].astype("int")


x_train_cluster


metric= mean_absolute_percentage_error


model_XGB=  XGBRegressor(n_estimators=100, learning_rate=0.05,early_stopping_rounds=10, random_state=0 )

model_XGB.fit(x_train_cluster, train_y,
           eval_set=[(x_val_cluster, val_y)]
)
pred_y_xbg = model_XGB.predict(x_val_cluster)

metric_1= metric(val_y,pred_y_xbg)
print(metric_1)

