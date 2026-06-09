import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error 
from sklearn.cluster import KMeans

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MSE


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df.head()


df.info()


OHE= OneHotEncoder(handle_unknown='ignore', sparse_output=False)


df.drop(['num_lanes', 'time_of_day','road_type'],axis=1)


s = (df.dtypes == 'object') | (df.dtypes == 'bool') 
object_cols = list(s[s].index)



df[object_cols]



OH_cols = pd.DataFrame(OHE.fit_transform(df[object_cols]))

# One-hot encoding removed index; put it back
OH_cols.index = df.index

# Remove categorical columns (will replace with one-hot encoding)
num_df = df.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_df = pd.concat([num_df, OH_cols], axis=1)



kmeans= KMeans(n_clusters= 6, max_iter=300, n_init=5)


X=OH_df.drop(['id', 'accident_risk'],axis=1)
y=OH_df.pop('accident_risk')


X_train, X_val, train_y, val_y = train_test_split(X, y, random_state = 0)


kmeans= KMeans(n_clusters= 6, max_iter=300, n_init=5)


train_X=pd.DataFrame(X_train)
train_X.columns = X_train.columns.astype(str)
val_X=pd.DataFrame(X_val)
val_X.columns = X_val.columns.astype(str)


train_X["Cluster"] = kmeans.fit_predict(train_X)
train_X["Cluster"] = train_X["Cluster"].astype("int")
val_X["Cluster"] = kmeans.fit_predict(val_X)
val_X["Cluster"] = val_X["Cluster"].astype("int")


train_X


model=keras.Sequential([
    layers.Dense(units=25,input_shape=(25,)),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.5),
    layers.Dense(units=12, activation= 'relu'),
    layers.Dense(units=6, activation= 'relu'),
    layers.Dense (units=1, activation= 'sigmoid')
                 
])

model.compile(loss = "MSE",
                  optimizer = Adam(learning_rate=0.01),
                  metrics=[keras.metrics.RootMeanSquaredError()]
                  )

early_stopping = EarlyStopping(
  	   min_delta=0.001, # minimium amount of change to count as an improvement
   	  patience=15, # how many epochs to wait before stopping
   	  restore_best_weights=True,
)
history = model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    batch_size=128,
    epochs=100,
    callbacks=[early_stopping], # put your callbacks in a list
)


history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['root_mean_squared_error', 'val_root_mean_squared_error']].plot();


history_frame


df_test.drop(['num_lanes', 'time_of_day','road_type'],axis=1)


OH_cols_test = pd.DataFrame(OHE.fit_transform(df_test[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_test.index = df_test.index

# Remove categorical columns (will replace with one-hot encoding)
num_df_test = df_test.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_df_test_1 = pd.concat([num_df_test, OH_cols_test], axis=1)



id=OH_df_test_1.pop('id')


OH_df_test=pd.DataFrame(OH_df_test_1)
OH_df_test.columns = OH_df_test_1.columns.astype(str)
OH_df_test["Cluster"] = kmeans.fit_predict(OH_df_test)
OH_df_test["Cluster"] = OH_df_test["Cluster"].astype("int")


OH_df_test


prediction=model.predict(OH_df_test)


predictions = prediction.flatten()


output = pd.DataFrame({ 'id':id,
                       'Target': predictions})


output.set_index('id')


output.to_csv('submission.csv', index=False)

