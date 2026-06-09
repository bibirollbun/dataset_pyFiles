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


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



df=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


df.head()


df_test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_test.head()


x=df.drop(columns=["id","day","rainfall"],axis=1)
y=df["rainfall"]





x_submission=df_test.drop(columns=["id","day",],axis=1)


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.ensemble import RandomForestClassifier 
rfc=RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(x_train,y_train)



rfc.score(x_test,y_test)


from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)



xgb.score(x_test,y_test)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.transform(x_test)


from sklearn.ensemble import RandomForestClassifier 
rfc=RandomForestClassifier(n_estimators=100,criterion="entropy")
rfc.fit(x_train_scaled,y_train)



rfc.score(x_test_scaled,y_test)


from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train_scaled,y_train)



xgb.score(x_test_scaled,y_test)


rfc=RandomForestClassifier(n_estimators=100,criterion="entropy")
rfc.fit(x,y)


x_submission.info()


x_submission["winddirection"] = df["winddirection"].mean()  # parantezli: mean()

x_submission["winddirection"].fillna(df["winddirection"].mean(), inplace=True)
y_pred = rfc.predict(x_submission)
#77 public score


df_submission=pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


df_submission["rainfall"] = y_pred
df_submission.to_csv("submissionrainfall.csv", index=False)#76 public score


df_submission


import tensorflow as tf



from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2



ann=tf.keras.models.Sequential()
from keras.callbacks import ReduceLROnPlateau

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)


ann.add(layers.Dense(256,activation="relu"))
ann.add(Dropout(0.3)) 



ann.add(layers.Dense(256,activation="relu"))
ann.add(Dropout(0.3))  



ann.add(layers.Dense(512,activation="relu",kernel_regularizer=l2(0.001))), 
ann.add(Dropout(0.3))  



ann.add(layers.Dense(512,activation="relu", kernel_regularizer=l2(0.001)))
ann.add(Dropout(0.3))  



ann.add(layers.Dense(256,activation="relu",kernel_regularizer=l2(0.001)))
ann.add(Dropout(0.3)) 


ann.add(layers.Dense(128,activation="relu"))
ann.add(Dropout(0.3))  # 1. dropout



ann.add(layers.Dense(1,activation="sigmoid"))


from keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)

ann.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])



from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model=ann.fit(x,y,batch_size=32,epochs=50)


y_pred = ann.predict(x_submission)
df_submission["rainfall"] = y_pred
df_submission.to_csv("submissionrainfall4.csv", index=False)#85 public score 89 private score


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x, y)
y_pred = model.predict(x_submission)
df_submission["rainfall"] = y_pred
df_submission.to_csv("submissionrainfall5.csv", index=False)#77 public score


import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(x, y)
y_pred = model.predict(x_submission)
df_submission["rainfall"] = y_pred
df_submission.to_csv("submissionrainfall6.csv", index=False)#71 public score


from sklearn.svm import SVC
model = SVC(kernel='linear', random_state=42)
model.fit(x, y)
y_pred = model.predict(x_submission)
df_submission["rainfall"] = y_pred
df_submission.to_csv("submissionrainfall7.csv", index=False)#75 public score


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x, y)
y_pred = model.predict(x_submission)
df_submission["rainfall"] = y_pred
df_submission.to_csv("submissionrainfall8.csv", index=False)#62 public score


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Grid Search iÃ§in parametre aralÄ±ÄŸÄ±nÄ± belirleme
param_grid = {
    'n_estimators': [50, 100, 200],  # AÄŸaÃ§ sayÄ±sÄ±
    'learning_rate': [0.01, 0.1, 0.2],  # Ã–ÄŸrenme hÄ±zÄ±
    'max_depth': [3, 5, 7],  # AÄŸaÃ§ derinliÄŸi
    'subsample': [0.8, 1.0],  # Ã–rneklem oranÄ±
    'colsample_bytree': [0.8, 1.0]  # Her aÄŸaÃ§ iÃ§in sÃ¼tun Ã¶rnekleme oranÄ±
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)

grid_search.fit(x, y)

print("En iyi hiperparametreler:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_submission)

best_model.score(x_test,y_test)
df_submission["rainfall"] = y_pred
df_submission.to_csv("submissionrainfall9.csv", index=False)#76S public score


import shap

explainer = shap.KernelExplainer(ann.predict, x[:100])
shap_values = explainer.shap_values(x[:100])

# SHAP deÄŸerlerini gÃ¶rselleÅŸtirmek
shap.summary_plot(shap_values, x[:100])


x_shap=x=df.drop(columns=["id","day","rainfall","maxtemp","mintemp","pressure"],axis=0)



df


ann2=tf.keras.models.Sequential()
from keras.callbacks import ReduceLROnPlateau

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)


ann2.add(layers.Dense(256,activation="relu"))
ann2.add(Dropout(0.3)) 



ann2.add(layers.Dense(256,activation="relu"))
ann2.add(Dropout(0.3))  



ann2.add(layers.Dense(512,activation="relu",kernel_regularizer=l2(0.001))), 
ann2.add(Dropout(0.3))  



ann2.add(layers.Dense(512,activation="relu", kernel_regularizer=l2(0.001)))
ann2.add(Dropout(0.3))  



ann2.add(layers.Dense(256,activation="relu",kernel_regularizer=l2(0.001)))
ann2.add(Dropout(0.3)) 


ann2.add(layers.Dense(128,activation="relu"))
ann2.add(Dropout(0.3))  # 1. dropout



ann2.add(layers.Dense(1,activation="sigmoid"))


from keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)

ann2.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])



from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model=ann2.fit(x_shap,y,epochs=50)


y_pred = ann2.predict(x_submission.drop(columns=["maxtemp","mintemp","pressure"]))
df_submission["rainfall"] = y_pred
df_submission.to_csv("submissionrainfall12.csv", index=False)#85 public score 89 private score




