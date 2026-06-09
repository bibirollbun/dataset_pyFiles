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


#from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout,BatchNormalization
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau


df_train=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


num_cols=[col for col in df_train.columns if df_train[col].dtype=='int64']
num_cols.remove('id')
num_cols.remove('y')
obj_cols=[col for col in df_train.columns if df_train[col].dtype=='O']


#df_train['duration']=np.log(df_train['duration']+1)
#df_train['campaign']=np.log(df_train['campaign']+1)


df_train[obj_cols].nunique()
one_hot_cols=[col for col in obj_cols if (col!='job' and col!='month')]
ord_cols=['job','month']


X=df_train[num_cols+one_hot_cols+ord_cols]
y=df_train['y']


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,stratify=y)


oneHot_col_pipeline=Pipeline(steps=[("OneHot",OneHotEncoder(handle_unknown='ignore'))])
ord_col_pipeline=Pipeline(steps=[("Ordinal",OrdinalEncoder())])
preprocessor=ColumnTransformer(transformers=[('one_hot',oneHot_col_pipeline,one_hot_cols),
                                           ('ordinal',ord_col_pipeline,ord_cols)],remainder='passthrough')


X_train=preprocessor.fit_transform(X_train)


X_val=preprocessor.transform(X_val)


X_train.shape,X_val.shape


model=Sequential()
model.add(Input((29,)))
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Dense(16,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',AUC(name='AUC')])
callbacks=[EarlyStopping(monitor='val_AUC',patience=4,restore_best_weights=True,verbose=1),
          ReduceLROnPlateau(monitor='val_AUC',patience=3,factor=0.5,verbose=1)]


model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=30,callbacks=callbacks,verbose=1)


model.save("/kaggle/working/neural_playground.h5")


df_test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


X_test=df_test[num_cols+one_hot_cols+ord_cols]


X_test=preprocessor.transform(X_test)


y_pred=model.predict(X_test)


y_pred=y_pred.flatten()


y_pred.shape


res=pd.DataFrame({'id':df_test['id'].values,'y':y_pred})


res.to_csv("more_complex_normalized.csv",index=False)


from sklearn.metrics import roc_auc_score


y_pred=model.predict(X_val).flatten()


roc_score=roc_auc_score(y_val,y_pred)


print(f"The roc score to this is {roc_score*100}%")


df_train[one_hot_cols+ord_cols].nunique().sum()+len(num_cols)


df_train['Duration_per_camp']=df_train['campaign']/df_train['duration']
df_test['Duration_per_camp']=df_test['campaign']/df_test['duration']


num_cols=num_cols+['Duration_per_camp']


X=df_train[num_cols+one_hot_cols+ord_cols]
y=df_train['y']


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,stratify=y)


one_hot_cols2=one_hot_cols+ord_cols


print(f"{len(num_cols)+len(one_hot_cols2)} and dataframe number of cols {len(df_train.columns)}")


from sklearn.preprocessing import MinMaxScaler


oneHot_col_pipeline=Pipeline(steps=[("OneHot",OneHotEncoder(handle_unknown='ignore'))])
preprocessor2=ColumnTransformer(transformers=[('one_hot',oneHot_col_pipeline,one_hot_cols2),('numerical',MinMaxScaler(),num_cols)],remainder='passthrough')


X_train=preprocessor2.fit_transform(X_train)


X_val=preprocessor2.transform(X_val)


model2=Sequential()
model2.add(Input((52,)))
model2.add(Dense(256,activation='relu'))
model2.add(BatchNormalization())
#model.add(Dropout(0.2))
model2.add(Dense(128,activation='relu'))
model2.add(BatchNormalization())
#model.add(Dropout(0.2))
model2.add(Dense(64,activation='relu'))
model2.add(BatchNormalization())
#model.add(Dropout(0.2))
model2.add(Dense(32,activation='relu'))
model2.add(BatchNormalization())
#model.add(Dropout(0.2))
model2.add(Dense(16,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dense(1,activation='sigmoid'))
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',AUC(name='AUC')])
callbacks=[EarlyStopping(monitor='val_AUC',patience=4,restore_best_weights=True,verbose=1),
          ReduceLROnPlateau(monitor='val_AUC',patience=3,factor=0.5,verbose=1)]


model2.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=10,callbacks=callbacks,verbose=1)


df_test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df_test['Duration_per_camp']=df_test['campaign']/df_test['duration']
tot=one_hot_cols2+num_cols
X_test=df_test[tot]
X_test=preprocessor2.transform(X_test)
y_pred=model2.predict(X_test).flatten()
res=pd.DataFrame({'id':df_test['id'].values,'y':y_pred})
res.to_csv("more_complex_normalized2.csv",index=False)


y_pred=model2.predict(X_val).flatten()
roc_score=roc_auc_score(y_val,y_pred)
print(f"The roc score to this is {roc_score*100}%")


df_train.nunique()

