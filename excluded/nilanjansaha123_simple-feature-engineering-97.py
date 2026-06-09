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


df_train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")



df_train.isnull().sum()


df_train.head()


df_train.shape


df_train.dtypes


drained_list=df_train['Drained_after_socializing'].unique()
stage_fear_list=df_train['Stage_fear'].unique()
personality_list=df_train['Personality'].unique()
drained_map={}
stage_fear_map={}
personality_map={}
for i in range(len(drained_list)):
    drained_map[drained_list[i]]=i
for i in range(len(stage_fear_list)):
    stage_fear_map[stage_fear_list[i]]=i
for i in range(len(personality_list)):
    personality_map[personality_list[i]]=i


df_train['Drained_after_socializing']=df_train['Drained_after_socializing'].map(drained_map)
df_train['Stage_fear']=df_train['Stage_fear'].map(stage_fear_map)
df_train['Personality']=df_train['Personality'].map(personality_map)


df_train[['Drained_after_socializing','Stage_fear','Personality']].isnull().sum()


def feature_engineering(df_train):
    df_train['Time_spent_alone_n_going_outside_interaction']=df_train['Time_spent_Alone']*df_train['Going_outside']
    df_train['Time_spent_alone_n_friend_circle_size_interaction']=df_train['Time_spent_Alone']*df_train['Friends_circle_size']
    df_train['Post_frequency_n_friend_circle_size_interaction']=df_train['Friends_circle_size']*df_train['Post_frequency']
    df_train['Social_event_attendance_n_Going_outside_interaction']=df_train['Going_outside']*df_train['Social_event_attendance']
    return df_train


df_train=feature_engineering(df_train)


df_train.dtypes


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
df_train=df_train.drop('id',axis=1)
one_hot_cols=['Stage_fear','Drained_after_socializing']
num_cols=[col for col in df_train.columns if df_train[col].dtype=='float64']
numerical_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='mean'))])
one_hot_pipeline=Pipeline(steps=[('hot_encoder',OneHotEncoder(handle_unknown='ignore'))])
preprocessor=ColumnTransformer(transformers=[('num',numerical_pipeline,num_cols),
                                            ('one_hot',one_hot_pipeline,one_hot_cols)],remainder='passthrough')


X=df_train.drop('Personality',axis=1)
y=df_train['Personality']


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,stratify=y)


X_train=preprocessor.fit_transform(X_train)
X_val=preprocessor.transform(X_val)


from xgboost import XGBClassifier
model=XGBClassifier()


model.fit(X_train,y_train)
y_pred=model.predict(X_val)


from sklearn.metrics import accuracy_score
print(f"The accuracy is {accuracy_score(y_val,y_pred)*100}%")


import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def objective(trial):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    # Train model on training set
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, preds)

    return val_acc  # Optuna will maximize this

# Create and run the study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print best result
print("Best Validation Accuracy:", study.best_value)
print("Best Parameters:", study.best_params)


parameters={'n_estimators': 335, 'max_depth': 12, 'learning_rate': 0.04942516926141748, 'subsample': 0.9598002661899309, 'colsample_bytree': 0.7806153773956759, 'gamma': 3.2010580121965413, 'reg_alpha': 2.495372711382577, 'reg_lambda': 0.047632520605919715}
final_model=XGBClassifier(**parameters)


df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
ids=df_test['id']
df_test=df_test.drop('id',axis=1)
df_test['Drained_after_socializing']=df_test['Drained_after_socializing'].map(drained_map)
df_test['Stage_fear']=df_test['Stage_fear'].map(stage_fear_map)
df_test=feature_engineering(df_test)
X_test=df_test
X_test=preprocessor.transform(X_test)


final_model.fit(X_train,y_train)
y_test_pred=final_model.predict(X_test)


personality_map


y_test_final=[]
for i in range(len(y_test_pred)):
    if (y_test_pred[i]==0):
        y_test_final.append("Extrovert")
    else:
        y_test_final.append("Introvert")


y_test_final=np.array(y_test_final)


res=pd.DataFrame({'id':ids,'Personality':y_test_final})


res.to_csv("personality_res.csv",index=False)


from tensorflow.keras.layers import Input,Dense,BatchNormalization
from sklearn.preprocessing import MinMaxScaler


df_train2=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test2=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df_train2['Drained_after_socializing']=df_train2['Drained_after_socializing'].map(drained_map)
df_train2['Stage_fear']=df_train2['Stage_fear'].map(stage_fear_map)
df_train2['Personality']=df_train2['Personality'].map(personality_map)
df_train2=feature_engineering(df_train2)
df_test2=feature_engineering(df_test2)


df_train2=df_train2.drop('id',axis=1)
one_hot_cols=['Stage_fear','Drained_after_socializing']
num_cols=[col for col in df_train2.columns if df_train2[col].dtype=='float64']
scaler=MinMaxScaler()
numerical_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='mean')),
                                  ('scaler',scaler)])
one_hot_pipeline=Pipeline(steps=[('hot_encoder',OneHotEncoder(handle_unknown='ignore'))])
preprocessor2=ColumnTransformer(transformers=[('num',numerical_pipeline,num_cols),
                                            ('one_hot',one_hot_pipeline,one_hot_cols)],remainder='passthrough')


X=df_train2.drop('Personality',axis=1)
y=df_train2['Personality']


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,stratify=y)


X_train=preprocessor2.fit_transform(X_train)
X_val=preprocessor2.transform(X_val)


X_train.shape,X_val.shape


from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau


model2=Sequential()
model2.add(Input(shape=(15,)))
model2.add(Dense(256,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dense(128,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dense(64,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dense(32,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dense(16,activation='relu'))
model2.add(Dense(1,activation='sigmoid'))
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


callbacks=[EarlyStopping(patience=5,monitor='val_accuracy',restore_best_weights=True,mode='max',verbose=1),
          ReduceLROnPlateau(patience=3,monitor='val_accuracy',factor=0.5,verbose=1)]


model2.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=15,callbacks=callbacks,verbose=1)


df_test2=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
ids=df_test2['id']
df_test2=df_test2.drop('id',axis=1)
df_test2['Drained_after_socializing']=df_test2['Drained_after_socializing'].map(drained_map)
df_test2['Stage_fear']=df_test2['Stage_fear'].map(stage_fear_map)
df_test2=feature_engineering(df_test2)
X_test=df_test2
X_test=preprocessor2.transform(X_test)


y_test_pred2=model2.predict(X_test)


y_temp=(y_test_pred2>0.5).astype(int)
y_test_pred2=y_temp


y_test_final2=[]
for i in range(len(y_test_pred2)):
    if (y_test_pred2[i]==0):
        y_test_final2.append("Extrovert")
    else:
        y_test_final2.append("Introvert")
y_test_final2=np.array(y_test_final2)
res2=pd.DataFrame({'id':ids,'Personality':y_test_final2})
res2.to_csv("personality_res4.csv",index=False)

