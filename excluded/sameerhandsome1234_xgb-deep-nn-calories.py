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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.preprocessing import StandardScaler


train_df=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission_df=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


def new_column(df):
    df['Cardio_Effort_Score'] = (df['Heart_Rate'] * df['Duration']) / df['Body_Temp']
    df['HR_Temp_Duration'] = df['Heart_Rate'] * df['Duration'] * df['Body_Temp']
new_column(train_df)
new_column(test_df)


def BMI_column(df):
    df['BMI'] = df['Weight'] / (df['Height']/100)**2
BMI_column(train_df)
BMI_column(test_df)


gender={
    'male':1,
    'female':0
}
train_df['Sex']=train_df['Sex'].map(gender)
test_df['Sex']=test_df['Sex'].map(gender)


features = ['Age', 'Duration', 'Heart_Rate', 'Body_Temp','BMI']


import itertools
def add_feature_cross_terms(df, features):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]  
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-5)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-5)
    return df_new


train_df = add_feature_cross_terms(train_df, features)
test_df = add_feature_cross_terms(test_df,features)
train_df = add_interaction_features(train_df, features)
test_df = add_interaction_features(test_df, features)


X=train_df.drop(columns=['id','Calories'])
y=train_df['Calories']
X_test=test_df.drop(columns=['id'])


numeric_cols=X.select_dtypes(include=np.number).columns.tolist()


scaler=StandardScaler()
scaler.fit(X[numeric_cols])
X[numeric_cols]=scaler.transform(X[numeric_cols])
X_test[numeric_cols]=scaler.transform(X_test[numeric_cols])


import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
    }

    model = XGBRegressor(**params, random_state=42, n_jobs=-1,tree_method="hist",device="cuda")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, None) 

        rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
        scores.append(rmsle)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)


params={'n_estimators': 752, 'max_depth': 10, 'learning_rate': 0.04480584309239781, 'subsample': 0.9059514929994765, 'colsample_bytree': 0.7284962415032582, 'gamma': 2.817896845923885, 'min_child_weight': 2}

model=XGBRegressor(**params,random_state=42,n_jobs=-1,tree_method="hist",device="cuda")
kf = KFold(n_splits=5, shuffle=True, random_state=42) 
global1=None
scores = []

for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, None)  
    global1= model.predict(X_test)
    global1= np.clip(global1,0,None)
    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
    scores.append(rmsle)
print(f"Cross Validation RMSLE: {scores}")



global2 = None
scores = []
desired_fold = 2 

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold == desired_fold:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, None)

        global2 = model.predict(X_test)
        global2 = np.clip(global2, 0, None)

        rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
        scores.append(rmsle)

print(f"Cross-validation RMSLE score: {scores}")


submission_df['Calories']=global2
submission_df.to_csv('csubmission16.csv',index=False)
from IPython.display import FileLink

FileLink("csubmission16.csv")


importance_dict = model.get_booster().get_score(importance_type='gain')

feature_importance_df = pd.DataFrame({
    'Feature': list(importance_dict.keys()),
    'Importance': list(importance_dict.values())
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

lowest_35 = feature_importance_df.tail(35)['Feature'].tolist()

X = X.drop(columns=lowest_35)
X_test = X_test.drop(columns=lowest_35)
X.head()


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

# Custom RMSLE loss using tf.math instead of K.log1p
def rmsle(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, tf.reduce_max(y_true))
    y_pred = tf.clip_by_value(y_pred, 0.0, tf.reduce_max(y_pred))
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))

# Define the model
def create_model(num_features):
    model = models.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model

# Create and compile the model
num_features = X_train.shape[1]
model = create_model(num_features)
model.compile(optimizer='adam', loss=rmsle)

# Fit the model
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)



global3=None
global3=model.predict(X_test)
global3= np.clip(global3,0,None)

submission_df['Calories']=global3
submission_df.to_csv('csubmission17.csv',index=False)
from IPython.display import FileLink

FileLink("csubmission17.csv")


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_model(num_features):
    he_init = 'he_normal'

    model = models.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(128, activation='relu', kernel_initializer=he_init),
        layers.Dense(64, activation='relu', kernel_initializer=he_init),
        layers.Dense(32, activation='relu', kernel_initializer=he_init),

        layers.Dense(1)
    ])
    return model


num_features = X_train.shape[1]
model = create_model(num_features)
model.compile(optimizer='adam', loss=rmsle)


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,           
    patience=2,         
    min_lr=1e-6,
    verbose=1
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=35,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)



global3=None
global3=model.predict(X_test)
global3= np.clip(global3,0,None)

submission_df['Calories']=global3
submission_df.to_csv('csubmission18.csv',index=False)
from IPython.display import FileLink

FileLink("csubmission18.csv")

